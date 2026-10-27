# GLAM unified local mapping:
# - Focal reference mask (`window_radius_voxels` from MapWindowSizeCM): physical sphere in mm
#   (axis-aligned ellipsoid in voxel index space when voxel spacing is anisotropic).
# - Extended target/normalization mask (`ext_radii = [max(int(r), MapRDFMaxRadius) for r in window_radius_voxels]`):
#   node-centered axis-aligned ellipsoid in voxel index space (sphere when radii are equal)
#   bounding RDF target shells, shell denominators, rho_beta normalization, CoordNum/SPI densities,
#   and spatial label permutations in CDI across both CPU and GPU.
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from .config import get_config, load_config
from . import core
from .utils import reformat_dict_to_matrix
from .rdf_fast import calculate_rdf_3d as _fast_calculate_rdf_3d
from .mapping_profile import profile_mapping, limit_nodes, allow_interpolation, stage

worker_quantized_array = None
worker_mask_array = None
worker_settings = {}
worker_sphere_mask = None
worker_ext_sphere_mask = None
worker_ext_radii = None
worker_canonical_focal_in_ext = None


def _get_spherical_mask(radius_voxels):
    """Build an axis-aligned ellipsoidal mask in voxel space with per-axis semi-axes
    `radius_voxels = (rz, ry, rx)` (a sphere in voxel space when rz == ry == rx).
    """
    rz, ry, rx = radius_voxels
    z, y, x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return ((z/max(rz,1))**2 + (y/max(ry,1))**2 + (x/max(rx,1))**2) <= 1


def _get_extended_radii(window_radius_voxels, map_max_radius):
    """Compute per-axis extended target semi-axes `max(r_axis, map_max_radius)`.
    Produces a voxel-space sphere when `map_max_radius >= max(window_radius_voxels)`
    and an axis-aligned ellipsoid when anisotropic `window_radius_voxels` exceed `map_max_radius`.
    """
    r_max = int(map_max_radius)
    return [max(int(r), r_max) for r in window_radius_voxels]


def _get_canonical_focal_in_ext(focal_radii, focal_sphere, ext_radii, ext_sphere):
    """Build the fixed canonical focal reference mask embedded at the center of `ext_sphere.shape`."""
    out = np.zeros(ext_sphere.shape, dtype=bool)
    sl = tuple(slice(int(re) - int(rf), int(re) + int(rf) + 1) for rf, re in zip(focal_radii, ext_radii))
    out[sl] = focal_sphere & ext_sphere[sl]
    return out


def _init_worker_mapping(quantized_array, mask_array, settings_dict, config_path):
    global worker_quantized_array, worker_mask_array, worker_settings, worker_sphere_mask
    global worker_ext_sphere_mask, worker_ext_radii, worker_canonical_focal_in_ext
    load_config(config_path)
    core.HAS_GPU = False
    worker_quantized_array = quantized_array
    worker_mask_array = mask_array
    worker_settings = settings_dict
    worker_sphere_mask = _get_spherical_mask(settings_dict['window_radius_voxels'])
    worker_ext_radii = _get_extended_radii(settings_dict['window_radius_voxels'], settings_dict['map_max_radius'])
    worker_ext_sphere_mask = _get_spherical_mask(worker_ext_radii)
    worker_canonical_focal_in_ext = _get_canonical_focal_in_ext(
        settings_dict['window_radius_voxels'], worker_sphere_mask, worker_ext_radii, worker_ext_sphere_mask
    )


def _calculate_local_meta_feature(matrix, method="Mean"):
    if matrix is None:
        return np.nan
    matrix = np.asarray(matrix, dtype=float)
    if method == "DiagMean":
        values = np.diag(matrix)
    elif method == "OffDiagMean":
        values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    else:
        values = matrix.ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    if method in ("Mean", "DiagMean", "OffDiagMean"):
        return float(np.mean(values))
    if method == "Variance":
        return float(np.var(values))
    if method == "Maximum":
        return float(np.max(values))
    if method == "Energy":
        return float(np.sum(values**2))
    raise ValueError(f"Unsupported mapping meta method: {method}")


def _extract_focal_and_target_window(quantized_array, mask_array, coord, focal_radii, focal_sphere, ext_radii, ext_sphere):
    """Extract:
    1. `focal_patch, focal_mask` on the clipped focal box `focal_radii` (preserving exact box-counting
       dimensions for FractalDimension & min_voxels check).
    2. `ext_patch, ext_mask, focal_in_ext` embedded in the fixed canonical box `ext_sphere.shape`
       `(2*rz_e+1, 2*ry_e+1, 2*rx_e+1)` (with out-of-volume regions masked to False / -1),
       so every node in a scan shares the exact same extended array shape and canonical support masks.
    """
    z, y, x = map(int, coord)
    dims = mask_array.shape
    centers = (z, y, x)

    f_starts = [max(0, c - r) for c, r in zip(centers, focal_radii)]
    f_ends = [min(d, c + r + 1) for d, c, r in zip(dims, centers, focal_radii)]
    f_slices = tuple(slice(a, b) for a, b in zip(f_starts, f_ends))
    f_sph_starts = [a - (c - r) for a, c, r in zip(f_starts, centers, focal_radii)]
    f_sph_slices = tuple(slice(a, a + b - c) for a, b, c in zip(f_sph_starts, f_ends, f_starts))
    focal_mask = (mask_array[f_slices] > 0) & focal_sphere[f_sph_slices]
    focal_patch = np.full(focal_mask.shape, -1, dtype=np.int16)
    focal_patch[focal_mask] = quantized_array[f_slices][focal_mask]

    e_starts = [max(0, c - r) for c, r in zip(centers, ext_radii)]
    e_ends = [min(d, c + r + 1) for d, c, r in zip(dims, centers, ext_radii)]
    e_slices = tuple(slice(a, b) for a, b in zip(e_starts, e_ends))
    e_sph_starts = [a - (c - r) for a, c, r in zip(e_starts, centers, ext_radii)]
    e_sph_slices = tuple(slice(a, a + b - c) for a, b, c in zip(e_sph_starts, e_ends, e_starts))

    sub_ext_mask = (mask_array[e_slices] > 0) & ext_sphere[e_sph_slices]
    if sub_ext_mask.shape == ext_sphere.shape:
        ext_mask = sub_ext_mask
        ext_patch = np.full(ext_sphere.shape, -1, dtype=np.int16)
        ext_patch[ext_mask] = quantized_array[e_slices][ext_mask]
    else:
        ext_mask = np.zeros(ext_sphere.shape, dtype=bool)
        ext_patch = np.full(ext_sphere.shape, -1, dtype=np.int16)
        ext_mask[e_sph_slices] = sub_ext_mask
        sub_q = quantized_array[e_slices]
        ext_patch[e_sph_slices] = np.where(sub_ext_mask, sub_q, -1)

    focal_in_ext = np.zeros(ext_sphere.shape, dtype=bool)
    f_in_e_slices = tuple(
        slice(es_start + (fs - es), es_start + (fe - es))
        for es_start, fs, fe, es in zip(e_sph_starts, f_starts, f_ends, e_starts)
    )
    focal_in_ext[f_in_e_slices] = focal_mask

    return focal_patch, focal_mask, ext_patch, ext_mask, focal_in_ext


def _local_random_rdf(patch, sample_mask, levels, radius, counts, total, samples, repeats):
    """Shuffle labels across the active target domain (`patch >= 0`, or `sample_mask` if
    `patch >= 0` is not masked) while restricting references to `sample_mask`.
    Averages strictly over supported replicates at each (pair, radius) via
    `average_supported_rdf_replicates` so unsupported zero tails never dilute the baseline.
    """
    from .rdf_smoothing import average_supported_rdf_replicates
    if repeats < 1:
        raise ValueError("Local RDF baseline needs at least one randomization")
    nonneg = patch >= 0
    sm = np.asarray(sample_mask) > 0
    target_roi = nonneg if (int(nonneg.sum()) == int(total) and np.all(sm <= nonneg)) else sm
    values = patch[target_roi]
    frames = []
    support_masks = []
    for _ in range(repeats):
        shuffled = np.full(patch.shape, -1, dtype=np.int16)
        shuffled[target_roi] = np.random.permutation(values)
        frame = _fast_calculate_rdf_3d(
            shuffled, levels, radius, list(counts), total, 1, samples,
            sample_mask=sm)
        if frame is None or frame.empty:
            raise RuntimeError("Empty local randomized RDF")
        frames.append(frame)
        if isinstance(getattr(frame, 'attrs', None), dict) and '_glam_support_mask' in frame.attrs:
            support_masks.append(np.asarray(frame.attrs['_glam_support_mask'], dtype=bool))
    cols = list(frames[0].columns)
    g_cols = [c for c in cols if c.startswith('g_')]
    r_vec = frames[0]['r'].to_numpy(dtype=np.float64, copy=True)
    for frame in frames[1:]:
        if not np.array_equal(frame['r'].values, r_vec):
            raise RuntimeError("Local random RDF radii differ")
    vals_stack = np.stack([f[g_cols].to_numpy(dtype=np.float64) for f in frames], axis=0)
    sup_stack = np.stack(support_masks, axis=0) if len(support_masks) == repeats else None
    return average_supported_rdf_replicates(r_vec, cols, vals_stack, sup_stack)


_BASELINE_REQUIRED_BASES = {
    'ConfigurationalDisorderIndex',
    'Wasserstein',
    'AssemblyCoupling',
    'B2',
    'InverseCorrelationLength',
    'FrustrationIndex',
}

_LOG1P_LN_BASES = {
    'Wasserstein',
    'Betti0',
    'Betti1',
    'Betti2',
    'Shape_CentroidDist',
    'Shape_InterfaceArea',
    'MaxClusterSize',
    'GranulometryEntropy',
}


def _strip_feature_transform(feature_name):
    for suffix in ('_Symlog', '_Log1p', '_Ln'):
        if feature_name.endswith(suffix):
            return feature_name[:-len(suffix)], suffix
    return feature_name, ''


def _process_single_voxel_worker(coords_z_y_x, *, _precomputed_node=None, _precomputed_scalars=None):
    try:
        z, y, x = map(int, coords_z_y_x)
        s = worker_settings
        levels = s['num_gray_levels']
        radii = s['window_radius_voxels']
        ext_radii = worker_ext_radii if worker_ext_radii is not None else _get_extended_radii(radii, s['map_max_radius'])
        ext_sphere = worker_ext_sphere_mask if worker_ext_sphere_mask is not None else _get_spherical_mask(ext_radii)

        pre_scalars = _precomputed_scalars or {}
        remaining_features = [
            f for f in s['features_to_map']
            if not all(f'{f}_{m}' in pre_scalars for m in s['meta_method'])
        ]
        if not remaining_features and pre_scalars:
            out = {f'{f}_{m}': float(pre_scalars[f'{f}_{m}']) for f in s['features_to_map'] for m in s['meta_method']}
            return (z, y, x, out)

        bases = {_strip_feature_transform(f)[0] for f in remaining_features}
        needs_baseline = bool(bases & _BASELINE_REQUIRED_BASES)

        if _precomputed_node is None:
            focal_patch, focal_mask, ext_patch, ext_mask, focal_in_ext = _extract_focal_and_target_window(
                worker_quantized_array, worker_mask_array, (z, y, x),
                radii, worker_sphere_mask, ext_radii, ext_sphere
            )
            total_focal = int(focal_mask.sum())
            if total_focal < s['min_voxels']:
                return (z, y, x, None)
            ext_values = ext_patch[ext_mask]
            if np.any(ext_values < 0) or np.any(ext_values >= levels):
                raise ValueError('Invalid local gray levels; check crop reconstruction')
            ext_counts = np.bincount(ext_values, minlength=levels).tolist()
            ext_total = int(ext_mask.sum())
            seed = np.random.SeedSequence([42, z, y, x])
            np.random.seed(int(seed.generate_state(1)[0]))
            rdf = _fast_calculate_rdf_3d(
                ext_patch, levels, s['map_max_radius'], list(ext_counts),
                ext_total, 1, s['map_rdf_samples'], sample_mask=focal_in_ext
            )
            if rdf is None or rdf.empty:
                return (z, y, x, None)
            baseline = None
            if needs_baseline:
                baseline = _local_random_rdf(
                    ext_patch, focal_in_ext, levels, s['map_max_radius'],
                    ext_counts, ext_total, s['map_rdf_samples'], s['randomizations']
                )
        else:
            focal_patch, rdf, baseline, ext_counts, ext_total = _precomputed_node

        if rdf is None or rdf.empty:
            return (z, y, x, None)

        rdf_shape_names = {
            'RDF_PeakPosition', 'RDF_DispersionRatio', 'LogRDF_Median',
            'LogRDF_Variance', 'LogRDF_Skewness', 'LogRDF_Kurtosis', 'LogRDF_PeakHeight',
        }
        matrices = core.calculate_rdf_shape_matrices(rdf, levels) if bool(bases & rdf_shape_names) else {}
        absent_ext = np.asarray(ext_counts) <= 0

        if 'PotentialEnergy' in bases or 'EnergyDensity' in bases:
            pe_dict = core.calculate_glam_potential_energy(rdf, levels)
            pe_mat = reformat_dict_to_matrix(pe_dict, levels, 'GLAM_PotentialEnergy_', None)
            if 'PotentialEnergy' in bases:
                matrices['PotentialEnergy'] = pe_mat
            if 'EnergyDensity' in bases:
                matrices['EnergyDensity'] = (
                    pe_mat / float(ext_total)
                    if (pe_mat is not None and ext_total > 0)
                    else np.full((levels, levels), np.nan)
                )

        spi_dict = None
        if 'StructuralPressureIndex' in bases or 'FrustrationIndex' in bases:
            spi_dict = core.calculate_glam_structural_pressure_index(rdf, levels, ext_counts, ext_total)
            if 'StructuralPressureIndex' in bases:
                matrices['StructuralPressureIndex'] = reformat_dict_to_matrix(
                    spi_dict, levels, 'GLAM_StructuralPressureIndex_', None
                )

        cdi_dict = None
        if 'ConfigurationalDisorderIndex' in bases or 'FrustrationIndex' in bases:
            if baseline is None:
                raise RuntimeError('Missing randomized baseline RDF for CDI/FrustrationIndex')
            cdi_dict = core.calculate_configurational_disorder_index(rdf, baseline, levels)
            if 'ConfigurationalDisorderIndex' in bases:
                matrices['ConfigurationalDisorderIndex'] = reformat_dict_to_matrix(
                    cdi_dict, levels, 'GLAM_ConfigurationalDisorderIndex_', None
                )

        if 'FrustrationIndex' in bases:
            frust_dict = core.calculate_glam_frustration_index(spi_dict, cdi_dict, levels)
            matrices['FrustrationIndex'] = reformat_dict_to_matrix(
                frust_dict, levels, 'GLAM_FrustrationIndex_', None
            )

        if 'Wasserstein' in bases or 'AssemblyCoupling' in bases:
            if baseline is None:
                raise RuntimeError('Missing randomized baseline RDF for Wasserstein/AssemblyCoupling')
            wass_dict = core.calculate_glam_wasserstein_distance(rdf, baseline, levels, ext_counts, ext_total)
            if 'Wasserstein' in bases:
                matrices['Wasserstein'] = reformat_dict_to_matrix(
                    wass_dict, levels, 'GLAM_Wasserstein_', None
                )
            if 'AssemblyCoupling' in bases:
                ac_dict = core.calculate_glam_assembly_coupling_matrix(wass_dict, levels)
                matrices['AssemblyCoupling'] = reformat_dict_to_matrix(
                    ac_dict, levels, 'GLAM_AssemblyCoupling_', None
                )

        if 'PhenotypicDistance' in bases:
            pd_dict = core.calculate_glam_phenotypic_distance_matrix(rdf, levels, ext_counts, ext_total)
            pd_mat = reformat_dict_to_matrix(pd_dict, levels, 'GLAM_PhenotypicDistance_', None)
            if pd_mat is not None:
                np.fill_diagonal(pd_mat, np.nan)
            matrices['PhenotypicDistance'] = pd_mat

        if 'Anisotropy' in bases:
            aniso_cut = s.get('anisotropy_cutoff_radius', 5)
            aniso_dict = core.calculate_anisotropic_glam_features(focal_patch, levels, aniso_cut)
            matrices['Anisotropy'] = reformat_dict_to_matrix(
                aniso_dict, levels, 'GLAM_Anisotropy_', None
            )

        calls = {
            'CoordNum': lambda: reformat_dict_to_matrix(
                core.calculate_glam_coordination_number(rdf, levels, ext_counts, ext_total),
                levels, 'GLAM_CoordNum_', None
            ),
            'Compressibility': lambda: reformat_dict_to_matrix(
                core.calculate_glam_compressibility(rdf, levels),
                levels, 'GLAM_Compressibility_', None
            ),
            'LocalPackingFraction': lambda: reformat_dict_to_matrix(
                core.calculate_glam_local_packing_fraction(rdf, levels, ext_counts, ext_total),
                levels, 'GLAM_LocalPackingFraction_', None
            ),
            'B2': lambda: reformat_dict_to_matrix(
                core.calculate_glam_b2_3d(rdf, baseline, levels),
                levels, 'GLAM_B2_for_', None
            ),
            'InverseCorrelationLength': lambda: reformat_dict_to_matrix(
                core.calculate_glam_inverse_correlation_length(rdf, baseline, levels),
                levels, 'GLAM_InverseCorrelationLength_', None
            ),
            'FractalDimension': lambda: reformat_dict_to_matrix(
                core.calculate_glam_fractal_dimension(focal_patch, levels),
                levels, 'GLAM_InterfaceFD_', 'GLAM_VolumeFD_'
            ),
        }
        for name, call in calls.items():
            if name in bases:
                matrices[name] = call()

        result = dict(pre_scalars)
        for feature in remaining_features:
            base, suffix = _strip_feature_transform(feature)
            if base not in matrices:
                raise ValueError(f'Unsupported MapFeatures entry: {feature}')
            matrix = np.asarray(matrices[base], dtype=float).copy()
            if base not in ('FractalDimension', 'Anisotropy') and np.any(absent_ext):
                matrix[absent_ext, :] = np.nan
                matrix[:, absent_ext] = np.nan

            if suffix == '_Symlog':
                matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
            elif suffix == '_Log1p' or (suffix == '_Ln' and base in _LOG1P_LN_BASES):
                with np.errstate(divide='ignore', invalid='ignore'):
                    matrix = np.log1p(np.maximum(matrix, 0.0))
                matrix[~np.isfinite(matrix)] = np.nan
            elif suffix == '_Ln':
                with np.errstate(divide='ignore', invalid='ignore'):
                    matrix = np.log(matrix)
                matrix[~np.isfinite(matrix)] = np.nan
            for method in s['meta_method']:
                result[f'{feature}_{method}'] = _calculate_local_meta_feature(matrix, method)

        ordered = {f'{f}_{m}': float(result[f'{f}_{m}']) for f in s['features_to_map'] for m in s['meta_method']}
        return (z, y, x, ordered)
    except Exception as exc:
        raise RuntimeError(f'Local mapping failed at node {tuple(coords_z_y_x)}: {exc}') from exc



def _scale_array_to_uint8_robust(array, valid_mask=None):
    """Scale finite voxels into 1..255 uint8 using robust [0.5, 99.5] percentiles
    so isolated boundary outlier spikes cannot flatten the 8-bit visualization map.
    """
    valid = np.isfinite(array) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & np.isfinite(array))
    scaled = np.zeros(array.shape, dtype=np.uint8)
    if not np.any(valid):
        return scaled
    vals = array[valid]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-6:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if (hi - lo) > 1e-6:
        clipped = np.clip(vals, lo, hi)
        scaled[valid] = (1 + 254.0 * (clipped - lo) / (hi - lo)).astype(np.uint8)
    else:
        scaled[valid] = 128
    return scaled


@profile_mapping
def generate_feature_maps(image_sitk, binary_mask_sitk, quantized_image_array,
                          num_gray_levels, prefix, output_dir, config_path):
    print("  --- Starting 3D Feature Map Generation ---")
    workers = int(get_config('NumWorkers'))
    window_cm = float(get_config('MapWindowSizeCM'))
    features = get_config('MapFeatures')
    methods = get_config('MapMetaMethod')
    repeats = int(get_config('NumRandomisations'))
    radius = int(get_config('MapRDFMaxRadius'))
    samples = int(get_config('MapRDFSamplePoints'))
    min_voxels = int(get_config('MapMinWindowVoxels'))
    if min(workers, repeats, radius, samples, min_voxels) < 1 or window_cm <= 0:
        raise ValueError("Invalid mapping settings")
    if not isinstance(features, list) or not isinstance(methods, list) or not features or not methods:
        raise ValueError("MapFeatures and MapMetaMethod must be nonempty lists")
    mask = sitk.GetArrayFromImage(binary_mask_sitk)
    if mask.shape != quantized_image_array.shape:
        raise ValueError("Mapping image/mask shape mismatch")
    spacing = np.asarray(image_sitk.GetSpacing()[::-1], dtype=float)
    radii = np.ceil(window_cm * 5.0 / spacing).astype(int).tolist()
    overlap = float(np.clip(get_config('MapOverlapPercent'), 0, 95))
    strides = [max(1, int(2 * r * (1 - overlap / 100))) for r in radii]
    grid = np.meshgrid(*(range(0, d, t) for d, t in zip(mask.shape, strides)), indexing='ij')
    coords = np.stack([g.ravel() for g in grid], axis=1)
    coords = coords[mask[tuple(coords.T)] > 0]
    coords = limit_nodes(coords)
    if not len(coords):
        print("  - No grid sampling points inside the target mask.")
        return
    aniso_r = int(get_config('AnisotropyCutoffRadius'))
    settings = dict(num_gray_levels=num_gray_levels, window_radius_voxels=radii, anisotropy_cutoff_radius=aniso_r,
                    min_voxels=min_voxels, map_max_radius=radius, map_rdf_samples=samples,
                    features_to_map=features, meta_method=methods, randomizations=repeats)
    maps = {f'{f}_{m}': np.full(mask.shape, np.nan, dtype=np.float32) for f in features for m in methods}
    print(f"  - Processing {len(coords)} nodes; local CDI uses {repeats} shuffled RDFs per node.")
    backend = get_config('MapBackend').lower()
    if backend == 'gpu':
        import sys
        from .mapping_gpu import gpu_results
        with stage('map_nodes_gpu', cuda=True):
            results = gpu_results(sys.modules[__name__], coords, quantized_image_array, mask, settings, config_path)
            for z, y, x, values in tqdm(results, total=len(coords)):
                if values is not None:
                    for key, value in values.items():
                        maps[key][z, y, x] = value
    elif backend == 'cpu':
        with stage('map_nodes_cpu', cuda=False):
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_mapping,
                                     initargs=(quantized_image_array, mask, settings, config_path)) as executor:
                for z, y, x, values in tqdm(executor.map(_process_single_voxel_worker, coords, chunksize=1), total=len(coords)):
                    if values is not None:
                        for key, value in values.items():
                            maps[key][z, y, x] = value
    else:
        raise ValueError('MapBackend must be CPU or GPU')
    with stage('save_and_interpolate_maps', cuda=False):
        for name, array in maps.items():
            valid = np.isfinite(array)
            if not valid.any():
                print(f"  - WARNING: Map for {name} has no finite values. Skipping save.")
                continue
            image = sitk.GetImageFromArray(array)
            image.CopyInformation(image_sitk)
            path = os.path.join(output_dir, f'{prefix}_MAP_{name}.nii.gz')
            sitk.WriteImage(image, path)
            print(f"  - Saved quantitative map: {os.path.basename(path)}")
            if get_config('MapSaveVisualization'):
                scaled = _scale_array_to_uint8_robust(array, valid)
                viz = sitk.GetImageFromArray(scaled)
                viz.CopyInformation(image_sitk)
                sitk.WriteImage(viz, os.path.join(output_dir, f'{prefix}_MAP_{name}_uint8.nii.gz'))
            if allow_interpolation():
                try:
                    from .map_interpolation import save_interpolated_map
                    save_interpolated_map(array, mask, strides, image_sitk, prefix, name,
                                          output_dir, get_config('MapSaveVisualization'))
                except Exception:
                    import traceback
                    print(f"  - ERROR: Interpolation failed for {name}; sampled map retained.")
                    traceback.print_exc()

# GLAM_RDF_FAST_PATCH_V2
# GLAM_MAP_INTERPOLATION_V4
# GLAM_BATCHED_GPU_MAPPING_V5
# GLAM_PROFILE_MAPPING_FIRST_V6

# GLAM_CPU_REDUCTION_ONLY_V1

# GLAM_GPU_RDF_OFFLOAD_V1

# GLAM_CDI_GUARD_VIZ_ROBUST_V1
