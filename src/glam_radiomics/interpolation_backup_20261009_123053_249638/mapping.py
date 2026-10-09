# GLAM priority fixes: local maps with a genuine shuffled RDF baseline.
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from .config import get_config, load_config
from . import core
from .utils import reformat_dict_to_matrix

worker_quantized_array = None
worker_mask_array = None
worker_settings = {}
worker_sphere_mask = None

def _get_spherical_mask(radius_voxels):
    rz, ry, rx = radius_voxels
    z, y, x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return ((z/max(rz,1))**2 + (y/max(ry,1))**2 + (x/max(rx,1))**2) <= 1

def _init_worker_mapping(quantized_array, mask_array, settings_dict, config_path):
    global worker_quantized_array, worker_mask_array, worker_settings, worker_sphere_mask
    load_config(config_path)
    core.HAS_GPU = False
    worker_quantized_array = quantized_array
    worker_mask_array = mask_array
    worker_settings = settings_dict
    worker_sphere_mask = _get_spherical_mask(settings_dict['window_radius_voxels'])

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

def _local_random_rdf(patch, sample_mask, levels, radius, counts, total, samples, repeats):
    """Shuffle labels inside the same window; preserve geometry and level counts."""
    if repeats < 1:
        raise ValueError("Local RDF baseline needs at least one randomization")
    roi = sample_mask > 0
    values = patch[roi]
    frames = []
    for _ in range(repeats):
        shuffled = np.full(patch.shape, -1, dtype=np.int16)
        shuffled[roi] = np.random.permutation(values)
        frame = _fast_calculate_rdf_3d(
            shuffled, levels, radius, list(counts), total, 1, samples,
            sample_mask=sample_mask)
        if frame is None or frame.empty:
            raise RuntimeError("Empty local randomized RDF")
        frames.append(frame)
    baseline = frames[0].copy()
    cols = [c for c in baseline.columns if c.startswith('g_')]
    for frame in frames[1:]:
        if not np.array_equal(frame['r'].values, baseline['r'].values):
            raise RuntimeError("Local random RDF radii differ")
    baseline[cols] = np.mean(np.stack([f[cols].to_numpy() for f in frames]), axis=0)
    return baseline

def _process_single_voxel_worker(coords_z_y_x):
    try:
        z, y, x = map(int, coords_z_y_x)
        s = worker_settings
        levels = s['num_gray_levels']
        radii = s['window_radius_voxels']
        dims = worker_mask_array.shape
        centers = (z,y,x)
        starts = [max(0,c-r) for c,r in zip(centers,radii)]
        ends = [min(d,c+r+1) for d,c,r in zip(dims,centers,radii)]
        slices = tuple(slice(a,b) for a,b in zip(starts,ends))
        sphere_starts = [a-(c-r) for a,c,r in zip(starts,centers,radii)]
        sphere_slices = tuple(slice(a,a+b-c) for a,b,c in zip(sphere_starts,ends,starts))
        sample_mask = ((worker_mask_array[slices] > 0) & worker_sphere_mask[sphere_slices])
        total = int(sample_mask.sum())
        if total < s['min_voxels']:
            return z,y,x,None
        patch = np.full(sample_mask.shape,-1,dtype=np.int16)
        patch[sample_mask] = worker_quantized_array[slices][sample_mask]
        values = patch[sample_mask]
        if np.any(values < 0) or np.any(values >= levels):
            raise ValueError("Invalid local gray levels; check crop reconstruction")
        counts = np.bincount(values,minlength=levels).tolist()
        # Node-local seed makes CPU maps independent of scheduling/worker count.
        seed = np.random.SeedSequence([42,z,y,x])
        np.random.seed(int(seed.generate_state(1)[0]))
        rdf = _fast_calculate_rdf_3d(patch,levels,s['map_max_radius'],list(counts),
                                  total,1,s['map_rdf_samples'],sample_mask=sample_mask)
        if rdf is None or rdf.empty:
            return z,y,x,None
        matrices = core.calculate_rdf_shape_matrices(rdf,levels)
        bases = {f.removesuffix('_Symlog').removesuffix('_Ln') for f in s['features_to_map']}
        if 'ConfigurationalDisorderIndex' in bases:
            baseline = _local_random_rdf(
                patch,sample_mask,levels,s['map_max_radius'],counts,total,
                s['map_rdf_samples'],s['randomizations'])
            features = core.calculate_configurational_disorder_index(rdf,baseline,levels)
            matrices['ConfigurationalDisorderIndex'] = reformat_dict_to_matrix(
                features,levels,'GLAM_ConfigurationalDisorderIndex_',None)
        calls = {
            'CoordNum': lambda: core.calculate_glam_coordination_number(rdf,levels,counts,total),
            'PotentialEnergy': lambda: core.calculate_glam_potential_energy(rdf,levels),
            'StructuralPressureIndex': lambda: core.calculate_glam_structural_pressure_index(rdf,levels,counts,total),
            'Compressibility': lambda: core.calculate_glam_compressibility(rdf,levels),
            'FractalDimension': lambda: core.calculate_glam_fractal_dimension(patch,levels),
        }
        for name,call in calls.items():
            if name in bases:
                if name == 'FractalDimension':
                    matrices[name] = reformat_dict_to_matrix(call(),levels,'GLAM_InterfaceFD_','GLAM_VolumeFD_')
                else:
                    matrices[name] = reformat_dict_to_matrix(call(),levels,f'GLAM_{name}_',None)
        result = {}
        for feature in s['features_to_map']:
            base = feature.removesuffix('_Symlog').removesuffix('_Ln')
            if base not in matrices:
                raise ValueError(f"Unsupported MapFeatures entry: {feature}")
            matrix = np.asarray(matrices[base],dtype=float).copy()
            if feature.endswith('_Symlog'):
                matrix = np.sign(matrix)*np.log1p(np.abs(matrix))
            elif feature.endswith('_Ln'):
                with np.errstate(divide='ignore',invalid='ignore'):
                    matrix = np.log(matrix)
                matrix[~np.isfinite(matrix)] = np.nan
            for method in s['meta_method']:
                result[f'{feature}_{method}'] = _calculate_local_meta_feature(matrix,method)
        return z,y,x,result
    except Exception as exc:
        raise RuntimeError(f"Local mapping failed at node {tuple(coords_z_y_x)}: {exc}") from exc

def generate_feature_maps(image_sitk,binary_mask_sitk,quantized_image_array,
                          num_gray_levels,prefix,output_dir,config_path):
    print("  --- Starting 3D Feature Map Generation ---")
    workers = int(get_config('NumWorkers'))
    window_cm = float(get_config('MapWindowSizeCM'))
    features = get_config('MapFeatures')
    methods = get_config('MapMetaMethod')
    repeats = int(get_config('NumRandomisations'))
    radius = int(get_config('MapRDFMaxRadius'))
    samples = int(get_config('MapRDFSamplePoints'))
    min_voxels = int(get_config('MapMinWindowVoxels'))
    if min(workers,repeats,radius,samples,min_voxels) < 1 or window_cm <= 0:
        raise ValueError("Invalid mapping settings")
    if not isinstance(features,list) or not isinstance(methods,list) or not features or not methods:
        raise ValueError("MapFeatures and MapMetaMethod must be nonempty lists")
    mask = sitk.GetArrayFromImage(binary_mask_sitk)
    if mask.shape != quantized_image_array.shape:
        raise ValueError("Mapping image/mask shape mismatch")
    spacing = np.asarray(image_sitk.GetSpacing()[::-1],dtype=float)
    radii = np.ceil(window_cm*5.0/spacing).astype(int).tolist()
    overlap = float(np.clip(get_config('MapOverlapPercent'),0,95))
    strides = [max(1,int(2*r*(1-overlap/100))) for r in radii]
    grid = np.meshgrid(*(range(0,d,t) for d,t in zip(mask.shape,strides)),indexing='ij')
    coords = np.stack([g.ravel() for g in grid],axis=1)
    coords = coords[mask[tuple(coords.T)] > 0]
    if not len(coords):
        print("  - No grid sampling points inside the target mask.")
        return
    settings = dict(num_gray_levels=num_gray_levels,window_radius_voxels=radii,
                    min_voxels=min_voxels,map_max_radius=radius,map_rdf_samples=samples,
                    features_to_map=features,meta_method=methods,randomizations=repeats)
    maps = {f'{f}_{m}':np.full(mask.shape,np.nan,dtype=np.float32) for f in features for m in methods}
    print(f"  - Processing {len(coords)} nodes; local CDI uses {repeats} shuffled RDFs per node.")
    # Keep worker initialization in a child process, even with one worker,
    # so disabling its GPU does not disable the parent extraction GPU.
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker_mapping,
                             initargs=(quantized_image_array,mask,settings,config_path)) as executor:
        for z,y,x,values in tqdm(executor.map(_process_single_voxel_worker,coords,chunksize=1),total=len(coords)):
            if values is not None:
                for key,value in values.items():
                    maps[key][z,y,x] = value
    for name,array in maps.items():
        valid = np.isfinite(array)
        if not valid.any():
            print(f"  - WARNING: Map for {name} has no finite values. Skipping save.")
            continue
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(image_sitk)
        path = os.path.join(output_dir,f'{prefix}_MAP_{name}.nii.gz')
        sitk.WriteImage(image,path)
        print(f"  - Saved quantitative map: {os.path.basename(path)}")
        if get_config('MapSaveVisualization'):
            scaled = np.zeros(array.shape,dtype=np.uint8)
            lo,hi = np.min(array[valid]),np.max(array[valid])
            # Reserve zero for unobserved voxels in the visualization only.
            scaled[valid] = (1+254*(array[valid]-lo)/(hi-lo)).astype(np.uint8) if hi-lo > 1e-6 else 128
            viz = sitk.GetImageFromArray(scaled)
            viz.CopyInformation(image_sitk)
            sitk.WriteImage(viz,os.path.join(output_dir,f'{prefix}_MAP_{name}_uint8.nii.gz'))

# GLAM_RDF_FAST_PATCH_V2
from .rdf_fast import calculate_rdf_3d as _fast_calculate_rdf_3d
