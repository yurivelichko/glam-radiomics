# src/glam_radiomics/mapping.py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Import configuration helpers
from .config import get_config, load_config 

# Import core GLAM calculation functions
from . import core
from .core import (
    calculate_rdf_3d,
    calculate_glam_coordination_number,
    calculate_glam_potential_energy,
    calculate_glam_structural_pressure_index,
    calculate_configurational_disorder_index,
    calculate_glam_compressibility,
    calculate_rdf_shape_matrices,
    calculate_glam_fractal_dimension
)
from .utils import reformat_dict_to_matrix

# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
worker_quantized_array = None
worker_mask_array = None
worker_settings = {}
worker_sphere_mask = None      


def _get_spherical_mask(radius_voxels):
    """Creates a boolean 3D sphere mask."""
    rz, ry, rx = radius_voxels
    z, y, x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    dist_sq = (z / max(rz, 1))**2 + (y / max(ry, 1))**2 + (x / max(rx, 1))**2
    return dist_sq <= 1.0


def _init_worker_mapping(quantized_array, mask_array, settings_dict, config_path):
    """
    Initializer: Sets worker arrays, disables GPU within multiprocessing workers
    to prevent VRAM collisions, and precomputes invariant masks.
    """
    global worker_quantized_array, worker_mask_array, worker_settings, worker_sphere_mask
    
    # CRITICAL: Prevent CUDA collisions across multiprocessing workers.
    # Small local sliding window patches run faster and safer on CPU cKDTree.
    core.HAS_GPU = False

    worker_quantized_array = quantized_array
    worker_mask_array = mask_array
    worker_settings = settings_dict

    # Pre-calculate sphere mask
    worker_sphere_mask = _get_spherical_mask(worker_settings['window_radius_voxels'])

    try:
        load_config(config_path)
    except Exception as e:
        print(f"[Worker {os.getpid()}] Error loading config: {e}")


def _calculate_local_meta_feature(matrix, method="Mean"):
    """Reduces an NxN matrix to a single scalar value."""
    if matrix is None or np.all(np.isnan(matrix)):
        return np.nan
    
    if method == "Mean":
        return np.nanmean(matrix)
    elif method == "Variance":
        return np.nanvar(matrix)
    elif method == "DiagMean":
        diag = np.diag(matrix)
        return np.nanmean(diag[~np.isnan(diag)])
    elif method == "OffDiagMean":
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        return np.nanmean(off_diag[~np.isnan(off_diag)])
    else:
        return np.nanmean(matrix)


def _process_single_voxel_worker(coords_z_y_x):
    try:
        z, y, x = coords_z_y_x
        
        global worker_quantized_array, worker_mask_array, worker_settings, worker_sphere_mask
        
        num_gray_levels = worker_settings['num_gray_levels']
        window_radius_voxels = worker_settings['window_radius_voxels']
        min_voxels = worker_settings['min_voxels']
        map_max_radius = worker_settings['map_max_radius']
        map_rdf_samples = worker_settings['map_rdf_samples']
        features_to_map = worker_settings['features_to_map']
        meta_method = worker_settings['meta_method']

        req_str = " ".join(features_to_map)
        need_coord = "CoordNum" in req_str
        need_potential = "Potential" in req_str
        need_pressure = "Pressure" in req_str or "SPI" in req_str
        need_compress = "Compress" in req_str
        need_config_disorder = "ConfigurationalDisorderIndex" in req_str or "ConfigDisorder" in req_str
        need_fractal = "Fractal" in req_str
        
        D, H, W = worker_mask_array.shape
        
        # 1. Define Window Bounding Box
        rz, ry, rx = window_radius_voxels
        z_start = max(0, z - rz)
        z_end = min(D, z + rz + 1)
        y_start = max(0, y - ry)
        y_end = min(H, y + ry + 1)
        x_start = max(0, x - rx)
        x_end = min(W, x + rx + 1)

        local_mask = worker_mask_array[z_start:z_end, y_start:y_end, x_start:x_end]
        local_quantized = worker_quantized_array[z_start:z_end, y_start:y_end, x_start:x_end]

        # 2. Extract Matching Sphere Subregion
        sp_z_start = z_start - (z - rz)
        sp_z_end = sp_z_start + (z_end - z_start)
        sp_y_start = y_start - (y - ry)
        sp_y_end = sp_y_start + (y_end - y_start)
        sp_x_start = x_start - (x - rx)
        sp_x_end = sp_x_start + (x_end - x_start)

        sphere_sub = worker_sphere_mask[sp_z_start:sp_z_end, sp_y_start:sp_y_end, sp_x_start:sp_x_end]
        sample_mask = ((local_mask > 0) & sphere_sub).astype(np.uint8)

        active_ref_voxels = int(np.sum(sample_mask))
        if active_ref_voxels < min_voxels:
            return (z, y, x, None)

        # 3. Quantized System Voxels
        local_structured_patch = np.full(sample_mask.shape, -1, dtype=np.int16)
        local_structured_patch[sample_mask > 0] = local_quantized[sample_mask > 0]

        roi_voxels = local_structured_patch[sample_mask > 0]
        local_level_counts = [int(np.sum(roi_voxels == i)) for i in range(num_gray_levels)]
        local_total_voxels = active_ref_voxels

        # 4. Calculate Local RDF
        local_rdf_df = calculate_rdf_3d(
            local_structured_patch, 
            num_gray_levels, 
            map_max_radius,
            local_level_counts, 
            local_total_voxels,
            num_randomisations=1, 
            rdf_sample_points=map_rdf_samples,
            sample_mask=sample_mask
        )
        
        if local_rdf_df is None or local_rdf_df.empty:
            return (z, y, x, None)

        # 5. Build Null Model (Random Baseline Proxy) for CDI
        g_cols = [col for col in local_rdf_df.columns if col.startswith('g_')]
        random_proxy_mat = np.ones((len(local_rdf_df), len(g_cols)), dtype=np.float64)
        local_random_proxy = pd.DataFrame(random_proxy_mat, columns=g_cols)
        local_random_proxy.insert(0, 'r', local_rdf_df['r'].values)
        
        # 6. Calculate Matrices
        local_matrices = {}
        shape_matrices = calculate_rdf_shape_matrices(local_rdf_df, num_gray_levels)
        local_matrices.update(shape_matrices)

        if need_config_disorder:
            features_cdi = calculate_configurational_disorder_index(
                local_rdf_df, local_random_proxy, num_gray_levels
            )
            local_matrices["ConfigurationalDisorderIndex"] = reformat_dict_to_matrix(
                features_cdi, num_gray_levels, "GLAM_ConfigurationalDisorderIndex_", None
            )
        
        if need_coord:
            features_coord = calculate_glam_coordination_number(
                local_rdf_df, num_gray_levels, local_level_counts, local_total_voxels
            )
            local_matrices["CoordNum"] = reformat_dict_to_matrix(
                features_coord, num_gray_levels, "GLAM_CoordNum_", None
            )
            
        if need_potential:
            features_pot = calculate_glam_potential_energy(local_rdf_df, num_gray_levels)
            local_matrices["PotentialEnergy"] = reformat_dict_to_matrix(
                features_pot, num_gray_levels, "GLAM_PotentialEnergy_", None
            )
            
        if need_pressure:
            features_spi = calculate_glam_structural_pressure_index(
                local_rdf_df, num_gray_levels, local_level_counts, local_total_voxels
            )
            local_matrices["StructuralPressureIndex"] = reformat_dict_to_matrix(
                features_spi, num_gray_levels, "GLAM_StructuralPressureIndex_", None
            )            
        
        if need_compress:
            features_comp = calculate_glam_compressibility(local_rdf_df, num_gray_levels)
            local_matrices["Compressibility"] = reformat_dict_to_matrix(
                features_comp, num_gray_levels, "GLAM_Compressibility_", None
            )

        if need_fractal:
            features_fractal = calculate_glam_fractal_dimension(local_structured_patch, num_gray_levels)
            local_matrices["FractalDimension"] = reformat_dict_to_matrix(
                features_fractal, num_gray_levels, "GLAM_InterfaceFD_", "GLAM_VolumeFD_"
            )

        # 7. Apply Transformations & Meta-Feature Reduction
        output_feature_values = {}
        for feat_name in features_to_map:
            matrix = None
            if feat_name.endswith("_Symlog"):
                base_name = feat_name.replace("_Symlog", "")
                matrix = local_matrices.get(base_name)
                if matrix is not None:
                    matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
            elif feat_name.endswith("_Ln"):
                base_name = feat_name.replace("_Ln", "")
                matrix = local_matrices.get(base_name)
                if matrix is not None:
                    with np.errstate(divide='ignore'):
                        matrix = np.log(matrix)
                    matrix[np.isneginf(matrix)] = np.nan
            else:
                matrix = local_matrices.get(feat_name)

            if matrix is None:
                output_feature_values[feat_name] = np.nan
            else:
                output_feature_values[feat_name] = _calculate_local_meta_feature(matrix, meta_method)

        return (z, y, x, output_feature_values)

    except Exception as e:
        return (coords_z_y_x[0], coords_z_y_x[1], coords_z_y_x[2], None)


def generate_feature_maps(image_sitk, binary_mask_sitk, quantized_image_array, 
                          num_gray_levels, prefix, output_dir, config_path):
    """
    Generates 3D quantitative feature maps using parallel sliding-window analysis.
    """
    print("  --- Starting 3D Feature Map Generation ---")
    
    try:
        num_workers = get_config('NumWorkers')
        window_cm = get_config('MapWindowSizeCM')
        features_to_map = get_config('MapFeatures')
        meta_method = get_config('MapMetaMethod')
        map_max_radius = get_config('MapRDFMaxRadius')
        map_rdf_samples = get_config('MapRDFSamplePoints')
        min_voxels = get_config('MapMinWindowVoxels')
        save_viz_map = get_config('MapSaveVisualization')
        overlap_percent = get_config('MapOverlapPercent')
    except KeyError as e:
        print(f"  - ERROR: Missing config key: {e}")
        return

    # Compute window radii in voxel units
    spacing_mm = image_sitk.GetSpacing()
    window_mm = window_cm * 10.0
    target_radius_mm = window_mm / 2.0
    window_radius_voxels = [
        int(np.ceil(target_radius_mm / spacing_mm[2])), # z
        int(np.ceil(target_radius_mm / spacing_mm[1])), # y
        int(np.ceil(target_radius_mm / spacing_mm[0]))  # x
    ]
    print(f"  - Window: {window_cm} cm | Radii (voxels z,y,x): {window_radius_voxels}")

    mask_array = sitk.GetArrayFromImage(binary_mask_sitk)
    
    overlap_percent = np.clip(overlap_percent, 0.0, 95.0)
    step_fraction = 1.0 - (overlap_percent / 100.0)

    stride_z = max(1, int((window_radius_voxels[0] * 2) * step_fraction))
    stride_y = max(1, int((window_radius_voxels[1] * 2) * step_fraction))
    stride_x = max(1, int((window_radius_voxels[2] * 2) * step_fraction))

    print(f"  - Sampling Grid Stride: z={stride_z}, y={stride_y}, x={stride_x} ({overlap_percent}% overlap)")

    z_range = range(0, mask_array.shape[0], stride_z)
    y_range = range(0, mask_array.shape[1], stride_y)
    x_range = range(0, mask_array.shape[2], stride_x)

    z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, x_range, indexing='ij')
    candidate_coords = np.vstack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()]).T

    is_in_mask = mask_array[candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2]] > 0
    roi_coords = candidate_coords[is_in_mask]

    if len(roi_coords) == 0:
        print("  - Skipping: No grid sampling points inside the target mask.")
        return

    output_maps = {
        feat: np.full(mask_array.shape, np.nan, dtype=np.float32) 
        for feat in features_to_map
    }

    worker_init_settings = {
        "num_gray_levels": num_gray_levels,
        "window_radius_voxels": window_radius_voxels,
        "min_voxels": min_voxels,
        "map_max_radius": map_max_radius,
        "map_rdf_samples": map_rdf_samples,
        "features_to_map": features_to_map,
        "meta_method": meta_method
    }
    
    print(f"  - Processing {len(roi_coords)} map nodes using {num_workers} parallel CPU workers...")
    
    init_args = (quantized_image_array, mask_array, worker_init_settings, config_path)
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=_init_worker_mapping, 
                             initargs=init_args) as executor:
        results = list(tqdm(executor.map(_process_single_voxel_worker, roi_coords, chunksize=25), total=len(roi_coords)))

    # Reconstruct 3D Volumes
    for result in results:
        if result is None:
            continue
        z, y, x, feature_dict = result
        if feature_dict is None:
            continue
            
        for feat_name, value in feature_dict.items():
            if feat_name in output_maps:
                output_maps[feat_name][z, y, x] = value

    # Save Output Maps
    for feat_name, map_array in output_maps.items():
        if np.all(np.isnan(map_array)):
            print(f"  - WARNING: Map for {feat_name} is completely NaN. Skipping save.")
            continue
        
        output_sitk = sitk.GetImageFromArray(map_array)
        output_sitk.CopyInformation(image_sitk)
        
        output_path = os.path.join(output_dir, f"{prefix}_MAP_{feat_name}.nii.gz")
        sitk.WriteImage(output_sitk, output_path)
        print(f"  - Saved quantitative map: {os.path.basename(output_path)}")

        if save_viz_map:
            try:
                scaled_map_uint8 = np.zeros(map_array.shape, dtype=np.uint8)
                valid_mask = ~np.isnan(map_array)
                if not np.any(valid_mask):
                    continue

                valid_voxels = map_array[valid_mask]
                min_val = np.min(valid_voxels)
                max_val = np.max(valid_voxels)
                data_range = max_val - min_val
                
                if data_range > 1e-6:
                    scaled_values = num_gray_levels * (valid_voxels - min_val) / data_range
                    scaled_map_uint8[valid_mask] = np.clip(scaled_values, 0, num_gray_levels).astype(np.uint8)
                else:
                    scaled_map_uint8[valid_mask] = num_gray_levels // 2
                
                output_sitk_uint8 = sitk.GetImageFromArray(scaled_map_uint8)
                output_sitk_uint8.CopyInformation(image_sitk)
                
                output_path_uint8 = os.path.join(output_dir, f"{prefix}_MAP_{feat_name}_uint8.nii.gz")
                sitk.WriteImage(output_sitk_uint8, output_path_uint8)
                print(f"  - Saved visualization map: {os.path.basename(output_path_uint8)}")
            except Exception as e:
                print(f"  - WARNING: Could not save visualization map for {feat_name}: {e}")