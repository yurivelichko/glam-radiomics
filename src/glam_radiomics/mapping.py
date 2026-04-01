# src/glam_radiomics/mapping.py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Import from your existing code
from .config import get_config, load_config 
from .core import (
    calculate_rdf_3d,
    calculate_glam_coordination_number,
    calculate_glam_potential_energy,
    calculate_glam_pressure_virial,
    calculate_effective_temperature,
    calculate_glam_compressibility,
    calculate_rdf_shape_matrices,
    calculate_glam_fractal_dimension,
    calculate_geometric_factor,   
    apply_geometric_correction
)
from .utils import reformat_dict_to_matrix

# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
worker_quantized_array = None
worker_mask_array = None
worker_settings = {}
worker_sphere_mask = None      
worker_full_geom_factor = None 
# ---

def _get_spherical_mask(radius_voxels):
    """Creates a boolean 3D sphere mask."""
    L = 2 * radius_voxels + 1
    z, y, x = np.ogrid[:L, :L, :L]
    center = radius_voxels
    dist_sq = (z - center)**2 + (y - center)**2 + (x - center)**2
    return dist_sq <= radius_voxels**2


def _init_worker_mapping(quantized_array, mask_array, settings_dict, config_path):
    """
    Initializer: Sets large arrays and pre-calculates invariant data.
    """
    global worker_quantized_array, worker_mask_array, worker_settings
    global worker_sphere_mask, worker_full_geom_factor
    
    worker_quantized_array = quantized_array
    worker_mask_array = mask_array
    worker_settings = settings_dict

    # --- OPTIMIZATION 1: Pre-calculate Sphere Mask ---
    # We generate this once per worker, instead of once per voxel
    radius = worker_settings['window_radius_voxels'][0]
    worker_sphere_mask = _get_spherical_mask(radius)

    # --- OPTIMIZATION 2: Pre-calculate "Perfect" Geometric Factor ---
    # We calculate the factor for a fully solid window (no boundary/edge effects)
    try:
        # Create a dummy mask that is 1 everywhere inside the sphere
        dummy_mask = worker_sphere_mask.astype(np.int16)
        
        worker_full_geom_factor = calculate_geometric_factor(
            dummy_mask, 
            worker_settings['map_max_radius'], 
            worker_settings['map_rdf_samples']
        )
    except Exception as e:
        print(f"[Worker Init] Warning: Could not pre-calc geom factor. {e}")
        worker_full_geom_factor = None

    try:
        load_config(config_path)
    except Exception as e:
        print(f"[Worker {os.getpid()}] FATAL ERROR: Could not load config. {e}")


def _calculate_local_meta_feature(matrix, method="Mean"):
    """Helper to reduce an NxN matrix to a single scalar value."""
    # (This function is unchanged from before)
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

# src/glam_radiomics/mapping.py

def _process_single_voxel_worker(coords_z_y_x):
    try:
        # 1. Unpack coordinates
        z, y, x = coords_z_y_x
        
        # 2. Access global data
        global worker_quantized_array, worker_mask_array, worker_settings
        global worker_sphere_mask, worker_full_geom_factor
        
        # Unpack settings
        num_gray_levels = worker_settings['num_gray_levels']
        window_radius_voxels = worker_settings['window_radius_voxels'] # e.g., 5
        min_voxels = worker_settings['min_voxels']
        map_max_radius = worker_settings['map_max_radius'] # e.g., 10 (Interaction range)
        map_rdf_samples = worker_settings['map_rdf_samples']
        features_to_map = worker_settings['features_to_map']
        meta_method = worker_settings['meta_method']

        req_str = " ".join(features_to_map)
        need_coord = "CoordNum" in req_str
        need_potential = "Potential" in req_str
        need_pressure = "Pressure" in req_str
        need_compress = "Compress" in req_str
        need_temp = "Temp" in req_str or "Effective" in req_str
        need_fractal = "Fractal" in req_str
        
        D, H, W = worker_mask_array.shape
        
        # --- NEW LOGIC: Define "Load" bounds vs "Window" bounds ---
        
        # 1. LOAD Bounds (Window + Halo)
        # We must load enough data for the center pixels to see neighbors 'map_max_radius' away.
        load_radius_z = window_radius_voxels[0] + map_max_radius
        load_radius_y = window_radius_voxels[1] + map_max_radius
        load_radius_x = window_radius_voxels[2] + map_max_radius

        z_min_load = max(0, z - load_radius_z)
        z_max_load = min(D, z + load_radius_z + 1)
        y_min_load = max(0, y - load_radius_y)
        y_max_load = min(H, y + load_radius_y + 1)
        x_min_load = max(0, x - load_radius_x)
        x_max_load = min(W, x + load_radius_x + 1)
        
        # Extract the large patch (System)
        local_mask_patch = worker_mask_array[z_min_load:z_max_load, y_min_load:y_max_load, x_min_load:x_max_load]
        local_quantized_patch = worker_quantized_array[z_min_load:z_max_load, y_min_load:y_max_load, x_min_load:x_max_load]
        
        # 2. WINDOW Bounds (The strict ROI) relative to the loaded patch
        # We need a mask that is 1 ONLY inside the strict sliding window.
        patch_depth, patch_height, patch_width = local_mask_patch.shape
        sample_mask = np.zeros((patch_depth, patch_height, patch_width), dtype=np.uint8)

        # Calculate offsets to locate the window inside the patch
        # The window starts at (z - radius), but the patch starts at z_min_load
        win_z_start = max(0, z - window_radius_voxels[0]) - z_min_load
        win_z_end   = min(D, z + window_radius_voxels[0] + 1) - z_min_load
        win_y_start = max(0, y - window_radius_voxels[1]) - y_min_load
        win_y_end   = min(H, y + window_radius_voxels[1] + 1) - y_min_load
        win_x_start = max(0, x - window_radius_voxels[2]) - x_min_load
        win_x_end   = min(W, x + window_radius_voxels[2] + 1) - x_min_load

        # Ensure indices are valid (should always be, given max/min logic above)
        win_z_start = max(0, win_z_start); win_z_end = min(patch_depth, win_z_end)
        win_y_start = max(0, win_y_start); win_y_end = min(patch_height, win_y_end)
        win_x_start = max(0, win_x_start); win_x_end = min(patch_width, win_x_end)

        # Set the active window region to 1
        sample_mask[win_z_start:win_z_end, win_y_start:win_y_end, win_x_start:win_x_end] = 1

        # 3. Apply Sphere Mask Logic (Optional but recommended for consistency)
        # We want the window to be spherical, so we mask the sample_mask with a sphere
        # Note: worker_sphere_mask corresponds to window_radius_voxels, NOT load_radius
        
        # We need to place the sphere mask into the center of our sample_mask
        sphere_d, sphere_h, sphere_w = worker_sphere_mask.shape
        
        # Only apply if dimensions match the window cut we just made
        cut_d = win_z_end - win_z_start
        cut_h = win_y_end - win_y_start
        cut_w = win_x_end - win_x_start
        
        if (cut_d, cut_h, cut_w) == (sphere_d, sphere_h, sphere_w):
             current_window = sample_mask[win_z_start:win_z_end, win_y_start:win_y_end, win_x_start:win_x_end]
             sample_mask[win_z_start:win_z_end, win_y_start:win_y_end, win_x_start:win_x_end] = current_window * worker_sphere_mask
        elif (cut_d <= sphere_d) and (cut_h <= sphere_h) and (cut_w <= sphere_w):
             # Handle boundary cases where window is cropped
             cropped_sphere = worker_sphere_mask[:cut_d, :cut_h, :cut_w]
             current_window = sample_mask[win_z_start:win_z_end, win_y_start:win_y_end, win_x_start:win_x_end]
             sample_mask[win_z_start:win_z_end, win_y_start:win_y_end, win_x_start:win_x_end] = current_window * cropped_sphere

        # 4. Final Constraints
        # The sample_mask must ALSO respect the original tissue mask (local_mask_patch)
        # i.e., we only sample from pixels that are (Inside Window) AND (Inside Tumor)
        sample_mask = sample_mask * local_mask_patch

        # Count voxels strictly inside the sampling region
        # (Neighbors outside this count are still valid neighbors, but not reference points)
        active_ref_voxels = np.sum(sample_mask)
        
        if active_ref_voxels < min_voxels:
            return (z, y, x, None)

        # "System" voxels are everything in the loaded patch that is tumor
        local_roi_voxels = local_quantized_patch[local_mask_patch > 0]
        local_level_counts = [np.sum(local_roi_voxels == i) for i in range(num_gray_levels)]
        local_total_voxels = local_roi_voxels.size
        
        local_structured_patch = np.full(local_mask_patch.shape, -1, dtype=np.int16)
        local_structured_patch[local_mask_patch > 0] = local_roi_voxels
        
        # 5. Calculate Local RDF (Passing the new sample_mask)
        local_rdf_df = calculate_rdf_3d(
            local_structured_patch, num_gray_levels, map_max_radius,
            local_level_counts, local_total_voxels,
            num_randomisations=1, 
            rdf_sample_points=map_rdf_samples,
            sample_mask=sample_mask # <--- UPDATE
        )
        
        if local_rdf_df.empty:
            return (z, y, x, None)

        # --- Geometric Factor Logic ---
        # Note: We must calculate geom factor for the SAMPLE MASK geometry
        # because that's where we are measuring from.
        local_geom_factor = calculate_geometric_factor(
            sample_mask, # Use the window+tumor mask
            map_max_radius, 
            map_rdf_samples
        )
        
        # --- Proxy Generation ---
        g_cols = [col for col in local_rdf_df.columns if col.startswith('g_')]
        geom_values = local_geom_factor.reindex(local_rdf_df['r']).fillna(0.01).values
        geom_matrix = np.tile(geom_values[:, np.newaxis], (1, len(g_cols)))
        local_random_proxy = pd.DataFrame(geom_matrix, columns=g_cols)
        local_random_proxy.insert(0, 'r', local_rdf_df['r'].values)
        
        # 8. Calculate Features (Lazy Evaluation)
        local_matrices = {}
        
        local_rdf_df = apply_geometric_correction(local_rdf_df, local_geom_factor)
        shape_matrices = calculate_rdf_shape_matrices(local_rdf_df, num_gray_levels)
        local_matrices.update(shape_matrices)

        if need_temp:
             features_eff_temp = calculate_effective_temperature(local_rdf_df, local_random_proxy, num_gray_levels)
             local_matrices["EffectiveTemp"] = reformat_dict_to_matrix(features_eff_temp, num_gray_levels, "GLAM_EffectiveTemp_", None)

        if need_coord:
            features_coordnum = calculate_glam_coordination_number(local_rdf_df, num_gray_levels, local_level_counts, local_total_voxels)
            local_matrices["CoordNum"] = reformat_dict_to_matrix(features_coordnum, num_gray_levels, "GLAM_CoordNum_", None)
            
        if need_potential:
            features_potential = calculate_glam_potential_energy(local_rdf_df, num_gray_levels)
            local_matrices["PotentialEnergy"] = reformat_dict_to_matrix(features_potential, num_gray_levels, "GLAM_PotentialEnergy_", None)
            
        if need_pressure:
            features_pressure = calculate_glam_pressure_virial(local_rdf_df, num_gray_levels, local_level_counts, local_total_voxels)
            local_matrices["PressureVirial"] = reformat_dict_to_matrix(features_pressure, num_gray_levels, "GLAM_PressureVirial_", None)
            
        if need_compress:
            features_compress = calculate_glam_compressibility(local_rdf_df, num_gray_levels)
            local_matrices["Compressibility"] = reformat_dict_to_matrix(features_compress, num_gray_levels, "GLAM_Compressibility_", None)

        if need_fractal:
            # Fractal dimension is usually structural, so we might still use the sample_mask area
            # Or the whole patch. Usually local FD implies local structure.
            # Let's mask the patch to the window for strict local FD
            window_only_patch = local_structured_patch.copy()
            window_only_patch[sample_mask == 0] = -1
            features_fractal = calculate_glam_fractal_dimension(window_only_patch, num_gray_levels)
            local_matrices["FractalDimension"] = reformat_dict_to_matrix(
                features_fractal, num_gray_levels, "GLAM_InterfaceFD_", "GLAM_VolumeFD_"
            )

        # 9. Apply Transforms and Reduce
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
        print(f"ERROR in voxel worker for {coords_z_y_x}: {e}")
        return (coords_z_y_x[0], coords_z_y_x[1], coords_z_y_x[2], None)

def generate_feature_maps(image_sitk, binary_mask_sitk, quantized_image_array, 
                          num_gray_levels, prefix, output_dir, config_path):
    """
    Main function to generate 3D feature maps using a sliding window.
    This version uses a ProcessPoolExecutor to parallelize the voxel loop.
    """
    print("  --- Starting 3D Feature Map Generation (Parallel) ---")
    
    # --- 1. Get Settings from Config ---
    try:
        num_workers = get_config('NumWorkers') # Get this from [System]
        window_cm = get_config('MapWindowSizeCM')
        features_to_map = get_config('MapFeatures')
        meta_method = get_config('MapMetaMethod')
        map_max_radius = get_config('MapRDFMaxRadius')
        map_rdf_samples = get_config('MapRDFSamplePoints')
        min_voxels = get_config('MapMinWindowVoxels')
        save_viz_map = get_config('MapSaveVisualization')
    except KeyError as e:
        print(f"  - ERROR: Missing a config setting. {e}")
        return

    # --- 2. Calculate Window Size in Voxels ---
    spacing_mm = image_sitk.GetSpacing()
    window_mm = window_cm * 10.0
    target_radius_mm = window_mm / 2.0
    window_radius_voxels = [
        int(np.ceil(target_radius_mm / spacing_mm[2])), # z
        int(np.ceil(target_radius_mm / spacing_mm[1])), # y
        int(np.ceil(target_radius_mm / spacing_mm[0]))  # x
    ]
    print(f"  - Target: {window_cm} cm. Spacing (mm): {spacing_mm}")
    print(f"  - Using window radius (vox... z,y,x): {window_radius_voxels}")
    


    # --- 3. Prepare Arrays and Task List ---
    mask_array = sitk.GetArrayFromImage(binary_mask_sitk)
    
    # Define stride. 
    overlap_percent = get_config('MapOverlapPercent')
    
    # Safety check: Prevent infinite loops if user sets 100% overlap
    if overlap_percent >= 100: 
        print("  - WARNING: Overlap cannot be 100%. Clamping to 99%.")
        overlap_percent = 99.0
    if overlap_percent < 0:
        overlap_percent = 0.0

    # Calculate the fraction of the window we need to STEP
    # e.g., 50% overlap -> step 0.5 of window
    # e.g., 0% overlap -> step 1.0 of window
    step_fraction = 1.0 - (overlap_percent / 100.0)

    # Window Size is effectively (2 * radius)
    stride_z = max(1, int((window_radius_voxels[0] * 2) * step_fraction))
    stride_y = max(1, int((window_radius_voxels[1] * 2) * step_fraction))
    stride_x = max(1, int((window_radius_voxels[2] * 2) * step_fraction))

    print(f"  - Overlap Strategy: {overlap_percent}% (Step Fraction: {step_fraction:.2f})")
    print(f"  - Generating grid with stride: z={stride_z}, y={stride_y}, x={stride_x}")

    # Generate grid ranges
    z_range = range(0, mask_array.shape[0], stride_z)
    y_range = range(0, mask_array.shape[1], stride_y)
    x_range = range(0, mask_array.shape[2], stride_x)

    # Create the meshgrid of candidate centers
    z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, x_range, indexing='ij')
    candidate_coords = np.vstack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()]).T

    # Filter: Only keep grid points that are actually inside the Mask
    # (Fast numpy indexing to check the mask at these coordinates)
    is_in_mask = mask_array[candidate_coords[:,0], candidate_coords[:,1], candidate_coords[:,2]] > 0
    roi_coords = candidate_coords[is_in_mask]

    if len(roi_coords) == 0:
        print("  - Skipping: No grid points fell inside the mask.")
        return

    # Create empty output maps
    output_maps = {
        feat: np.full(mask_array.shape, np.nan, dtype=np.float32) 
        for feat in features_to_map
    }



    # --- 4. Pack all read-only settings for the worker ---
    worker_init_settings = {
        "num_gray_levels": num_gray_levels,
        "window_radius_voxels": window_radius_voxels,
        "min_voxels": min_voxels,
        "map_max_radius": map_max_radius,
        "map_rdf_samples": map_rdf_samples,
        "features_to_map": features_to_map,
        "meta_method": meta_method
    }
    
    # --- 5. The PARALLEL Sliding Window Loop ---
    print(f"  - Sliding window over {len(roi_coords)} voxels using {num_workers} workers...")
    
    # Pack the initargs
    init_args = (quantized_image_array, mask_array, worker_init_settings, config_path)
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=_init_worker_mapping, 
                             initargs=init_args) as executor:
        
        # Use executor.map with chunksize for efficiency
        # The tqdm wrapper will give us a progress bar
        results = list(tqdm(executor.map(_process_single_voxel_worker, roi_coords, chunksize=10), total=len(roi_coords)))

    # --- 6. Reconstruct Maps from Results ---
    print("  - Parallel processing complete. Reconstructing maps...")
    for result in results:
        if result is None: continue # Handle a potential worker crash
        
        z, y, x, feature_dict = result
        
        if feature_dict is None:
            continue # Voxel was skipped (e.g., < min_voxels)
            
        for feat_name, value in feature_dict.items():
            if feat_name in output_maps:
                output_maps[feat_name][z, y, x] = value

    # --- 7. Save all generated maps to NIfTI files ---
    for feat_name, map_array in output_maps.items():
        if np.all(np.isnan(map_array)):
            print(f"  - WARNING: Map for {feat_name} is all NaN. Skipping save.")
            continue
        
        # --- 1. Save the Quantitative (Float32) Map (ALWAYS) ---
        output_sitk = sitk.GetImageFromArray(map_array)
        output_sitk.CopyInformation(image_sitk)
        
        output_path = os.path.join(output_dir, f"{prefix}_MAP_{feat_name}.nii.gz")
        sitk.WriteImage(output_sitk, output_path)
        print(f"  - Saved feature map: {os.path.basename(output_path)}")

        # --- 2. Save the Visualization (UInt8) Map (OPTIONAL) ---
        if save_viz_map:
            try:
                # Create a new array for the 8-bit map, starting with all 0s
                scaled_map_uint8 = np.zeros(map_array.shape, dtype=np.uint8)
                
                # Find the mask of all valid (non-NaN) voxels
                valid_mask = ~np.isnan(map_array)
                if not np.any(valid_mask):
                    continue # Should be impossible, but safe to check

                # Get the float values from the valid region
                valid_voxels = map_array[valid_mask]
                
                min_val = np.min(valid_voxels)
                max_val = np.max(valid_voxels)
                data_range = max_val - min_val
                
                if data_range > 1e-6: # Avoid divide-by-zero for flat maps
                    # Min-Max Normalization: Scale 0-num_gray_levels
                    scaled_values = num_gray_levels * (valid_voxels - min_val) / data_range
                    scaled_values = np.clip(scaled_values, 0, num_gray_levels) # Ensure it's in range
                    # Copy the scaled values into the uint8 map
                    scaled_map_uint8[valid_mask] = scaled_values.astype(np.uint8)

                else: # It's a flat map
                    # Set all valid voxels to mid-gray 
                    scaled_map_uint8[valid_mask] = num_gray_levels // 2
                
                # Create and save the new SITK image
                output_sitk_uint8 = sitk.GetImageFromArray(scaled_map_uint8)
                output_sitk_uint8.CopyInformation(image_sitk)
                
                output_path_uint8 = os.path.join(output_dir, f"{prefix}_MAP_{feat_name}_uint8.nii.gz")
                sitk.WriteImage(output_sitk_uint8, output_path_uint8)
                print(f"  - Saved visualization map: {os.path.basename(output_path_uint8)}")
            
            except Exception as e:
                print(f"  - WARNING: Could not save visualization map for {feat_name}. Error: {e}")