# src/glam_radiomics/run.py
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import traceback
import argparse
from concurrent.futures import ProcessPoolExecutor

from . import mapping
from .config import load_config, get_config

# --- Import ALL core functions ---
from .core import (
    calculate_glam_shape_matrices,
    calculate_shape_features_3d,
    calculate_first_order_stats_from_matrix,
    # Custom Radiomics Implementations
    calculate_glcm_3d, calculate_glrlm_3d, calculate_glszm_3d, calculate_gldm_3d, calculate_ngtdm_3d,
    calculate_glrlm_features, calculate_glszm_features, calculate_gldm_features, calculate_ngtdm_features,
    calculate_first_order_features, 
    calculate_glcm_style_meta_features, 
    # Original GLAM functions
    calculate_rdf_3d,
    calculate_glam_b2_3d,
    calculate_glam_inverse_correlation_length,
    calculate_glam_coordination_number,
    calculate_glam_compressibility,
    calculate_anisotropic_glam_features,
    calculate_glam_fractal_dimension,
    calculate_glam_multifractal_spectrum,
    calculate_glam_lacunarity,
    calculate_glam_topology,
    calculate_glam_potential_energy,
    calculate_glam_structural_pressure_index,
    calculate_glam_shape_matrices,
    calculate_configurational_disorder_index,
    calculate_nematic_order_parameter,
    calculate_nematic_order_per_gray_level,
    calculate_local_nematic_alignment,
    calculate_stress_features,
    calculate_orientational_correlation_length,
    calculate_advanced_eigen_features,
    calculate_diagonal_features,
    calculate_symmetry_features,
    calculate_rdf_shape_matrices,
    calculate_glam_wasserstein_distance,
    calculate_glam_assembly_coupling_matrix,
    calculate_glam_phenotypic_distance_matrix,
    calculate_cluster_features,
    calculate_profile_shape_features,
    calculate_glam_percolation,
    calculate_glam_granulometry,
    calculate_glam_frustration_index,
    calculate_glam_local_packing_fraction
)

from .utils import (
    find_scan_mask_pairs,
    generate_binary_mask,
    save_feature_dataframes,
    reformat_dict_to_matrix,
    save_matrix
)

# =============================================================================
# === HELPER FUNCTIONS ===
# =============================================================================

def perform_quantization(image_array, mask_array, method, num_levels,
                         q_min=None, q_max=None, bin_width=None,
                         global_min=None, global_max=None):
    """Normalize the whole crop consistently; quantize only the ROI."""
    image_array = np.asarray(image_array, dtype=np.float64)
    roi = np.asarray(mask_array) > 0
    if image_array.shape != roi.shape:
        raise ValueError("Image/mask array shapes differ")
    if not roi.any():
        return None
    if not np.isfinite(image_array).all():
        raise ValueError("Nonfinite image intensities: clean input before extraction")
    if not 1 <= num_levels <= 32767:
        raise ValueError("NumGrayLevels must be in 1..32767 for int16 storage")
    method = method.lower()
    print(f"  - Starting GLAM data preparation (Quantization: {method})...")
    if method == 'fixedcount':
        if (global_min is None) != (global_max is None):
            raise ValueError("Supply both global intensity bounds or neither")
        if global_min is None:
            image_min, image_max = np.percentile(image_array[roi], [1.0, 99.0])
        else:
            image_min, image_max = float(global_min), float(global_max)
        if not np.isfinite([image_min, image_max]).all() or image_max < image_min:
            raise ValueError("Invalid fixed-count intensity bounds")
        if image_max - image_min > 1e-6:
            rescaled_image_array = (
                np.clip(image_array, image_min, image_max) - image_min
            ) / (image_max - image_min)
        else:
            rescaled_image_array = np.zeros_like(image_array)
        q_vals = np.floor(rescaled_image_array[roi] * num_levels)
    elif method == 'fixedwidth':
        if q_min is None or q_max is None or bin_width is None:
            raise ValueError("Fixed-width quantization requires bounds and bin width")
        if not np.isfinite([q_min, q_max, bin_width]).all() or bin_width <= 0 or q_max <= q_min:
            raise ValueError("Invalid fixed-width bounds/bin width")
        expected_levels = int(np.ceil((q_max - q_min) / bin_width))
        if num_levels != expected_levels:
            raise ValueError("NumGrayLevels does not match fixed-width bounds")
        # Preserve physical intensity units, with clipping applied to the whole crop.
        rescaled_image_array = np.clip(image_array, q_min, q_max)
        q_vals = np.floor((rescaled_image_array[roi] - q_min) / bin_width)
    else:
        raise ValueError(f"Unknown QuantizationMethod: {method}")
    q_vals = np.clip(q_vals, 0, num_levels - 1).astype(np.int16)
    quantized_image = np.zeros(image_array.shape, dtype=np.int16)
    quantized_image[roi] = q_vals
    structured_glam_image = np.full(image_array.shape, -1, dtype=np.int16)
    structured_glam_image[roi] = q_vals
    return {
        "rescaled_image_array": rescaled_image_array,
        "quantized_image": quantized_image,
        "structured_glam_image": structured_glam_image,
        "roi_quantized_voxels": q_vals,
        "level_counts": np.bincount(q_vals, minlength=num_levels).tolist(),
        "total_roi_voxels": int(q_vals.size),
    }


def calculate_glam_rdfs(structured_glam_image, roi_quantized_voxels, num_levels,
                        max_radius, level_counts, total_roi_voxels):
    """Use identical ROI geometry and centered lattice shells for both RDFs."""
    n_random = int(get_config('NumRandomisations'))
    samples = int(get_config('RdfSamplePoints'))
    if n_random < 1:
        raise ValueError("NumRandomisations must be at least 1")
    roi = structured_glam_image >= 0
    if int(roi.sum()) != total_roi_voxels:
        raise ValueError("RDF ROI count mismatch")
    print("  - Calculating boundary-corrected structured RDF...")
    structured = calculate_rdf_3d(
        structured_glam_image, num_levels, max_radius, list(level_counts),
        total_roi_voxels, n_random, samples, sample_mask=roi)
    baseline_frames = []
    for i in range(n_random):
        shuffled = np.random.permutation(roi_quantized_voxels)
        randomized = np.full(structured_glam_image.shape, -1, dtype=np.int16)
        randomized[roi] = shuffled
        frame = calculate_rdf_3d(
            randomized, num_levels, max_radius, list(level_counts),
            total_roi_voxels, n_random, samples, sample_mask=roi)
        if frame is None or frame.empty:
            raise RuntimeError(f"Empty randomized RDF at replicate {i + 1}")
        if not np.array_equal(frame['r'].values, structured['r'].values):
            raise RuntimeError("Structured/random RDF radii differ")
        baseline_frames.append(frame)
    baseline = structured.copy()
    g_cols = [c for c in structured.columns if c.startswith('g_')]
    baseline[g_cols] = np.mean(
        np.stack([f[g_cols].to_numpy() for f in baseline_frames]), axis=0)
    return structured, baseline


def calculate_primary_glam_features(rdf_structured_df, rdf_random_df, structured_glam_image,
                                    num_levels, level_counts, total_roi_voxels, spacing): 
    """
    Calculates all primary (matrix-forming) GLAM features by calling core functions.
    """
    max_rdf_radius = get_config('MaxRdfRadius')
    anisotropy_cutoff_radius = get_config('AnisotropyCutoffRadius')
    
    spi_feats = calculate_glam_structural_pressure_index(rdf_structured_df, num_levels, level_counts, total_roi_voxels)
    cdi_feats = calculate_configurational_disorder_index(rdf_structured_df, rdf_random_df, num_levels)
    frustration_feats = calculate_glam_frustration_index(spi_feats, cdi_feats, num_levels)
    wasserstein_distance = calculate_glam_wasserstein_distance(rdf_structured_df, rdf_random_df, num_levels, level_counts, total_roi_voxels)

    print("  - Starting Primary GLAM Feature calculation...")
    glam_features = {
        **calculate_glam_b2_3d(rdf_structured_df, rdf_random_df, num_levels),
        **calculate_glam_inverse_correlation_length(rdf_structured_df, rdf_random_df, num_levels),
        **calculate_glam_coordination_number(rdf_structured_df, num_levels, level_counts, total_roi_voxels),
        **calculate_glam_compressibility(rdf_structured_df, num_levels),
        **calculate_anisotropic_glam_features(structured_glam_image, num_levels, anisotropy_cutoff_radius), 
        **calculate_glam_fractal_dimension(structured_glam_image, num_levels),
        **calculate_glam_multifractal_spectrum(structured_glam_image, num_levels),
        **calculate_glam_lacunarity(structured_glam_image, num_levels),
        **calculate_glam_topology(structured_glam_image, num_levels),
        **calculate_glam_potential_energy(rdf_structured_df, num_levels),
        **spi_feats,
        **cdi_feats,
        **frustration_feats,
        **calculate_glam_local_packing_fraction(rdf_structured_df, num_levels, level_counts, total_roi_voxels),
        **calculate_glam_shape_matrices(structured_glam_image, num_levels, spacing),
        **wasserstein_distance,
        **calculate_glam_assembly_coupling_matrix(wasserstein_distance, num_levels),
        **calculate_glam_phenotypic_distance_matrix(rdf_structured_df, num_levels, level_counts, total_roi_voxels),
        **calculate_glam_percolation(structured_glam_image, num_levels, total_roi_voxels),
        **calculate_glam_granulometry(structured_glam_image, num_levels, max_radius=10)
    }
    return glam_features

def calculate_scalar_glam_features(rescaled_image_array, mask_array, quantized_image,
                                   num_levels, max_radius):
    """Calculates all scalar-only GLAM features by calling core functions."""
    anisotropy_cutoff_radius = get_config('AnisotropyCutoffRadius')

    print("  - Starting Scalar GLAM Feature calculation...")
    scalar_features = calculate_nematic_order_parameter(rescaled_image_array, mask_array)

    nematic_s_per_gl_features = calculate_nematic_order_per_gray_level(
        rescaled_image_array, mask_array, quantized_image, num_levels
    )
    scalar_features.update(nematic_s_per_gl_features)

    scalar_features.update(calculate_local_nematic_alignment(
        rescaled_image_array, mask_array, anisotropy_cutoff_radius 
    ))
    scalar_features.update(calculate_stress_features(rescaled_image_array, mask_array))

    scalar_features.update(calculate_orientational_correlation_length(rescaled_image_array, mask_array, max_radius))

    return scalar_features

def calculate_advanced_meta_features(data, prefix, is_matrix=True):
    """
    Runs all meta-feature calculations (FirstOrder, SecondOrder, Advanced)
    on a given matrix or vector by calling core functions.
    """
    meta_features = {}
    meta_features.update(calculate_first_order_stats_from_matrix(data, f"{prefix}.FirstOrder"))
    meta_features.update(calculate_cluster_features(data, f"{prefix}.Advanced"))

    if is_matrix and data is not None and data.ndim == 2:
        meta_features.update(calculate_glcm_style_meta_features(data, f"{prefix}.SecondOrder"))
        meta_features.update(calculate_advanced_eigen_features(data, f"{prefix}.Advanced"))
        meta_features.update(calculate_diagonal_features(data, f"{prefix}.Advanced"))
        meta_features.update(calculate_symmetry_features(data, f"{prefix}.Advanced"))

    return meta_features

def calculate_random_baseline_features(structured_glam_image, roi_quantized_voxels, num_levels):
    """
    Generates a spatially randomized version of the tumor and calculates 
    Baseline Topology and Multifractal features.
    """
    print("  - Calculating Random Baseline (Null Model) for Topology & Fractals...")
    
    shuffled_voxels = roi_quantized_voxels.copy()
    np.random.shuffle(shuffled_voxels)
    
    randomized_image = np.full(structured_glam_image.shape, -1, dtype=np.int16)
    mask_indices = structured_glam_image > -1
    randomized_image[mask_indices] = shuffled_voxels
    
    # A. Topology
    random_topology = calculate_glam_topology(randomized_image, num_levels)
    random_topology = {f"Random_{k.replace('GLAM_', '')}": v for k, v in random_topology.items()}
    
    # B. Multifractals
    random_fractals = calculate_glam_multifractal_spectrum(randomized_image, num_levels)
    random_fractals = {f"Random_{k.replace('GLAM_', '')}": v for k, v in random_fractals.items()}
    
    return {**random_topology, **random_fractals}

def build_and_analyze_glam_matrices(primary_glam_features, scalar_glam_features,
                                    total_roi_voxels, num_levels, prefix, output_folder,
                                    rdf_structured_df): 
    """
    Builds all GLAM matrices, saves them, and calculates all meta-features.
    """
    print("  - Building, saving, and analyzing GLAM matrices...")
    glam_matrix_defs = {
        "B2": ("GLAM_B2_for_", None),
        "InverseCorrelationLength": ("GLAM_InverseCorrelationLength_", None),
        "CoordNum": ("GLAM_CoordNum_", None),
        "Anisotropy": ("GLAM_Anisotropy_", None),
        "PotentialEnergy": ("GLAM_PotentialEnergy_", None),
        "StructuralPressureIndex": ("GLAM_StructuralPressureIndex_", None),
        "ConfigurationalDisorderIndex": ("GLAM_ConfigurationalDisorderIndex_", None),
        "FrustrationIndex": ("GLAM_FrustrationIndex_", None), 
        "LocalPackingFraction": ("GLAM_LocalPackingFraction_", None),
        "Wasserstein": ("GLAM_Wasserstein_", None),
        "AssemblyCoupling": ("GLAM_AssemblyCoupling_", None),
        "PhenotypicDistance": ("GLAM_PhenotypicDistance_", None), 
        "FractalDimension": ("GLAM_InterfaceFD_", "GLAM_VolumeFD_"),
        "MultifractalWidth": ("GLAM_InterfaceMultifractal_Width_", "GLAM_VolumeMultifractal_Width_"),
        "MultifractalAlpha0": ("GLAM_InterfaceMultifractal_Alpha0_", "GLAM_VolumeMultifractal_Alpha0_"),
        "MultifractalD2": ("GLAM_InterfaceMultifractal_D2_", "GLAM_VolumeMultifractal_D2_"),
        "Lacunarity": ("GLAM_InterfaceLacunarity_", "GLAM_VolumeLacunarity_"),
        "Betti0": ("GLAM_InterfaceBetti0_", "GLAM_VolumeBetti0_"),
        "Betti1": ("GLAM_InterfaceBetti1_", "GLAM_VolumeBetti1_"),
        "Betti2": ("GLAM_InterfaceBetti2_", "GLAM_VolumeBetti2_"),
        "Euler": ("GLAM_InterfaceEuler_", "GLAM_VolumeEuler_"),
        "Shape_Sphericity":     (None, "GLAM_Shape_Sphericity_"),
        "Shape_Solidity":       (None, "GLAM_Shape_Solidity_"),
        "Shape_RadialMean":     (None, "GLAM_Shape_RadialMean_"),
        "Shape_RadialVariance": (None, "GLAM_Shape_RadialVariance_"),
        "Shape_RadialSkewness": (None, "GLAM_Shape_RadialSkewness_"),
        "Shape_RadialKurtosis": (None, "GLAM_Shape_RadialKurtosis_"),
        "Shape_CentroidDist":   ("GLAM_Shape_CentroidDist_", None),
        "Shape_InterfaceArea":  ("GLAM_Shape_InterfaceArea_", None),
        "MaxClusterSize": ("GLAM_InterfaceMaxClusterSize_", "GLAM_VolumeMaxClusterSize_"),
        "ClusterNumberDensity": ("GLAM_InterfaceClusterNumberDensity_", "GLAM_VolumeClusterNumberDensity_"),
        "PercolationStrength": ("GLAM_InterfacePercolationStrength_", "GLAM_VolumePercolationStrength_"),
        "GranulometryMean": ("GLAM_InterfaceGranulometryMean_", "GLAM_VolumeGranulometryMean_"),
        "GranulometryVariance": ("GLAM_InterfaceGranulometryVariance_", "GLAM_VolumeGranulometryVariance_"),
        "GranulometrySkewness": ("GLAM_InterfaceGranulometrySkewness_", "GLAM_VolumeGranulometrySkewness_"),
        "GranulometryKurtosis": ("GLAM_InterfaceGranulometryKurtosis_", "GLAM_VolumeGranulometryKurtosis_"),
        "GranulometryEntropy": ("GLAM_InterfaceGranulometryEntropy_", "GLAM_VolumeGranulometryEntropy_"),
    }

    comparison_targets = ['Betti0', 'Betti1', 'Betti2', 'Euler']

    glam_matrices = {}
    all_meta_features = {}

    # Build Standard Matrices
    for name, (prefix_str, diag_prefix) in glam_matrix_defs.items():
        matrix = reformat_dict_to_matrix(primary_glam_features, num_levels, prefix_str, diag_prefix)
        glam_matrices[name] = matrix
        
        if name in comparison_targets:
            rand_prefix = f"Random_Interface{name}_"
            rand_diag = f"Random_Volume{name}_"
            
            random_matrix = reformat_dict_to_matrix(primary_glam_features, num_levels, rand_prefix, rand_diag)
            
            if random_matrix is not None:
                glam_matrices[f"{name}_Random"] = random_matrix
                
                excess_matrix = matrix - random_matrix
                glam_matrices[f"{name}_Excess"] = excess_matrix
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_matrix = matrix / random_matrix
                    ratio_matrix[~np.isfinite(ratio_matrix)] = 1.0 
                    ratio_matrix[random_matrix == 0] = np.nan 
                    
                glam_matrices[f"{name}_Ratio"] = ratio_matrix

    # Robust EnergyDensity Calculation
    energy_density_matrix = np.full((num_levels, num_levels), np.nan) 
    if total_roi_voxels > 0 and 'PotentialEnergy' in glam_matrices:
        potential_energy_matrix = glam_matrices.get('PotentialEnergy')
        if potential_energy_matrix is not None and not np.all(np.isnan(potential_energy_matrix)):
            energy_density_matrix = potential_energy_matrix / total_roi_voxels
    glam_matrices['EnergyDensity'] = energy_density_matrix

    print("  - Calculating RDF shape matrices...")
    rdf_shape_matrices = calculate_rdf_shape_matrices(rdf_structured_df, num_levels)
    glam_matrices.update(rdf_shape_matrices)

    # --- Transformations (Symlog & Ln) ---
    symlog_targets = [  'StructuralPressureIndex', 'ConfigurationalDisorderIndex', 'B2', 'PotentialEnergy', 
                        'RDF_Kurtosis', 'RDF_Skewness', 'Euler', 
                        'Betti0_Excess', 'Betti1_Excess', 'Betti2_Excess', 'Euler_Excess',
                        'GranulometrySkewness', 'GranulometryKurtosis', 'AssemblyCoupling']
    
    for name in symlog_targets:
        if name in glam_matrices:
            mat = glam_matrices[name]
            if mat is not None and not np.all(np.isnan(mat)):
                glam_matrices[f'{name}_Symlog'] = np.sign(mat) * np.log1p(np.abs(mat))
            else:
                glam_matrices[f'{name}_Symlog'] = np.full_like(mat, np.nan) if mat is not None else None

    ln_targets = ['Lacunarity', 'RDF_Median', 'RDF_PeakHeight', 'CoordNum', 'RDF_Variance', 'RDF_DispersionRatio',
                  'GranulometryMean', 'GranulometryVariance']
    for name in ln_targets:
        if name in glam_matrices:
            mat = glam_matrices[name]
            if mat is not None and not np.all(np.isnan(mat)):
                with np.errstate(divide='ignore'):
                    ln_mat = np.log(mat)
                ln_mat[np.isneginf(ln_mat)] = np.nan
                glam_matrices[f'{name}_Ln'] = ln_mat
            else:
                glam_matrices[f'{name}_Ln'] = np.full_like(mat, np.nan) if mat is not None else None

    # Apply Log1p (Ln(1+x)) to Betti numbers AND Shape Interactions
    log1p_targets = ['Betti0', 'Betti1', 'Betti2', 'Shape_CentroidDist', 
                     'Shape_InterfaceArea', 'Wasserstein', 'MaxClusterSize',
                     'GranulometryEntropy']
    
    for metric in log1p_targets:
        if metric in glam_matrices:
            mat = glam_matrices[metric]
            glam_matrices[f'{metric}_Ln'] = np.log1p(mat) if mat is not None else None

    # --- Save and Analyze ---
    for name, matrix in glam_matrices.items():
        save_matrix(matrix, prefix, output_folder, f"GLAM_{name}_matrix")

        if name.endswith("_Random") or name.endswith("_Excess") or name.endswith("_Ratio"):
            continue

        all_meta_features.update(
            calculate_advanced_meta_features(matrix, f"GLAM.{name}", is_matrix=True)
        )

    s_per_gl_vector = np.array([
        scalar_glam_features.get(f'GLAM_NematicOrder_S_per_GL_{i}') for i in range(num_levels)
    ])
    all_meta_features.update(calculate_advanced_meta_features(
        s_per_gl_vector, "GLAM.NematicOrder.S_per_GL", is_matrix=False
    ))

    print("  - Calculating Advanced Profile Shape features (Peaks/Bimodality)...")
    shape_targets = [
        'Betti0_Ln', 'Betti1_Ln', 'Betti2_Ln', 'Euler_Symlog', 
        'FractalDimension', 'Lacunarity_Ln', 'MultifractalWidth', 'MultifractalAlpha0', 'MultifractalD2',
        'InverseCorrelationLength', 'Anisotropy',  # <--- UPDATED HERE
        'PotentialEnergy_Symlog', 'CoordNum_Ln', 'ConfigurationalDisorderIndex_Symlog', 'StructuralPressureIndex_Symlog',   
        'Shape_CentroidDist_Ln', 'Shape_InterfaceArea_Ln', 'MaxClusterSize_Ln'
    ]
    
    for name in shape_targets:
        if name in glam_matrices and glam_matrices[name] is not None:
            shape_feats = calculate_profile_shape_features(glam_matrices[name], f"GLAM.{name}")
            all_meta_features.update(shape_feats)

    return glam_matrices, all_meta_features

# =============================================================================
# === CUSTOM RADIOMICS PIPELINE ===
# =============================================================================

def process_custom_radiomics(image_array, mask_array, quantized_image, 
                             num_levels, prefix, output_dir, baseline_name="Original"):
    """
    Calculates 5 Texture Matrices (GLCM, GLRLM, GLSZM, GLDM, NGTDM).
    Saves Raw and Ln versions only. (SymLog removed).
    Calculates Scalar Features for all versions.
    """
    
    # --- CRITICAL FIX for 0-based vs 1-based indexing ---
    # The 'quantized_image' input is 0-based (0 to 15).
    # The matrix calculation functions (calculate_glcm_3d etc.) assume 1-based (1 to 16),
    # treating 0 as background.
    # We must shift the ROI voxels by +1 so they map to 1..16.
    
    radiomics_image = quantized_image.copy()
    # Only shift voxels inside the mask. Background (0) remains 0.
    radiomics_image[mask_array > 0] += 1 
    # ----------------------------------------------------

    features = {}
    
    matrix_pipeline = {
        "GLCM": (calculate_glcm_3d, lambda m, p: calculate_glcm_style_meta_features(m, p)),
        "GLRLM": (calculate_glrlm_3d, calculate_glrlm_features),
        "GLSZM": (calculate_glszm_3d, calculate_glszm_features),
        "GLDM": (calculate_gldm_3d, calculate_gldm_features),
        "NGTDM": (calculate_ngtdm_3d, calculate_ngtdm_features)
    }
    
    for mat_name, (calc_func, feat_func) in matrix_pipeline.items():
        # A. Calculate Raw Matrix using the shifted image
        raw_matrix = calc_func(radiomics_image, num_levels)
        
        # Save Raw
        mat_prefix = f"Radiomics_{baseline_name}_{mat_name}"
        save_matrix(raw_matrix, prefix, output_dir, f"{mat_prefix}_Raw")
        
        # B. Calculate Features on Raw
        raw_feats = feat_func(raw_matrix, f"Radiomics.{baseline_name}.{mat_name}.Raw")
        features.update(raw_feats)
        
        # C. Apply Transforms (Ln Only)
        # NGTDM is a feature table (N x 3), not a probability map.
        if mat_name == "NGTDM": 
            continue
            
        # Transform: Ln (Natural Log)
        with np.errstate(divide='ignore'):
            ln_matrix = np.log(raw_matrix)
        ln_matrix[np.isneginf(ln_matrix)] = 0 
        
        save_matrix(ln_matrix, prefix, output_dir, f"{mat_prefix}_Ln")
        
        ln_feats = feat_func(ln_matrix, f"Radiomics.{baseline_name}.{mat_name}.Ln")
        features.update(ln_feats)

    return features

# =============================================================================
# === WORKER FUNCTIONS ===
# =============================================================================

def _init_worker_config(config_path_for_worker):
    """Initializer for parallel workers."""
    print(f"[Worker {os.getpid()}] Initializing and loading config...")
    try:
        load_config(config_path_for_worker)
    except Exception as e:
        print(f"[Worker {os.getpid()}] FATAL ERROR: Could not load config. {e}")

def process_patient_folder_worker(patient_folder_tuple):
    """Worker function to process a single patient folder."""
    patient_folder_name, input_dir, output_dir, project_name, config_path = patient_folder_tuple

    print("-" * 60)
    print(f"Processing patient folder: {patient_folder_name} [Worker {os.getpid()}]")
    patient_dir_path = os.path.join(input_dir, patient_folder_name)

    scan_pairs = find_scan_mask_pairs(patient_dir_path)

    if not scan_pairs:
        print(f"  > DEBUG: No valid scan/mask pairs found in folder '{patient_folder_name}'. Skipping.")
        return [], []
        
    patient_output_dir = os.path.join(output_dir, patient_folder_name)
    try:
        os.makedirs(patient_output_dir, exist_ok=True)
    except OSError as e:
        print(f"  > ERROR: Could not create patient output directory {patient_output_dir}: {e}. Skipping.")
        return [], []
    
    all_primary_rows_worker = []
    all_meta_rows_worker = []

    for prefix, paths in scan_pairs.items(): 
        print(f"Processing scan set: {prefix} from {project_name}...")
        try:
            list_of_primary_rows, list_of_meta_rows = process_single_scan(
                prefix, paths, patient_output_dir, config_path 
            )
            
            if list_of_primary_rows:
                for row in list_of_primary_rows: row['project_name'] = project_name
                for row in list_of_meta_rows: row['project_name'] = project_name
                all_primary_rows_worker.extend(list_of_primary_rows)
                all_meta_rows_worker.extend(list_of_meta_rows)

        except Exception as e:
            print(f"!! FATAL ERROR processing {prefix}. Error: {e}")
            traceback.print_exc()

    return all_primary_rows_worker, all_meta_rows_worker

# =============================================================================
# === ORCHESTRATOR ===
# =============================================================================

def process_scans(input_dir, output_dir, project_name="DefaultProject", config_path=None):
    """Main processing loop (Orchestrator) - SYNCHRONOUS OVERRIDE."""
    all_primary_features = []
    all_meta_and_radiomics = []

    if config_path is None:
        print("Error: 'config_path' was not passed to process_scans.")
        return

    print("="*60)
    print(f"STARTING PROJECT: {project_name}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print("  Execution: FORCED SYNCHRONOUS (Bypassing Windows OS Pool Bug)")
    print("="*60)
    
    if not os.path.exists(input_dir): return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    try:
        patient_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    except Exception as e:
        return

    _init_worker_config(config_path)
    
    for patient_folder_name in patient_folders:
        task = (patient_folder_name, input_dir, output_dir, project_name, config_path)
        primary_rows, meta_rows = process_patient_folder_worker(task)
        if primary_rows: all_primary_features.extend(primary_rows)
        if meta_rows: all_meta_and_radiomics.extend(meta_rows)

    save_feature_dataframes(all_primary_features, all_meta_and_radiomics, output_dir)

def process_single_scan(prefix, paths, output_dir, config_path):
    """Processes a single patient."""
    label_mapping = get_config('LabelMapping')
    labels_for_analysis = get_config('LabelsForAnalysis')

    mask_path = paths.get('mask')
    norm_mask_path = paths.get('norm_mask')  # <--- NEW: Grab the norm mask path
    image_paths_dict = paths.get('images', {})
    
    if not mask_path or not image_paths_dict:
        print(f"  - ERROR: Incomplete file set for {prefix}. Skipping.")
        return [], []

    try:
        multilabel_mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    except Exception as e:
        print(f"  - ERROR: Could not read mask file {mask_path}: {e}")
        return [], []

    primary_rows_list = []
    meta_rows_list = []
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(multilabel_mask_sitk)
    present_labels = {l for l in stats.GetLabels() if l != 0}
    labels_to_process = {}

    if any(l in present_labels for l in label_mapping if l != 99):
        print(f"  - Multi-label mask detected. Processing {len(labels_for_analysis)} defined labels.")
        labels_to_process = labels_for_analysis
    elif present_labels == {1}:
        print("  - Simple binary mask (0/1) detected. Processing as single 'ROI'.")
        labels_to_process = {1: 'ROI'}
    else:
        print(f"  - WARNING: Skipping scan {prefix}. Unknown mask format.")
        return [], []

    # --- UPDATED: Precompute global intensity bounds (Whole Brain or Whole Tumor) ---
    method = get_config('QuantizationMethod').lower()
    global_bounds = {}

    if method == 'fixedcount':
        ref_mask_array = None
        
        # 1. Try to load the Normalization Mask first
        if norm_mask_path:
            print(f"  - Loading Normalization Mask for global intensity bounds: {os.path.basename(norm_mask_path)}")
            try:
                norm_mask_sitk = sitk.ReadImage(norm_mask_path, sitk.sitkUInt8)
                ref_mask_array = sitk.GetArrayFromImage(norm_mask_sitk)
            except Exception as e:
                print(f"  - WARNING: Could not read normalization mask {norm_mask_path}: {e}. Falling back to Whole Tumor.")
        
        # 2. Fall back to Whole Tumor if no norm mask is provided (or if it failed to load)
        if ref_mask_array is None:
            print("  - Precomputing global intensity bounds (Whole Tumor) for consistent quantization...")
            whole_tumor_mask_sitk = generate_binary_mask(multilabel_mask_sitk, 99)
            ref_mask_array = sitk.GetArrayFromImage(whole_tumor_mask_sitk)

        for seq_name, image_path in image_paths_dict.items():
            try:
                img_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
                img_arr = sitk.GetArrayFromImage(img_sitk)
                ref_voxels = img_arr[ref_mask_array > 0]

                if ref_voxels.size > 0:
                    seq_min = np.percentile(ref_voxels, 1.0)
                    seq_max = np.percentile(ref_voxels, 99.0)
                    global_bounds[seq_name] = (seq_min, seq_max)
                else:
                    global_bounds[seq_name] = (None, None)
            except Exception as e:
                print(f"  - WARNING: Could not precompute bounds for {seq_name}: {e}")
    # -----------------------------------------------------------------------------------

    for label_id, label_name in labels_to_process.items():
        print(f"  --- Processing Label: {label_name} ({label_id}) ---")
        
        binary_mask_sitk = generate_binary_mask(multilabel_mask_sitk, label_id)

        stats.Execute(binary_mask_sitk)
        if not stats.HasLabel(1):
            print(f"  - SKIPPING label {label_name}: No voxels found.")
            continue

        for seq_name, image_path in image_paths_dict.items():
            print(f"    --- Processing Sequence: {seq_name} ---")
            try:
                image_sitk = sitk.ReadImage(image_path, sitk.sitkFloat32)
            except Exception as e:
                print(f"    - ERROR: Could not read image file {image_path}: {e}")
                continue

            output_prefix = f"{prefix}_{label_name}_{seq_name}"
            try:
                # Fetch the precomputed global bounds for this sequence
                g_min, g_max = global_bounds.get(seq_name, (None, None))

                primary_row, meta_row = process_single_label(
                    output_prefix, image_sitk, binary_mask_sitk,
                    label_id, label_name, seq_name, output_dir, config_path,
                    global_min=g_min, global_max=g_max  
                )
                if primary_row and meta_row:
                    primary_rows_list.append(primary_row)
                    meta_rows_list.append(meta_row)
            except Exception as e:
                print(f"    - ERROR processing {output_prefix}: {e}")
                import traceback
                traceback.print_exc()

    return primary_rows_list, meta_rows_list

def process_single_label(prefix, image_sitk, binary_mask_sitk, label_id, label_name, seq_name, output_dir, config_path, global_min=None, global_max=None):
    """
    Processes a single image/label pair using Custom Radiomics + GLAM.
    """
    # Fetch parameters (num_gray_levels is now dynamically determined for FixedWidth)
    num_gray_levels = get_config('NumGrayLevels')
    max_rdf_radius = get_config('MaxRdfRadius')
    
    method = get_config('QuantizationMethod').lower()
    q_min = get_config('QuantizationMin') if method == 'fixedwidth' else None
    q_max = get_config('QuantizationMax') if method == 'fixedwidth' else None
    bin_width = get_config('BinWidth') if method == 'fixedwidth' else None

    # --- 1. Prepare Data ---
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array = sitk.GetArrayFromImage(binary_mask_sitk)
        if image_array.shape != mask_array.shape: return None, None

        # --- BOUNDING BOX CROP ---
        from scipy import ndimage
        slices = ndimage.find_objects((mask_array > 0).astype(int))
        crop_slices = None
        if slices:
            z_s, y_s, x_s = slices[0]
            pad = 2
            z_start, z_end = max(0, z_s.start - pad), min(image_array.shape[0], z_s.stop + pad)
            y_start, y_end = max(0, y_s.start - pad), min(image_array.shape[1], y_s.stop + pad)
            x_start, x_end = max(0, x_s.start - pad), min(image_array.shape[2], x_s.stop + pad)
            
            crop_slices = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
            
            image_array = image_array[crop_slices]
            mask_array = mask_array[crop_slices]
        # ----------------------------------

        # Pass the new arguments here
        prep_data = perform_quantization(
            image_array, mask_array, method, num_gray_levels, q_min, q_max, bin_width,
            global_min=global_min, global_max=global_max
        )
        if not prep_data: return None, None
        
    except Exception as e:
        print(f"  - ERROR during data preparation: {e}")
        return None, None

    # --- 2. Custom Radiomics (Original ROI) ---
    print("  - Calculating Custom Radiomics (Texture, Shape, FirstOrder)...")
    
    radiomic_features = {}
    
    # A. Shape
    shape_feats = calculate_shape_features_3d(binary_mask_sitk, "Radiomics.Original.Shape")
    radiomic_features.update(shape_feats)
    print("Shape features calculated.")

    # B. First Order
    fo_feats = calculate_first_order_features(
        prep_data['rescaled_image_array'], mask_array, "Radiomics.Original.FirstOrder"
    )
    radiomic_features.update(fo_feats)
    print("First Order features calculated.")


    # C. Texture Matrices
    texture_feats = process_custom_radiomics(
        image_array, mask_array, prep_data['quantized_image'],
        num_gray_levels, prefix, output_dir, baseline_name="Original"
    )
    radiomic_features.update(texture_feats)
    print("Radiomic features calculated.")

    # --- 3. Custom Radiomics (Randomized Baseline) ---
    print("  - Calculating Custom Radiomics (Randomized)...")
    roi_voxels = prep_data['quantized_image'][mask_array > 0]
    shuffled_voxels = roi_voxels.copy()
    np.random.shuffle(shuffled_voxels)
    
    random_quantized_image = np.zeros_like(prep_data['quantized_image'])
    random_quantized_image[mask_array > 0] = shuffled_voxels
    
    random_texture_feats = process_custom_radiomics(
        image_array, mask_array, random_quantized_image,
        num_gray_levels, prefix, output_dir, baseline_name="Random"
    )
    radiomic_features.update(random_texture_feats)

    # --- 4. GLAM Analysis ---
    rdf_structured_df, rdf_random_df = calculate_glam_rdfs(
        prep_data['structured_glam_image'], prep_data['roi_quantized_voxels'],
        num_gray_levels, max_rdf_radius, prep_data['level_counts'],
        prep_data['total_roi_voxels']
    )
    
    # geom_factor = calculate_geometric_factor(mask_array, max_rdf_radius, get_config('RdfSamplePoints'))
    # rdf_structured_corr = apply_geometric_correction(rdf_structured_df, geom_factor)
    # rdf_random_corr = apply_geometric_correction(rdf_random_df, geom_factor)

    save_matrix(rdf_structured_df, prefix, output_dir, "GLAM_RDF_structured_Corrected", columns=rdf_structured_df.columns)

    # --- GET SPACING ---
    spacing = image_sitk.GetSpacing() 
    # -------------------

    primary_glam_features = calculate_primary_glam_features(
        rdf_structured_df, rdf_random_df, prep_data['structured_glam_image'],
        num_gray_levels, prep_data['level_counts'], prep_data['total_roi_voxels'],
        spacing # <--- PASS IT
    )
    
    primary_glam_features.update(calculate_random_baseline_features(
        prep_data['structured_glam_image'], prep_data['roi_quantized_voxels'], num_gray_levels
    ))

    scalar_glam_features = calculate_scalar_glam_features(
        prep_data['rescaled_image_array'], mask_array, prep_data['quantized_image'],
        num_gray_levels, max_rdf_radius
    )

    glam_matrices, glam_meta_features = build_and_analyze_glam_matrices(
        primary_glam_features, scalar_glam_features, prep_data['total_roi_voxels'],
        num_gray_levels, prefix, output_dir, rdf_structured_df
    )
    
    # Combine everything
    all_primary_glam_features = {**primary_glam_features, **scalar_glam_features}
    
    primary_feature_row = {
        'scan_prefix': prefix, 'label_id': label_id, 'label_name': label_name,
        'sequence_name': seq_name, **all_primary_glam_features
    }
    
    meta_feature_row = {
        'scan_prefix': prefix, 'label_id': label_id, 'label_name': label_name,
        'sequence_name': seq_name, 
        **radiomic_features, 
        **glam_meta_features
    }

    # --- 5. Feature Mapping ---
    if get_config('EnableMapping'):
        try:
            full_mask_array = sitk.GetArrayFromImage(binary_mask_sitk)
            cropped_quantized = prep_data['quantized_image']
            
            if crop_slices is None:
                mapping_quantized = cropped_quantized
            else:
                mapping_quantized = np.full(full_mask_array.shape, -1, dtype=np.int16)
                mapping_quantized[crop_slices] = cropped_quantized
                
            if mapping_quantized.shape != full_mask_array.shape:
                raise ValueError(
                    f"Mapping shape mismatch: quantized={mapping_quantized.shape}, mask={full_mask_array.shape}"
                )
                
            mapping.generate_feature_maps(
                image_sitk, binary_mask_sitk, mapping_quantized,
                num_gray_levels, prefix, output_dir, config_path
            )
        except Exception:
            print("  - ERROR: Feature Mapping failed:")
            import traceback
            traceback.print_exc()

    return primary_feature_row, meta_feature_row

def main():
    parser = argparse.ArgumentParser(description="Run GLAM Radiomics Analysis")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file (config.ini).")
    parser.add_argument("input_dir", help="Path to the directory containing NIfTI scans and masks.")
    parser.add_argument("output_dir", help="Path to the directory where results will be saved.")
    parser.add_argument("-p", "--project_name", default="GLAM_Analysis", help="Optional name for this analysis run.")
    args = parser.parse_args()

    try:
        load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return 
    
    print("="*60)
    print(f"     3D GLAM & RADIOMICS ANALYZER ({args.project_name})")
    print("="*60)
    np.random.seed(42)

    process_scans(args.input_dir, args.output_dir, args.project_name, config_path=args.config)

    print("Analysis finished.")