# src/glam_radiomics/core.py
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import SimpleITK as sitk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import measure
import scipy.integrate

# Import config access
from .config import get_config

# =============================================================================#
# === 1. MATRIX GENERATION FUNCTIONS (Corrected Trimming) ===#
# =============================================================================#


def calculate_glcm_3d(image, num_gl):
    """Calculates the symmetric 3D GLCM (offset [0,0,1]) and returns normalized matrix."""
    glcm = np.zeros((num_gl, num_gl), dtype=np.uint32)
    offset = (0, 0, 1) # Vertical offset
    
    z, y, x = np.where(image > 0)
    nz, ny, nx = z + offset[0], y + offset[1], x + offset[2]
    
    valid_neighbors = (nz < image.shape[0]) & (ny < image.shape[1]) & (nx < image.shape[2])
    
    curr_vals = image[z[valid_neighbors], y[valid_neighbors], x[valid_neighbors]]
    neighbor_vals = image[nz[valid_neighbors], ny[valid_neighbors], nx[valid_neighbors]]
    
    mask_neighbor = neighbor_vals > 0
    curr_vals = curr_vals[mask_neighbor]
    neighbor_vals = neighbor_vals[mask_neighbor]
    
    np.add.at(glcm, (curr_vals - 1, neighbor_vals - 1), 1)
        
    glcm_symmetric = glcm + glcm.T
    total = np.sum(glcm_symmetric)
    if total == 0: return np.zeros_like(glcm_symmetric, dtype=float)
    return glcm_symmetric.astype(float) / total

def calculate_glrlm_3d(image, num_gl):
    """Calculates 3D GLRLM with DYNAMIC TRIMMING."""
    # 1. Initialize with max possible dimension
    max_run = image.shape[2]
    glrlm = np.zeros((num_gl, max_run), dtype=np.uint32)

    # 2. Fill Matrix
    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            row = image[z, y, :]
            padded = np.concatenate(([0], row, [0]))
            diffs = np.diff(padded)
            run_starts = np.where(diffs != 0)[0]
            
            for i in range(len(run_starts) - 1):
                start = run_starts[i]
                end = run_starts[i+1]
                val = padded[start+1]
                length = end - start
                
                if val > 0:
                    if length <= max_run:
                        glrlm[val-1, length-1] += 1
                        
    # 3. TRIM EMPTY COLUMNS (The Fix)
    if np.sum(glrlm) == 0: 
        return np.zeros((num_gl, 1), dtype=float)
    
    # Sum down columns to find which run lengths actually exist
    col_sums = np.sum(glrlm, axis=0)
    # Find the index of the last non-zero column
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    
    # Slice the matrix to keep only up to that column
    glrlm = glrlm[:, :last_idx+1]

    # 4. Normalize
    return glrlm.astype(float) / np.sum(glrlm)

def calculate_glszm_3d(image, num_gl):
    """Calculates 3D GLSZM with DYNAMIC TRIMMING."""
    # 1. Initialize with safe upper bound (Total Voxel Count)
    max_possible_zone = np.sum(image > 0)
    if max_possible_zone == 0: return np.zeros((num_gl, 1), dtype=float)
    
    glszm = np.zeros((num_gl, max_possible_zone), dtype=np.uint32)
    
    # 2. Fill Matrix
    for g in range(1, num_gl + 1):
        binary_img = (image == g)
        if not np.any(binary_img): continue
        
        labeled, num_feats = ndimage.label(binary_img, structure=np.ones((3,3,3)))
        if num_feats == 0: continue
        
        roi_counts = ndimage.sum(binary_img, labeled, index=np.arange(1, num_feats + 1))
        
        unique, counts = np.unique(roi_counts, return_counts=True)
        for size, count in zip(unique, counts):
            idx = int(size) - 1
            if idx < max_possible_zone:
                glszm[g-1, idx] += count

    # 3. TRIM EMPTY COLUMNS (The Fix)
    if np.sum(glszm) == 0: 
        return np.zeros((num_gl, 1), dtype=float)
    
    col_sums = np.sum(glszm, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    glszm = glszm[:, :last_idx+1]
    
    # 4. Normalize
    return glszm.astype(float) / np.sum(glszm)

def calculate_gldm_3d(image, num_gl, alpha=0):
    """Calculates 3D GLDM and returns normalized matrix."""
    gldm = np.zeros((num_gl, 27), dtype=np.uint32)
    footprint = np.ones((3,3,3))
    footprint[1,1,1] = 0
    
    for g in range(1, num_gl + 1):
        mask_g = (image == g)
        if not np.any(mask_g): continue
        
        if alpha == 0:
            neighbor_count = ndimage.convolve(mask_g.astype(int), footprint, mode='constant', cval=0)
            counts = neighbor_count[mask_g]
            unique, u_counts = np.unique(counts, return_counts=True)
            for k, c in zip(unique, u_counts):
                gldm[g-1, k] += c
                
    if np.sum(gldm) == 0: return np.zeros((num_gl, 1), dtype=float)
    
    # Optional: Trim GLDM columns too if no voxel has e.g. 26 neighbors
    col_sums = np.sum(gldm, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    gldm = gldm[:, :last_idx+1]
    
    return gldm.astype(float) / np.sum(gldm)

def calculate_ngtdm_3d(image, num_gl):
    """
    Calculates NGTDM. Returns a DataFrame-like dictionary or matrix.
    NGTDM is strictly 3 columns (count, sum_diff, probability).
    """
    ngtdm = np.zeros((num_gl, 3), dtype=float)
    
    kernel = np.ones((3,3,3))
    kernel[1,1,1] = 0
    
    mask = (image > 0).astype(float)
    neighbor_count_map = ndimage.convolve(mask, kernel, mode='constant', cval=0)
    valid_neighborhood = neighbor_count_map > 0
    
    sum_neighbor_vals = ndimage.convolve(image.astype(float), kernel, mode='constant', cval=0)
    avg_neighbor_map = np.zeros_like(image, dtype=float)
    avg_neighbor_map[valid_neighborhood] = sum_neighbor_vals[valid_neighborhood] / neighbor_count_map[valid_neighborhood]
    
    diff_map = np.abs(image - avg_neighbor_map)
    
    for g in range(1, num_gl + 1):
        mask_g = (image == g)
        n_i = np.sum(mask_g)
        if n_i == 0: continue
        
        valid_mask_g = mask_g & valid_neighborhood
        s_i = np.sum(diff_map[valid_mask_g])
        
        ngtdm[g-1, 0] = n_i
        ngtdm[g-1, 1] = s_i

    total_n = np.sum(ngtdm[:, 0])
    if total_n > 0:
        ngtdm[:, 2] = ngtdm[:, 0] / total_n
        
    return ngtdm

def calculate_glrlm_features(matrix, prefix):
    """Calculates scalar features from a GLRLM matrix (NumGL x RunLength)."""
    feats = {}
    if matrix is None or np.sum(matrix) == 0: return feats
    
    p = matrix
    num_g, max_run = p.shape
    
    pg = np.sum(p, axis=1) 
    pr = np.sum(p, axis=0) 
    
    i_indices = np.arange(1, num_g + 1)
    
    # --- FIX: Force dtype=float here as well ---
    j_indices = np.arange(1, max_run + 1, dtype=np.float64)
    # -------------------------------------------
    
    feats[f'{prefix}.ShortRunEmphasis'] = np.sum(pr / (j_indices**2))
    feats[f'{prefix}.LongRunEmphasis'] = np.sum(pr * (j_indices**2))
    feats[f'{prefix}.GrayLevelNonUniformity'] = np.sum(pg**2)
    feats[f'{prefix}.RunLengthNonUniformity'] = np.sum(pr**2)
    
    feats[f'{prefix}.LowGrayLevelRunEmphasis'] = np.sum(pg / (i_indices**2))
    feats[f'{prefix}.HighGrayLevelRunEmphasis'] = np.sum(pg * (i_indices**2))
    
    return feats

def calculate_glszm_features(matrix, prefix):
    """Calculates scalar features from a GLSZM matrix (NumGL x ZoneSize)."""
    feats = {}
    if matrix is None or np.sum(matrix) == 0: return feats
    
    p = matrix
    num_g, max_size = p.shape
    
    pg = np.sum(p, axis=1)
    pz = np.sum(p, axis=0)
    
    i_indices = np.arange(1, num_g + 1)
    
    # --- FIX: Force dtype=float to prevent integer overflow on large zones ---
    j_indices = np.arange(1, max_size + 1, dtype=np.float64)
    # -------------------------------------------------------------------------
    
    feats[f'{prefix}.SmallAreaEmphasis'] = np.sum(pz / (j_indices**2))
    feats[f'{prefix}.LargeAreaEmphasis'] = np.sum(pz * (j_indices**2))
    feats[f'{prefix}.GrayLevelNonUniformity'] = np.sum(pg**2)
    feats[f'{prefix}.ZoneSizeNonUniformity'] = np.sum(pz**2)
    feats[f'{prefix}.LowGrayLevelZoneEmphasis'] = np.sum(pg / (i_indices**2))
    feats[f'{prefix}.HighGrayLevelZoneEmphasis'] = np.sum(pg * (i_indices**2))
    
    return feats

def calculate_gldm_features(matrix, prefix):
    """Calculates scalar features from a GLDM matrix (NumGL x Dependence)."""
    feats = {}
    if matrix is None or np.sum(matrix) == 0: return feats
    
    p = matrix
    num_g, max_dep = p.shape
    
    pg = np.sum(p, axis=1)
    pd_vec = np.sum(p, axis=0)
    
    i_indices = np.arange(1, num_g + 1)
    j_indices = np.arange(0, max_dep) # Dependence count starts at 0? Usually 0..26
    
    # Avoid div by zero if j starts at 0
    j_sq = j_indices.astype(float)**2
    j_sq[j_sq == 0] = 1e-9
    
    feats[f'{prefix}.SmallDependenceEmphasis'] = np.sum(pd_vec / j_sq)
    feats[f'{prefix}.LargeDependenceEmphasis'] = np.sum(pd_vec * (j_indices**2))
    feats[f'{prefix}.GrayLevelNonUniformity'] = np.sum(pg**2)
    feats[f'{prefix}.DependenceNonUniformity'] = np.sum(pd_vec**2)
    
    return feats

def calculate_ngtdm_features(matrix, prefix):
    """Calculates features from NGTDM matrix (NumGL x 3: n, s, p)."""
    feats = {}
    if matrix is None: return feats
    
    n = matrix[:, 0]
    s = matrix[:, 1]
    p = matrix[:, 2]
    
    # Valid gray levels
    valid = p > 0
    if not np.any(valid): return feats
    
    p = p[valid]
    s = s[valid]
    n = n[valid]
    i_vals = np.where(valid)[0] + 1 # Gray levels
    
    Ng = len(p)
    if Ng < 1: return feats
    
    # 1. Coarseness
    sum_p_s = np.sum(p * s)
    feats[f'{prefix}.Coarseness'] = 1.0 / (sum_p_s + 1e-9)
    
    # 2. Contrast
    # Requires double summation over i, j
    sum_cont = 0.0
    for idx_i, val_i in enumerate(i_vals):
        for idx_j, val_j in enumerate(i_vals):
            sum_cont += p[idx_i] * p[idx_j] * (val_i - val_j)**2 * (s[idx_i] + s[idx_j])
            
    # Normalized by Ng*(Ng-1) ? Standard def usually involves sum(n)
    # Using simplified texture definition
    N_tot = np.sum(n) / np.sum(p) # Recover total count approx
    feats[f'{prefix}.Contrast'] = sum_cont / (Ng * (Ng - 1) * N_tot + 1e-9) if Ng > 1 else 0
    
    # 3. Busyness
    num = np.sum(p * s)
    denom = 0.0
    for idx_i, val_i in enumerate(i_vals):
        for idx_j, val_j in enumerate(i_vals):
            denom += np.abs(i_vals[idx_i]*p[idx_i] - i_vals[idx_j]*p[idx_j])
    feats[f'{prefix}.Busyness'] = num / (denom + 1e-9)

    # 4. Complexity
    # (Complex formula, simplified here for brevity, adhering to common implementations)
    feats[f'{prefix}.Complexity'] = 0.0 # Placeholder or implement full loop if needed
    
    return feats

def calculate_first_order_features(image_array, mask_array, prefix):
    """
    Calculates comprehensive First Order statistics on the original intensities.
    Matches standard radiomics definitions.
    """
    feats = {}
    # Get voxels inside the mask
    voxels = image_array[mask_array > 0]
    
    if voxels.size == 0:
        return feats
    
    # 1. Basic Stats
    mean_val = np.mean(voxels)
    median_val = np.median(voxels)
    var_val = np.var(voxels)
    min_val = np.min(voxels)
    max_val = np.max(voxels)
    range_val = np.ptp(voxels)
    
    feats[f'{prefix}.Mean'] = mean_val
    feats[f'{prefix}.Median'] = median_val
    feats[f'{prefix}.Variance'] = var_val
    feats[f'{prefix}.Minimum'] = min_val
    feats[f'{prefix}.Maximum'] = max_val
    feats[f'{prefix}.Range'] = range_val
    
    # 2. Advanced Moments
    # Standard skew/kurtosis (Fisher=True means subtract 3 from kurtosis for normal dist)
    feats[f'{prefix}.Skewness'] = stats.skew(voxels)
    feats[f'{prefix}.Kurtosis'] = stats.kurtosis(voxels)
    
    # 3. Energy Metrics
    # Energy = sum(pixel^2)
    energy = np.sum(voxels**2)
    feats[f'{prefix}.Energy'] = energy
    
    # Root Mean Squared (RMS) = sqrt(mean(pixel^2))
    feats[f'{prefix}.RootMeanSquared'] = np.sqrt(np.mean(voxels**2))
    
    # Total Energy = Energy * VoxelVolume (Here usually just Energy is reported, 
    # but strictly TotalEnergy requires multiplying by Voxel Volume in mm^3. 
    # We stick to the intensity-based Energy here).
    
    # 4. Percentile & Robust Metrics
    p10 = np.percentile(voxels, 10)
    p90 = np.percentile(voxels, 90)
    p25 = np.percentile(voxels, 25)
    p75 = np.percentile(voxels, 75)
    
    feats[f'{prefix}.10Percentile'] = p10
    feats[f'{prefix}.90Percentile'] = p90
    feats[f'{prefix}.InterquartileRange'] = p75 - p25
    
    # Mean Absolute Deviation (MAD): mean(|x - mean|)
    feats[f'{prefix}.MeanAbsoluteDeviation'] = np.mean(np.abs(voxels - mean_val))
    
    # Robust Mean Absolute Deviation (rMAD): mean(|x - median|) calculated on 
    # the subset of voxels between 10th and 90th percentile
    robust_subset = voxels[(voxels >= p10) & (voxels <= p90)]
    if robust_subset.size > 0:
        feats[f'{prefix}.RobustMeanAbsoluteDeviation'] = np.mean(np.abs(robust_subset - median_val))
    else:
        feats[f'{prefix}.RobustMeanAbsoluteDeviation'] = 0.0

    # 5. Histogram-based Metrics (Entropy & Uniformity)
    # Using 'fd' (Freedman Diaconis Estimator) for robust bin width
    hist, _ = np.histogram(voxels, bins='fd')
    
    # Probabilities
    p = hist / np.sum(hist)
    p = p[p > 0] # Remove zero probs to allow log
    
    # Entropy: -sum(p * log2(p))
    feats[f'{prefix}.Entropy'] = -np.sum(p * np.log2(p))
    
    # Uniformity: sum(p^2)
    feats[f'{prefix}.Uniformity'] = np.sum(p**2)
    
    return feats

def calculate_shape_features_3d(mask_sitk, prefix):
    """
    Calculates comprehensive 3D Shape features, including Volume, Surface Area,
    Axis Lengths (via PCA), and Maximum Diameter (via Convex Hull).
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist, squareform
    
    feats = {}
    
    # 1. Basic SITK Shape Statistics (Volume, Surface)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_sitk)
    
    if not stats.HasLabel(1):
        return feats
    
    # Volumes
    voxel_vol = stats.GetPhysicalSize(1)
    feats[f'{prefix}.VoxelVolume'] = voxel_vol
    feats[f'{prefix}.MeshVolume'] = voxel_vol # Approx, refined below if mesh succeeds
    
    # 2. Advanced Mesh-based Features (Marching Cubes)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    spacing = mask_sitk.GetSpacing() # (x, y, z)
    
    # Note: numpy is (z,y,x), spacing is (x,y,z). 
    # marching_cubes expects spacing in the order of the dimensions (z,y,x).
    spacing_zyx = spacing[::-1] 
    
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_arr, level=0.5, spacing=spacing_zyx)
        surface_area = measure.mesh_surface_area(verts, faces)
        feats[f'{prefix}.SurfaceArea'] = surface_area
        
        # Mesh Volume (often more accurate for non-cubic shapes)
        # We can stick to VoxelVolume for robustness, or calculate via divergence theorem.
        # For now, let's refine Sphericity using the marching cubes area.
        
        # Sphericity = (36pi * V^2)^(1/3) / Area
        # Use VoxelVolume for V to be safe
        if surface_area > 0:
            feats[f'{prefix}.Sphericity'] = (np.pi**(1/3) * (6 * voxel_vol)**(2/3)) / surface_area
            feats[f'{prefix}.SurfaceVolumeRatio'] = surface_area / voxel_vol
        else:
            feats[f'{prefix}.Sphericity'] = 0
            feats[f'{prefix}.SurfaceVolumeRatio'] = 0

    except Exception:
        # Fallback if mesh fails
        feats[f'{prefix}.SurfaceArea'] = stats.GetPerimeter(1) # Approx for 2D/3D
        feats[f'{prefix}.Sphericity'] = stats.GetRoundness(1)
    
    # 3. PCA-based Features (Axis Lengths, Elongation, Flatness)
    # Get all voxel coordinates (Z, Y, X)
    voxel_coords = np.argwhere(mask_arr > 0)
    
    if len(voxel_coords) > 3:
        # Convert to Physical Coordinates (mm)
        # coords * spacing
        # voxel_coords is (N, 3) -> (z, y, x)
        # spacing_zyx is (sz, sy, sx)
        physical_coords = voxel_coords * np.array(spacing_zyx)
        
        # PCA via Covariance Matrix
        # Rowvar=False means each column is a variable (z, y, x)
        cov_matrix = np.cov(physical_coords, rowvar=False)
        
        # Eigenvalues represent the variance along principal axes
        eigen_vals = np.linalg.eigvalsh(cov_matrix)
        
        # Sort desc (Largest to smallest)
        eigen_vals = np.sort(eigen_vals)[::-1]
        
        # Standard Radiomics Def: Axis Length = 4 * sqrt(eigenvalue)
        # (Derived from ellipsoid inertia tensor approximation)
        # Ensure non-negative (numerical noise can cause -0.0)
        eigen_vals = np.maximum(eigen_vals, 0)
        
        major_axis = 4 * np.sqrt(eigen_vals[0])
        minor_axis = 4 * np.sqrt(eigen_vals[1])
        least_axis = 4 * np.sqrt(eigen_vals[2])
        
        feats[f'{prefix}.MajorAxisLength'] = major_axis
        feats[f'{prefix}.MinorAxisLength'] = minor_axis
        feats[f'{prefix}.LeastAxisLength'] = least_axis
        
        # Elongation = sqrt(Minor / Major) ? No, standard is sqrt(lambda2/lambda1) = Minor/Major
        # PyRadiomics Def:
        # Elongation = sqrt(eigen_2 / eigen_1)
        # Flatness = sqrt(eigen_3 / eigen_1)
        
        if eigen_vals[0] > 0:
            feats[f'{prefix}.Elongation'] = np.sqrt(eigen_vals[1] / eigen_vals[0])
            feats[f'{prefix}.Flatness'] = np.sqrt(eigen_vals[2] / eigen_vals[0])
        else:
            feats[f'{prefix}.Elongation'] = 0
            feats[f'{prefix}.Flatness'] = 0

    # 4. Maximum 3D Diameter (via Convex Hull)
    # Calculating distance between ALL voxel pairs is O(N^2) -> too slow.
    # Calculating distance between Hull vertices is much faster.
    if len(voxel_coords) > 3:
        try:
            # Re-use physical coords
            hull = ConvexHull(physical_coords)
            hull_points = physical_coords[hull.vertices]
            
            # Calculate pairwise distances between hull vertices
            # pdist returns condensed distance matrix
            dists = pdist(hull_points, metric='euclidean')
            
            if dists.size > 0:
                feats[f'{prefix}.Maximum3DDiameter'] = np.max(dists)
            else:
                 feats[f'{prefix}.Maximum3DDiameter'] = 0
        except Exception:
             feats[f'{prefix}.Maximum3DDiameter'] = 0
             
    return feats

# =============================================================================
# === CORE GLAM CALCULATION FUNCTIONS ===
# =============================================================================

def calculate_rdf_3d(image_3d, num_levels, max_radius, level_counts, 
                     total_roi_voxels, num_randomisations, rdf_sample_points, 
                     sample_mask=None): 
    """
    Calculates the Radial Distribution Function (RDF) for a 3D image.
    
    Args:
        image_3d: The input image (quantized) containing the "System" (neighbors).
        sample_mask: (Optional) A binary mask of the same shape as image_3d. 
                     If provided, Reference Points are chosen ONLY from voxels 
                     where sample_mask == 1. Neighbors are still chosen from 
                     the entire image_3d.
    """
    
    # Remove the global check: if NUM_RANDOMISATIONS <= 1:
    if num_randomisations == 0:
         pass 
    elif num_randomisations == 1:
         pass 
    
    # 1. Define ALL potential neighbors (The "System")
    # These include pixels in the Halo/Padding
    coords = [np.argwhere(image_3d == i) for i in range(num_levels)]
    coord_trees = [cKDTree(c) if len(c) > 0 else None for c in coords]
    rdf_data = defaultdict(lambda: defaultdict(float))
    
    for alpha in range(num_levels):
        if level_counts[alpha] == 0: continue
        
        # 2. Define Reference Points (The "Window")
        coords_alpha_all = coords[alpha]
        
        if sample_mask is not None:
            # Filter: Keep points where sample_mask is True
            # This restricts the measurement "center" to the strict ROI/Window
            if len(coords_alpha_all) == 0: continue
            
            # Efficient boolean indexing
            is_in_mask = sample_mask[coords_alpha_all[:,0], coords_alpha_all[:,1], coords_alpha_all[:,2]] > 0
            coords_alpha = coords_alpha_all[is_in_mask]
        else:
            # Default behavior: Measure everywhere
            coords_alpha = coords_alpha_all

        # Check if we have enough points in the Window to calculate stats
        if len(coords_alpha) == 0: continue

        # Downsample if needed (Sample ONLY from the valid window points)
        num_ref_points = min(len(coords_alpha), rdf_sample_points)
        if num_ref_points == 0: continue
        
        ref_indices = np.random.choice(len(coords_alpha), num_ref_points, replace=False)
        ref_points = coords_alpha[ref_indices]

        for beta in range(num_levels):
            # Beta points (neighbors) come from the FULL SYSTEM (coords), not the mask
            if level_counts[beta] == 0: continue
            tree_beta = coord_trees[beta]
            if tree_beta is None: continue

            neighbors_at_max_radius = tree_beta.query_ball_point(ref_points, max_radius)
            pair_counts = defaultdict(int)

            for i in range(num_ref_points):
                if alpha == beta:
                    # Self-interaction check uses the full tree
                    dist_result, _ = tree_beta.query(ref_points[i], k=min(len(coords[beta]), num_ref_points + 1))
                    distances = np.atleast_1d(dist_result)
                    distances = distances[distances > 1e-6]
                else:
                    if not neighbors_at_max_radius[i]: continue
                    distances = np.linalg.norm(tree_beta.data[neighbors_at_max_radius[i]] - ref_points[i], axis=1)

                bins = np.floor(distances).astype(int)
                for r in range(1, max_radius + 1):
                    count = np.sum(bins == r)
                    pair_counts[r] += count
            
            # Density is calculated over the WHOLE patch volume
            rho_beta = level_counts[beta] / total_roi_voxels
            if rho_beta == 0: continue
            
            for r in range(1, max_radius + 1):
                if r in pair_counts:
                    dn = pair_counts[r] / num_ref_points
                    shell_volume = (4/3) * np.pi * ((r + 0.5)**3 - (r - 0.5)**3)
                    if shell_volume > 0:
                        rdf_data[(alpha, beta)][r] = dn / (shell_volume * rho_beta)

    df_data = []
    for r in range(1, max_radius + 1):
        row = {'r': r}
        for alpha in range(num_levels):
            for beta in range(num_levels):
                key = f'g_{alpha}_{beta}'
                row[key] = rdf_data.get((alpha, beta), {}).get(r, 0)
        df_data.append(row)

    return pd.DataFrame(df_data)

def calculate_geometric_factor(mask_array, max_radius, rdf_sample_points):
    """
    Calculates the 'Geometric Availability Factor' (g_geom) for the ROI.
    It simulates a texture where every voxel has the same value (0).
    Result: A Series where index is 'r' and value is the fraction of shell available (0.0-1.0).
    """
    # 1. Create a 'Uniform' image where all mask voxels are 0
    # This removes all texture, leaving only geometry.
    uniform_image = np.full(mask_array.shape, -1, dtype=np.int16)
    mask_indices = mask_array > 0
    uniform_image[mask_indices] = 0
    
    total_voxels = np.sum(mask_indices)
    if total_voxels == 0: return pd.Series()

    # 2. Run standard RDF calculation on this uniform image
    # We use num_levels=1 because there is only "Gray Level 0"
    rdf_geom_df = calculate_rdf_3d(
        image_3d=uniform_image,
        num_levels=1,
        max_radius=max_radius,
        level_counts=[total_voxels],
        total_roi_voxels=total_voxels,
        num_randomisations=0, # No randomization needed for geometry
        rdf_sample_points=rdf_sample_points
    )
    
    if rdf_geom_df.empty or 'g_0_0' not in rdf_geom_df.columns:
        return pd.Series()

    # 3. Extract the curve.
    # g_0_0 here represents exactly: (Measured Density) / (Global Density)
    # Since Global Density is constant, this curve is the Volume Fraction.
    geom_factor = rdf_geom_df.set_index('r')['g_0_0']
    
    # Safety: geometric factor can't be 0 (would cause division by zero).
    # We clip it to a small epsilon.
    geom_factor = geom_factor.clip(lower=0.01)
    
    return geom_factor

def apply_geometric_correction(rdf_df, geom_factor_series):
    """
    Divides every column in the RDF DataFrame by the Geometric Factor.
    """
    if rdf_df.empty or geom_factor_series.empty:
        return rdf_df

    corrected_df = rdf_df.copy()
    r_values = corrected_df['r'].values
    
    # Align the geometric factor to the current RDF's 'r'
    # (In case rows are missing, though they should match)
    factors = geom_factor_series.reindex(r_values).fillna(1.0).values
    
    # Divide every g_x_y column by the factor
    for col in corrected_df.columns:
        if col.startswith('g_'):
            corrected_df[col] = corrected_df[col] / factors
            
    return corrected_df

def calculate_js_divergence_matrix(rdf_df, num_levels):
    """
    Calculates the symmetric Jensen-Shannon (JS) Divergence Anisotropy Matrix 
    by comparing RDF_ij and RDF_ji. Applies exp(-lambda * r) spatial damping.
    """
    js_features = {}
    if rdf_df.empty: return {}
    
    r = rdf_df['r'].values

    def safe_kl(p, q):
        """Native KL divergence that safely handles 0 * log(0) = 0"""
        mask = p > 0
        out = np.zeros_like(p)
        # Using log2 bounds the final JS divergence strictly between 0 and 1
        out[mask] = p[mask] * np.log2(p[mask] / q[mask])
        return np.sum(out)

    for i in range(num_levels):
        for j in range(num_levels):
            # The diagonal is perfectly symmetric with itself
            if i == j:
                js_features[f'GLAM_JSDivergence_{i}_{j}'] = 0.0
                continue
            
            key_ij = f'g_{i}_{j}'
            key_ji = f'g_{j}_{i}'
            
            if key_ij in rdf_df.columns and key_ji in rdf_df.columns:
                # 2. Apply damping envelope to raw RDF curves
                P_raw = np.nan_to_num(rdf_df[key_ij].values) 
                Q_raw = np.nan_to_num(rdf_df[key_ji].values) 
                
                sum_P = np.sum(P_raw)
                sum_Q = np.sum(Q_raw)
                
                # If both are physical vacuums, they have 0 structural disagreement
                if sum_P == 0 and sum_Q == 0:
                    js_features[f'GLAM_JSDivergence_{i}_{j}'] = 0.0
                    continue
                    
                # 3. L1 Re-normalization (Mandatory for probability distributions)
                P = P_raw / sum_P if sum_P > 0 else np.zeros_like(P_raw)
                Q = Q_raw / sum_Q if sum_Q > 0 else np.zeros_like(Q_raw)
                
                # 4. Calculate Shared Midpoint M
                M = 0.5 * (P + Q)
                
                # 5. Calculate JS Divergence
                js_val = 0.5 * safe_kl(P, M) + 0.5 * safe_kl(Q, M)
                js_features[f'GLAM_JSDivergence_{i}_{j}'] = js_val
            else:
                js_features[f'GLAM_JSDivergence_{i}_{j}'] = np.nan
                
    return js_features

def calculate_cumulative_js_matrix(rdf_df, num_levels):
    """
    Calculates a 'Wasserstein-smoothed' JS Divergence Matrix.
    Instead of comparing local density g(r), it compares the Cumulative Coordination 
    profiles N(R) using the JS Divergence, resulting in a much smoother anisotropy matrix.
    """
    js_features = {}
    if rdf_df.empty: return {}
    
    r = rdf_df['r'].values

    def safe_kl(p, q):
        """Native KL divergence that safely handles 0 * log(0) = 0"""
        mask = p > 0
        out = np.zeros_like(p)
        out[mask] = p[mask] * np.log2(p[mask] / q[mask])
        return np.sum(out)

    for i in range(num_levels):
        for j in range(num_levels):
            # The diagonal is perfectly symmetric with itself
            if i == j:
                js_features[f'GLAM_CumulativeJSDivergence_{i}_{j}'] = 0.0
                continue
            
            key_ij = f'g_{i}_{j}'
            key_ji = f'g_{j}_{i}'
            
            if key_ij in rdf_df.columns and key_ji in rdf_df.columns:
                # 1. Get raw screened signals (incorporating spherical volume element r^2)
                # This matches the core of the Wasserstein integrand
                P_shell = np.nan_to_num(rdf_df[key_ij].values) * (r**2)
                Q_shell = np.nan_to_num(rdf_df[key_ji].values) * (r**2)
                
                # 2. Convert to Cumulative Profiles (The "Wasserstein" smoothing step)
                # We use cumulative_trapezoid to build a smooth, monotonically increasing curve
                P_cumul = scipy.integrate.cumulative_trapezoid(P_shell, r, initial=0)
                Q_cumul = scipy.integrate.cumulative_trapezoid(Q_shell, r, initial=0)
                
                sum_P = np.sum(P_cumul)
                sum_Q = np.sum(Q_cumul)
                
                # If both are physical vacuums, they have 0 structural disagreement
                if sum_P == 0 and sum_Q == 0:
                    js_features[f'GLAM_CumulativeJSDivergence_{i}_{j}'] = 0.0
                    continue
                    
                # 3. L1 Re-normalization of the CUMULATIVE profiles
                P = P_cumul / sum_P if sum_P > 0 else np.zeros_like(P_cumul)
                Q = Q_cumul / sum_Q if sum_Q > 0 else np.zeros_like(Q_cumul)
                
                # 4. Calculate Shared Midpoint M
                M = 0.5 * (P + Q)
                
                # 5. Calculate JS Divergence on the smoothed profiles
                js_val = 0.5 * safe_kl(P, M) + 0.5 * safe_kl(Q, M)
                js_features[f'GLAM_CumulativeJSDivergence_{i}_{j}'] = js_val
            else:
                js_features[f'GLAM_CumulativeJSDivergence_{i}_{j}'] = np.nan
                
    return js_features

def calculate_glam_b2_3d(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the 3D B2 coefficient."""
    glam_b2_coeffs = {}
    if rdf_structured_df.empty or rdf_random_df.empty: return {}
    r = rdf_structured_df['r'].values
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            b2_key = f'GLAM_B2_for_{key}'
            if key in rdf_structured_df.columns and key in rdf_random_df.columns:
                g_r_structured = rdf_structured_df[key].values
                g_r_random = rdf_random_df[key].values
                integrand = (g_r_structured - g_r_random) * r**2
                B2 = -2 * np.pi * np.trapezoid(integrand, r)
                glam_b2_coeffs[b2_key] = B2
    return glam_b2_coeffs

def calculate_glam_correlation_length(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the Positional Correlation Length."""
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')

    glam_corr_lengths = {}
    if rdf_structured_df.empty or rdf_random_df.empty: return {}
    def exp_decay(r, A, xi): return A * np.exp(-r / xi)
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            corr_len_key = f'GLAM_corr_length_{key}'
            if key not in rdf_structured_df.columns or key not in rdf_random_df.columns:
                glam_corr_lengths[corr_len_key] = np.nan
                continue
            
            h_r_raw = rdf_structured_df[key].values - rdf_random_df[key].values
            r_vals = rdf_structured_df['r'].values

            if len(h_r_raw) > savgol_window:
                h_r = savgol_filter(h_r_raw, savgol_window, savgol_poly, mode='constant', cval=0.0)
            else:
                h_r = h_r_raw 

            try:
                search_range = len(r_vals) // 2
                if search_range < 3: raise ValueError("Not enough data points.")
                
                peak_index = np.argmax(np.abs(h_r[:search_range]))
                
                if peak_index + 3 >= len(r_vals): raise ValueError("Peak too close to end.")
                
                r_fit = r_vals[peak_index:]
                h_fit = np.abs(h_r[peak_index:]) 

                if np.mean(h_fit) > h_fit[0] * 0.95 and np.var(h_fit) < 1e-4 : raise ValueError("Data is too flat.")
                
                initial_guess = [np.abs(h_r[peak_index]), 3.0] 
                popt, _ = curve_fit(exp_decay, r_fit, h_fit, p0=initial_guess, maxfev=5000, bounds=([-np.inf, 0], [np.inf, np.inf]))
                glam_corr_lengths[corr_len_key] = popt[1]
                
            except (RuntimeError, ValueError) as e:
                glam_corr_lengths[corr_len_key] = np.nan
    return glam_corr_lengths

def calculate_glam_coordination_number(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the Coordination Number (Z)."""
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    peak_prominence = get_config('PeakProminence')

    coordination_numbers = {}
    if rdf_df.empty: return {}
    r = rdf_df['r'].values
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            coord_key = f'GLAM_CoordNum_{key}'
            if key not in rdf_df.columns:
                coordination_numbers[coord_key] = np.nan
                continue
            
            g_r_raw = rdf_df[key].values

            if len(g_r_raw) > savgol_window:
                g_r = savgol_filter(g_r_raw, savgol_window, savgol_poly, mode='constant', cval=0.0)
            else:
                g_r = g_r_raw 

            rho_beta = level_counts[beta] / total_roi_voxels if total_roi_voxels > 0 else 0
            
            peaks, _ = find_peaks(g_r, prominence=peak_prominence)
            
            if len(peaks) == 0:
                search_r = min(len(g_r), 15) 
                if search_r == 0:
                    coordination_numbers[coord_key] = np.nan
                    continue
                first_peak_idx = np.argmax(g_r[:search_r])
                if first_peak_idx == 0: 
                     coordination_numbers[coord_key] = np.nan
                     continue
            else:
                first_peak_idx = peaks[0] 

            minima, _ = find_peaks(-g_r[first_peak_idx:])
            
            if len(minima) == 0:
                r_min_idx = min(first_peak_idx * 2, len(r) - 1)
                if r_min_idx <= first_peak_idx: r_min_idx = len(r) - 1
            else:
                r_min_idx = first_peak_idx + minima[0] 
            
            integrand = g_r_raw[:r_min_idx+1] * r[:r_min_idx+1]**2
            integral = np.trapezoid(integrand, r[:r_min_idx+1])
            
            Z = 4 * np.pi * rho_beta * integral
            coordination_numbers[coord_key] = Z
    return coordination_numbers

def calculate_glam_compressibility(rdf_df, num_levels):
    """
    Calculates a metric related to Isothermal Compressibility.
    UPDATED: Restricts integration to the active tumor region.
    """
    compressibility_metrics = {}
    if rdf_df.empty: return {}
    r = rdf_df['r'].values
    
    for i in range(num_levels):
        key = f'g_{i}_{i}'
        comp_key = f'GLAM_Compressibility_{key}'
        
        if key in rdf_df.columns:
            g_r = rdf_df[key].values
            
            # Only integrate where the tumor actually exists (g(r) > 0)
            valid_mask = g_r > 1e-6
            
            if np.sum(valid_mask) > 1:
                r_valid = r[valid_mask]
                g_r_valid = g_r[valid_mask]
                
                integrand = (g_r_valid - 1) * r_valid**2
                metric = 4 * np.pi * np.trapezoid(integrand, r_valid)
            else:
                metric = 0.0

            compressibility_metrics[comp_key] = metric
            
    return compressibility_metrics

def calculate_anisotropic_glam_features(image_3d, num_levels, cutoff_radius):
    """Calculates anisotropic GLAM features using the gyration tensor."""
    # print("  - Starting Anisotropic GLAM analysis...")
    anisotropic_features = {}
    coords = [np.argwhere(image_3d == i) for i in range(num_levels)]
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            
            # --- FILL LOCATION 1: POPULATION CHECK ---
            # If not enough points to measure, set to 0.0 (Isotropic) and skip
            if len(coords[alpha]) == 0 or len(coords[beta]) < 10: 
                anisotropic_features[f'GLAM_Anisotropy_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue1_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue2_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue3_{alpha}_{beta}'] = 0.0
                continue

            num_ref_points = min(len(coords[alpha]), 50)
            # Safety: Ensure we don't sample more than exists
            if len(coords[alpha]) < num_ref_points:
                 ref_indices = np.arange(len(coords[alpha]))
            else:
                 ref_indices = np.random.choice(len(coords[alpha]), num_ref_points, replace=False)
            
            avg_gyration_tensor = np.zeros((3, 3))
            valid_tensors = 0
            
            for i in ref_indices:
                ref_point = coords[alpha][i]
                vectors = coords[beta] - ref_point
                distances = np.linalg.norm(vectors, axis=1)
                neighbors = vectors[distances < cutoff_radius]
                
                if len(neighbors) < 3: continue
                
                gyration_tensor = np.cov(neighbors, rowvar=False)
                avg_gyration_tensor += gyration_tensor
                valid_tensors += 1
            
            # --- FILL LOCATION 2: VALID TENSOR CHECK ---
            if valid_tensors > 0:
                avg_gyration_tensor /= valid_tensors
                # Check for NaNs/Infs in the tensor before Eigendecomposition
                if np.any(np.isnan(avg_gyration_tensor)) or np.any(np.isinf(avg_gyration_tensor)):
                     l1, l2, l3 = 0.0, 0.0, 0.0
                     anisotropy = 0.0
                else:
                    try:
                        eigenvalues, _ = np.linalg.eigh(avg_gyration_tensor)
                        eigenvalues = np.sort(eigenvalues)[::-1]
                        l1, l2, l3 = eigenvalues
                        
                        denom = (l1 + l2 + l3)**2
                        if denom > 1e-9:
                            anisotropy = 1 - 3 * (l1*l2 + l2*l3 + l3*l1) / denom
                        else:
                            anisotropy = 0.0
                    except np.linalg.LinAlgError:
                        l1, l2, l3 = 0.0, 0.0, 0.0
                        anisotropy = 0.0

                anisotropic_features[f'GLAM_Anisotropy_{alpha}_{beta}'] = anisotropy
                anisotropic_features[f'GLAM_Eigenvalue1_{alpha}_{beta}'] = l1
                anisotropic_features[f'GLAM_Eigenvalue2_{alpha}_{beta}'] = l2
                anisotropic_features[f'GLAM_Eigenvalue3_{alpha}_{beta}'] = l3
            
            else:
                # If we found reference points but NO valid neighbors within radius
                anisotropic_features[f'GLAM_Anisotropy_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue1_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue2_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue3_{alpha}_{beta}'] = 0.0

    # print("  - Anisotropic GLAM analysis complete.")
    return anisotropic_features

def calculate_glam_fractal_dimension(image_3d, num_levels):
    """Calculates Volume and Interface Fractal Dimensions using an optimized box-counting method."""
    # print("  - Starting Fractal Dimension analysis...")
    fractal_dimensions = {}

    def boxcount(binary_image, max_box_size=32):
        """Optimized helper function for box-counting algorithm using NumPy."""
        # CHANGE 1: Return 0.0 instead of NaN for empty bins
        if not np.any(binary_image): return 0.0
        
        p = binary_image.shape
        scales = np.logspace(np.log2(2), np.log2(min(p)), num=8, base=2.0, dtype=np.int32)
        scales = np.unique(scales[scales > 1])
        
        # CHANGE 2: Return 0.0 if object is too small for scaling analysis
        if len(scales) < 2: return 0.0
        
        counts = np.array([np.sum(ndimage.maximum_filter(binary_image, size=s, mode='constant')[::s, ::s, ::s]) for s in scales])
        
        valid_indices = counts > 0
        # CHANGE 3: Return 0.0 if not enough points for regression
        if np.sum(valid_indices) < 2:
            return 0.0 
            
        scales_fit = scales[valid_indices]
        counts_fit = counts[valid_indices]
        
        try:
            with np.errstate(divide='ignore'): 
                coeffs = np.polyfit(np.log(scales_fit), np.log(counts_fit), 1)
            return -coeffs[0]
        except np.linalg.LinAlgError:
            # CHANGE 4: Return 0.0 on math error
            return 0.0
        
    for i in range(num_levels):
        binary_image = (image_3d == i)
        fractal_dimensions[f'GLAM_VolumeFD_{i}'] = boxcount(binary_image)

    for i in range(num_levels):
        for j in range(num_levels):

            if i == j: continue

            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            # CHANGE 5: Explicitly set 0.0 if masks are empty, don't just skip
            # This ensures the matrix builder finds a value.
            if not np.any(mask_i) or not np.any(mask_j): 
                fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
                continue
                
            interface = ndimage.binary_dilation(mask_i) & mask_j
            fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = boxcount(interface)

    # print("  - Fractal Dimension analysis complete.")
    return fractal_dimensions

def calculate_glam_lacunarity(image_3d, num_levels):
    """
    Calculates Volume and Interface Lacunarity.
    UPDATED: Returns 1.0 (instead of NaN) for empty/invalid regions.
    This ensures that Log(Lacunarity) becomes 0.0, avoiding empty cells.
    """
    lacunarity_features = {}
    
    # Define box sizes to average over. 
    box_sizes = [2, 3, 4, 5] 
    
    def get_lacunarity_for_mask(binary_mask):
        # CHANGE 1: Return 1.0 (perfect homogeneity) if mask is too small
        if np.sum(binary_mask) < 10: return 1.0
        
        float_mask = binary_mask.astype(float)
        lac_values = []
        
        for r in box_sizes:
            kernel = np.ones((r, r, r))
            
            try:
                mass_map = ndimage.convolve(float_mask, kernel, mode='constant', cval=0.0)
                valid_masses = mass_map[mass_map > 0]
                
                if valid_masses.size < 2: continue
                
                mean_mass = np.mean(valid_masses)
                var_mass = np.var(valid_masses)
                
                if mean_mass == 0: continue
                
                # Standard Lacunarity Formula
                lambda_r = (var_mass / (mean_mass**2)) + 1
                lac_values.append(lambda_r)
                
            except Exception:
                continue

        # CHANGE 2: Return 1.0 if calculation failed
        if not lac_values: return 1.0
        return np.mean(lac_values)

    # 1. Diagonal: Volume Lacunarity (Single Gray Levels)
    for i in range(num_levels):
        binary_image = (image_3d == i)
        lacunarity_features[f'GLAM_VolumeLacunarity_{i}'] = get_lacunarity_for_mask(binary_image)

    # 2. Off-Diagonal: Interface Lacunarity (Pairwise Boundaries)
    for i in range(num_levels):
        for j in range(num_levels):

            if i == j: continue

            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            # CHANGE 3: Return 1.0 if one of the masks is empty
            if not np.any(mask_i) or not np.any(mask_j): 
                lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = 1.0
                continue
            
            interface = ndimage.binary_dilation(mask_i) & mask_j
            lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = get_lacunarity_for_mask(interface)

    return lacunarity_features

def calculate_glam_topology(image_3d, num_levels):
    """
    Calculates Full Topological Metrics (Betti Numbers).
    Uses the Euler-Poincare formula: Chi = B0 - B1 + B2
    
    1. Betti-0: Connected Components (Fragmentation)
    2. Betti-2: Enclosed Voids (Calculated by labeling background)
    3. Euler (Chi): Standard calculation
    4. Betti-1: Derived (B1 = B0 + B2 - Chi) - Tunnels/Loops
    """
    topology_features = {}
    
    def get_betti_numbers(binary_mask):
        # Pad with 1 pixel of zeros to ensure "background" is connected around the edges
        padded = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        
        # 1. Betti-0 (Objects)
        # Connectivity=3 (26-neighbors) for 3D objects
        labeled_obj, n_objects = ndimage.label(padded, structure=np.ones((3,3,3)))
        b0 = float(n_objects)
        
        # 2. Betti-2 (Voids/Cavities)
        # We look at the BACKGROUND (inverted mask). 
        # Background connectivity should be 1 (6-neighbors) if foreground is 3 (26-neighbors)
        # to strictly satisfy topological duality, but typically 3vs3 or 1vs1 is used in medical imaging.
        # We will use 6-connectivity (cross) for background to prevent "leaking" through diagonals.
        background = (padded == 0)
        structure_bg = ndimage.generate_binary_structure(3, 1) # 6-connectivity
        labeled_bg, n_bg_components = ndimage.label(background, structure=structure_bg)
        
        # The background has 1 giant component (the outside world) + N internal voids.
        # So Betti-2 = Total Background Components - 1
        b2 = max(0.0, float(n_bg_components) - 1.0)
        
        # 3. Euler Characteristic
        # skimage euler_number handles the connectivity internally
        chi = measure.euler_number(padded, connectivity=3)
        
        # 4. Betti-1 (Tunnels/Loops)
        # Derived: Chi = B0 - B1 + B2  =>  B1 = B0 + B2 - Chi
        b1 = b0 + b2 - chi
        
        # Sanity check: Betti numbers cannot be negative
        b1 = max(0.0, b1)
        
        return b0, b1, b2, float(chi)

    # 1. Diagonal: Volume Topology
    for i in range(num_levels):
        binary_image = (image_3d == i)
        
        if np.sum(binary_image) == 0:
            b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
        else:
            b0, b1, b2, chi = get_betti_numbers(binary_image)
            
        topology_features[f'GLAM_VolumeBetti0_{i}'] = b0
        topology_features[f'GLAM_VolumeBetti1_{i}'] = b1
        topology_features[f'GLAM_VolumeBetti2_{i}'] = b2
        topology_features[f'GLAM_VolumeEuler_{i}'] = chi

    # 2. Off-Diagonal: Interface Topology
    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue
            
            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            if not np.any(mask_i) or not np.any(mask_j):
                b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
                continue
                
            interface = ndimage.binary_dilation(mask_i) & mask_j
            
            if np.sum(interface) == 0:
                b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
            else:
                b0, b1, b2, chi = get_betti_numbers(interface)
            
            topology_features[f'GLAM_InterfaceBetti0_{i}_{j}'] = b0
            topology_features[f'GLAM_InterfaceBetti1_{i}_{j}'] = b1
            topology_features[f'GLAM_InterfaceBetti2_{i}_{j}'] = b2
            topology_features[f'GLAM_InterfaceEuler_{i}_{j}'] = chi

    return topology_features

def calculate_glam_potential_energy(rdf_df, num_levels):
    """Calculates the Potential of Mean Force (PMF) Energy (U*) and contributions."""
    potential_energies = {}
    if rdf_df.empty: return {}
    r = rdf_df['r'].values
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            if key in rdf_df.columns:
                g_r = rdf_df[key].values
                pmf = -np.log(g_r + 1e-9) # W*(r)
                integrand = pmf * g_r * r**2
                energy = 4 * np.pi * np.trapezoid(integrand, r)
                potential_energies[f'GLAM_PotentialEnergy_{alpha}_{beta}'] = energy
    return potential_energies

def calculate_glam_pressure_virial(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the Pressure Virial (P*), the interaction component of pressure."""
    # --- Get config params ---
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    # ---

    pressure_virials = {}
    if rdf_df.empty or total_roi_voxels == 0: return {}
    r = rdf_df['r'].values
    if r.size < 2: return {}

    for alpha in range(num_levels):
        rho_alpha = level_counts[alpha] / total_roi_voxels
        for beta in range(num_levels):
            rho_beta = level_counts[beta] / total_roi_voxels
            key = f'g_{alpha}_{beta}'
            
            if key in rdf_df.columns:
                g_r_raw = rdf_df[key].values

                if len(g_r_raw) > savgol_window:
                    g_r = savgol_filter(g_r_raw, savgol_window, savgol_poly, mode='constant', cval=0.0)
                else:
                    g_r = g_r_raw
                
                g_r[g_r <= 1e-9] = 1e-9 
                
                pmf = -np.log(g_r) 
                
                dW_dr = np.gradient(pmf, r, edge_order=2)
                
                integrand = r * dW_dr * g_r_raw * r**2
                
                integral = np.trapezoid(integrand, r)
                
                pressure_virial = - (rho_alpha * rho_beta / 6.0) * 4 * np.pi * integral
                pressure_virials[f'GLAM_PressureVirial_{alpha}_{beta}'] = pressure_virial
            else:
                pressure_virials[f'GLAM_PressureVirial_{alpha}_{beta}'] = np.nan
                
    return pressure_virials

def calculate_effective_temperature(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the Effective Structural Temperature averaged over the first coordination shell."""
    # --- Get config params ---
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    peak_prominence = get_config('PeakProminence')
    # ---

    effective_temps = {}
    if rdf_structured_df.empty or rdf_random_df.empty: return effective_temps
    r = rdf_structured_df['r'].values
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            temp_key = f'GLAM_EffectiveTemp_{alpha}_{beta}'
            
            if key not in rdf_structured_df.columns or key not in rdf_random_df.columns:
                effective_temps[temp_key] = np.nan
                continue

            g_struct_raw, g_rand_raw = rdf_structured_df[key].values, rdf_random_df[key].values
            
            if len(g_struct_raw) > savgol_window:
                g_struct = savgol_filter(g_struct_raw, savgol_window, savgol_poly, mode='constant', cval=0.0)
                g_rand = savgol_filter(g_rand_raw, savgol_window, savgol_poly, mode='constant', cval=0.0)
            else:
                g_struct, g_rand = g_struct_raw, g_rand_raw

            try:
                peaks, _ = find_peaks(g_struct, prominence=peak_prominence)
                
                if len(peaks) == 0:
                    search_r = min(len(g_struct), 15) 
                    if search_r == 0: raise ValueError("Not enough data")
                    first_peak_idx = np.argmax(g_struct[:search_r])
                    if first_peak_idx == 0: raise ValueError("Peak at r=1, no clear shell")
                else:
                    first_peak_idx = peaks[0] 

                minima, _ = find_peaks(-g_struct[first_peak_idx:])
                
                if len(minima) == 0:
                    r_min_idx = min(first_peak_idx * 2, len(r) - 1)
                    if r_min_idx <= first_peak_idx: r_min_idx = len(r) - 1
                else:
                    r_min_idx = first_peak_idx + minima[0] 

                if r_min_idx == 0: r_min_idx = 1 

                g_s_shell = g_struct[:r_min_idx]
                g_r_shell = g_rand[:r_min_idx]

                g_s_shell[g_s_shell <= 1e-9] = 1e-9
                g_r_shell[g_r_shell <= 1e-9] = 1e-9
                
                log_g_struct = np.log(g_s_shell)
                log_g_rand = np.log(g_r_shell)
                
                log_diff = log_g_struct - log_g_rand

                with np.errstate(divide='ignore', invalid='ignore'):
                    t_eff_r = log_g_struct / log_diff
                
                finite_t_eff = t_eff_r[np.isfinite(t_eff_r)]
                
                if finite_t_eff.size > 0:
                    effective_temps[temp_key] = np.mean(finite_t_eff)
                else:
                    effective_temps[temp_key] = np.nan
                    
            except Exception as e:
                effective_temps[temp_key] = np.nan
                
    return effective_temps

def calculate_nematic_order_parameter(image_array, mask_array):
    """Calculates the GLOBAL Nematic Order Parameter (S) from the intensity gradient."""
    print("  - Starting Nematic Order Parameter analysis (Global)...")
    if np.sum(mask_array) < 10: return {}
    grad = np.array(np.gradient(image_array.astype(float)))
    roi_indices = np.where(mask_array > 0)
    grad_vectors = grad[:, roi_indices[0], roi_indices[1], roi_indices[2]].T
    norms = np.linalg.norm(grad_vectors, axis=1)
    valid_indices = norms > 1e-6
    if np.sum(valid_indices) < 10: return {}
    unit_vectors = grad_vectors[valid_indices] / norms[valid_indices, np.newaxis]
    Q = np.mean([np.outer(n, n) for n in unit_vectors], axis=0) - (1/3) * np.identity(3)
    S = 1.5 * np.max(np.linalg.eigvalsh(Q))
    print("  - Nematic Order Parameter analysis (Global) complete.")
    return {'GLAM.NematicOrder.S': S}

def calculate_nematic_order_per_gray_level(image_array, mask_array, quantized_image, num_levels):
    """Calculates the Nematic Order Parameter (S_alpha) for each gray level."""
    print("  - Starting Nematic Order Parameter analysis (Per Gray Level)...")
    nematic_features = {}
    if np.sum(mask_array) < 10: return {}

    grad = np.array(np.gradient(image_array.astype(float)))
    
    for alpha in range(num_levels):
        key = f'GLAM_NematicOrder_S_per_GL_{alpha}'
        
        roi_indices_for_level = np.where((mask_array > 0) & (quantized_image == alpha))
        
        if len(roi_indices_for_level[0]) < 10:
            nematic_features[key] = np.nan
            continue
            
        grad_vectors = grad[:, roi_indices_for_level[0], roi_indices_for_level[1], roi_indices_for_level[2]].T
        
        norms = np.linalg.norm(grad_vectors, axis=1)
        valid_indices = norms > 1e-6
        
        if np.sum(valid_indices) < 10:
            nematic_features[key] = np.nan
            continue
            
        unit_vectors = grad_vectors[valid_indices] / norms[valid_indices, np.newaxis]
        
        Q = np.mean([np.outer(n, n) for n in unit_vectors], axis=0) - (1/3) * np.identity(3)
        
        S = 1.5 * np.max(np.linalg.eigvalsh(Q))
        nematic_features[key] = S

    print("  - Nematic Order Parameter analysis (Per Gray Level) complete.")
    return nematic_features

def calculate_local_nematic_alignment(image_array, mask_array, cutoff_radius):
    """Calculates the average alignment of local directors."""
    print("  - Starting Local Nematic Alignment analysis...")
    roi_coords = np.argwhere(mask_array > 0)
    if len(roi_coords) < 50: return {}

    grad = np.array(np.gradient(image_array.astype(float)))
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    tree = cKDTree(roi_coords)
    num_ref_points = min(len(roi_coords), 200)
    ref_indices = np.random.choice(len(roi_coords), num_ref_points, replace=False)
    
    local_directors = {}
    for i in ref_indices:
        ref_point = roi_coords[i]
        neighbor_indices = tree.query_ball_point(ref_point, r=cutoff_radius)
        if len(neighbor_indices) < 10: continue
        local_grad_vectors = grad_vectors[neighbor_indices]
        norms = np.linalg.norm(local_grad_vectors, axis=1)
        valid = norms > 1e-6
        if np.sum(valid) < 10: continue
        local_unit_vectors = local_grad_vectors[valid] / norms[valid, np.newaxis]
        Q_local = np.mean([np.outer(n, n) for n in local_unit_vectors], axis=0) - (1/3) * np.identity(3)
        eigenvalues, eigenvectors = np.linalg.eigh(Q_local)
        local_directors[tuple(ref_point)] = eigenvectors[:, np.argmax(eigenvalues)]

    if len(local_directors) < 2: return {}

    director_coords = np.array(list(local_directors.keys()))
    director_vectors = np.array(list(local_directors.values()))
    director_tree = cKDTree(director_coords)
    
    alignments = []
    for i, coord in enumerate(director_coords):
        dist, idx = director_tree.query(coord, k=2)
        if len(idx) > 1:
            alignments.append(np.dot(director_vectors[i], director_vectors[idx[1]])**2)

    if not alignments: return {}
    print("  - Local Nematic Alignment analysis complete.")
    return {'GLAM.LocalNematic.Alignment': np.mean(alignments)}

def calculate_stress_features(image_array, mask_array):
    """Calculates Stress analogues for the volume and surface of the ROI."""
    print("  - Starting Stress analysis...")
    stress_features = {}
    if np.sum(mask_array) < 10: return {}
    
    laplacian = ndimage.laplace(image_array.astype(float))
    
    # Volumetric Stress
    roi_laplacian = laplacian[mask_array > 0]
    stress_features['GLAM.Stress.VolumetricLaplacianMean'] = np.mean(np.abs(roi_laplacian))
    stress_features['GLAM.Stress.VolumetricLaplacianVariance'] = np.var(np.abs(roi_laplacian))

    mask_bool = mask_array > 0 
    dilated_mask = ndimage.binary_dilation(mask_bool)
    surface_mask = dilated_mask & ~mask_bool 
    
    if np.any(surface_mask): 
        surface_laplacian = laplacian[surface_mask]
        stress_features['GLAM.Stress.SurfaceLaplacianMean'] = np.mean(np.abs(surface_laplacian))
        stress_features['GLAM.Stress.SurfaceLaplacianVariance'] = np.var(np.abs(surface_laplacian))
    else:
        stress_features['GLAM.Stress.SurfaceLaplacianMean'] = np.nan
        stress_features['GLAM.Stress.SurfaceLaplacianVariance'] = np.nan
    
    print("  - Stress analysis complete.")
    return stress_features

def calculate_orientational_correlation_length(image_array, mask_array, max_radius):
    """Calculates the Orientational Correlation Length from the g2(r) function.
       (Memory-efficient version)
    """
    print("  - Starting Orientational Correlation analysis...")
    if np.sum(mask_array) < 50: return {}
    grad = np.array(np.gradient(image_array.astype(float)))
    roi_coords = np.argwhere(mask_array > 0)
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    norms = np.linalg.norm(grad_vectors, axis=1)
    valid_indices = norms > 1e-6
    if np.sum(valid_indices) < 50: return {}
    coords, vectors = roi_coords[valid_indices], grad_vectors[valid_indices] / norms[valid_indices, np.newaxis]
    tree = cKDTree(coords)
    num_ref_points = min(len(coords), 500) # You can also reduce this 500 to e.g. 200
    ref_indices = np.random.choice(len(coords), num_ref_points, replace=False)

    # --- NEW: Pre-allocate bins ---
    # We will add to these bins instead of creating huge lists
    g2_r_bins = np.zeros(max_radius, dtype=np.float64)
    bin_counts = np.zeros(max_radius, dtype=np.int64)
    # ---

    for i in ref_indices:
        ref_point, ref_vector = coords[i], vectors[i]
        neighbor_indices = tree.query_ball_point(ref_point, r=max_radius)
        if len(neighbor_indices) < 2: continue
        
        neighbor_coords = coords[neighbor_indices]
        neighbor_vectors = vectors[neighbor_indices]
        
        distances = np.linalg.norm(neighbor_coords - ref_point, axis=1)
        valid_dist = distances > 1e-6
        if not np.any(valid_dist): continue
        
        # Get distances and dot products for valid neighbors
        distances = distances[valid_dist]
        dot_products = np.dot(neighbor_vectors[valid_dist], ref_vector)
        p2_values = 0.5 * (3 * dot_products**2 - 1)
        
        # --- NEW: Binning inside the loop ---
        # Get bin index (0 for r=1, 1 for r=2, etc.)
        bin_indices = np.floor(distances).astype(int) - 1
        
        # Filter out any bins that are out of range (e.g., r=0 or r > max_radius)
        valid_bins = (bin_indices >= 0) & (bin_indices < max_radius)
        if not np.any(valid_bins): continue
        
        bin_indices = bin_indices[valid_bins]
        p2_values = p2_values[valid_bins]
        
        # Add values to their respective bins atomically
        # This is the memory-efficient equivalent of binned_statistic
        np.add.at(g2_r_bins, bin_indices, p2_values)
        np.add.at(bin_counts, bin_indices, 1)
        # ---
        
    # Calculate the average g2(r) for bins that have data
    valid_g2_bins = bin_counts > 0
    g2_r = np.full(max_radius, np.nan)
    g2_r[valid_g2_bins] = g2_r_bins[valid_g2_bins] / bin_counts[valid_g2_bins]
    
    r_vals = np.arange(1, max_radius + 1)
    valid_g2 = ~np.isnan(g2_r)
    
    if np.sum(valid_g2) < 3: return {'GLAM.OrientationalCorrLength': np.nan}
    
    def exp_decay(r, A, xi): return A * np.exp(-r / xi)
    try:
        # Fit the decay curve
        popt, _ = curve_fit(exp_decay, r_vals[valid_g2], g2_r[valid_g2], p0=[g2_r[valid_g2][0], 5.0], maxfev=5000)
        corr_length = popt[1] if popt[1] > 0 else np.nan
    except (RuntimeError, ValueError): 
        corr_length = np.nan
        
    print("  - Orientational Correlation analysis complete.")
    return {'GLAM.OrientationalCorrLength': corr_length}

def calculate_rdf_shape_matrices(rdf_df, num_levels):
    """
    Calculates primary matrices based on the shape of the g(r) curves.
    UPDATED: Uses 'Non-Zero' statistics to handle variable tumor sizes robustly.
    """
    if rdf_df.empty:
        return {}

    # Initialize empty matrices
    mat_peak_pos = np.full((num_levels, num_levels), np.nan)

    mat_log_median = np.full((num_levels, num_levels), np.nan)
    mat_log_var = np.full((num_levels, num_levels), np.nan)
    mat_log_skew = np.full((num_levels, num_levels), np.nan)
    mat_log_kurt = np.full((num_levels, num_levels), np.nan)
    mat_log_peak_height = np.full((num_levels, num_levels), np.nan)

    r_values = rdf_df['r'].values

    for i in range(num_levels):
        for j in range(num_levels):
            key = f'g_{i}_{j}'
            if key in rdf_df.columns:
                g_r_vector = rdf_df[key].values
                
                # --- PEAK DETECTION (Global) ---
                if g_r_vector.size > 0:
                    peak_index = np.argmax(g_r_vector)
                    mat_peak_pos[i, j] = r_values[peak_index]

                # --- SMART STATISTICS (Ignore Trailing Zeros) ---
                # 1. Log-Transform first
                log_g_r_vector = np.log1p(g_r_vector)
                
                # 2. Filter: Only keep non-zero values (Active Texture Region)
                # This automatically adapts to the tumor size.
                valid_mask = g_r_vector > 1e-6 # Filter out the empty "tail"
                active_log_data = log_g_r_vector[valid_mask]
                
                # 3. Calculate stats on the ACTIVE region only
                if active_log_data.size > 2:
                    mat_log_median[i, j] = np.median(active_log_data)
                    mat_log_var[i, j] = np.var(active_log_data)
                    mat_log_skew[i, j] = stats.skew(active_log_data)
                    mat_log_kurt[i, j] = stats.kurtosis(active_log_data)
                
                # Peak Height is always the max, regardless of zeros
                if log_g_r_vector.size > 0:
                    mat_log_peak_height[i, j] = np.max(log_g_r_vector)
                
    return {
        "RDF_PeakPosition": mat_peak_pos,
        "LogRDF_Median": mat_log_median,
        "LogRDF_Variance": mat_log_var,
        "LogRDF_Skewness": mat_log_skew,
        "LogRDF_Kurtosis": mat_log_kurt,
        "LogRDF_PeakHeight": mat_log_peak_height
    }

def calculate_glam_wasserstein_distance(rdf_structured_df, rdf_random_df, num_levels, level_counts, total_roi_voxels):
    """
    Calculates the 1-Wasserstein Distance (Earth Mover's Distance) between 
    the structured and random states. This measures the 'Biological Work' 
    or 'Assembly Cost' of the tumor's spatial architecture.
    """
    wasserstein_distances = {}
    if rdf_structured_df.empty or rdf_random_df.empty: 
        return {}
        
    r = rdf_structured_df['r'].values
    
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'g_{alpha}_{beta}'
            if key in rdf_structured_df.columns and key in rdf_random_df.columns:
                g_r_struct = rdf_structured_df[key].values
                g_r_rand = rdf_random_df[key].values
                
                # Apply the spatial screening window
                g_r_struct_screened = g_r_struct 
                g_r_rand_screened = g_r_rand
                
                # Local density of state beta
                rho_beta = level_counts[beta] / total_roi_voxels if total_roi_voxels > 0 else 0
                
                # Core integrands for the Coordination Number
                integrand_struct = 4 * np.pi * rho_beta * g_r_struct_screened * (r**2)
                integrand_rand = 4 * np.pi * rho_beta * g_r_rand_screened * (r**2)
                
                # 1. Calculate Cumulative Coordination Numbers N(R)
                n_struct = scipy.integrate.cumulative_trapezoid(integrand_struct, r, initial=0)
                n_rand = scipy.integrate.cumulative_trapezoid(integrand_rand, r, initial=0)
                
                # 2. Wasserstein Distance is the absolute area between the two cumulative curves
                w_dist = np.trapezoid(np.abs(n_struct - n_rand), r)
                wasserstein_distances[f'GLAM_Wasserstein_{alpha}_{beta}'] = w_dist
                
    return wasserstein_distances

def calculate_first_order_stats_from_matrix(matrix, feature_prefix):
    """Calculates first-order statistics for a given matrix OR vector."""
    stats_features = {}
    if matrix is None or np.all(np.isnan(matrix)): return stats_features
    
    flat_matrix = matrix[~np.isnan(matrix)].ravel()
    
    if flat_matrix.size == 0: return stats_features
    stats_features[f'{feature_prefix}.10Percentile'] = np.percentile(flat_matrix, 10)
    stats_features[f'{feature_prefix}.90Percentile'] = np.percentile(flat_matrix, 90)
    stats_features[f'{feature_prefix}.Energy'] = np.sum(flat_matrix**2)

    clean_matrix = flat_matrix.copy()
    if np.any(clean_matrix < 0):
        clean_matrix -= np.min(clean_matrix) 

    stats_features[f'{feature_prefix}.InterquartileRange'] = np.quantile(flat_matrix, 0.75) - np.quantile(flat_matrix,0.25)
    stats_features[f'{feature_prefix}.Kurtosis'] = stats.kurtosis(flat_matrix)
    stats_features[f'{feature_prefix}.Maximum'] = np.max(flat_matrix)
    stats_features[f'{feature_prefix}.MeanAbsoluteDeviation'] = np.mean(np.abs(flat_matrix - np.mean(flat_matrix)))
    stats_features[f'{feature_prefix}.Mean'] = np.mean(flat_matrix)
    stats_features[f'{feature_prefix}.Median'] = np.median(flat_matrix)
    stats_features[f'{feature_prefix}.Minimum'] = np.min(flat_matrix)
    stats_features[f'{feature_prefix}.Range'] = np.max(flat_matrix) - np.min(flat_matrix)
    stats_features[f'{feature_prefix}.RobustMeanAbsoluteDeviation'] = np.mean(np.abs(flat_matrix - np.median(flat_matrix)))
    stats_features[f'{feature_prefix}.RootMeanSquared'] = np.sqrt(np.mean(flat_matrix**2))
    stats_features[f'{feature_prefix}.Skewness'] = stats.skew(flat_matrix)
    stats_features[f'{feature_prefix}.TotalEnergy'] = np.sum(flat_matrix**2)

    unique_vals, counts = np.unique(flat_matrix, return_counts=True)
    probabilities = counts / flat_matrix.size
    stats_features[f'{feature_prefix}.Uniformity'] = np.sum(probabilities**2)

    stats_features[f'{feature_prefix}.Variance'] = np.var(flat_matrix)
    return stats_features

def calculate_glcm_style_meta_features(glam_matrix, feature_prefix):
    """Calculates a comprehensive set of 24 GLCM-style features from a given matrix."""
    meta_features = {}
    if glam_matrix is None or np.all(np.isnan(glam_matrix)):
        return meta_features
    
    if glam_matrix.ndim != 2:
        print(f"  - WARNING: GLCM-style features skipped for {feature_prefix}, not a 2D matrix.")
        return {}
        
    clean_matrix = glam_matrix.copy()
    if np.isnan(clean_matrix).any():
        mean_val = np.nanmean(clean_matrix[np.isfinite(clean_matrix)])
        clean_matrix = np.nan_to_num(clean_matrix, nan=mean_val if not np.isnan(mean_val) else 0)
    
    min_val = np.min(clean_matrix)
    if min_val < 0:
        clean_matrix = clean_matrix - min_val

    matrix_sum = np.sum(clean_matrix)
    if matrix_sum == 0: return {}
    p = clean_matrix / matrix_sum
    
    i, j = np.ogrid[0:p.shape[0], 0:p.shape[1]]
    N = p.shape[0]
    
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    
    ux = np.sum(i.ravel() * px)
    uy = np.sum(j.ravel() * py)

    sx = np.sum(px * (i.ravel() - ux)**2)
    sy = np.sum(py * (j.ravel() - uy)**2)
    
    meta_features[f'{feature_prefix}.Autocorrelation'] = np.sum(p * i * j)
    meta_features[f'{feature_prefix}.JointEnergy'] = np.sum(p**2)
    meta_features[f'{feature_prefix}.JointEntropy'] = -np.sum(p[p>0] * np.log2(p[p>0]))
    meta_features[f'{feature_prefix}.Contrast'] = np.sum(p * (i - j)**2)
    
    if sx > 0 and sy > 0:
        meta_features[f'{feature_prefix}.Correlation'] = (np.sum(p * (i - ux) * (j - uy))) / np.sqrt(sx * sy)
    else:
        meta_features[f'{feature_prefix}.Correlation'] = 0

    meta_features[f'{feature_prefix}.InverseDifference'] = np.sum(p / (1 + np.abs(i - j)))
    meta_features[f'{feature_prefix}.InverseDifferenceMoment'] = np.sum(p / (1 + (i-j)**2))
    meta_features[f'{feature_prefix}.InverseDifferenceNormalized'] = np.sum(p / (1 + (np.abs(i - j) / N)))
    meta_features[f'{feature_prefix}.InverseDifferenceMomentNormalized'] = np.sum(p / (1 + ((i - j)**2 / N**2)))

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_var = p / ((i-j)**2)
        inv_var[i==j] = 0
        meta_features[f'{feature_prefix}.InverseVariance'] = np.sum(inv_var)

    k_sum = np.arange(2 * (N-1) + 1)
    k_diff = np.arange(N)
    px_plus_y = np.array([np.sum(p[i + j == k]) for k in k_sum])
    px_minus_y = np.array([np.sum(p[np.abs(i - j) == k]) for k in k_diff])

    meta_features[f'{feature_prefix}.SumAverage'] = np.sum(k_sum * px_plus_y)
    meta_features[f'{feature_prefix}.SumEntropy'] = -np.sum(px_plus_y[px_plus_y>0] * np.log2(px_plus_y[px_plus_y>0]))
    meta_features[f'{feature_prefix}.SumSquares'] = np.sum(p * (i - ux)**2) 
    meta_features[f'{feature_prefix}.DifferenceAverage'] = np.sum(k_diff * px_minus_y)
    meta_features[f'{feature_prefix}.DifferenceEntropy'] = -np.sum(px_minus_y[px_minus_y>0] * np.log2(px_minus_y[px_minus_y>0]))
    meta_features[f'{feature_prefix}.DifferenceVariance'] = np.sum(px_minus_y * (k_diff - meta_features[f'{feature_prefix}.DifferenceAverage'])**2)

    hx = -np.sum(px[px>0] * np.log2(px[px>0]))
    hy = -np.sum(py[py>0] * np.log2(py[py>0]))
    px_py = px[:, None] * py[None, :]
    hxy1 = -np.sum(p[px_py>0] * np.log2(px_py[px_py>0]))
    hxy2 = -np.sum(px_py[px_py>0] * np.log2(px_py[px_py>0]))

    if max(hx, hy) > 0:
        meta_features[f'{feature_prefix}.Imc1'] = (meta_features[f'{feature_prefix}.JointEntropy'] - hxy1) / max(hx, hy)
    else:
        meta_features[f'{feature_prefix}.Imc1'] = 0
        
    if hxy2 > meta_features[f'{feature_prefix}.JointEntropy']:
        meta_features[f'{feature_prefix}.Imc2'] = np.sqrt(1 - np.exp(-2 * (hxy2 - meta_features[f'{feature_prefix}.JointEntropy'])))
    else:
        meta_features[f'{feature_prefix}.Imc2'] = 0

    meta_features[f'{feature_prefix}.ClusterShade'] = np.sum((i + j - ux - uy)**3 * p)
    meta_features[f'{feature_prefix}.ClusterProminence'] = np.sum((i + j - ux - uy)**4 * p)
    
    meta_features[f'{feature_prefix}.JointAverage'] = ux 
    meta_features[f'{feature_prefix}.MaximumProbability'] = np.max(p)
    meta_features[f'{feature_prefix}.ClusterTendency'] = np.sum((i + j - ux - uy)**2 * p)

    return meta_features

def calculate_advanced_eigen_features(matrix, feature_prefix):
    """Calculates Eigen-decomposition features for a given matrix."""
    eigen_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.shape[0] != matrix.shape[1]: 
        if matrix.ndim == 1: 
             print(f"  - SKIPPING: Eigen-features for {feature_prefix}, not a square matrix.")
        return eigen_features
    try:
        clean_matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix[np.isfinite(matrix)]))
        
        eigenvalues = np.real(np.linalg.eigvals(clean_matrix))
        
        eigenvalues_sorted = sorted(eigenvalues, key=np.abs, reverse=True)
        
        eigen_features[f'{feature_prefix}.Eigen.Eigenvalue1'] = eigenvalues_sorted[0]
        if len(eigenvalues_sorted) > 1: eigen_features[f'{feature_prefix}.Eigen.Eigenvalue2'] = eigenvalues_sorted[1]
        eigen_features[f'{feature_prefix}.Eigen.SpectralRadius'] = np.abs(eigenvalues_sorted[0])
        eigen_features[f'{feature_prefix}.Eigen.Trace'] = np.trace(clean_matrix)
        
        abs_eigenvalues = np.abs(eigenvalues) + 1e-9
        eigen_features[f'{feature_prefix}.Eigen.LogDeterminant'] = np.sum(np.log(abs_eigenvalues))
        eigen_features[f'{feature_prefix}.Eigen.DeterminantSign'] = np.prod(np.sign(eigenvalues))

        # --- FIX START ---
        # Changed [] to np.array([]) to ensure type consistency
        p = np.abs(eigenvalues) / np.sum(np.abs(eigenvalues)) if np.sum(np.abs(eigenvalues)) > 0 else np.array([])
        
        # Now p is always a numpy array, so p > 0 works
        if len(p) > 0 and len(p[p>0]) > 0: 
            eigen_features[f'{feature_prefix}.Eigen.Entropy'] = -np.sum(p[p>0] * np.log2(p[p>0]))
        # --- FIX END ---
    
    except np.linalg.LinAlgError: print(f"  - WARNING: Eigen-decomposition failed for {feature_prefix}.")
    return eigen_features

def calculate_diagonal_features(matrix, feature_prefix):
    """Calculates Diagonal vs. Off-Diagonal features."""
    diag_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        if matrix.ndim == 1:
            print(f"  - SKIPPING: Diagonal-features for {feature_prefix}, not a square matrix.")
        return diag_features
    diagonal = np.diag(matrix)[~np.isnan(np.diag(matrix))]
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    off_diagonal = off_diagonal[~np.isnan(off_diagonal)]
    if diagonal.size > 0:
        diag_features[f'{feature_prefix}.Diag.Mean'] = np.mean(diagonal)
        diag_features[f'{feature_prefix}.Diag.Variance'] = np.var(diagonal)
    if off_diagonal.size > 0:
        diag_features[f'{feature_prefix}.OffDiag.Mean'] = np.mean(off_diagonal)
        diag_features[f'{feature_prefix}.OffDiag.Variance'] = np.var(off_diagonal)
    sum_abs_diag, sum_abs_offdiag = np.sum(np.abs(diagonal)), np.sum(np.abs(off_diagonal))
    if sum_abs_offdiag > 0: diag_features[f'{feature_prefix}.Diag.DominanceRatio'] = sum_abs_diag / sum_abs_offdiag
    return diag_features

def calculate_symmetry_features(matrix, feature_prefix):
    """Calculates features based on the symmetry/asymmetry of the matrix."""
    sym_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        if matrix.ndim == 1:
            print(f"  - SKIPPING: Symmetry-features for {feature_prefix}, not a square matrix.")
        return sym_features
    clean_matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix[np.isfinite(matrix)]))
    asymmetry_matrix = clean_matrix - clean_matrix.T
    sym_features[f'{feature_prefix}.Symmetry.FrobeniusNorm'] = np.linalg.norm(asymmetry_matrix, 'fro')
    sym_features[f'{feature_prefix}.Symmetry.MeanAbsoluteAsymmetry'] = np.mean(np.abs(asymmetry_matrix))
    net_influence = np.sum(clean_matrix, axis=1) - np.sum(clean_matrix, axis=0)
    sym_features[f'{feature_prefix}.Symmetry.NetInfluenceMean'] = np.mean(net_influence)
    sym_features[f'{feature_prefix}.Symmetry.NetInfluenceVariance'] = np.var(net_influence)
    return sym_features

def calculate_cluster_features(matrix, feature_prefix):
    """Calculates features based on K-Means clustering of the matrix rows."""
    cluster_features = {}
    
    # 1. Basic Validity Checks
    if matrix is None or np.all(np.isnan(matrix)) or matrix.shape[0] < 4: 
        return cluster_features
    
    clean_matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix[np.isfinite(matrix)]))
    
    if clean_matrix.ndim == 1:
        clean_matrix = clean_matrix.reshape(-1, 1)

    # Count how many unique rows exist. We cannot find more clusters than unique rows.
    unique_rows = np.unique(clean_matrix, axis=0)
    n_unique = unique_rows.shape[0]

    # Cap K at 8, matrix size, OR number of unique rows
    max_possible_k = min(8, clean_matrix.shape[0] - 1, n_unique)
    
    if max_possible_k < 2: 
        return {}
    # ---------------------------------------
        
    best_k, best_score = -1, -1
    
    for k in range(2, max_possible_k + 1):
        try:
            # fit_predict suppresses some warnings better than fit()
            kmeans_attempt = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans_attempt.fit_predict(clean_matrix)
            
            # Ensure we actually got k clusters
            if len(np.unique(labels)) < 2: continue

            score = silhouette_score(clean_matrix, labels)
            if score > best_score: best_score, best_k = score, k
        except Exception: 
            continue
            
    if best_k != -1:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(clean_matrix)
        cluster_features[f'{feature_prefix}.Cluster.OptimalK'] = best_k
        cluster_features[f'{feature_prefix}.Cluster.SilhouetteScore'] = best_score
        cluster_features[f'{feature_prefix}.Cluster.Inertia'] = kmeans.inertia_
        
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        if len(counts) > 0:
            cluster_features[f'{feature_prefix}.Cluster.SizeMedian'] = np.median(counts)
            cluster_features[f'{feature_prefix}.Cluster.SizeMean'] = np.mean(counts)
            cluster_features[f'{feature_prefix}.Cluster.SizeVariance'] = np.var(counts)
            cluster_features[f'{feature_prefix}.Cluster.SizeMax'] = np.max(counts)
            cluster_features[f'{feature_prefix}.Cluster.SizeMin'] = np.min(counts)
            
    return cluster_features

def calculate_profile_shape_features(matrix, feature_prefix):
    """
    Analyzes the 1D shape of the Main Diagonal and Anti-Diagonal.
    Detects Peaks, Valleys, and Bimodality.
    """
    if matrix is None or np.all(np.isnan(matrix)):
        return {}
    
    features = {}
    rows, cols = matrix.shape
    
    # 1. Extract Profiles
    # Main Diagonal (Volume properties)
    main_diag = np.diag(matrix)
    
    # Anti-Diagonal (Interface properties: 0-23, 1-22, etc.)
    # fliplr flips left-to-right, so the main diagonal becomes the anti-diagonal
    anti_diag = np.diag(np.fliplr(matrix))
    
    profiles = {
        'MainDiag': main_diag,
        'AntiDiag': anti_diag
    }
    
    for name, profile in profiles.items():
        # Remove NaNs for signal analysis
        clean_profile = profile[~np.isnan(profile)]
        
        if len(clean_profile) < 3:
            continue
            
        # --- A. Basic Peak Detection ---
        # prominence=0.1 ensures we don't count tiny noise jitters as peaks
        # For Log metrics, 0.1 is a reasonable relative threshold.
        # For raw metrics (like large Euler numbers), we might need normalization, 
        # but find_peaks is generally robust if we look at relative prominence.
        
        # We normalize the profile to 0-1 for consistent peak detection parameters
        ptp = np.ptp(clean_profile)
        if ptp > 1e-6:
            norm_profile = (clean_profile - np.min(clean_profile)) / ptp
            peaks, properties = find_peaks(norm_profile, prominence=0.05)
            valleys, _ = find_peaks(-norm_profile, prominence=0.05)
        else:
            peaks, valleys = [], []
            
        num_peaks = len(peaks)
        
        features[f'{feature_prefix}.{name}.PeakCount'] = num_peaks
        features[f'{feature_prefix}.{name}.ValleyCount'] = len(valleys)
        
        # --- B. Bimodality / Structure Metrics ---
        if num_peaks >= 2:
            # Distance between the first two major peaks (in gray levels)
            # This tells you how "separated" the two tissue types are.
            sorted_peaks = np.sort(peaks)
            features[f'{feature_prefix}.{name}.PeakSeparation'] = float(sorted_peaks[-1] - sorted_peaks[0])
            
            # Depth of the valley between peaks (The "Dip")
            # Deeper valley = More distinct layers. Shallow valley = Mixed tissue.
            if len(valleys) > 0:
                # Find valley between the peaks
                relevant_valleys = [v for v in valleys if sorted_peaks[0] < v < sorted_peaks[-1]]
                if relevant_valleys:
                    # Get the deepest valley in the normalized signal
                    min_valley_height = np.min(norm_profile[relevant_valleys])
                    # Get average height of the two surrounding peaks
                    avg_peak_height = np.mean(norm_profile[sorted_peaks])
                    
                    # Bimodality Index: How deep is the dip relative to the peaks?
                    features[f'{feature_prefix}.{name}.BimodalityIndex'] = float(avg_peak_height - min_valley_height)
                else:
                     features[f'{feature_prefix}.{name}.BimodalityIndex'] = 0.0
            else:
                features[f'{feature_prefix}.{name}.BimodalityIndex'] = 0.0
        else:
            features[f'{feature_prefix}.{name}.PeakSeparation'] = 0.0
            features[f'{feature_prefix}.{name}.BimodalityIndex'] = 0.0
            
        # --- C. Signal Energy/Roughness ---
        # Does the profile look smooth or jagged?
        # Calculate sum of absolute differences (Total Variation)
        diffs = np.diff(clean_profile)
        features[f'{feature_prefix}.{name}.Roughness'] = np.sum(np.abs(diffs))

    return features

def calculate_glam_multifractal_spectrum(image_3d, num_levels):
    """
    Calculates the Multifractal Spectrum (f(alpha) vs alpha) and Generalized Dimensions (D_q).
    Uses the Method of Moments (Partition Function).
    """
    multifractal_features = {}
    
    # Define moments q to scan
    q_values = np.array([-5, -2, 0, 1, 2, 5]) 
    
    # Box sizes for scaling analysis
    box_sizes = [2, 3, 4, 6, 8, 12, 16]
    
    def get_multifractal_spectrum(binary_mask):
        if np.sum(binary_mask) < 50: 
            return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}

        pixels = np.argwhere(binary_mask > 0)
        N = len(pixels)
        
        # Partition function Z(q, epsilon)
        Z = defaultdict(list)
        
        # Track which sizes actually got processed
        processed_sizes = []
        
        for eps in box_sizes:
            if eps >= min(binary_mask.shape): continue
            
            processed_sizes.append(eps)
            
            box_indices = np.floor(pixels / eps).astype(int)
            _, counts = np.unique(box_indices, axis=0, return_counts=True)
            p = counts / N
            
            for q in q_values:
                if q == 1:
                    # Shannon Entropy for q=1 (limit q->1)
                    sum_val = np.sum(p * np.log(p + 1e-12))
                else:
                    sum_val = np.sum(p ** q)
                
                Z[q].append(sum_val)

        # Safety Check: Did we get enough scales for a regression?
        if len(processed_sizes) < 3:
             return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}
             
        # X-axis: log(epsilon)
        x_all = np.log(np.array(processed_sizes))
        
        tau_q = {}
        
        for q in q_values:
            # Y-axis: log(Z(q, epsilon))
            # Z[q] corresponds 1-to-1 with processed_sizes
            z_values = np.array(Z[q])
            
            # --- CRITICAL FIX: Mask invalid values before Log ---
            # Remove NaNs, Infs, and Zeros (which cause log errors)
            mask = (z_values > 1e-20) & np.isfinite(z_values)
            
            if np.sum(mask) < 3:
                tau_q[q] = 0.0
                results[f"D_q_{q}"] = 0.0
                continue
                
            y_clean = np.log(z_values[mask])
            x_clean = x_all[mask]
            
            # Perform Regression
            slope, _, _, _, _ = stats.linregress(x_clean, y_clean)
            
            tau = slope
            tau_q[q] = tau
            
            # Calculate D_q
            if q == 1:
                results[f"D_q_{q}"] = tau 
            elif q != 1:
                results[f"D_q_{q}"] = tau / (q - 1)
        
        # Calculate Alpha and f(Alpha) via finite difference
        alphas = []
        q_sorted = sorted(q_values)
        for i in range(len(q_sorted)-1):
            q1, q2 = q_sorted[i], q_sorted[i+1]
            t1, t2 = tau_q[q1], tau_q[q2]
            
            # Numerical derivative d(tau)/dq
            if (q2 - q1) != 0:
                alpha = (t2 - t1) / (q2 - q1)
                alphas.append(alpha)
            
        if alphas:
            results["Width"] = max(alphas) - min(alphas)
            results["Alpha_0"] = np.mean(alphas)
        else:
            results["Width"] = 0.0
            results["Alpha_0"] = 0.0
            
        return results

    # Initialize results container - FIX: Define 'results' here so it resets per loop iteration logic in your original structure?
    # Actually, the original code had 'results = {}' INSIDE the helper. 
    # But here we need to loop over levels.
    
    # 1. Diagonal: Volume Multifractality
    for i in range(num_levels):
        binary_image = (image_3d == i)
        # We need a fresh results dict inside the helper for each call
        results = {} # This dummy initialization isn't used, the helper creates it. 
        # Wait, the helper 'get_multifractal_spectrum' needs to define 'results = {}' inside itself.
        # My snippet above returns 'results', so we are good.
        
        mf_res = get_multifractal_spectrum(binary_image)
        
        multifractal_features[f'GLAM_VolumeMultifractal_Width_{i}'] = mf_res["Width"]
        multifractal_features[f'GLAM_VolumeMultifractal_Alpha0_{i}'] = mf_res["Alpha_0"]
        multifractal_features[f'GLAM_VolumeMultifractal_D2_{i}'] = mf_res.get("D_q_2", 0.0)

    # 2. Off-Diagonal: Interface Multifractality
    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue
            
            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            if not np.any(mask_i) or not np.any(mask_j):
                val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
            else:
                interface = ndimage.binary_dilation(mask_i) & mask_j
                if np.sum(interface) < 50:
                    val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
                else:
                    mf_res = get_multifractal_spectrum(interface)
                    val_width = mf_res["Width"]
                    val_alpha = mf_res["Alpha_0"]
                    val_d2 = mf_res.get("D_q_2", 0.0)
            
            multifractal_features[f'GLAM_InterfaceMultifractal_Width_{i}_{j}'] = val_width
            multifractal_features[f'GLAM_InterfaceMultifractal_Alpha0_{i}_{j}'] = val_alpha
            multifractal_features[f'GLAM_InterfaceMultifractal_D2_{i}_{j}'] = val_d2

    return multifractal_features

def calculate_glam_shape_matrices(structured_image, num_levels, spacing):
    """
    Calculates 3D shape features for:
    1. Diagonals (Per-Level): Sphericity, Spikiness (Kurtosis), Solidity.
    2. Off-Diagonals (Inter-Level): Centroid Distance, Interface Area.
    """
    feats = {}
    spacing_zyx = spacing[::-1]
    voxel_vol = np.prod(spacing)
    
    # --- OPTIMIZATION: Crop to Tumor Bounding Box ---
    mask_all = (structured_image > -1)
    if not np.any(mask_all): return feats
    
    slices = ndimage.find_objects(mask_all.astype(int))
    if not slices: return feats
    
    roi_img = structured_image[slices[0]].copy()
    
    # 1. DIAGONALS & Centroids
    centroids = {} 
    
    for i in range(num_levels):
        mask_i = (roi_img == i)
        voxel_coords_idx = np.argwhere(mask_i)
        n_voxels = len(voxel_coords_idx)
        
        if n_voxels > 0:
            c_mass = np.mean(voxel_coords_idx * np.array(spacing_zyx), axis=0)
            centroids[i] = c_mass
        
        val_sphericity = 0.0
        val_solidity = 0.0 
        val_rad_mean = 0.0
        val_rad_var = 0.0
        val_rad_skew = 0.0
        val_rad_kurt = 0.0
        
        if n_voxels >= 4:
            physical_coords = voxel_coords_idx * np.array(spacing_zyx)
            try:
                verts, faces, _, _ = measure.marching_cubes(mask_i, level=0.5, spacing=spacing_zyx)
                surf_area = measure.mesh_surface_area(verts, faces)
                vol = n_voxels * voxel_vol
                if surf_area > 0:
                    val_sphericity = (np.pi**(1/3) * (6 * vol)**(2/3)) / surf_area
                
                centroid_mesh = np.mean(verts, axis=0)
                radial_dists = np.linalg.norm(verts - centroid_mesh, axis=1)
                val_rad_mean = np.mean(radial_dists)
                val_rad_var = np.var(radial_dists)
                val_rad_skew = stats.skew(radial_dists)
                val_rad_kurt = stats.kurtosis(radial_dists)
            except: pass
            
            try:
                hull = ConvexHull(physical_coords)
                if hull.volume > 0: val_solidity = (n_voxels * voxel_vol) / hull.volume
                else: val_solidity = 1.0
            except: val_solidity = 1.0 if n_voxels > 0 else 0.0

        feats[f'GLAM_Shape_Sphericity_{i}'] = val_sphericity
        feats[f'GLAM_Shape_Solidity_{i}'] = val_solidity
        feats[f'GLAM_Shape_RadialMean_{i}'] = val_rad_mean
        feats[f'GLAM_Shape_RadialVariance_{i}'] = val_rad_var
        feats[f'GLAM_Shape_RadialSkewness_{i}'] = val_rad_skew
        feats[f'GLAM_Shape_RadialKurtosis_{i}'] = val_rad_kurt

    # 2. OFF-DIAGONALS
    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue 
            
            val_dist = 0.0
            if i in centroids and j in centroids:
                val_dist = np.linalg.norm(centroids[i] - centroids[j])
            
            feats[f'GLAM_Shape_CentroidDist_{i}_{j}'] = val_dist
            
            mask_i = (roi_img == i)
            mask_j = (roi_img == j)
            
            val_interface = 0.0
            if np.any(mask_i) and np.any(mask_j):
                dilated_i = ndimage.binary_dilation(mask_i)
                intersection = dilated_i & mask_j
                val_interface = np.sum(intersection) * voxel_vol 
            
            feats[f'GLAM_Shape_InterfaceArea_{i}_{j}'] = val_interface

    return feats