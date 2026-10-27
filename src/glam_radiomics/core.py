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
import scipy.signal

# Import config access
from .config import get_config

# Safely attempt to load CuPy for CUDA acceleration
try:
    import cupy as cp
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    HAS_GPU = False

#------------------------------------------------------------------------------
def calculate_glcm_3d(image, num_gl):
    """Calculates the symmetric 3D GLCM (offset [0,0,1]). GPU Accelerated."""
    if not HAS_GPU: return _calculate_glcm_3d_cpu(image, num_gl)
    
    image_gpu = cp.array(image, dtype=cp.int32)
    glcm = cp.zeros((num_gl, num_gl), dtype=cp.uint32)
    
    # Fast GPU offset slicing
    img_curr = image_gpu[:, :, :-1]
    img_next = image_gpu[:, :, 1:]
    
    mask = (img_curr > 0) & (img_next > 0)
    
    if cp.any(mask):
        vals_curr = img_curr[mask] - 1
        vals_next = img_next[mask] - 1
        
        # Flatten and bincount for instant 2D matrix accumulation
        flat_indices = vals_curr * num_gl + vals_next
        counts = cp.bincount(flat_indices, minlength=num_gl*num_gl)
        glcm += counts.reshape((num_gl, num_gl)).astype(cp.uint32)
        
    glcm_symmetric = glcm + glcm.T
    total = cp.sum(glcm_symmetric)
    
    if total == 0: 
        res = np.zeros((num_gl, num_gl), dtype=float)
    else:
        res = (glcm_symmetric.astype(cp.float64) / float(total)).get()
        
    del image_gpu, img_curr, img_next, mask, glcm, glcm_symmetric
    cp.get_default_memory_pool().free_all_blocks()
    
    return res

def _calculate_glcm_3d_cpu(image, num_gl):
    """Fallback CPU function for GLCM."""
    glcm = np.zeros((num_gl, num_gl), dtype=np.uint32)
    offset = (0, 0, 1) 
    z, y, x = np.where(image > 0)
    nz, ny, nx = z + offset[0], y + offset[1], x + offset[2]
    valid = (nz < image.shape[0]) & (ny < image.shape[1]) & (nx < image.shape[2])
    curr_vals = image[z[valid], y[valid], x[valid]]
    neigh_vals = image[nz[valid], ny[valid], nx[valid]]
    mask_neigh = neigh_vals > 0
    curr_vals = curr_vals[mask_neigh]
    neigh_vals = neigh_vals[mask_neigh]
    np.add.at(glcm, (curr_vals - 1, neigh_vals - 1), 1)
    glcm_sym = glcm + glcm.T
    total = np.sum(glcm_sym)
    if total == 0: return np.zeros_like(glcm_sym, dtype=float)
    return glcm_sym.astype(float) / total


def calculate_glrlm_3d(image, num_gl):
    """Calculates 3D GLRLM. GPU Accelerated via boolean edge detection (No Python Loops)."""
    if not HAS_GPU: return _calculate_glrlm_3d_cpu(image, num_gl)
    
    image_gpu = cp.array(image, dtype=cp.int32)
    max_run = image_gpu.shape[2]
    glrlm = cp.zeros((num_gl, max_run), dtype=cp.uint32)
    
    for g in range(1, num_gl + 1):
        mask = (image_gpu == g)
        if not cp.any(mask): continue
        
        # Pad along the X-axis to find natural boundaries
        padded_mask = cp.pad(mask, ((0,0), (0,0), (1,1)), mode='constant', constant_values=False)
        
        # Detect rising (+1) and falling (-1) edges natively on GPU
        diffs = padded_mask[:, :, 1:].astype(cp.int8) - padded_mask[:, :, :-1].astype(cp.int8)
        
        starts = cp.argwhere(diffs == 1)
        ends = cp.argwhere(diffs == -1)
        
        # Calculate lengths
        lengths = ends[:, 2] - starts[:, 2]
        
        if lengths.size > 0:
            counts = cp.bincount(lengths)
            length_indices = cp.arange(1, len(counts))
            valid_counts = counts[1:]
            
            valid_mask = length_indices <= max_run
            glrlm[g-1, length_indices[valid_mask] - 1] += valid_counts[valid_mask].astype(cp.uint32)

    glrlm_cpu = glrlm.get()
    
    del image_gpu, glrlm
    cp.get_default_memory_pool().free_all_blocks()
    
    if np.sum(glrlm_cpu) == 0: return np.zeros((num_gl, 1), dtype=float)
    
    col_sums = np.sum(glrlm_cpu, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    glrlm_cpu = glrlm_cpu[:, :last_idx+1]
    
    return glrlm_cpu.astype(float) / np.sum(glrlm_cpu)

def _calculate_glrlm_3d_cpu(image, num_gl):
    """Fallback CPU function for GLRLM."""
    max_run = image.shape[2]
    glrlm = np.zeros((num_gl, max_run), dtype=np.uint32)
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
    if np.sum(glrlm) == 0: 
        return np.zeros((num_gl, 1), dtype=float)
    col_sums = np.sum(glrlm, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    glrlm = glrlm[:, :last_idx+1]
    return glrlm.astype(float) / np.sum(glrlm)


def calculate_glszm_3d(image, num_gl):
    """Calculates 3D GLSZM with GPU Acceleration and batched counting."""
    if not HAS_GPU: return _calculate_glszm_3d_cpu(image, num_gl)
    import cupyx.scipy.ndimage as cupy_ndimage
    
    image_gpu = cp.array(image, dtype=cp.int32)
    max_possible_zone = int(cp.sum(image_gpu > 0))
    if max_possible_zone == 0: return np.zeros((num_gl, 1), dtype=float)
    
    glszm = cp.zeros((num_gl, max_possible_zone), dtype=cp.uint32)
    structure_3d = cp.ones((3,3,3), dtype=cp.int32)
    
    for g in range(1, num_gl + 1):
        binary_img = (image_gpu == g)
        if not cp.any(binary_img): continue
        
        labeled, num_feats = cupy_ndimage.label(binary_img, structure=structure_3d)
        if num_feats == 0: continue
        
        # GPU Vectorized region counting (Massive speedup over ndimage.sum)
        roi_counts = cp.bincount(labeled.ravel())[1:] 
        unique, counts = cp.unique(roi_counts, return_counts=True)
        
        # Ensure sizes don't exceed max bound
        valid_idx = unique <= max_possible_zone
        unique, counts = unique[valid_idx], counts[valid_idx]
        
        glszm[g-1, unique - 1] += counts.astype(cp.uint32)

    glszm_cpu = glszm.get()
    
    del image_gpu, glszm, structure_3d
    cp.get_default_memory_pool().free_all_blocks()
    
    if np.sum(glszm_cpu) == 0: return np.zeros((num_gl, 1), dtype=float)
    
    col_sums = np.sum(glszm_cpu, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    glszm_cpu = glszm_cpu[:, :last_idx+1]
    
    return glszm_cpu.astype(float) / np.sum(glszm_cpu)

def _calculate_glszm_3d_cpu(image, num_gl):
    max_possible_zone = np.sum(image > 0)
    if max_possible_zone == 0: return np.zeros((num_gl, 1), dtype=float)
    glszm = np.zeros((num_gl, max_possible_zone), dtype=np.uint32)
    for g in range(1, num_gl + 1):
        binary_img = (image == g)
        if not np.any(binary_img): continue
        labeled, num_feats = ndimage.label(binary_img, structure=np.ones((3,3,3)))
        if num_feats == 0: continue
        roi_counts = ndimage.sum(binary_img, labeled, index=np.arange(1, num_feats + 1))
        unique, counts = np.unique(roi_counts, return_counts=True)
        for size, count in zip(unique, counts):
            idx = int(size) - 1
            if idx < max_possible_zone: glszm[g-1, idx] += count
    if np.sum(glszm) == 0: return np.zeros((num_gl, 1), dtype=float)
    col_sums = np.sum(glszm, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    glszm = glszm[:, :last_idx+1]
    return glszm.astype(float) / np.sum(glszm)


def calculate_gldm_3d(image, num_gl, alpha=0):
    """Calculates 3D GLDM using GPU 3D Convolutions."""
    if not HAS_GPU: return _calculate_gldm_3d_cpu(image, num_gl, alpha)
    import cupyx.scipy.ndimage as cupy_ndimage
    
    gldm = cp.zeros((num_gl, 27), dtype=cp.uint32)
    footprint = cp.ones((3,3,3), dtype=cp.int32)
    footprint[1,1,1] = 0
    image_gpu = cp.array(image, dtype=cp.int32)
    
    for g in range(1, num_gl + 1):
        mask_g = (image_gpu == g)
        if not cp.any(mask_g): continue
        if alpha == 0:
            neighbor_count = cupy_ndimage.convolve(mask_g.astype(cp.int32), footprint, mode='constant', cval=0)
            counts = neighbor_count[mask_g]
            unique, u_counts = cp.unique(counts, return_counts=True)
            gldm[g-1, unique] += u_counts.astype(cp.uint32)
            
    gldm_cpu = gldm.get()
    
    del image_gpu, gldm, footprint
    cp.get_default_memory_pool().free_all_blocks()
    
    if np.sum(gldm_cpu) == 0: return np.zeros((num_gl, 1), dtype=float)

    col_sums = np.sum(gldm_cpu, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    gldm_cpu = gldm_cpu[:, :last_idx+1]
    return gldm_cpu.astype(float) / np.sum(gldm_cpu)

def _calculate_gldm_3d_cpu(image, num_gl, alpha=0):
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
            for k, c in zip(unique, u_counts): gldm[g-1, k] += c
    if np.sum(gldm) == 0: return np.zeros((num_gl, 1), dtype=float)
    col_sums = np.sum(gldm, axis=0)
    last_idx = np.max(np.where(col_sums > 0)[0]) if np.any(col_sums) else 0
    gldm = gldm[:, :last_idx+1]
    return gldm.astype(float) / np.sum(gldm)


def calculate_ngtdm_3d(image, num_gl):
    """Calculates NGTDM using GPU 3D Convolutions."""
    if not HAS_GPU: return _calculate_ngtdm_3d_cpu(image, num_gl)
    import cupyx.scipy.ndimage as cupy_ndimage
    
    ngtdm = cp.zeros((num_gl, 3), dtype=cp.float64)
    kernel = cp.ones((3,3,3), dtype=cp.float32)
    kernel[1,1,1] = 0
    
    image_gpu = cp.array(image, dtype=cp.float32)
    mask_gpu = (image_gpu > 0).astype(cp.float32)
    
    neighbor_count_map = cupy_ndimage.convolve(mask_gpu, kernel, mode='constant', cval=0)
    valid_neighborhood = neighbor_count_map > 0
    
    sum_neighbor_vals = cupy_ndimage.convolve(image_gpu, kernel, mode='constant', cval=0)
    avg_neighbor_map = cp.zeros_like(image_gpu)
    avg_neighbor_map[valid_neighborhood] = sum_neighbor_vals[valid_neighborhood] / neighbor_count_map[valid_neighborhood]
    
    diff_map = cp.abs(image_gpu - avg_neighbor_map)
    
    for g in range(1, num_gl + 1):
        mask_g = (image_gpu == g)
        n_i = cp.sum(mask_g)
        if n_i == 0: continue
        
        valid_mask_g = mask_g & valid_neighborhood
        s_i = cp.sum(diff_map[valid_mask_g])
        
        ngtdm[g-1, 0] = n_i
        ngtdm[g-1, 1] = s_i

    ngtdm_cpu = ngtdm.get()
    
    del image_gpu, mask_gpu, neighbor_count_map, valid_neighborhood, sum_neighbor_vals, avg_neighbor_map, diff_map
    cp.get_default_memory_pool().free_all_blocks()
    
    total_n = np.sum(ngtdm_cpu[:, 0])
    
    if total_n > 0: ngtdm_cpu[:, 2] = ngtdm_cpu[:, 0] / total_n
    return ngtdm_cpu

def _calculate_ngtdm_3d_cpu(image, num_gl):
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
    if total_n > 0: ngtdm[:, 2] = ngtdm[:, 0] / total_n
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
    
    # Crop mask to bounding box with padding to prevent marching_cubes segfaults
    from scipy import ndimage
    slices = ndimage.find_objects((mask_arr > 0).astype(int))
    if slices:
        z_s, y_s, x_s = slices[0]
        mask_arr = mask_arr[
            max(0, z_s.start-2):min(mask_arr.shape[0], z_s.stop+2),
            max(0, y_s.start-2):min(mask_arr.shape[1], y_s.stop+2),
            max(0, x_s.start-2):min(mask_arr.shape[2], x_s.stop+2)
        ]
    
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
            # Downsample massive whole-organ masks to prevent CPU hanging
            if len(physical_coords) > 25000:
                indices = np.linspace(0, len(physical_coords) - 1, 25000, dtype=int)
                hull_coords = physical_coords[indices]
            else:
                hull_coords = physical_coords
                
            hull = ConvexHull(hull_coords)
            hull_points = hull_coords[hull.vertices]
            
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

def get_shell_kernel(r):
    """Centered lattice shell: r-0.5 <= distance < r+0.5, excluding self."""
    if int(r) != r or r < 1:
        raise ValueError("Shell radius must be a positive integer")
    size = int(np.ceil(r + 0.5))
    z, y, x = np.ogrid[-size:size+1, -size:size+1, -size:size+1]
    distance = np.sqrt(z*z + y*y + x*x)
    return ((distance >= r - 0.5) & (distance < r + 0.5)).astype(float)


_LAST_GPU_RDF_GEOMETRY = None


class _GpuRdfGeometry:
    """Cached CuPy 3D FFT shell denominators for global RDF calculation."""
    def __init__(self, mask, radius):
        from scipy.fft import next_fast_len
        from .rdf_fast import _shell_kernel

        self.mask = mask.copy()
        self.radius = int(radius)
        self.shape = mask.shape
        self.coords = np.argwhere(mask)
        self.indices_cpu = tuple(self.coords.T)
        self.indices_gpu = tuple(cp.asarray(idx) for idx in self.indices_cpu)
        self.extent = np.minimum(self.radius, np.asarray(self.shape) - 1).astype(int)
        self.kernel_shape = tuple(2 * self.extent + 1)
        self.fft_shape = tuple(
            next_fast_len(d + k - 1) for d, k in zip(self.shape, self.kernel_shape)
        )
        self.crop = tuple(
            slice(int(e), int(e) + d) for e, d in zip(self.extent, self.shape)
        )
        self._shell_kernel = _shell_kernel
        self.kernels_gpu = {}
        self.cached_bytes = 0
        self.max_cache_bytes = 512 * 1024 * 1024
        self.denominators_cpu = np.zeros((self.radius, len(self.coords)), dtype=np.float64)

        mask_gpu = cp.asarray(mask, dtype=cp.float64)
        mask_fft = cp.fft.rfftn(mask_gpu, s=self.fft_shape)
        for r in range(1, self.radius + 1):
            kfft = self.kernel_fft(r)
            counts_gpu = self.counts(mask_fft, kfft)
            self.denominators_cpu[r - 1] = cp.asnumpy(counts_gpu[self.indices_gpu])
        del mask_gpu, mask_fft

    def kernel_fft(self, r):
        if r in self.kernels_gpu:
            return self.kernels_gpu[r]
        kernel = self._shell_kernel(r, self.shape)
        pad = tuple(
            (int(e) - (k - 1) // 2, int(e) - (k - 1) // 2)
            for e, k in zip(self.extent, kernel.shape)
        )
        kernel = np.pad(kernel, pad)
        kfft = cp.fft.rfftn(cp.asarray(kernel, dtype=cp.float64), s=self.fft_shape)
        if self.cached_bytes + kfft.nbytes <= self.max_cache_bytes:
            self.kernels_gpu[r] = kfft
            self.cached_bytes += kfft.nbytes
        return kfft

    def counts(self, image_fft, kernel_fft):
        raw = cp.fft.irfftn(image_fft * kernel_fft, s=self.fft_shape)[self.crop]
        rounded = cp.rint(raw)
        if float(cp.max(cp.abs(raw - rounded)).item()) > 1e-4:
            raise FloatingPointError("GPU FFT shell counts exceed integer-rounding tolerance")
        return cp.maximum(rounded, 0.0)


def _get_gpu_rdf_geometry(mask, radius):
    global _LAST_GPU_RDF_GEOMETRY
    old = _LAST_GPU_RDF_GEOMETRY
    if (
        old is not None
        and old.radius == radius
        and old.shape == mask.shape
        and np.array_equal(old.mask, mask)
    ):
        return old
    _LAST_GPU_RDF_GEOMETRY = None
    old = None
    cp.get_default_memory_pool().free_all_blocks()
    _LAST_GPU_RDF_GEOMETRY = _GpuRdfGeometry(mask, radius)
    return _LAST_GPU_RDF_GEOMETRY


def calculate_rdf_3d(image_3d, num_levels, max_radius, level_counts,
                     total_roi_voxels, num_randomisations, rdf_sample_points,
                     sample_mask=None):
    """
    Calculates the boundary-corrected Radial Distribution Function (RDF) for a 3D image.
    Uses deterministic CuPy 3D FFTs over all supported alpha voxels when HAS_GPU is True,
    with automatic fallback to CPU rdf_fast.calculate_rdf_3d when HAS_GPU is False.
    Enforces identical `sample_mask` focal-reference vs target-domain semantics on GPU and CPU.
    """
    if not HAS_GPU:
        return _calculate_rdf_3d_cpu(
            image_3d, num_levels, max_radius, level_counts,
            total_roi_voxels, num_randomisations, rdf_sample_points, sample_mask
        )

    from .rdf_fast import _support_params, _required_median_volumes, _resolve_target_and_ref_masks
    from .rdf_smoothing import attach_geometric_support

    image = np.asarray(image_3d)
    if image.ndim != 3:
        raise ValueError("RDF requires a 3D image")
    for name, value in [
        ("num_levels", num_levels),
        ("max_radius", max_radius),
        ("rdf_sample_points", rdf_sample_points),
    ]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    levels, radius = int(num_levels), int(max_radius)
    target_mask, ref_mask = _resolve_target_and_ref_masks(image, sample_mask, total_roi_voxels)
    values = image[target_mask]
    if not values.size or values.size != total_roi_voxels:
        raise ValueError("Empty RDF ROI or total_roi_voxels mismatch")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values >= levels)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("Invalid RDF gray levels")

    labels = values.astype(np.int64)
    counts = np.bincount(labels, minlength=levels)
    is_ref = ref_mask[target_mask]

    try:
        geom = _get_gpu_rdf_geometry(target_mask, radius)
        min_support_n, min_support_fraction, min_median_volume, min_ref_volume, min_shell_fraction = _support_params()
        req_median_by_r = _required_median_volumes(radius, min_median_volume, min_shell_fraction)
        alpha_radius_valid = np.zeros((radius, levels), dtype=bool)
        ref_weights = np.zeros((radius, labels.size), dtype=np.float64)

        for alpha in range(levels):
            alpha_ids = np.flatnonzero(is_ref & (labels == alpha))
            if not alpha_ids.size:
                continue
            tail_failed = False
            for r in range(1, radius + 1):
                if tail_failed:
                    continue
                volumes = geom.denominators_cpu[r - 1, alpha_ids]
                valid_mask = volumes >= min_ref_volume
                supported = volumes[valid_mask]
                total_shell_vol = float(np.sum(supported))
                valid = (
                    supported.size >= min_support_n
                    and supported.size / alpha_ids.size >= min_support_fraction
                    and total_shell_vol > 0.0
                    and float(np.median(supported)) >= req_median_by_r[r - 1]
                )
                if valid:
                    alpha_radius_valid[r - 1, alpha] = True
                    ref_weights[r - 1, alpha_ids[valid_mask]] = 1.0 / total_shell_vol
                else:
                    tail_failed = True

        any_valid_at_r = np.any(alpha_radius_valid, axis=1)
        result_gpu = cp.zeros((radius, levels, levels), dtype=cp.float64)
        image_gpu = cp.asarray(image)
        mask_gpu = cp.asarray(target_mask)

        for beta in range(levels):
            if not counts[beta]:
                continue
            beta_mask_gpu = (mask_gpu & (image_gpu == beta)).astype(cp.float64)
            beta_fft = cp.fft.rfftn(beta_mask_gpu, s=geom.fft_shape)
            rho = counts[beta] / values.size
            for r in range(1, radius + 1):
                if not any_valid_at_r[r - 1]:
                    continue
                neighbor_counts = cp.asnumpy(
                    geom.counts(beta_fft, geom.kernel_fft(r))[geom.indices_gpu]
                )
                result_gpu[r - 1, :, beta] = cp.asarray(
                    np.bincount(
                        labels,
                        weights=neighbor_counts * ref_weights[r - 1],
                        minlength=levels,
                    )
                    / rho
                )

        result = cp.asnumpy(result_gpu)
        del result_gpu, image_gpu, mask_gpu
        cp.get_default_memory_pool().free_all_blocks()
    except cp.cuda.memory.OutOfMemoryError:
        print('  - WARNING: GPU out of memory in RDF FFT; falling back to CPU...')
        global _LAST_GPU_RDF_GEOMETRY
        _LAST_GPU_RDF_GEOMETRY = None
        cp.get_default_memory_pool().free_all_blocks()
        return _calculate_rdf_3d_cpu(
            image_3d, num_levels, max_radius, level_counts,
            total_roi_voxels, num_randomisations, rdf_sample_points, sample_mask
        )

    r_vec = np.arange(1, radius + 1, dtype=np.float64)
    cols = ["r"] + [f"g_{a}_{b}" for a in range(levels) for b in range(levels)]
    flat_pairs = result.reshape(radius, levels * levels)
    df = pd.DataFrame(np.hstack((r_vec[:, None], flat_pairs)), columns=cols, copy=False)
    df.attrs["_glam_fast_raw"] = (r_vec, np.ascontiguousarray(flat_pairs.T))
    return attach_geometric_support(df, alpha_radius_valid, counts)


def _calculate_rdf_3d_cpu(image_3d, num_levels, max_radius, level_counts,
                          total_roi_voxels, num_randomisations, rdf_sample_points,
                          sample_mask=None):
    from .rdf_fast import calculate_rdf_3d as fast_rdf
    return fast_rdf(image_3d, num_levels, max_radius, level_counts,
                    total_roi_voxels, num_randomisations, rdf_sample_points,
                    sample_mask=sample_mask)





def calculate_glam_b2_3d(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the 3D B2 coefficient strictly over each pair's common supported domain
    `L = min(cut_s, cut_r)`, validating that structured and baseline radius grids match.
    """
    glam_b2_coeffs = {}
    if rdf_structured_df.empty or rdf_random_df.empty:
        return {}
    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r_s, raw_s, _, s_map = extract_fast_raw_rdf(rdf_structured_df, num_levels)
    r_r, raw_r, _, r_map = extract_fast_raw_rdf(rdf_random_df, num_levels)
    if not np.array_equal(r_s, r_r):
        raise ValueError('Structured and baseline RDF radius grids differ')
    cut_s = get_row_cutoff_indices(raw_s, rdf_structured_df)
    cut_r = get_row_cutoff_indices(raw_r, rdf_random_df)

    for alpha in range(num_levels):
        for beta in range(num_levels):
            b2_key = f'GLAM_B2_for_g_{alpha}_{beta}'
            si = s_map.get((alpha, beta))
            ri = r_map.get((alpha, beta))
            if si is None or ri is None:
                glam_b2_coeffs[b2_key] = np.nan
                continue
            L = min(int(cut_s[si]), int(cut_r[ri]))
            if L < 2:
                glam_b2_coeffs[b2_key] = np.nan
                continue
            r_sub = r_s[:L]
            integrand = (raw_s[si, :L] - raw_r[ri, :L]) * (r_sub ** 2)
            glam_b2_coeffs[b2_key] = float(-2.0 * np.pi * np.trapezoid(integrand, r_sub))
    return glam_b2_coeffs


def calculate_glam_inverse_correlation_length(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the Inverse Positional Correlation Length (Decay Rate, kappa)
    strictly within each pair's common supported domain `L = min(cut_s, cut_r)`.
    """
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')

    glam_inv_corr_lengths = {}
    if rdf_structured_df.empty or rdf_random_df.empty:
        return {}

    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r_vals, raw_s, _, s_map = extract_fast_raw_rdf(rdf_structured_df, num_levels)
    r_rand, raw_r, _, r_map = extract_fast_raw_rdf(rdf_random_df, num_levels)
    if not np.array_equal(r_vals, r_rand):
        raise ValueError('Structured and baseline RDF radius grids differ')
    cut_s = get_row_cutoff_indices(raw_s, rdf_structured_df)
    cut_r = get_row_cutoff_indices(raw_r, rdf_random_df)

    def exp_decay_rate(r, A, kappa):
        return A * np.exp(-kappa * r)

    for alpha in range(num_levels):
        for beta in range(num_levels):
            inv_corr_len_key = f'GLAM_InverseCorrelationLength_{alpha}_{beta}'
            si = s_map.get((alpha, beta))
            ri = r_map.get((alpha, beta))
            if si is None or ri is None:
                glam_inv_corr_lengths[inv_corr_len_key] = np.nan
                continue

            L = min(int(cut_s[si]), int(cut_r[ri]))
            if L < 6:
                glam_inv_corr_lengths[inv_corr_len_key] = np.nan
                continue

            h_r_raw = raw_s[si, :L] - raw_r[ri, :L]
            if not np.any(np.abs(h_r_raw) > 1e-9):
                glam_inv_corr_lengths[inv_corr_len_key] = np.nan
                continue

            r_sub = r_vals[:L]
            if L > savgol_window:
                h_r = savgol_filter(h_r_raw, savgol_window, savgol_poly, mode='interp')
            else:
                h_r = h_r_raw

            try:
                search_range = L // 2
                if search_range < 3:
                    raise ValueError("Not enough supported data points.")
                peak_index = int(np.argmax(np.abs(h_r[:search_range])))
                if peak_index + 3 >= L:
                    raise ValueError("Peak too close to support cutoff.")

                r_fit = r_sub[peak_index:]
                h_fit = np.abs(h_r[peak_index:])
                if float(np.max(h_fit)) <= 1e-9:
                    raise ValueError("Zero amplitude in fit window.")

                initial_guess = [float(np.abs(h_r[peak_index])), 0.3]
                popt, _ = curve_fit(
                    exp_decay_rate, r_fit, h_fit, p0=initial_guess,
                    maxfev=1000, ftol=1e-8, method='trf',
                    bounds=([0, 0], [np.inf, np.inf]),
                )
                glam_inv_corr_lengths[inv_corr_len_key] = float(popt[1])
            except (RuntimeError, ValueError):
                glam_inv_corr_lengths[inv_corr_len_key] = np.nan

    return glam_inv_corr_lengths


def _resolve_contact_peak_rmin(g_r, L):
    """Resolve first coordination shell horizon `r_min_idx` when the RDF peaks at
    nearest-neighbor contact (`first_peak_idx == 0`, i.e., r = 1):
    1. First strict interior local minimum (`diffs[:-1] < 0` and `diffs[1:] > 0`).
    2. First radius where `g_r <= 1.0` (crossing the random-packing baseline).
    3. Default short-range contact coordination horizon `min(4, L - 1)`.
    """
    if L <= 2:
        return 1
    diffs = np.diff(g_r)
    if L >= 3:
        strict_min = np.flatnonzero((diffs[:-1] < 0) & (diffs[1:] > 0))
        if strict_min.size > 0:
            return int(strict_min[0]) + 1
    below_unity = np.flatnonzero(g_r[1:] <= 1.0)
    if below_unity.size > 0:
        return max(1, min(int(below_unity[0]) + 1, L - 1))
    return min(4, L - 1)


def _find_first_shell_minima_cutoff_safe(g_mat, cutoff_idx, active_mask, max_search_r, peak_prominence):
    """Vectorized cutoff-safe solver for first coordination shell minimum `r_min_idx`
    strictly within each row's supported prefix `:L` (`L = cutoff_idx[idx]`).
    Supports both interior peaks (`first_peak_idx > 0`) and nearest-neighbor contact
    peaks (`first_peak_idx == 0`, where g(r) is highest at r = 1).
    """
    n_pairs, _ = g_mat.shape
    r_min_indices = np.zeros(n_pairs, dtype=np.int32)
    row_active = np.zeros(n_pairs, dtype=bool)
    active_rows = np.flatnonzero(active_mask & (cutoff_idx >= 2))
    if not active_rows.size:
        return r_min_indices, row_active

    for L in np.unique(cutoff_idx[active_rows]):
        L = int(L)
        if L < 2:
            continue
        rows = active_rows[cutoff_idx[active_rows] == L]
        sub = g_mat[rows, :L]
        m = len(rows)
        s_r = min(L, int(max_search_r))
        if s_r == 0:
            continue

        argmax_s = np.argmax(sub[:, :s_r], axis=1).astype(np.int32)
        diffs = np.diff(sub, axis=1)
        has_zero_diff = np.any(diffs == 0, axis=1)

        if L >= 3:
            strict_min = (diffs[:, :-1] < 0) & (diffs[:, 1:] > 0)
            prefix_min = np.minimum.accumulate(sub, axis=1)[:, :-2]
            suffix_min = np.minimum.accumulate(sub[:, ::-1], axis=1)[:, ::-1][:, 2:]
            interior_max = (diffs[:, :-1] > 0) & (diffs[:, 1:] <= 0)
            prom_upper = np.where(interior_max, sub[:, 1:-1] - np.maximum(prefix_min, suffix_min), -1.0)
            can_prom = np.max(prom_upper, axis=1) >= peak_prominence
        else:
            strict_min = np.zeros((m, 0), dtype=bool)
            can_prom = np.zeros(m, dtype=bool)

        fast_mask = (~can_prom) & (~has_zero_diff)
        if np.any(fast_mask):
            f_idx = np.flatnonzero(fast_mask)
            fp = argmax_s[f_idx]

            # Case A: Interior peak (fp > 0)
            interior_sel = fp > 0
            if np.any(interior_sel):
                fi_int = f_idx[interior_sel]
                fp_int = fp[interior_sel]
                if L >= 3:
                    col_grid = np.arange(L - 2, dtype=np.int32)[None, :]
                    after_min = strict_min[fi_int] & (col_grid >= fp_int[:, None])
                    has_m = np.any(after_min, axis=1)
                    rm = np.where(has_m, np.argmax(after_min, axis=1) + 1, np.minimum(fp_int * 2, L - 1))
                else:
                    rm = np.minimum(fp_int * 2, L - 1)
                rm = np.where(rm <= fp_int, L - 1, rm)
                rm = np.minimum(rm, L - 1).astype(np.int32)
                r_min_indices[rows[fi_int]] = rm
                row_active[rows[fi_int]] = True

            # Case B: Contact peak at r = 1 (fp == 0)
            contact_sel = (fp == 0) & (sub[f_idx, 0] > 0)
            if np.any(contact_sel):
                fi_con = f_idx[contact_sel]
                if L >= 3:
                    has_m0 = np.any(strict_min[fi_con], axis=1)
                    first_m0 = np.argmax(strict_min[fi_con], axis=1) + 1
                else:
                    has_m0 = np.zeros(len(fi_con), dtype=bool)
                    first_m0 = np.ones(len(fi_con), dtype=np.int32)
                below_u = sub[fi_con, 1:] <= 1.0
                has_bu = np.any(below_u, axis=1)
                first_bu = np.argmax(below_u, axis=1) + 1
                fallback_r = min(4, L - 1)
                rm0 = np.where(has_m0, first_m0, np.where(has_bu, first_bu, fallback_r))
                rm0 = np.clip(rm0, 1, L - 1).astype(np.int32)
                r_min_indices[rows[fi_con]] = rm0
                row_active[rows[fi_con]] = True

        for k in np.flatnonzero(~fast_mask):
            g_r = sub[k]
            peaks, _ = find_peaks(g_r, prominence=peak_prominence) if can_prom[k] else ((), None)
            if len(peaks) == 0:
                first_peak_idx = int(argmax_s[k])
            else:
                first_peak_idx = int(peaks[0])

            if first_peak_idx == 0:
                if g_r[0] <= 0:
                    continue
                r_min_idx = _resolve_contact_peak_rmin(g_r, L)
            elif not has_zero_diff[k]:
                sub_min = strict_min[k, first_peak_idx:]
                if np.any(sub_min):
                    r_min_idx = first_peak_idx + int(np.argmax(sub_min)) + 1
                else:
                    r_min_idx = min(first_peak_idx * 2, L - 1)
                    if r_min_idx <= first_peak_idx:
                        r_min_idx = L - 1
            else:
                minima, _ = find_peaks(-g_r[first_peak_idx:])
                if len(minima) == 0:
                    r_min_idx = min(first_peak_idx * 2, L - 1)
                    if r_min_idx <= first_peak_idx:
                        r_min_idx = L - 1
                else:
                    r_min_idx = first_peak_idx + int(minima[0])
            r_min_idx = min(int(r_min_idx), L - 1)
            r_min_indices[rows[k]] = r_min_idx
            row_active[rows[k]] = True

    return r_min_indices, row_active


def calculate_glam_coordination_number(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the Coordination Number (Z) strictly within each pair's supported
    radial domain (:L, where L = smoothed.cutoff_idx[row]), vectorized across all pairs.
    """
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    peak_prominence = get_config('PeakProminence')

    coordination_numbers = {}
    if rdf_df.empty or total_roi_voxels <= 0:
        return {}
    from .rdf_smoothing import prepare_smoothed_rdf
    smoothed = prepare_smoothed_rdf(rdf_df, num_levels, savgol_window, savgol_poly)
    r = smoothed.radius
    n_r = r.size
    pairs = smoothed.pairs
    n_pairs = len(pairs)
    if not n_pairs or n_r == 0:
        return coordination_numbers

    raw_mat = smoothed.raw
    cutoff_idx = smoothed.cutoff_idx
    col_grid = np.arange(n_r, dtype=np.int32)[None, :]
    in_sup = col_grid < cutoff_idx[:, None]
    has_pos = (cutoff_idx >= 2) & np.any(np.where(in_sup, raw_mat, 0.0) > 0, axis=1)

    r_min_indices, row_active = _find_first_shell_minima_cutoff_safe(
        smoothed.smooth, cutoff_idx, has_pos, 15, peak_prominence
    )
    valid_min = r_min_indices >= 1
    row_active &= valid_min

    # Retain the original per-prefix trapezoidal reduction order.
    integrals = np.zeros(n_pairs, dtype=np.float64)
    for idx in np.flatnonzero(row_active):
        end = int(r_min_indices[idx]) + 1
        integrand = raw_mat[idx, :end] * (r[:end] ** 2)
        integrals[idx] = float(np.trapezoid(integrand, r[:end]))
    rho = np.asarray(level_counts, dtype=np.float64) / float(total_roi_voxels)
    row_lookup = smoothed.row_index
    for alpha in range(num_levels):
        for beta in range(num_levels):
            coord_key = f'GLAM_CoordNum_g_{alpha}_{beta}'
            row_i = row_lookup.get((alpha, beta))
            if row_i is not None and row_active[row_i] and rho[beta] > 0:
                z_val = float(4.0 * np.pi * rho[beta] * integrals[row_i])
                coordination_numbers[coord_key] = z_val if z_val > 0 else np.nan
            else:
                coordination_numbers[coord_key] = np.nan
    return coordination_numbers



def calculate_glam_compressibility(rdf_df, num_levels):
    """Calculates a metric related to Isothermal Compressibility over each state's supported radial domain."""
    compressibility_metrics = {}
    if rdf_df.empty:
        return {}
    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r, raw, _, row_index = extract_fast_raw_rdf(rdf_df, num_levels)
    cutoff_idx = get_row_cutoff_indices(raw, rdf_df)
    for i in range(num_levels):
        comp_key = f'GLAM_Compressibility_g_{i}_{i}'
        row_i = row_index.get((i, i))
        if row_i is not None:
            L = int(cutoff_idx[row_i])
            if L >= 2 and np.any(raw[row_i, :L] > 1e-6):
                r_valid = r[:L]
                g_r_valid = raw[row_i, :L]
                integrand = (g_r_valid - 1.0) * (r_valid ** 2)
                metric = float(4.0 * np.pi * np.trapezoid(integrand, r_valid))
            else:
                metric = 0.0
            compressibility_metrics[comp_key] = metric
    return compressibility_metrics


def calculate_anisotropic_glam_features(image_3d, num_levels, cutoff_radius):
    """
    Calculates anisotropic GLAM features using the gyration tensor.
    Uses fast vectorized CPU covariance for small local mapping windows (<= 65536 voxels)
    and batched CuPy covariance for large global volumes.
    """
    is_small_window = int(np.prod(image_3d.shape)) <= 65536
    if not HAS_GPU or is_small_window:
        return _calculate_anisotropic_glam_features_cpu(image_3d, num_levels, cutoff_radius)

    anisotropic_features = {}
    coords = [np.argwhere(image_3d == i) for i in range(num_levels)]
    coords_gpu = [cp.array(c, dtype=cp.float32) if len(c) > 0 else None for c in coords]
    cutoff_sq = float(cutoff_radius) ** 2

    for alpha in range(num_levels):
        for beta in range(num_levels):
            if coords_gpu[alpha] is None or coords_gpu[beta] is None or len(coords_gpu[beta]) < 10:
                anisotropic_features[f'GLAM_Anisotropy_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue1_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue2_{alpha}_{beta}'] = 0.0
                anisotropic_features[f'GLAM_Eigenvalue3_{alpha}_{beta}'] = 0.0
                continue

            num_ref_points = min(len(coords_gpu[alpha]), 50)
            if len(coords_gpu[alpha]) < num_ref_points:
                ref_indices = np.arange(len(coords_gpu[alpha]))
            else:
                ref_indices = np.linspace(0, len(coords_gpu[alpha]) - 1, num_ref_points, dtype=int)

            ref_coords_gpu = coords_gpu[alpha][ref_indices]
            target_coords_gpu = coords_gpu[beta]

            vectors = target_coords_gpu[cp.newaxis, :, :] - ref_coords_gpu[:, cp.newaxis, :]
            dist_sq = cp.sum(vectors ** 2, axis=2)
            mask = dist_sq < cutoff_sq
            valid_counts = cp.sum(mask, axis=1)
            valid_refs = valid_counts >= 3

            if cp.any(valid_refs):
                v_counts = valid_counts[valid_refs]
                masked_vectors = vectors[valid_refs] * mask[valid_refs, :, cp.newaxis]
                means = cp.sum(masked_vectors, axis=1) / v_counts[:, cp.newaxis]
                centered = masked_vectors - means[:, cp.newaxis, :]
                centered *= mask[valid_refs, :, cp.newaxis]
                cov_tensors = cp.matmul(centered.transpose(0, 2, 1), centered) / (v_counts[:, cp.newaxis, cp.newaxis] - 1)
                avg_gyration_cpu = cp.mean(cov_tensors, axis=0).get()

                if np.any(np.isnan(avg_gyration_cpu)) or np.any(np.isinf(avg_gyration_cpu)):
                    l1, l2, l3, anisotropy = 0.0, 0.0, 0.0, 0.0
                else:
                    try:
                        eigenvalues, _ = np.linalg.eigh(avg_gyration_cpu)
                        eigenvalues = np.sort(eigenvalues)[::-1]
                        l1, l2, l3 = eigenvalues
                        denom = (l1 + l2 + l3) ** 2
                        anisotropy = 1.0 - 3.0 * (l1 * l2 + l2 * l3 + l3 * l1) / denom if denom > 1e-9 else 0.0
                    except np.linalg.LinAlgError:
                        l1, l2, l3, anisotropy = 0.0, 0.0, 0.0, 0.0
            else:
                l1, l2, l3, anisotropy = 0.0, 0.0, 0.0, 0.0

            anisotropic_features[f'GLAM_Anisotropy_{alpha}_{beta}'] = float(anisotropy)
            anisotropic_features[f'GLAM_Eigenvalue1_{alpha}_{beta}'] = float(l1)
            anisotropic_features[f'GLAM_Eigenvalue2_{alpha}_{beta}'] = float(l2)
            anisotropic_features[f'GLAM_Eigenvalue3_{alpha}_{beta}'] = float(l3)

    if HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()

    return anisotropic_features

def _calculate_anisotropic_glam_features_cpu(image_3d, num_levels, cutoff_radius):
    """Ultra-fast exact BLAS moment solver for anisotropic GLAM features (gyration tensor).
    Uses translation invariance of sample covariance and `np.add.reduceat` across reference
    states to eliminate inner Python loops while preserving exact NaN semantics.
    """
    anisotropic_features = {}
    is_local_window = int(np.prod(image_3d.shape)) <= 65536
    absent_default = np.nan if is_local_window else 0.0

    roi_mask = image_3d >= 0
    if not np.any(roi_mask):
        for a in range(num_levels):
            for b in range(num_levels):
                anisotropic_features[f'GLAM_Anisotropy_{a}_{b}'] = absent_default
                anisotropic_features[f'GLAM_Eigenvalue1_{a}_{b}'] = absent_default
                anisotropic_features[f'GLAM_Eigenvalue2_{a}_{b}'] = absent_default
                anisotropic_features[f'GLAM_Eigenvalue3_{a}_{b}'] = absent_default
        return anisotropic_features

    all_coords = np.argwhere(roi_mask).astype(np.float64)
    all_labels = image_3d[roi_mask].astype(np.int32)
    counts = np.bincount(all_labels, minlength=num_levels)
    cutoff_sq = float(cutoff_radius) ** 2
    min_target_pts = 3 if is_local_window else 10

    aniso_mat = np.full((num_levels, num_levels), absent_default, dtype=np.float64)
    l1_mat = np.full((num_levels, num_levels), absent_default, dtype=np.float64)
    l2_mat = np.full((num_levels, num_levels), absent_default, dtype=np.float64)
    l3_mat = np.full((num_levels, num_levels), absent_default, dtype=np.float64)

    active_alphas = [a for a in range(num_levels) if counts[a] > 0]
    active_betas = [b for b in range(num_levels) if counts[b] >= min_target_pts]
    if not active_alphas or not active_betas:
        for a in range(num_levels):
            for b in range(num_levels):
                anisotropic_features[f'GLAM_Anisotropy_{a}_{b}'] = float(aniso_mat[a, b])
                anisotropic_features[f'GLAM_Eigenvalue1_{a}_{b}'] = float(l1_mat[a, b])
                anisotropic_features[f'GLAM_Eigenvalue2_{a}_{b}'] = float(l2_mat[a, b])
                anisotropic_features[f'GLAM_Eigenvalue3_{a}_{b}'] = float(l3_mat[a, b])
        return anisotropic_features

    ref_indices_in_all = []
    ref_starts = []
    offset = 0
    for a in active_alphas:
        a_idx = np.flatnonzero(all_labels == a)
        n_a = len(a_idx)
        n_ref = min(n_a, 50)
        sel = a_idx if n_a <= n_ref else a_idx[np.linspace(0, n_a - 1, n_ref, dtype=int)]
        ref_indices_in_all.append(sel)
        ref_starts.append(offset)
        offset += n_ref

    ref_idx_vec = np.concatenate(ref_indices_in_all)
    ref_starts = np.asarray(ref_starts, dtype=np.intp)
    ref_pts = all_coords[ref_idx_vec]

    beta_arr = np.asarray(active_betas, dtype=np.int32)
    beta_mask = np.isin(all_labels, beta_arr)
    tgt_pts_unsorted = all_coords[beta_mask]
    tgt_lbl_unsorted = all_labels[beta_mask]
    order_b = np.argsort(tgt_lbl_unsorted, kind='mergesort')
    tgt_pts = tgt_pts_unsorted[order_b]
    tgt_labels = tgt_lbl_unsorted[order_b]
    tgt_orig_idx = np.flatnonzero(beta_mask)[order_b]

    r_n2 = np.sum(ref_pts * ref_pts, axis=1, keepdims=True)
    t_n2 = np.sum(tgt_pts * tgt_pts, axis=1, keepdims=True).T
    d2 = r_n2 + t_n2 - 2.0 * (ref_pts @ tgt_pts.T)
    adj_all = d2 < (cutoff_sq - 1e-12)
    adj_all &= ~(ref_idx_vec[:, None] == tgt_orig_idx[None, :])

    z, y, x = tgt_pts[:, 0], tgt_pts[:, 1], tgt_pts[:, 2]
    moments_tgt = np.column_stack((
        np.ones(len(tgt_pts), dtype=np.float64),
        z, y, x,
        z * z, y * y, x * x,
        z * y, z * x, y * x,
    ))

    b_changes = np.flatnonzero(np.diff(tgt_labels) != 0) + 1
    b_starts = np.concatenate(([0], b_changes))
    b_ends = np.concatenate((b_changes, [len(tgt_labels)]))
    n_B = len(active_betas)
    m_ref_tot = len(ref_idx_vec)

    sums_all = np.empty((m_ref_tot, n_B, 10), dtype=np.float64)
    for bi, (bs, be) in enumerate(zip(b_starts, b_ends)):
        sums_all[:, bi, :] = adj_all[:, bs:be].astype(np.float64) @ moments_tgt[bs:be]

    cnt = sums_all[:, :, 0]
    valid_r = cnt >= 3.0
    inv_n = np.where(valid_r, 1.0 / np.maximum(cnt, 1.0), 0.0)
    inv_nm1 = np.where(valid_r, 1.0 / np.maximum(cnt - 1.0, 1.0), 0.0)

    sz, sy, sx = sums_all[:, :, 1], sums_all[:, :, 2], sums_all[:, :, 3]
    cov_6 = np.empty((m_ref_tot, n_B, 6), dtype=np.float64)
    cov_6[:, :, 0] = (sums_all[:, :, 4] - sz * sz * inv_n) * inv_nm1
    cov_6[:, :, 1] = (sums_all[:, :, 5] - sy * sy * inv_n) * inv_nm1
    cov_6[:, :, 2] = (sums_all[:, :, 6] - sx * sx * inv_n) * inv_nm1
    cov_6[:, :, 3] = (sums_all[:, :, 7] - sz * sy * inv_n) * inv_nm1
    cov_6[:, :, 4] = (sums_all[:, :, 8] - sz * sx * inv_n) * inv_nm1
    cov_6[:, :, 5] = (sums_all[:, :, 9] - sy * sx * inv_n) * inv_nm1
    cov_6[~valid_r] = 0.0

    valid_cnt_ab = np.add.reduceat(valid_r.astype(np.int32), ref_starts, axis=0)
    cov_sum_ab = np.add.reduceat(cov_6, ref_starts, axis=0)

    active_ab = valid_cnt_ab > 0
    if np.any(active_ab):
        ai_idx, bi_idx = np.nonzero(active_ab)
        mean_cov = cov_sum_ab[ai_idx, bi_idx] / valid_cnt_ab[ai_idx, bi_idx, None]
        finite_row = np.all(np.isfinite(mean_cov), axis=1)
        ai_idx = ai_idx[finite_row]
        bi_idx = bi_idx[finite_row]
        mean_cov = mean_cov[finite_row]

        if len(mean_cov):
            K = len(mean_cov)
            gyr_stack = np.empty((K, 3, 3), dtype=np.float64)
            gyr_stack[:, 0, 0] = mean_cov[:, 0]
            gyr_stack[:, 1, 1] = mean_cov[:, 1]
            gyr_stack[:, 2, 2] = mean_cov[:, 2]
            gyr_stack[:, 0, 1] = gyr_stack[:, 1, 0] = mean_cov[:, 3]
            gyr_stack[:, 0, 2] = gyr_stack[:, 2, 0] = mean_cov[:, 4]
            gyr_stack[:, 1, 2] = gyr_stack[:, 2, 1] = mean_cov[:, 5]

            try:
                eig_all = np.linalg.eigvalsh(gyr_stack)[:, ::-1]
            except np.linalg.LinAlgError:
                eig_all = np.full((K, 3), absent_default, dtype=np.float64)
                for idx_g, g_m in enumerate(gyr_stack):
                    try:
                        eig_all[idx_g] = np.linalg.eigvalsh(g_m)[::-1]
                    except np.linalg.LinAlgError:
                        pass

            l1_v, l2_v, l3_v = eig_all[:, 0], eig_all[:, 1], eig_all[:, 2]
            denom_v = (l1_v + l2_v + l3_v) ** 2
            aniso_v = np.where(
                denom_v > 1e-9,
                1.0 - 3.0 * (l1_v * l2_v + l2_v * l3_v + l3_v * l1_v) / np.maximum(denom_v, 1e-300),
                absent_default,
            )
            if is_local_window:
                invalid_tensor = (~np.all(np.isfinite(eig_all), axis=1)) | (denom_v <= 1e-9)
                aniso_v[invalid_tensor] = np.nan
                l1_v[invalid_tensor] = np.nan
                l2_v[invalid_tensor] = np.nan
                l3_v[invalid_tensor] = np.nan

            a_orig = np.asarray(active_alphas, dtype=np.intp)[ai_idx]
            b_orig = np.asarray(active_betas, dtype=np.intp)[bi_idx]
            aniso_mat[a_orig, b_orig] = aniso_v
            l1_mat[a_orig, b_orig] = l1_v
            l2_mat[a_orig, b_orig] = l2_v
            l3_mat[a_orig, b_orig] = l3_v

    for a in range(num_levels):
        for b in range(num_levels):
            anisotropic_features[f'GLAM_Anisotropy_{a}_{b}'] = float(aniso_mat[a, b])
            anisotropic_features[f'GLAM_Eigenvalue1_{a}_{b}'] = float(l1_mat[a, b])
            anisotropic_features[f'GLAM_Eigenvalue2_{a}_{b}'] = float(l2_mat[a, b])
            anisotropic_features[f'GLAM_Eigenvalue3_{a}_{b}'] = float(l3_mat[a, b])

    return anisotropic_features




def calculate_glam_fractal_dimension(image_3d, num_levels):
    """
    Calculates Volume and Interface Fractal Dimensions.
    Uses GPU acceleration for large global volumes and fast batched 3D integral
    box-counting for small local mapping windows without per-voxel console spam.
    """
    is_small_window = int(np.prod(image_3d.shape)) <= 65536
    if not HAS_GPU or is_small_window:
        return _calculate_glam_fractal_dimension_cpu(image_3d, num_levels, verbose=not is_small_window)

    import cupyx.scipy.ndimage as cupy_ndimage
    print("  - Starting Fractal Dimension analysis (GPU Accelerated)...")
    fractal_dimensions = {}

    image_gpu = cp.array(image_3d, dtype=cp.int16)
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}

    def boxcount_gpu(binary_mask_gpu):
        if not cp.any(binary_mask_gpu):
            return 0.0
        p = binary_mask_gpu.shape
        scales = np.logspace(np.log2(2), np.log2(min(p)), num=8, base=2.0, dtype=np.int32)
        scales = np.unique(scales[scales > 1])
        if len(scales) < 2:
            return 0.0
        counts = []
        for s in scales:
            s = int(s)
            filtered = cupy_ndimage.maximum_filter(binary_mask_gpu, size=s, mode='constant')
            count = cp.sum(filtered[::s, ::s, ::s])
            counts.append(float(count))
        counts = np.array(counts)
        valid_indices = counts > 0
        if np.sum(valid_indices) < 2:
            return 0.0
        scales_fit = scales[valid_indices]
        counts_fit = counts[valid_indices]
        try:
            with np.errstate(divide='ignore'):
                coeffs = np.polyfit(np.log(scales_fit), np.log(counts_fit), 1)
            return -coeffs[0]
        except np.linalg.LinAlgError:
            return 0.0

    for i in range(num_levels):
        fractal_dimensions[f'GLAM_VolumeFD_{i}'] = boxcount_gpu(masks_gpu[i])

    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
            continue
        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])
        for j in range(num_levels):
            if i == j:
                continue
            if not cp.any(masks_gpu[j]):
                fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
                continue
            interface_gpu = dilated_i_gpu & masks_gpu[j]
            fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = boxcount_gpu(interface_gpu)

    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return fractal_dimensions


def _calculate_glam_fractal_dimension_cpu(image_3d, num_levels, verbose=True):
    """Bitpacked 3D box-counting for local mapping windows (<= 32 gray levels)
    and standard fallback for global volumes.
    """
    if verbose:
        print("  - Starting Fractal Dimension analysis (CPU)...")
    fractal_dimensions = {}
    p = image_3d.shape
    min_dim = min(p)
    scales = np.logspace(np.log2(2), np.log2(max(min_dim, 2)), num=8, base=2.0, dtype=np.int32)
    scales = np.unique(scales[scales > 1])

    if len(scales) < 2:
        for i in range(num_levels):
            fractal_dimensions[f'GLAM_VolumeFD_{i}'] = 0.0
            for j in range(num_levels):
                if i != j:
                    fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
        return fractal_dimensions

    # Fast bitpacked box-counting when num_levels <= 32 and window <= 65536 voxels:
    # Pack all num_levels dilated masks into a single uint32 3D volume D_bits where
    # bit i is 1 iff dilated[i][z, y, x] is True.
    if int(np.prod(p)) <= 65536 and num_levels <= 32:
        dz, dy, dx = p
        roi_vals = image_3d[image_3d >= 0]
        if not roi_vals.size:
            for i in range(num_levels):
                fractal_dimensions[f'GLAM_VolumeFD_{i}'] = 0.0
                for j in range(num_levels):
                    if i != j:
                        fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
            return fractal_dimensions

        # Build single uint32 bitmask of labels: (1 << label) at each in-mask voxel
        label_bits = np.where(
            image_3d >= 0,
            np.left_shift(np.uint32(1), np.clip(image_3d, 0, 31).astype(np.uint32)),
            np.uint32(0),
        )
        # 6-connected dilation of all gray levels simultaneously in uint32!
        d_bits = label_bits.copy()
        d_bits[1:, :, :] |= label_bits[:-1, :, :]
        d_bits[:-1, :, :] |= label_bits[1:, :, :]
        d_bits[:, 1:, :] |= label_bits[:, :-1, :]
        d_bits[:, :-1, :] |= label_bits[:, 1:, :]
        d_bits[:, :, 1:] |= label_bits[:, :, :-1]
        d_bits[:, :, :-1] |= label_bits[:, :, 1:]

        present_j = np.flatnonzero(np.bincount(roi_vals.astype(int), minlength=num_levels) > 0)
        n_pres = len(present_j)
        # W_bits[idx_j, z, y, x] holds d_bits where image_3d == j, else 0
        w_bits = np.where(
            image_3d[None, :, :, :] == present_j[:, None, None, None],
            d_bits[None, :, :, :],
            np.uint32(0),
        )

        bit_shifts = np.arange(num_levels, dtype=np.uint32)[:, None]
        # counts_mat has shape (num_levels, n_pres, len(scales)) where [i, idx_j, s]
        # is the box count for (VolumeFD_j when i == j, InterfaceFD_i_j when i != j)
        counts_mat = np.zeros((num_levels, n_pres, len(scales)), dtype=np.float64)

        for s_idx, s in enumerate(scales):
            s = int(s)
            half = s // 2
            z0 = np.clip(np.arange(0, dz, s) - s + 1 + half, 0, dz)
            z_end = int(np.clip((len(z0) - 1) * s + half + 1, 0, dz))
            y0 = np.clip(np.arange(0, dy, s) - s + 1 + half, 0, dy)
            y_end = int(np.clip((len(y0) - 1) * s + half + 1, 0, dy))
            x0 = np.clip(np.arange(0, dx, s) - s + 1 + half, 0, dx)
            x_end = int(np.clip((len(x0) - 1) * s + half + 1, 0, dx))

            sub = w_bits[:, :z_end, :y_end, :x_end]
            rz = np.bitwise_or.reduceat(sub, z0, axis=1)
            ry = np.bitwise_or.reduceat(rz, y0, axis=2)
            box_or = np.bitwise_or.reduceat(ry, x0, axis=3).reshape(n_pres, -1)  # (n_pres, N_boxes)

            # Unpack bits 0..num_levels-1 and count active boxes per (i, idx_j)
            active_bits = ((box_or[None, :, :] >> bit_shifts[:, None, :]) & np.uint32(1)) != 0
            counts_mat[:, :, s_idx] = np.sum(active_bits, axis=2)

        counts_flat = counts_mat.reshape(num_levels * n_pres, len(scales))
        log_scales = np.log(scales.astype(np.float64))
        valid = counts_flat > 0
        n_valid = np.sum(valid, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_counts = np.where(valid, np.log(np.maximum(counts_flat, 1.0)), 0.0)

        x_val = np.where(valid, log_scales[None, :], 0.0)
        x_mean = np.sum(x_val, axis=1, keepdims=True) / np.maximum(n_valid[:, None], 1)
        y_mean = np.sum(log_counts, axis=1, keepdims=True) / np.maximum(n_valid[:, None], 1)
        dx_mat = np.where(valid, log_scales[None, :] - x_mean, 0.0)
        dy_mat = np.where(valid, log_counts - y_mean, 0.0)
        denom = np.sum(dx_mat * dx_mat, axis=1)
        slopes = np.where(
            (n_valid >= 2) & (denom > 1e-12),
            -np.sum(dx_mat * dy_mat, axis=1) / np.maximum(denom, 1e-300),
            0.0,
        ).reshape(num_levels, n_pres)

        # Initialize absent gray levels to np.nan; present pairs with no contact stay 0.0
        for i in range(num_levels):
            fractal_dimensions[f'GLAM_VolumeFD_{i}'] = np.nan
            for j in range(num_levels):
                if i != j:
                    fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = np.nan

        for idx_j, j in enumerate(present_j):
            j = int(j)
            fractal_dimensions[f'GLAM_VolumeFD_{j}'] = float(slopes[j, idx_j])
            for idx_i, i in enumerate(present_j):
                i = int(i)
                if i != j:
                    fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = float(slopes[i, idx_j])
        return fractal_dimensions

    # Standard fallback for large global volumes on CPU
    def boxcount(binary_image):
        if not np.any(binary_image):
            return 0.0
        counts = np.array([np.sum(ndimage.maximum_filter(binary_image, size=s, mode='constant')[::s, ::s, ::s]) for s in scales])
        valid_indices = counts > 0
        if np.sum(valid_indices) < 2:
            return 0.0
        scales_fit = scales[valid_indices]
        counts_fit = counts[valid_indices]
        try:
            with np.errstate(divide='ignore'):
                coeffs = np.polyfit(np.log(scales_fit), np.log(counts_fit), 1)
            return -coeffs[0]
        except np.linalg.LinAlgError:
            return 0.0

    for i in range(num_levels):
        binary_image = (image_3d == i)
        fractal_dimensions[f'GLAM_VolumeFD_{i}'] = boxcount(binary_image)

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j:
                continue
            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            if not np.any(mask_i) or not np.any(mask_j):
                fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = 0.0
                continue
            interface = ndimage.binary_dilation(mask_i) & mask_j
            fractal_dimensions[f'GLAM_InterfaceFD_{i}_{j}'] = boxcount(interface)

    return fractal_dimensions


def calculate_glam_lacunarity(image_3d, num_levels):
    """
    Calculates Volume and Interface Lacunarity.
    GPU Accelerated using CuPy NDImage for ultrafast 3D convolutions and dilations.
    """
    if not HAS_GPU:
        return _calculate_glam_lacunarity_cpu(image_3d, num_levels)

    # Import the GPU-accelerated equivalent of scipy.ndimage
    import cupyx.scipy.ndimage as cupy_ndimage

    print("  - Starting Lacunarity analysis (GPU Accelerated)...")
    lacunarity_features = {}
    box_sizes = [2, 3, 4, 5]
    
    # 1. Move the entire image to the GPU once
    image_gpu = cp.array(image_3d, dtype=cp.int16)

    def get_lacunarity_for_mask_gpu(binary_mask_gpu):
        if cp.sum(binary_mask_gpu) < 10: return 1.0
        
        float_mask_gpu = binary_mask_gpu.astype(cp.float32)
        lac_values = []
        
        for r in box_sizes:
            kernel_gpu = cp.ones((r, r, r), dtype=cp.float32)
            try:
                # Execute 3D Convolution natively on GPU
                mass_map_gpu = cupy_ndimage.convolve(float_mask_gpu, kernel_gpu, mode='constant', cval=0.0)
                
                valid_mask_gpu = mass_map_gpu > 0
                valid_masses_gpu = mass_map_gpu[valid_mask_gpu]
                
                if valid_masses_gpu.size < 2: continue
                
                # Calculate mean and variance simultaneously on GPU
                mean_mass = cp.mean(valid_masses_gpu)
                var_mass = cp.var(valid_masses_gpu)
                
                if mean_mass == 0: continue
                
                # Pull just the final scalar back to the CPU
                lambda_r = float((var_mass / (mean_mass**2)) + 1)
                lac_values.append(lambda_r)
            except Exception:
                continue
                
        if not lac_values: return 1.0
        return float(np.mean(lac_values))

    # 2. Pre-calculate all individual gray-level masks on the GPU
    # This saves us from evaluating (image == i) repeatedly in the nested loops
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}

    # A. Diagonal: Volume Lacunarity
    for i in range(num_levels):
        lacunarity_features[f'GLAM_VolumeLacunarity_{i}'] = get_lacunarity_for_mask_gpu(masks_gpu[i])

    # B. Off-Diagonal: Interface Lacunarity
    for i in range(num_levels):
        # Only dilate if the base mask actually has points
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = 1.0
            continue
            
        # Execute 3D Binary Dilation on the GPU
        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])
        
        for j in range(num_levels):
            if i == j: continue
            
            if not cp.any(masks_gpu[j]):
                lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = 1.0
                continue
            
            # Fast bitwise AND on the GPU
            interface_gpu = dilated_i_gpu & masks_gpu[j]
            lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = get_lacunarity_for_mask_gpu(interface_gpu)

    # Free GPU Memory explicitly
    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return lacunarity_features

def _calculate_glam_lacunarity_cpu(image_3d, num_levels):
    """
    Fallback CPU function for Volume and Interface Lacunarity.
    """
    print("  - Starting Lacunarity analysis (CPU)...")
    lacunarity_features = {}
    box_sizes = [2, 3, 4, 5]
    
    def get_lacunarity_for_mask(binary_mask):
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
                
                lambda_r = (var_mass / (mean_mass**2)) + 1
                lac_values.append(lambda_r)
                
            except Exception:
                continue

        if not lac_values: return 1.0
        return np.mean(lac_values)

    for i in range(num_levels):
        binary_image = (image_3d == i)
        lacunarity_features[f'GLAM_VolumeLacunarity_{i}'] = get_lacunarity_for_mask(binary_image)

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue

            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            if not np.any(mask_i) or not np.any(mask_j): 
                lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = 1.0
                continue
            
            interface = ndimage.binary_dilation(mask_i) & mask_j
            lacunarity_features[f'GLAM_InterfaceLacunarity_{i}_{j}'] = get_lacunarity_for_mask(interface)

    return lacunarity_features


# def calculate_glam_topology(image_3d, num_levels):
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

def calculate_glam_topology(image_3d, num_levels):
    """
    Calculates Full Topological Metrics (Betti Numbers).
    Uses GPU Connected Component Labeling for massive speedups.
    """
    if not HAS_GPU: return _calculate_glam_topology_cpu(image_3d, num_levels)
    import cupyx.scipy.ndimage as cupy_ndimage
    from skimage import measure 
    
    print("  - Starting Topology analysis (GPU Accelerated)...")
    topology_features = {}
    image_gpu = cp.array(image_3d, dtype=cp.int16)
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}
    
    def get_betti_numbers_gpu(binary_mask_gpu):
        padded_gpu = cp.pad(binary_mask_gpu, pad_width=1, mode='constant', constant_values=0)
        
        # Betti-0 (Objects)
        structure_3d = cp.ones((3,3,3), dtype=cp.int32)
        _, n_objects = cupy_ndimage.label(padded_gpu, structure=structure_3d)
        b0 = float(n_objects)
        
        # Betti-2 (Voids)
        background_gpu = (padded_gpu == 0)
        structure_bg = cupy_ndimage.generate_binary_structure(3, 1)
        _, n_bg_components = cupy_ndimage.label(background_gpu, structure=structure_bg)
        b2 = max(0.0, float(n_bg_components) - 1.0)
        
        # Euler (Skimage CPU)
        padded_cpu = padded_gpu.get()
        chi = measure.euler_number(padded_cpu, connectivity=3)
        b1 = max(0.0, b0 + b2 - chi)
        return b0, b1, b2, float(chi)

    # 1. Volume Topology
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
        else:
            b0, b1, b2, chi = get_betti_numbers_gpu(masks_gpu[i])
            
        topology_features[f'GLAM_VolumeBetti0_{i}'] = b0
        topology_features[f'GLAM_VolumeBetti1_{i}'] = b1
        topology_features[f'GLAM_VolumeBetti2_{i}'] = b2
        topology_features[f'GLAM_VolumeEuler_{i}'] = chi

    # 2. Interface Topology
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    topology_features[f'GLAM_InterfaceBetti0_{i}_{j}'] = 0.0
                    topology_features[f'GLAM_InterfaceBetti1_{i}_{j}'] = 0.0
                    topology_features[f'GLAM_InterfaceBetti2_{i}_{j}'] = 0.0
                    topology_features[f'GLAM_InterfaceEuler_{i}_{j}'] = 0.0
            continue
            
        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])
        for j in range(num_levels):
            if i == j: continue
            if not cp.any(masks_gpu[j]):
                b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
            else:
                interface_gpu = dilated_i_gpu & masks_gpu[j]
                if not cp.any(interface_gpu):
                    b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
                else:
                    b0, b1, b2, chi = get_betti_numbers_gpu(interface_gpu)
            
            topology_features[f'GLAM_InterfaceBetti0_{i}_{j}'] = b0
            topology_features[f'GLAM_InterfaceBetti1_{i}_{j}'] = b1
            topology_features[f'GLAM_InterfaceBetti2_{i}_{j}'] = b2
            topology_features[f'GLAM_InterfaceEuler_{i}_{j}'] = chi
            
    # Cleanup
    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return topology_features

def _calculate_glam_topology_cpu(image_3d, num_levels):
    """Fallback CPU function for Topology."""
    topology_features = {}
    def get_betti_numbers(binary_mask):
        padded = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        labeled_obj, n_objects = ndimage.label(padded, structure=np.ones((3,3,3)))
        b0 = float(n_objects)
        background = (padded == 0)
        structure_bg = ndimage.generate_binary_structure(3, 1)
        labeled_bg, n_bg_components = ndimage.label(background, structure=structure_bg)
        b2 = max(0.0, float(n_bg_components) - 1.0)
        chi = measure.euler_number(padded, connectivity=3)
        b1 = max(0.0, b0 + b2 - chi)
        return b0, b1, b2, float(chi)

    for i in range(num_levels):
        binary_image = (image_3d == i)
        if np.sum(binary_image) == 0: b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
        else: b0, b1, b2, chi = get_betti_numbers(binary_image)
        topology_features[f'GLAM_VolumeBetti0_{i}'] = b0
        topology_features[f'GLAM_VolumeBetti1_{i}'] = b1
        topology_features[f'GLAM_VolumeBetti2_{i}'] = b2
        topology_features[f'GLAM_VolumeEuler_{i}'] = chi

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue
            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            if not np.any(mask_i) or not np.any(mask_j):
                b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
                continue
            interface = ndimage.binary_dilation(mask_i) & mask_j
            if np.sum(interface) == 0: b0, b1, b2, chi = 0.0, 0.0, 0.0, 0.0
            else: b0, b1, b2, chi = get_betti_numbers(interface)
            topology_features[f'GLAM_InterfaceBetti0_{i}_{j}'] = b0
            topology_features[f'GLAM_InterfaceBetti1_{i}_{j}'] = b1
            topology_features[f'GLAM_InterfaceBetti2_{i}_{j}'] = b2
            topology_features[f'GLAM_InterfaceEuler_{i}_{j}'] = chi
    return topology_features


def calculate_glam_potential_energy(rdf_df, num_levels):
    """Calculates the Potential of Mean Force (PMF) Energy (U*) strictly over each pair's
    supported radial domain (eliminating the half-trapezoid step artifact at R_cutoff).
    """
    potential_energies = {}
    if rdf_df.empty:
        return {}
    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r, g_mat, pairs, row_index = extract_fast_raw_rdf(rdf_df, num_levels)
    if not pairs:
        return potential_energies

    cutoff_idx = get_row_cutoff_indices(g_mat, rdf_df)
    energies = np.full(len(pairs), np.nan, dtype=np.float64)

    for L in np.unique(cutoff_idx):
        L = int(L)
        if L < 2:
            continue
        rows = np.flatnonzero(cutoff_idx == L)
        r_sub = r[:L]
        g_sub = g_mat[rows, :L]
        pmf = -np.log(g_sub + 1e-9)
        integrand = pmf * g_sub * (r_sub[None, :] ** 2)
        energies[rows] = 4.0 * np.pi * np.trapezoid(integrand, r_sub, axis=-1)

    for alpha in range(num_levels):
        for beta in range(num_levels):
            idx = row_index.get((alpha, beta))
            val = float(energies[idx]) if idx is not None and np.isfinite(energies[idx]) else np.nan
            potential_energies[f'GLAM_PotentialEnergy_{alpha}_{beta}'] = val
    return potential_energies


def calculate_glam_structural_pressure_index(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the Structural Pressure Index (SPI) strictly inside each pair's supported domain."""
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')

    spi_metrics = {}
    if rdf_df.empty or total_roi_voxels == 0:
        return {}
    from .rdf_smoothing import prepare_smoothed_rdf
    smoothed = prepare_smoothed_rdf(rdf_df, num_levels, savgol_window, savgol_poly)
    r = smoothed.radius
    if r.size < 2:
        return {}

    n_pairs = len(smoothed.pairs)
    integrals = np.zeros(n_pairs, dtype=np.float64)
    if n_pairs:
        g_raw = smoothed.raw
        g_smooth = smoothed.smooth
        cutoff_idx = smoothed.cutoff_idx
        for L in np.unique(cutoff_idx):
            L = int(L)
            if L < 2:
                continue
            rows = np.flatnonzero(cutoff_idx == L)
            r_sub = r[:L]
            raw_sub = g_raw[rows, :L]
            sm_sub = np.maximum(g_smooth[rows, :L], 1e-9)
            pmf_sub = -np.log(sm_sub)
            edge_ord = 2 if L >= 3 else 1
            dW_dr = np.gradient(pmf_sub, r_sub, axis=-1, edge_order=edge_ord)
            integrand = (r_sub[None, :] ** 3) * dW_dr * raw_sub
            integrals[rows] = np.trapezoid(integrand, r_sub, axis=-1)

    rho = np.asarray(level_counts, dtype=np.float64) / float(total_roi_voxels)
    for alpha in range(num_levels):
        rho_alpha = rho[alpha]
        for beta in range(num_levels):
            row_idx = smoothed.row_index.get((alpha, beta))
            if row_idx is not None and smoothed.cutoff_idx[row_idx] >= 2 and rho_alpha > 0 and rho[beta] > 0:
                val = -(rho_alpha * rho[beta] / 6.0) * 4.0 * np.pi * integrals[row_idx]
                spi_metrics[f'GLAM_StructuralPressureIndex_{alpha}_{beta}'] = float(val)
            else:
                spi_metrics[f'GLAM_StructuralPressureIndex_{alpha}_{beta}'] = np.nan
    return spi_metrics


def calculate_configurational_disorder_index(rdf_structured_df, rdf_random_df, num_levels):
    """Calculates the Configurational Disorder Index strictly over the common supported
    radial domain `L = min(cutoff_s, cutoff_r)` using vectorized first-shell detection,
    with positive-shell guards (preventing empty/undershot Savitzky-Golay shells from
    injecting log(1e-9) = -20.72 spikes) and Tikhonov damping when g_s(r) -> g_r(r).
    """
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    peak_prominence = get_config('PeakProminence')

    try:
        max_shell_radius = int(get_config('MaxLocalShellRadius'))
    except Exception:
        max_shell_radius = 30

    config_disorder_indices = {}
    if rdf_structured_df.empty or rdf_random_df.empty:
        return config_disorder_indices
    from .rdf_smoothing import prepare_smoothed_rdf
    _cdi_struct_smoothing = prepare_smoothed_rdf(rdf_structured_df, num_levels, savgol_window, savgol_poly)
    _cdi_rand_smoothing = prepare_smoothed_rdf(rdf_random_df, num_levels, savgol_window, savgol_poly)
    if not np.array_equal(_cdi_struct_smoothing.radius, _cdi_rand_smoothing.radius):
        raise ValueError('Structured and baseline RDF radius grids differ')

    r = _cdi_struct_smoothing.radius
    n_r = r.size
    pairs = _cdi_struct_smoothing.pairs
    n_pairs = len(pairs)
    if not n_pairs:
        return config_disorder_indices

    struct_raw = _cdi_struct_smoothing.raw
    struct_smooth = _cdi_struct_smoothing.smooth
    cutoff_s = _cdi_struct_smoothing.cutoff_idx

    rand_raw = _cdi_rand_smoothing.raw
    rand_smooth = _cdi_rand_smoothing.smooth
    cutoff_r = _cdi_rand_smoothing.cutoff_idx

    if _cdi_rand_smoothing.pairs != pairs:
        r_map = _cdi_rand_smoothing.row_index
        rand_raw_aligned = np.full_like(struct_raw, np.nan)
        rand_smooth_aligned = np.full_like(struct_smooth, np.nan)
        cutoff_r_aligned = np.zeros(n_pairs, dtype=np.int32)
        valid_pair_row = np.zeros(n_pairs, dtype=bool)
        for i, p in enumerate(pairs):
            if p in r_map:
                ri = r_map[p]
                rand_raw_aligned[i] = rand_raw[ri]
                rand_smooth_aligned[i] = rand_smooth[ri]
                cutoff_r_aligned[i] = cutoff_r[ri]
                valid_pair_row[i] = True
        rand_raw = rand_raw_aligned
        rand_smooth = rand_smooth_aligned
        cutoff_r = cutoff_r_aligned
    else:
        valid_pair_row = np.ones(n_pairs, dtype=bool)

    common_cutoff = np.minimum(cutoff_s, cutoff_r)
    col_grid = np.arange(n_r, dtype=np.int32)[None, :]
    in_common = col_grid < common_cutoff[:, None]
    has_both_pos = (
        valid_pair_row
        & (common_cutoff >= 2)
        & np.any(np.where(in_common, struct_raw, 0.0) > 1e-6, axis=1)
        & np.any(np.where(in_common, rand_raw, 0.0) > 1e-6, axis=1)
    )

    r_min_indices, row_active = _find_first_shell_minima_cutoff_safe(
        struct_smooth, common_cutoff, has_both_pos, max_shell_radius, peak_prominence
    )
    r_min_indices = np.where(row_active & (r_min_indices == 0), 1, r_min_indices)

    # Require both raw and smoothed curves to be strictly positive inside the first shell
    # so empty shells (g=0) or Savitzky-Golay negative undershoots never inject log(1e-9) = -20.72
    pos_shell = (
        (struct_raw > 1e-6)
        & (rand_raw > 1e-6)
        & (struct_smooth > 1e-3)
        & (rand_smooth > 1e-3)
    )
    in_shell_mask = row_active[:, None] & (col_grid < r_min_indices[:, None]) & in_common & pos_shell
    has_valid_shell = np.any(in_shell_mask, axis=1)
    row_active &= has_valid_shell

    log_struct = np.where(in_shell_mask, np.log(np.maximum(struct_smooth, 1e-3)), 0.0)
    log_rand = np.where(in_shell_mask, np.log(np.maximum(rand_smooth, 1e-3)), 0.0)
    log_diff = log_struct - log_rand

    # Tikhonov-damped reciprocal `log_diff / (log_diff^2 + 1e-2)` prevents 1/0 explosion
    # when g_struct(r) is nearly identical to g_rand(r)
    damped_inv_diff = log_diff / (log_diff * log_diff + 1e-2)
    disorder_mat = log_struct * damped_inv_diff

    in_shell = in_shell_mask & np.isfinite(disorder_mat)
    safe_disorder = np.where(in_shell, disorder_mat, 0.0)
    weights = np.where(in_shell, np.abs(log_diff), 0.0)
    weighted_sum = np.sum(safe_disorder * weights, axis=1)
    weight_totals = np.sum(weights, axis=1)
    cdi_values = np.where(
        row_active,
        np.where(weight_totals > 1e-6, weighted_sum / np.maximum(weight_totals, 1e-300), 0.0),
        np.nan,
    )

    row_lookup = _cdi_struct_smoothing.row_index
    for alpha in range(num_levels):
        for beta in range(num_levels):
            row_i = row_lookup.get((alpha, beta))
            val = float(cdi_values[row_i]) if row_i is not None else np.nan
            config_disorder_indices[f'GLAM_ConfigurationalDisorderIndex_{alpha}_{beta}'] = val

    return config_disorder_indices




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
    """
    Calculates the average alignment of local directors.
    GPU Accelerated using batched CuPy tensor operations and eigendecomposition.
    """
    if not HAS_GPU:
        return _calculate_local_nematic_alignment_cpu(image_array, mask_array, cutoff_radius)

    print("  - Starting Local Nematic Alignment analysis (GPU)...")
    roi_coords = np.argwhere(mask_array > 0)
    if len(roi_coords) < 50: return {}

    # Calculate Gradients on CPU (Fast enough via NumPy)
    grad = np.array(np.gradient(image_array.astype(float)))
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    
    num_ref_points = min(len(roi_coords), 200)
    ref_indices = np.linspace(0, len(roi_coords) - 1, num_ref_points, dtype=int)
    
    # Move to GPU
    coords_gpu = cp.array(roi_coords, dtype=cp.float32)
    vectors_gpu = cp.array(grad_vectors, dtype=cp.float32)
    
    # Pre-calculate norms and unit vectors on GPU
    norms_gpu = cp.linalg.norm(vectors_gpu, axis=1)
    valid_mask = norms_gpu > 1e-6
    
    unit_vectors_gpu = cp.zeros_like(vectors_gpu)
    unit_vectors_gpu[valid_mask] = vectors_gpu[valid_mask] / norms_gpu[valid_mask, cp.newaxis]
    
    cutoff_sq = cutoff_radius ** 2
    I3 = cp.eye(3, dtype=cp.float32) / 3.0
    
    valid_q_tensors = []
    valid_ref_coords = []
    
    # Loop over the sampled reference points
    for i in ref_indices:
        ref_point = coords_gpu[i]
        
        # Broadcast Euclidean squared distance
        dist_sq = cp.sum((coords_gpu - ref_point)**2, axis=1)
        neighbor_mask = (dist_sq < cutoff_sq) & valid_mask
        
        if cp.sum(neighbor_mask) < 10: 
            continue
            
        local_unit_vecs = unit_vectors_gpu[neighbor_mask]
        
        # Vectorized outer product: (N, 3, 1) * (N, 1, 3) -> (N, 3, 3)
        outer_prods = local_unit_vecs[:, :, cp.newaxis] * local_unit_vecs[:, cp.newaxis, :]
        Q_local = cp.mean(outer_prods, axis=0) - I3
        
        valid_q_tensors.append(Q_local)
        valid_ref_coords.append(roi_coords[i]) # Keep original CPU coordinate for KDTree
        
    if len(valid_q_tensors) < 2:
        return {}
        
    # --- BATCHED EIGENDECOMPOSITION ---
    # Stack Q-tensors into a single array of shape (Batch, 3, 3)
    Q_batch = cp.stack(valid_q_tensors)
    
    # Solve eigenvalues and eigenvectors for ALL points simultaneously
    eigenvalues, eigenvectors = cp.linalg.eigh(Q_batch)
    
    # eigh returns ascending eigenvalues. The principal eigenvector (max eigenvalue)
    # is the 3rd column (index 2) for all batches.
    principal_eigenvectors = eigenvectors[:, :, 2].get()
    director_coords = np.array(valid_ref_coords)
    
    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    # --- CPU KD-Tree for Final Alignment ---
    # cKDTree is virtually instantaneous for < 200 points
    director_tree = cKDTree(director_coords)
    
    alignments = []
    for i, coord in enumerate(director_coords):
        # Find the 2 nearest neighbors (k=2 returns the point itself + 1 nearest neighbor)
        dist, idx = director_tree.query(coord, k=2)
        if len(idx) > 1:
            alignments.append(np.dot(principal_eigenvectors[i], principal_eigenvectors[idx[1]])**2)

    if not alignments: return {}
    print("  - Local Nematic Alignment analysis complete.")
    return {'GLAM.LocalNematic.Alignment': np.mean(alignments)}

def _calculate_local_nematic_alignment_cpu(image_array, mask_array, cutoff_radius):
    """
    Fallback CPU function for local nematic alignment.
    """
    print("  - Starting Local Nematic Alignment analysis (CPU)...")
    roi_coords = np.argwhere(mask_array > 0)
    if len(roi_coords) < 50: return {}

    grad = np.array(np.gradient(image_array.astype(float)))
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    tree = cKDTree(roi_coords)
    num_ref_points = min(len(roi_coords), 200)
    ref_indices = np.linspace(0, len(roi_coords) - 1, num_ref_points, dtype=int)
    
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
    """Calculates Stress analogues using GPU Accelerated 3D Laplacian filters."""
    if not HAS_GPU:
        return _calculate_stress_features_cpu(image_array, mask_array)
        
    import cupyx.scipy.ndimage as cupy_ndimage
    
    print("  - Starting Stress analysis (GPU Accelerated)...")
    stress_features = {}
    if np.sum(mask_array) < 10: return {}
    
    # Move raw float data to GPU
    image_gpu = cp.array(image_array, dtype=cp.float32)
    mask_gpu = cp.array(mask_array) > 0
    
    # Calculate massive 3D Laplacian instantly on GPU
    laplacian_gpu = cupy_ndimage.laplace(image_gpu)
    
    roi_laplacian = laplacian_gpu[mask_gpu]
    stress_features['GLAM.Stress.VolumetricLaplacianMean'] = float(cp.mean(cp.abs(roi_laplacian)))
    stress_features['GLAM.Stress.VolumetricLaplacianVariance'] = float(cp.var(cp.abs(roi_laplacian)))

    dilated_mask = cupy_ndimage.binary_dilation(mask_gpu)
    surface_mask = dilated_mask & ~mask_gpu 
    
    if cp.any(surface_mask): 
        surface_laplacian = laplacian_gpu[surface_mask]
        stress_features['GLAM.Stress.SurfaceLaplacianMean'] = float(cp.mean(cp.abs(surface_laplacian)))
        stress_features['GLAM.Stress.SurfaceLaplacianVariance'] = float(cp.var(cp.abs(surface_laplacian)))
    else:
        stress_features['GLAM.Stress.SurfaceLaplacianMean'] = np.nan
        stress_features['GLAM.Stress.SurfaceLaplacianVariance'] = np.nan
        
    # Clean up massive float arrays
    del image_gpu, mask_gpu, laplacian_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return stress_features

def _calculate_stress_features_cpu(image_array, mask_array):
    """Fallback CPU function for Stress analysis."""
    print("  - Starting Stress analysis (CPU)...")
    stress_features = {}
    if np.sum(mask_array) < 10: return {}
    
    laplacian = ndimage.laplace(image_array.astype(float))
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
        
    return stress_features



def calculate_orientational_correlation_length(image_array, mask_array, max_radius):
    """
    Calculates the Orientational Correlation Length from the g2(r) function.
    Automatically detects CUDA availability and scales memory batching.
    Uses pure cuBLAS matrix math for massive speedups.
    """
    if not HAS_GPU:
        print("  > DEBUG: No GPU detected. Falling back to CPU Orientational Correlation.")
        return _calculate_orientational_correlation_length_cpu(image_array, mask_array, max_radius)

    print("  - Starting Orientational Correlation analysis (GPU Batched)...")
    if np.sum(mask_array) < 50: return {}
    
    # 1. Calculate Gradients on CPU (Fast enough via NumPy)
    grad = np.array(np.gradient(image_array.astype(float)))
    roi_coords = np.argwhere(mask_array > 0)
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    norms = np.linalg.norm(grad_vectors, axis=1)
    
    valid_indices = norms > 1e-6
    if np.sum(valid_indices) < 50: return {}
    
    coords = roi_coords[valid_indices]
    vectors = grad_vectors[valid_indices] / norms[valid_indices, np.newaxis]

    # Sample reference points
    num_ref_points = min(len(coords), 500)
    ref_indices = np.linspace(0, len(coords) - 1, num_ref_points, dtype=int)
    
    # 2. Move active data to GPU
    coords_gpu = cp.array(coords, dtype=cp.float32)
    vectors_gpu = cp.array(vectors, dtype=cp.float32)
    
    ref_coords_gpu = coords_gpu[ref_indices]
    ref_vectors_gpu = vectors_gpu[ref_indices]

    # 3. Dynamic GPU Memory Batching
    dev = cp.cuda.Device(0)
    free_mem, _ = dev.mem_info
    safe_vram_bytes = free_mem * 0.5 
    
    num_targets = coords_gpu.shape[0]
    # We create 3 matrices per batch: dist_sq, dot_prods, valid_mask (4 bytes each)
    bytes_per_ref = num_targets * 12 
    
    batch_size = int(safe_vram_bytes // bytes_per_ref)
    batch_size = max(1, min(batch_size, num_ref_points))

    g2_r_bins = cp.zeros(max_radius, dtype=cp.float64)
    bin_counts = cp.zeros(max_radius, dtype=cp.int64)

    # Pre-calculate target squared norms for Euclidean distances
    B_sq = cp.sum(coords_gpu**2, axis=1)

    for i in range(0, num_ref_points, batch_size):
        batch_coords = ref_coords_gpu[i : i + batch_size]
        batch_vecs = ref_vectors_gpu[i : i + batch_size]
        
        # --- GPU Euclidean Distance ---
        A_sq = cp.sum(batch_coords**2, axis=1, keepdims=True)
        AB = cp.matmul(batch_coords, coords_gpu.T)
        dist_sq = A_sq + B_sq - 2.0 * AB
        dists = cp.sqrt(cp.maximum(dist_sq, 0.0))
        
        # --- GPU Dot Products (Angles) ---
        dot_prods = cp.matmul(batch_vecs, vectors_gpu.T)
        
        # Filter valid neighbors
        valid_mask = (dists > 1e-6) & (dists <= max_radius + 0.99)
        valid_dists = dists[valid_mask]
        valid_dots = dot_prods[valid_mask]
        
        if valid_dists.size > 0:
            # Legendre Polynomial P2(cos theta)
            p2_vals = 0.5 * (3 * valid_dots**2 - 1)
            
            # Bin distances (dist=1 -> bin=0)
            bins = cp.floor(valid_dists).astype(cp.int32) - 1
            
            # Ensure bins are within valid range
            valid_bin_mask = (bins >= 0) & (bins < max_radius)
            bins = bins[valid_bin_mask]
            p2_vals = p2_vals[valid_bin_mask]
            
            if bins.size > 0:
                # Accumulate sums and counts using fast batched bincount
                batch_g2 = cp.bincount(bins, weights=p2_vals, minlength=max_radius)
                batch_counts = cp.bincount(bins, minlength=max_radius)
                
                g2_r_bins += batch_g2[:max_radius]
                bin_counts += batch_counts[:max_radius]

    # Free memory block
    cp.get_default_memory_pool().free_all_blocks()

    # 4. Bring results back to CPU for curve fitting
    g2_r_bins_cpu = g2_r_bins.get()
    bin_counts_cpu = bin_counts.get()

    valid_g2_bins = bin_counts_cpu > 0
    g2_r = np.full(max_radius, np.nan)
    g2_r[valid_g2_bins] = g2_r_bins_cpu[valid_g2_bins] / bin_counts_cpu[valid_g2_bins]
    
    r_vals = np.arange(1, max_radius + 1)
    valid_g2 = ~np.isnan(g2_r)
    
    if np.sum(valid_g2) < 3: return {'GLAM.OrientationalCorrLength': np.nan}
    
    def exp_decay(r, A, xi): return A * np.exp(-r / xi)
    
    try:
        popt, _ = curve_fit(exp_decay, r_vals[valid_g2], g2_r[valid_g2], p0=[g2_r[valid_g2][0], 5.0], maxfev=5000)
        corr_length = popt[1] if popt[1] > 0 else np.nan
    except (RuntimeError, ValueError): 
        corr_length = np.nan
        
    print("  - Orientational Correlation analysis complete.")
    return {'GLAM.OrientationalCorrLength': corr_length}

def _calculate_orientational_correlation_length_cpu(image_array, mask_array, max_radius):
    """
    Fallback CPU function using SciPy cKDTree.
    """
    print("  - Starting Orientational Correlation analysis (CPU)...")
    if np.sum(mask_array) < 50: return {}
    
    grad = np.array(np.gradient(image_array.astype(float)))
    roi_coords = np.argwhere(mask_array > 0)
    grad_vectors = grad[:, roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].T
    norms = np.linalg.norm(grad_vectors, axis=1)
    
    valid_indices = norms > 1e-6
    if np.sum(valid_indices) < 50: return {}
    
    coords, vectors = roi_coords[valid_indices], grad_vectors[valid_indices] / norms[valid_indices, np.newaxis]
    tree = cKDTree(coords)
    num_ref_points = min(len(coords), 500)
    ref_indices = np.linspace(0, len(coords) - 1, num_ref_points, dtype=int)

    g2_r_bins = np.zeros(max_radius, dtype=np.float64)
    bin_counts = np.zeros(max_radius, dtype=np.int64)

    for i in ref_indices:
        ref_point, ref_vector = coords[i], vectors[i]
        neighbor_indices = tree.query_ball_point(ref_point, r=max_radius + 0.99)
        if len(neighbor_indices) < 2: continue
        
        neighbor_coords = coords[neighbor_indices]
        neighbor_vectors = vectors[neighbor_indices]
        
        distances = np.linalg.norm(neighbor_coords - ref_point, axis=1)
        valid_dist = (distances > 1e-6) & (distances <= max_radius + 0.99)
        if not np.any(valid_dist): continue
        
        distances = distances[valid_dist]
        dot_products = np.dot(neighbor_vectors[valid_dist], ref_vector)
        p2_values = 0.5 * (3 * dot_products**2 - 1)
        
        bin_indices = np.floor(distances).astype(int) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < max_radius)
        if not np.any(valid_bins): continue
        
        bin_indices = bin_indices[valid_bins]
        p2_values = p2_values[valid_bins]
        
        np.add.at(g2_r_bins, bin_indices, p2_values)
        np.add.at(bin_counts, bin_indices, 1)
        
    valid_g2_bins = bin_counts > 0
    g2_r = np.full(max_radius, np.nan)
    g2_r[valid_g2_bins] = g2_r_bins[valid_g2_bins] / bin_counts[valid_g2_bins]
    
    r_vals = np.arange(1, max_radius + 1)
    valid_g2 = ~np.isnan(g2_r)
    
    if np.sum(valid_g2) < 3: return {'GLAM.OrientationalCorrLength': np.nan}
    
    def exp_decay(r, A, xi): return A * np.exp(-r / xi)
    try:
        popt, _ = curve_fit(exp_decay, r_vals[valid_g2], g2_r[valid_g2], p0=[g2_r[valid_g2][0], 5.0], maxfev=5000)
        corr_length = popt[1] if popt[1] > 0 else np.nan
    except (RuntimeError, ValueError): 
        corr_length = np.nan
        
    print("  - Orientational Correlation analysis complete.")
    return {'GLAM.OrientationalCorrLength': corr_length}


def calculate_rdf_shape_matrices(rdf_df, num_levels):
    """Batch active RDF statistics over each pair's geometrically supported prefix `:L`
    (including geometrically supported shells with zero observed neighbors).
    """
    import numpy as np
    if rdf_df.empty:
        return {}
    names = ('RDF_PeakPosition', 'RDF_DispersionRatio', 'LogRDF_Median',
             'LogRDF_Variance', 'LogRDF_Skewness', 'LogRDF_Kurtosis',
             'LogRDF_PeakHeight')
    output = {name: np.full((num_levels, num_levels), np.nan) for name in names}
    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r, raw, pairs, _ = extract_fast_raw_rdf(rdf_df, num_levels)
    if not pairs or r.size == 0:
        return output
    cutoff_idx = get_row_cutoff_indices(raw, rdf_df)
    col_grid = np.arange(r.size, dtype=np.int32)[None, :]
    sup_mask = col_grid < cutoff_idx[:, None]
    raw_sup = np.where(sup_mask, raw, 0.0)

    aa, bb = np.asarray(pairs).T
    has_signal = (cutoff_idx > 0) & np.any(raw_sup > 1e-6, axis=1)
    log_values = np.log1p(np.maximum(raw, 0.0))
    if np.any(has_signal):
        raw_for_peak = np.where(sup_mask[has_signal], raw[has_signal], -np.inf)
        log_for_peak = np.where(sup_mask[has_signal], log_values[has_signal], -np.inf)
        output['RDF_PeakPosition'][aa[has_signal], bb[has_signal]] = r[np.argmax(raw_for_peak, axis=1)]
        output['LogRDF_PeakHeight'][aa[has_signal], bb[has_signal]] = np.max(log_for_peak, axis=1)

    eps = np.finfo(np.float64).eps
    valid_moment_rows = has_signal & (cutoff_idx > 2)
    for L in np.unique(cutoff_idx[valid_moment_rows]):
        L = int(L)
        ids = np.flatnonzero(valid_moment_rows & (cutoff_idx == L))
        x = raw[ids, :L]
        y = log_values[ids, :L]
        mean_x = np.mean(x, axis=1)
        var_x = np.var(x, axis=1)
        ratio = np.zeros(len(ids), dtype=float)
        np.divide(var_x, mean_x, out=ratio, where=mean_x > 0)

        mean_y = np.mean(y, axis=1, keepdims=True)
        d = y - mean_y
        d2 = d * d
        m2 = np.mean(d2, axis=1)
        m3 = np.mean(d2 * d, axis=1)
        m4 = np.mean(d2 * d2, axis=1)
        zero_var = m2 <= (eps * np.abs(mean_y[:, 0])) ** 2
        safe_m2 = np.where(zero_var, 1.0, m2)
        skew = np.where(zero_var, np.nan, m3 / (safe_m2 * np.sqrt(safe_m2)))
        kurt = np.where(zero_var, np.nan, (m4 / (safe_m2 * safe_m2)) - 3.0)

        a, b = aa[ids], bb[ids]
        output['RDF_DispersionRatio'][a, b] = ratio
        output['LogRDF_Median'][a, b] = np.median(y, axis=1)
        output['LogRDF_Variance'][a, b] = m2
        output['LogRDF_Skewness'][a, b] = skew
        output['LogRDF_Kurtosis'][a, b] = kurt
    return output


def calculate_glam_wasserstein_distance(rdf_structured_df, rdf_random_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the 1-Wasserstein Distance restricted to the common supported radial domain
    `L = min(cut_s, cut_r)`, vectorized across all (alpha, beta) pairs grouped by cutoff L.
    """
    wasserstein_distances = {}
    if rdf_structured_df.empty or rdf_random_df.empty or total_roi_voxels <= 0:
        return {}

    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r_s, raw_struct, pairs, row_index = extract_fast_raw_rdf(rdf_structured_df, num_levels)
    r_r, raw_rand, rand_pairs, rand_index = extract_fast_raw_rdf(rdf_random_df, num_levels)
    if not np.array_equal(r_s, r_r):
        raise ValueError('Structured and baseline RDF radius grids differ')
    cut_s = get_row_cutoff_indices(raw_struct, rdf_structured_df)
    cut_r = get_row_cutoff_indices(raw_rand, rdf_random_df)
    rho = np.asarray(level_counts, dtype=np.float64) / float(total_roi_voxels)

    if pairs == rand_pairs and len(pairs) == num_levels * num_levels:
        common_cut = np.minimum(cut_s, cut_r)
        aa = np.repeat(np.arange(num_levels), num_levels)
        bb = np.tile(np.arange(num_levels), num_levels)
        valid_row = (common_cut >= 2) & (rho[aa] > 0) & (rho[bb] > 0)
        w_vals = np.full(len(pairs), np.nan, dtype=np.float64)

        for L in np.unique(common_cut[valid_row]):
            L = int(L)
            rows = np.flatnonzero(valid_row & (common_cut == L))
            r_sub = r_s[:L]
            dr = np.diff(r_sub)[None, :]
            factor = (4.0 * np.pi * rho[bb[rows], None]) * (r_sub[None, :] ** 2)
            diff_integrand = (raw_struct[rows, :L] - raw_rand[rows, :L]) * factor
            # Cumulative trapezoid of (n_struct - n_rand) by linearity!
            seg = 0.5 * (diff_integrand[:, :-1] + diff_integrand[:, 1:]) * dr
            n_diff = np.hstack((np.zeros((len(rows), 1), dtype=np.float64), np.cumsum(seg, axis=1)))
            w_vals[rows] = np.trapezoid(np.abs(n_diff), r_sub, axis=1)

        for alpha in range(num_levels):
            base_i = alpha * num_levels
            for beta in range(num_levels):
                val = w_vals[base_i + beta]
                wasserstein_distances[f'GLAM_Wasserstein_{alpha}_{beta}'] = float(val) if np.isfinite(val) else np.nan
        return wasserstein_distances

    for alpha in range(num_levels):
        for beta in range(num_levels):
            w_key = f'GLAM_Wasserstein_{alpha}_{beta}'
            s_idx = row_index.get((alpha, beta))
            r_idx = rand_index.get((alpha, beta))
            if s_idx is None or r_idx is None or rho[alpha] <= 0 or rho[beta] <= 0:
                wasserstein_distances[w_key] = np.nan
                continue
            L = min(int(cut_s[s_idx]), int(cut_r[r_idx]))
            if L < 2:
                wasserstein_distances[w_key] = np.nan
                continue
            r_sub = r_s[:L]
            factor = 4.0 * np.pi * rho[beta] * (r_sub ** 2)
            n_struct = scipy.integrate.cumulative_trapezoid(raw_struct[s_idx, :L] * factor, r_sub, initial=0.0)
            n_rand = scipy.integrate.cumulative_trapezoid(raw_rand[r_idx, :L] * factor, r_sub, initial=0.0)
            wasserstein_distances[w_key] = float(np.trapezoid(np.abs(n_struct - n_rand), r_sub))

    return wasserstein_distances


def _nan_safe_gradient_2d(mat):
    """Compute mixed partial derivative d2/dalpha dbeta over finite states without NaN bleed."""
    def grad_1d(arr, axis):
        out = np.full_like(arr, np.nan, dtype=np.float64)
        if axis == 0:
            for j in range(arr.shape[1]):
                col = arr[:, j]
                fin = np.flatnonzero(np.isfinite(col))
                if fin.size >= 2:
                    out[fin, j] = np.gradient(col[fin]) if np.all(np.diff(fin) == 1) else np.gradient(col[fin], fin.astype(np.float64))
        else:
            for i in range(arr.shape[0]):
                row = arr[i, :]
                fin = np.flatnonzero(np.isfinite(row))
                if fin.size >= 2:
                    out[i, fin] = np.gradient(row[fin]) if np.all(np.diff(fin) == 1) else np.gradient(row[fin], fin.astype(np.float64))
        return out

    return grad_1d(grad_1d(mat, axis=0), axis=1)


def calculate_glam_assembly_coupling_matrix(wasserstein_distances, num_levels):
    """Calculates the Assembly Coupling Matrix (mixed partial derivative of Wasserstein Distance)."""
    coupling_features = {}
    if not wasserstein_distances:
        return coupling_features

    emd_matrix = np.full((num_levels, num_levels), np.nan, dtype=np.float64)
    for alpha in range(num_levels):
        for beta in range(num_levels):
            key = f'GLAM_Wasserstein_{alpha}_{beta}'
            emd_matrix[alpha, beta] = wasserstein_distances.get(key, np.nan)

    d2EMD = _nan_safe_gradient_2d(emd_matrix)
    for alpha in range(num_levels):
        for beta in range(num_levels):
            val = d2EMD[alpha, beta]
            coupling_features[f'GLAM_AssemblyCoupling_{alpha}_{beta}'] = float(val) if np.isfinite(val) else np.nan

    return coupling_features


def calculate_glam_phenotypic_distance_matrix(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Compare normalized auto-RDF cumulative curves on pairwise common support.
    Normalization is performed after selecting the common domain, not before.
    Valid self-distances are zero; undefined states/pairs return NaN.
    """
    result = {}
    if rdf_df.empty or total_roi_voxels <= 0:
        return result
    from .rdf_smoothing import extract_fast_raw_rdf, get_row_cutoff_indices
    r, raw, _, row_index = extract_fast_raw_rdf(rdf_df, num_levels)
    if r.size < 2:
        return result
    if not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0):
        raise ValueError('PhenotypicDistance requires finite, strictly increasing radii')
    cuts = get_row_cutoff_indices(raw, rdf_df)
    rho = np.asarray(level_counts, dtype=np.float64) / float(total_roi_voxels)
    cumulative = [None] * num_levels
    diag_cuts = np.zeros(num_levels, dtype=np.int32)
    for state in range(num_levels):
        idx = row_index.get((state, state))
        if idx is None or not np.isfinite(rho[state]) or rho[state] <= 0:
            continue
        L = int(cuts[idx])
        if L < 2:
            continue
        g = raw[idx, :L]
        if not np.all(np.isfinite(g)) or np.any(g < 0):
            continue
        integrand = 4.0 * np.pi * rho[state] * g * r[:L]**2
        cumulative[state] = scipy.integrate.cumulative_trapezoid(integrand, r[:L], initial=0.0)
        diag_cuts[state] = L
    for alpha in range(num_levels):
        for beta in range(alpha, num_levels):
            L = min(int(diag_cuts[alpha]), int(diag_cuts[beta]))
            value = np.nan
            if L >= 2:
                ca, cb = cumulative[alpha][:L], cumulative[beta][:L]
                ma, mb = float(ca[-1]), float(cb[-1])
                if np.isfinite(ma) and np.isfinite(mb) and ma > 1e-9 and mb > 1e-9:
                    value = 0.0 if alpha == beta else float(
                        scipy.integrate.trapezoid(np.abs(ca / ma - cb / mb), r[:L]))
            result[f'GLAM_PhenotypicDistance_{alpha}_{beta}'] = value
            result[f'GLAM_PhenotypicDistance_{beta}_{alpha}'] = value
    return result



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

def _extract_active_submatrix(matrix):
    """Drop completely empty (all-NaN) rows/columns before spectral/GLCM/symmetry/cluster analysis."""
    if matrix is None or matrix.ndim != 2:
        return None
    finite = np.isfinite(matrix)
    if not np.any(finite):
        return None
    if matrix.shape[0] == matrix.shape[1]:
        active = np.any(finite, axis=1) | np.any(finite, axis=0)
        sub = matrix[np.ix_(active, active)].copy()
    else:
        r_act = np.any(finite, axis=1)
        c_act = np.any(finite, axis=0)
        sub = matrix[np.ix_(r_act, c_act)].copy()
    if sub.size == 0 or not np.any(np.isfinite(sub)):
        return None
    if np.isnan(sub).any():
        fill_val = float(np.nanmean(sub[np.isfinite(sub)]))
        sub = np.nan_to_num(sub, nan=fill_val if np.isfinite(fill_val) else 0.0)
    return sub


def calculate_glcm_style_meta_features(glam_matrix, feature_prefix):
    """Calculates a comprehensive set of 24 GLCM-style features on the active submatrix."""
    meta_features = {}
    clean_matrix = _extract_active_submatrix(glam_matrix)
    if clean_matrix is None:
        return meta_features

    min_val = np.min(clean_matrix)
    if min_val < 0:
        clean_matrix = clean_matrix - min_val

    matrix_sum = np.sum(clean_matrix)
    if matrix_sum == 0:
        return {}
    p = clean_matrix / matrix_sum

    i, j = np.ogrid[0:p.shape[0], 0:p.shape[1]]
    N = p.shape[0]

    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)

    ux = np.sum(i.ravel() * px)
    uy = np.sum(j.ravel() * py)

    sx = np.sum(px * (i.ravel() - ux) ** 2)
    sy = np.sum(py * (j.ravel() - uy) ** 2)

    meta_features[f'{feature_prefix}.Autocorrelation'] = np.sum(p * i * j)
    meta_features[f'{feature_prefix}.JointEnergy'] = np.sum(p ** 2)
    meta_features[f'{feature_prefix}.JointEntropy'] = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    meta_features[f'{feature_prefix}.Contrast'] = np.sum(p * (i - j) ** 2)

    if sx > 0 and sy > 0:
        meta_features[f'{feature_prefix}.Correlation'] = (np.sum(p * (i - ux) * (j - uy))) / np.sqrt(sx * sy)
    else:
        meta_features[f'{feature_prefix}.Correlation'] = 0

    meta_features[f'{feature_prefix}.InverseDifference'] = np.sum(p / (1 + np.abs(i - j)))
    meta_features[f'{feature_prefix}.InverseDifferenceMoment'] = np.sum(p / (1 + (i - j) ** 2))
    meta_features[f'{feature_prefix}.InverseDifferenceNormalized'] = np.sum(p / (1 + (np.abs(i - j) / N)))
    meta_features[f'{feature_prefix}.InverseDifferenceMomentNormalized'] = np.sum(p / (1 + ((i - j) ** 2 / N ** 2)))

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_var = p / ((i - j) ** 2)
        inv_var[i == j] = 0
        meta_features[f'{feature_prefix}.InverseVariance'] = np.sum(inv_var)

    k_sum = np.arange(2 * (N - 1) + 1)
    k_diff = np.arange(N)
    px_plus_y = np.array([np.sum(p[i + j == k]) for k in k_sum])
    px_minus_y = np.array([np.sum(p[np.abs(i - j) == k]) for k in k_diff])

    meta_features[f'{feature_prefix}.SumAverage'] = np.sum(k_sum * px_plus_y)
    meta_features[f'{feature_prefix}.SumEntropy'] = -np.sum(px_plus_y[px_plus_y > 0] * np.log2(px_plus_y[px_plus_y > 0]))
    meta_features[f'{feature_prefix}.SumSquares'] = np.sum(p * (i - ux) ** 2)
    meta_features[f'{feature_prefix}.DifferenceAverage'] = np.sum(k_diff * px_minus_y)
    meta_features[f'{feature_prefix}.DifferenceEntropy'] = -np.sum(px_minus_y[px_minus_y > 0] * np.log2(px_minus_y[px_minus_y > 0]))
    meta_features[f'{feature_prefix}.DifferenceVariance'] = np.sum(px_minus_y * (k_diff - meta_features[f'{feature_prefix}.DifferenceAverage']) ** 2)

    hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    px_py = px[:, None] * py[None, :]
    hxy1 = -np.sum(p[px_py > 0] * np.log2(px_py[px_py > 0]))
    hxy2 = -np.sum(px_py[px_py > 0] * np.log2(px_py[px_py > 0]))

    if max(hx, hy) > 0:
        meta_features[f'{feature_prefix}.Imc1'] = (meta_features[f'{feature_prefix}.JointEntropy'] - hxy1) / max(hx, hy)
    else:
        meta_features[f'{feature_prefix}.Imc1'] = 0

    if hxy2 > meta_features[f'{feature_prefix}.JointEntropy']:
        meta_features[f'{feature_prefix}.Imc2'] = np.sqrt(1 - np.exp(-2 * (hxy2 - meta_features[f'{feature_prefix}.JointEntropy'])))
    else:
        meta_features[f'{feature_prefix}.Imc2'] = 0

    meta_features[f'{feature_prefix}.ClusterShade'] = np.sum((i + j - ux - uy) ** 3 * p)
    meta_features[f'{feature_prefix}.ClusterProminence'] = np.sum((i + j - ux - uy) ** 4 * p)

    meta_features[f'{feature_prefix}.JointAverage'] = ux
    meta_features[f'{feature_prefix}.MaximumProbability'] = np.max(p)
    meta_features[f'{feature_prefix}.ClusterTendency'] = np.sum((i + j - ux - uy) ** 2 * p)

    return meta_features


def calculate_advanced_eigen_features(matrix, feature_prefix):
    """Calculates Eigen-decomposition features on the active square submatrix."""
    eigen_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return eigen_features
    clean_matrix = _extract_active_submatrix(matrix)
    if clean_matrix is None or clean_matrix.shape[0] < 2:
        return eigen_features
    try:
        eigenvalues = np.real(np.linalg.eigvals(clean_matrix))
        eigenvalues_sorted = sorted(eigenvalues, key=np.abs, reverse=True)

        eigen_features[f'{feature_prefix}.Eigen.Eigenvalue1'] = eigenvalues_sorted[0]
        if len(eigenvalues_sorted) > 1:
            eigen_features[f'{feature_prefix}.Eigen.Eigenvalue2'] = eigenvalues_sorted[1]
        eigen_features[f'{feature_prefix}.Eigen.SpectralRadius'] = np.abs(eigenvalues_sorted[0])
        eigen_features[f'{feature_prefix}.Eigen.Trace'] = np.trace(clean_matrix)

        abs_eigenvalues = np.abs(eigenvalues) + 1e-9
        eigen_features[f'{feature_prefix}.Eigen.LogDeterminant'] = np.sum(np.log(abs_eigenvalues))
        eigen_features[f'{feature_prefix}.Eigen.DeterminantSign'] = np.prod(np.sign(eigenvalues))

        p = np.abs(eigenvalues) / np.sum(np.abs(eigenvalues)) if np.sum(np.abs(eigenvalues)) > 0 else np.array([])
        if len(p) > 0 and len(p[p > 0]) > 0:
            eigen_features[f'{feature_prefix}.Eigen.Entropy'] = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    except np.linalg.LinAlgError:
        print(f"  - WARNING: Eigen-decomposition failed for {feature_prefix}.")
    return eigen_features


def calculate_diagonal_features(matrix, feature_prefix):
    """Calculates Diagonal vs. Off-Diagonal features."""
    diag_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
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
    if sum_abs_offdiag > 0:
        diag_features[f'{feature_prefix}.Diag.DominanceRatio'] = sum_abs_diag / sum_abs_offdiag
    return diag_features


def calculate_symmetry_features(matrix, feature_prefix):
    """Calculates features based on the symmetry/asymmetry of the active square submatrix."""
    sym_features = {}
    if matrix is None or np.all(np.isnan(matrix)) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return sym_features
    clean_matrix = _extract_active_submatrix(matrix)
    if clean_matrix is None:
        return sym_features
    asymmetry_matrix = clean_matrix - clean_matrix.T
    sym_features[f'{feature_prefix}.Symmetry.FrobeniusNorm'] = np.linalg.norm(asymmetry_matrix, 'fro')
    sym_features[f'{feature_prefix}.Symmetry.MeanAbsoluteAsymmetry'] = np.mean(np.abs(asymmetry_matrix))
    net_influence = np.sum(clean_matrix, axis=1) - np.sum(clean_matrix, axis=0)
    sym_features[f'{feature_prefix}.Symmetry.NetInfluenceMean'] = np.mean(net_influence)
    sym_features[f'{feature_prefix}.Symmetry.NetInfluenceVariance'] = np.var(net_influence)
    return sym_features


def calculate_cluster_features(matrix, feature_prefix):
    """Calculates features based on K-Means clustering of active matrix rows."""
    cluster_features = {}
    if matrix is None or np.all(np.isnan(matrix)):
        return cluster_features

    if matrix.ndim == 1:
        valid = matrix[np.isfinite(matrix)]
        if valid.size < 4:
            return cluster_features
        clean_matrix = valid.reshape(-1, 1)
    else:
        clean_matrix = _extract_active_submatrix(matrix)
        if clean_matrix is None or clean_matrix.shape[0] < 4:
            return cluster_features

    unique_rows = np.unique(clean_matrix, axis=0)
    n_unique = unique_rows.shape[0]
    max_possible_k = min(8, clean_matrix.shape[0] - 1, n_unique)
    if max_possible_k < 2:
        return {}

    best_k, best_score = -1, -1
    for k in range(2, max_possible_k + 1):
        try:
            kmeans_attempt = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans_attempt.fit_predict(clean_matrix)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(clean_matrix, labels)
            if score > best_score:
                best_score, best_k = score, k
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


# def calculate_glam_multifractal_spectrum(image_3d, num_levels):
#     """
#     Calculates the Multifractal Spectrum (f(alpha) vs alpha) and Generalized Dimensions (D_q).
#     Uses the Method of Moments (Partition Function).
#     """
#     multifractal_features = {}
    
#     # Define moments q to scan
#     q_values = np.array([-5, -2, 0, 1, 2, 5]) 
    
#     # Box sizes for scaling analysis
#     box_sizes = [2, 3, 4, 6, 8, 12, 16]
    
#     def get_multifractal_spectrum(binary_mask):
#         if np.sum(binary_mask) < 50: 
#             return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}

#         pixels = np.argwhere(binary_mask > 0)
#         N = len(pixels)
        
#         # Partition function Z(q, epsilon)
#         Z = defaultdict(list)
        
#         # Track which sizes actually got processed
#         processed_sizes = []
#         results = {}

#         for eps in box_sizes:
#             if eps >= min(binary_mask.shape): continue
            
#             processed_sizes.append(eps)
            
#             box_indices = np.floor(pixels / eps).astype(int)
#             _, counts = np.unique(box_indices, axis=0, return_counts=True)
#             p = counts / N
            
#             for q in q_values:
#                 if q == 1:
#                     # Shannon Entropy for q=1 (limit q->1)
#                     sum_val = np.sum(p * np.log(p + 1e-12))
#                 else:
#                     sum_val = np.sum(p ** q)
                
#                 Z[q].append(sum_val)

#         # Safety Check: Did we get enough scales for a regression?
#         if len(processed_sizes) < 3:
#              return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}
             
#         # X-axis: log(epsilon)
#         x_all = np.log(np.array(processed_sizes))
        
#         tau_q = {}
        
#         for q in q_values:
#             # Y-axis: log(Z(q, epsilon))
#             # Z[q] corresponds 1-to-1 with processed_sizes
#             z_values = np.array(Z[q])
            
#             # --- CRITICAL FIX: Mask invalid values before Log ---
#             # Remove NaNs, Infs, and Zeros (which cause log errors)
#             mask = (z_values > 1e-20) & np.isfinite(z_values)
            
#             if np.sum(mask) < 3:
#                 tau_q[q] = 0.0
#                 results[f"D_q_{q}"] = 0.0
#                 continue
                
#             y_clean = np.log(z_values[mask])
#             x_clean = x_all[mask]
            
#             # Perform Regression
#             slope, _, _, _, _ = stats.linregress(x_clean, y_clean)
            
#             tau = slope
#             tau_q[q] = tau
            
#             # Calculate D_q
#             if q == 1:
#                 results[f"D_q_{q}"] = tau 
#             elif q != 1:
#                 results[f"D_q_{q}"] = tau / (q - 1)
        
#         # Calculate Alpha and f(Alpha) via finite difference
#         alphas = []
#         q_sorted = sorted(q_values)
#         for i in range(len(q_sorted)-1):
#             q1, q2 = q_sorted[i], q_sorted[i+1]
#             t1, t2 = tau_q[q1], tau_q[q2]
            
#             # Numerical derivative d(tau)/dq
#             if (q2 - q1) != 0:
#                 alpha = (t2 - t1) / (q2 - q1)
#                 alphas.append(alpha)
            
#         if alphas:
#             results["Width"] = max(alphas) - min(alphas)
#             results["Alpha_0"] = np.mean(alphas)
#         else:
#             results["Width"] = 0.0
#             results["Alpha_0"] = 0.0
            
#         return results

#     # Initialize results container - FIX: Define 'results' here so it resets per loop iteration logic in your original structure?
#     # Actually, the original code had 'results = {}' INSIDE the helper. 
#     # But here we need to loop over levels.
    
#     # 1. Diagonal: Volume Multifractality
#     for i in range(num_levels):
#         binary_image = (image_3d == i)
        
#         mf_res = get_multifractal_spectrum(binary_image)
        
#         multifractal_features[f'GLAM_VolumeMultifractal_Width_{i}'] = mf_res["Width"]
#         multifractal_features[f'GLAM_VolumeMultifractal_Alpha0_{i}'] = mf_res["Alpha_0"]
#         multifractal_features[f'GLAM_VolumeMultifractal_D2_{i}'] = mf_res.get("D_q_2", 0.0)

#     # 2. Off-Diagonal: Interface Multifractality
#     for i in range(num_levels):
#         for j in range(num_levels):
#             if i == j: continue
            
#             mask_i = (image_3d == i)
#             mask_j = (image_3d == j)
            
#             if not np.any(mask_i) or not np.any(mask_j):
#                 val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
#             else:
#                 interface = ndimage.binary_dilation(mask_i) & mask_j
#                 if np.sum(interface) < 50:
#                     val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
#                 else:
#                     mf_res = get_multifractal_spectrum(interface)
#                     val_width = mf_res["Width"]
#                     val_alpha = mf_res["Alpha_0"]
#                     val_d2 = mf_res.get("D_q_2", 0.0)
            
#             multifractal_features[f'GLAM_InterfaceMultifractal_Width_{i}_{j}'] = val_width
#             multifractal_features[f'GLAM_InterfaceMultifractal_Alpha0_{i}_{j}'] = val_alpha
#             multifractal_features[f'GLAM_InterfaceMultifractal_D2_{i}_{j}'] = val_d2

#     return multifractal_features

def calculate_glam_multifractal_spectrum(image_3d, num_levels):
    """
    Calculates the Multifractal Spectrum.
    GPU Accelerated using CuPy and flat-index 3D block counting.
    """
    if not HAS_GPU:
        return _calculate_glam_multifractal_spectrum_cpu(image_3d, num_levels)

    import cupyx.scipy.ndimage as cupy_ndimage

    print("  - Starting Multifractal Spectrum analysis (GPU Accelerated)...")
    multifractal_features = {}
    q_values = cp.array([-5, -2, 0, 1, 2, 5], dtype=cp.float64)
    q_values_cpu = [-5, -2, 0, 1, 2, 5]
    box_sizes = [2, 3, 4, 6, 8, 12, 16]

    image_gpu = cp.array(image_3d, dtype=cp.int16)
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}

    def get_multifractal_spectrum_gpu(binary_mask_gpu):
        total_pixels = int(cp.sum(binary_mask_gpu))
        if total_pixels < 50:
            return {f"D_q_{q}": 0.0 for q in q_values_cpu} | {"Width": 0.0, "Alpha_0": 0.0}

        pixels = cp.argwhere(binary_mask_gpu > 0)
        N = float(total_pixels)
        Z = defaultdict(list)
        processed_sizes = []
        results = {}

        shape_z, shape_y, shape_x = binary_mask_gpu.shape

        for eps in box_sizes:
            if eps >= min(binary_mask_gpu.shape): continue
            processed_sizes.append(eps)

            box_indices = cp.floor(pixels / eps).astype(cp.int32)

            # Fast GPU flat-indexing for unique counting
            max_y = (shape_y // eps) + 2
            max_x = (shape_x // eps) + 2
            flat_indices = box_indices[:, 0] * (max_y * max_x) + box_indices[:, 1] * max_x + box_indices[:, 2]

            _, counts = cp.unique(flat_indices, return_counts=True)
            p = counts / N

            for q in q_values_cpu:
                if q == 1:
                    sum_val = cp.sum(p * cp.log(p + 1e-12))
                else:
                    sum_val = cp.sum(p ** q)
                Z[q].append(float(sum_val))

        if len(processed_sizes) < 3:
             return {f"D_q_{q}": 0.0 for q in q_values_cpu} | {"Width": 0.0, "Alpha_0": 0.0}

        return _fit_multifractal_moments(processed_sizes, Z, q_values_cpu)

    # 1. Volume
    for i in range(num_levels):
        mf_res = get_multifractal_spectrum_gpu(masks_gpu[i])
        multifractal_features[f'GLAM_VolumeMultifractal_Width_{i}'] = mf_res["Width"]
        multifractal_features[f'GLAM_VolumeMultifractal_Alpha0_{i}'] = mf_res["Alpha_0"]
        multifractal_features[f'GLAM_VolumeMultifractal_D2_{i}'] = mf_res.get("D_q_2", 0.0)
        multifractal_features[f'GLAM_VolumeMultifractal_D1_{i}'] = mf_res.get("D_q_1", 0.0)

    # 2. Interface
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    multifractal_features[f'GLAM_InterfaceMultifractal_Width_{i}_{j}'] = 0.0
                    multifractal_features[f'GLAM_InterfaceMultifractal_Alpha0_{i}_{j}'] = 0.0
                    multifractal_features[f'GLAM_InterfaceMultifractal_D2_{i}_{j}'] = 0.0
            continue

        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])

        for j in range(num_levels):
            if i == j: continue

            if not cp.any(masks_gpu[j]):
                val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
            else:
                interface_gpu = dilated_i_gpu & masks_gpu[j]
                if cp.sum(interface_gpu) < 50:
                    val_width, val_alpha, val_d2 = 0.0, 0.0, 0.0
                else:
                    mf_res = get_multifractal_spectrum_gpu(interface_gpu)
                    val_width = mf_res["Width"]
                    val_alpha = mf_res["Alpha_0"]
                    val_d2 = mf_res.get("D_q_2", 0.0)

            multifractal_features[f'GLAM_InterfaceMultifractal_Width_{i}_{j}'] = val_width
            multifractal_features[f'GLAM_InterfaceMultifractal_Alpha0_{i}_{j}'] = val_alpha
            multifractal_features[f'GLAM_InterfaceMultifractal_D2_{i}_{j}'] = val_d2

    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return multifractal_features


def _calculate_glam_multifractal_spectrum_cpu(image_3d, num_levels):
    """Fallback CPU function for Multifractal Spectrum."""
    multifractal_features = {}
    q_values = np.array([-5, -2, 0, 1, 2, 5]) 
    box_sizes = [2, 3, 4, 6, 8, 12, 16]

    def get_multifractal_spectrum(binary_mask):
        if np.sum(binary_mask) < 50: 
            return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}

        pixels = np.argwhere(binary_mask > 0)
        N = len(pixels)
        Z = defaultdict(list)
        processed_sizes = []
        results = {}

        for eps in box_sizes:
            if eps >= min(binary_mask.shape): continue
            processed_sizes.append(eps)
            box_indices = np.floor(pixels / eps).astype(int)
            _, counts = np.unique(box_indices, axis=0, return_counts=True)
            p = counts / N
            for q in q_values:
                if q == 1: sum_val = np.sum(p * np.log(p + 1e-12))
                else: sum_val = np.sum(p ** q)
                Z[q].append(sum_val)

        if len(processed_sizes) < 3:
             return {f"D_q_{q}": 0.0 for q in q_values} | {"Width": 0.0, "Alpha_0": 0.0}

        return _fit_multifractal_moments(processed_sizes, Z, q_values)

    for i in range(num_levels):
        binary_image = (image_3d == i)
        mf_res = get_multifractal_spectrum(binary_image)
        multifractal_features[f'GLAM_VolumeMultifractal_Width_{i}'] = mf_res["Width"]
        multifractal_features[f'GLAM_VolumeMultifractal_Alpha0_{i}'] = mf_res["Alpha_0"]
        multifractal_features[f'GLAM_VolumeMultifractal_D2_{i}'] = mf_res.get("D_q_2", 0.0)
        multifractal_features[f'GLAM_VolumeMultifractal_D1_{i}'] = mf_res.get("D_q_1", 0.0)

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
                    val_width, val_alpha, val_d2 = mf_res["Width"], mf_res["Alpha_0"], mf_res.get("D_q_2", 0.0)

            multifractal_features[f'GLAM_InterfaceMultifractal_Width_{i}_{j}'] = val_width
            multifractal_features[f'GLAM_InterfaceMultifractal_Alpha0_{i}_{j}'] = val_alpha
            multifractal_features[f'GLAM_InterfaceMultifractal_D2_{i}_{j}'] = val_d2

    return multifractal_features



def calculate_glam_shape_matrices(structured_image, num_levels, spacing):
    """
    Calculates 3D shape features (Sphericity, Area, Radial Variance).
    Uses GPU acceleration to rapidly solve Interface Areas.
    """
    if not HAS_GPU:
        return _calculate_glam_shape_matrices_cpu(structured_image, num_levels, spacing)

    import cupyx.scipy.ndimage as cupy_ndimage

    print("  - Starting Shape Matrices analysis (GPU Accelerated)...")
    feats = {}
    spacing_zyx = spacing[::-1]
    voxel_vol = np.prod(spacing)

    # Crop to Bounding Box to minimize memory overhead
    mask_all = (structured_image > -1)
    if not np.any(mask_all): return feats

    slices = ndimage.find_objects(mask_all.astype(int))
    if not slices: return feats

    roi_img = structured_image[slices[0]].copy()

    # Move cropped image to GPU
    roi_img_gpu = cp.array(roi_img, dtype=cp.int16)
    masks_gpu = {i: (roi_img_gpu == i) for i in range(num_levels)}

    # 1. DIAGONALS & Centroids (Marching Cubes stays on CPU)
    centroids = {}
    for i in range(num_levels):
        mask_i_cpu = masks_gpu[i].get()
        voxel_coords_idx = np.argwhere(mask_i_cpu)
        n_voxels = len(voxel_coords_idx)

        if n_voxels > 0:
            c_mass = np.mean(voxel_coords_idx * np.array(spacing_zyx), axis=0)
            centroids[i] = c_mass

        val_sphericity, val_solidity = 0.0, 0.0
        val_rad_mean, val_rad_var = 0.0, 0.0
        val_rad_skew, val_rad_kurt = 0.0, 0.0

        if n_voxels >= 4:
            physical_coords = voxel_coords_idx * np.array(spacing_zyx)
            try:
                verts, faces, _, _ = measure.marching_cubes(mask_i_cpu, level=0.5, spacing=spacing_zyx)
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

    # 2. OFF-DIAGONALS (GPU Accelerated Dilation!)
    for i in range(num_levels):
        dilated_i_gpu = None
        if cp.any(masks_gpu[i]):
            dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])

        for j in range(num_levels):
            if i == j: continue

            val_dist = 0.0
            if i in centroids and j in centroids:
                val_dist = np.linalg.norm(centroids[i] - centroids[j])
            feats[f'GLAM_Shape_CentroidDist_{i}_{j}'] = val_dist

            val_interface = 0.0
            if dilated_i_gpu is not None and cp.any(masks_gpu[j]):
                intersection = dilated_i_gpu & masks_gpu[j]
                val_interface = float(cp.sum(intersection)) * voxel_vol

            feats[f'GLAM_Shape_InterfaceArea_{i}_{j}'] = val_interface

    # Cleanup
    del roi_img_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return feats

def _calculate_glam_shape_matrices_cpu(structured_image, num_levels, spacing):
    """Fallback CPU function for Shape matrices."""
    print("  - Starting Shape Matrices analysis (CPU)...")
    feats = {}
    spacing_zyx = spacing[::-1]
    voxel_vol = np.prod(spacing)
    
    mask_all = (structured_image > -1)
    if not np.any(mask_all): return feats
    
    slices = ndimage.find_objects(mask_all.astype(int))
    if not slices: return feats
    roi_img = structured_image[slices[0]].copy()
    
    centroids = {} 
    for i in range(num_levels):
        mask_i = (roi_img == i)
        voxel_coords_idx = np.argwhere(mask_i)
        n_voxels = len(voxel_coords_idx)
        
        if n_voxels > 0: centroids[i] = np.mean(voxel_coords_idx * np.array(spacing_zyx), axis=0)
        
        val_sphericity, val_solidity = 0.0, 0.0 
        val_rad_mean, val_rad_var, val_rad_skew, val_rad_kurt = 0.0, 0.0, 0.0, 0.0
        
        if n_voxels >= 4:
            physical_coords = voxel_coords_idx * np.array(spacing_zyx)
            try:
                verts, faces, _, _ = measure.marching_cubes(mask_i, level=0.5, spacing=spacing_zyx)
                surf_area = measure.mesh_surface_area(verts, faces)
                vol = n_voxels * voxel_vol
                if surf_area > 0: val_sphericity = (np.pi**(1/3) * (6 * vol)**(2/3)) / surf_area
                centroid_mesh = np.mean(verts, axis=0)
                radial_dists = np.linalg.norm(verts - centroid_mesh, axis=1)
                val_rad_mean, val_rad_var = np.mean(radial_dists), np.var(radial_dists)
                val_rad_skew, val_rad_kurt = stats.skew(radial_dists), stats.kurtosis(radial_dists)
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

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue 
            
            val_dist = 0.0
            if i in centroids and j in centroids: val_dist = np.linalg.norm(centroids[i] - centroids[j])
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


def calculate_glam_percolation(image_3d, num_levels, total_roi_voxels):
    """
    Calculates Percolation Metrics for the GLAM Soft Matter geometric classes.
    GPU Accelerated using CuPy connected component labeling.
    """
    if not HAS_GPU: 
        return _calculate_glam_percolation_cpu(image_3d, num_levels, total_roi_voxels)
        
    import cupyx.scipy.ndimage as cupy_ndimage
    print("  - Starting Percolation analysis (GPU Accelerated)...")
    percolation_features = {}
    
    image_gpu = cp.array(image_3d, dtype=cp.int16)
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}
    structure_3d = cp.ones((3, 3, 3), dtype=cp.int32) # 26-connectivity for 3D clusters
    
    def get_percolation_metrics_gpu(binary_mask_gpu):
        total_mask_voxels = float(cp.sum(binary_mask_gpu))
        if total_mask_voxels == 0:
            return 0.0, 0.0, 0.0
            
        labeled_gpu, num_features = cupy_ndimage.label(binary_mask_gpu, structure=structure_3d)
        if num_features == 0:
            return 0.0, 0.0, 0.0
            
        # Bincount to get sizes of all clusters (index 0 is the background, skip it)
        cluster_sizes = cp.bincount(labeled_gpu.ravel())[1:]
        if cluster_sizes.size == 0: 
            return 0.0, 0.0, 0.0
            
        max_cluster_size = float(cp.max(cluster_sizes))
        
        # 1. Cluster Number Density (n_c = N_clusters / V_ROI)
        cluster_density = float(num_features) / total_roi_voxels if total_roi_voxels > 0 else 0.0
        
        # 2. Percolation Strength (P = S_max / N_active) - static threshold surrogate
        percolation_strength = max_cluster_size / total_mask_voxels
        
        return max_cluster_size, cluster_density, percolation_strength

    # 1. Volume Percolation (Diagonal)
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            s_max, n_c, p_strength = 0.0, 0.0, 0.0
        else:
            s_max, n_c, p_strength = get_percolation_metrics_gpu(masks_gpu[i])
            
        percolation_features[f'GLAM_VolumeMaxClusterSize_{i}'] = s_max
        percolation_features[f'GLAM_VolumeClusterNumberDensity_{i}'] = n_c
        percolation_features[f'GLAM_VolumePercolationStrength_{i}'] = p_strength

    # 2. Interface Percolation (Off-Diagonal)
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    percolation_features[f'GLAM_InterfaceMaxClusterSize_{i}_{j}'] = 0.0
                    percolation_features[f'GLAM_InterfaceClusterNumberDensity_{i}_{j}'] = 0.0
                    percolation_features[f'GLAM_InterfacePercolationStrength_{i}_{j}'] = 0.0
            continue
            
        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])
        
        for j in range(num_levels):
            if i == j: continue
            
            if not cp.any(masks_gpu[j]):
                s_max, n_c, p_strength = 0.0, 0.0, 0.0
            else:
                interface_gpu = dilated_i_gpu & masks_gpu[j]
                if not cp.any(interface_gpu):
                    s_max, n_c, p_strength = 0.0, 0.0, 0.0
                else:
                    s_max, n_c, p_strength = get_percolation_metrics_gpu(interface_gpu)
            
            percolation_features[f'GLAM_InterfaceMaxClusterSize_{i}_{j}'] = s_max
            percolation_features[f'GLAM_InterfaceClusterNumberDensity_{i}_{j}'] = n_c
            percolation_features[f'GLAM_InterfacePercolationStrength_{i}_{j}'] = p_strength

    # Cleanup
    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return percolation_features

def _calculate_glam_percolation_cpu(image_3d, num_levels, total_roi_voxels):
    """Fallback CPU function for Percolation Metrics."""
    print("  - Starting Percolation analysis (CPU)...")
    from scipy import ndimage
    
    percolation_features = {}
    structure_3d = np.ones((3, 3, 3), dtype=int)
    
    def get_percolation_metrics(binary_mask):
        total_mask_voxels = float(np.sum(binary_mask))
        if total_mask_voxels == 0:
            return 0.0, 0.0, 0.0
            
        labeled, num_features = ndimage.label(binary_mask, structure=structure_3d)
        if num_features == 0:
            return 0.0, 0.0, 0.0
            
        cluster_sizes = np.bincount(labeled.ravel())[1:]
        if cluster_sizes.size == 0: 
            return 0.0, 0.0, 0.0
            
        max_cluster_size = float(np.max(cluster_sizes))
        cluster_density = float(num_features) / total_roi_voxels if total_roi_voxels > 0 else 0.0
        percolation_strength = max_cluster_size / total_mask_voxels
        
        return max_cluster_size, cluster_density, percolation_strength

    for i in range(num_levels):
        binary_image = (image_3d == i)
        s_max, n_c, p_strength = get_percolation_metrics(binary_image)
        percolation_features[f'GLAM_VolumeMaxClusterSize_{i}'] = s_max
        percolation_features[f'GLAM_VolumeClusterNumberDensity_{i}'] = n_c
        percolation_features[f'GLAM_VolumePercolationStrength_{i}'] = p_strength

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue
            
            mask_i = (image_3d == i)
            mask_j = (image_3d == j)
            
            if not np.any(mask_i) or not np.any(mask_j):
                s_max, n_c, p_strength = 0.0, 0.0, 0.0
            else:
                interface = ndimage.binary_dilation(mask_i) & mask_j
                s_max, n_c, p_strength = get_percolation_metrics(interface)
                
            percolation_features[f'GLAM_InterfaceMaxClusterSize_{i}_{j}'] = s_max
            percolation_features[f'GLAM_InterfaceClusterNumberDensity_{i}_{j}'] = n_c
            percolation_features[f'GLAM_InterfacePercolationStrength_{i}_{j}'] = p_strength

    return percolation_features


def calculate_glam_granulometry(image_3d, num_levels, max_radius=10):
    """
    Calculates Granulometry (Pattern Spectrum) Metrics.
    Extracts the Mean, Variance, Skewness, Kurtosis, and Entropy of local thickness.
    GPU Accelerated using morphological openings with spherical structuring elements.
    """
    if not HAS_GPU:
        return _calculate_glam_granulometry_cpu(image_3d, num_levels, max_radius)
        
    import cupyx.scipy.ndimage as cupy_ndimage
    print("  - Starting Granulometry analysis (GPU Accelerated)...")
    granulometry_features = {}
    
    image_gpu = cp.array(image_3d, dtype=cp.int16)
    masks_gpu = {i: (image_gpu == i) for i in range(num_levels)}
    
    # Pre-calculate spherical structuring elements to avoid recompiling them inside loops
    spheres_gpu = []
    for r in range(1, max_radius + 1):
        L = 2 * r + 1
        z, y, x = np.ogrid[:L, :L, :L]
        sphere_cpu = ((z - r)**2 + (y - r)**2 + (x - r)**2 <= r**2).astype(np.int32)
        spheres_gpu.append(cp.array(sphere_cpu))
        
    def get_granulometry_metrics_gpu(binary_mask_gpu):
        vol_prev = float(cp.sum(binary_mask_gpu))
        if vol_prev == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        spectrum = []
        for r_idx, footprint in enumerate(spheres_gpu):
            if vol_prev == 0: break
            # Morphological Opening: Erosion followed by Dilation
            opened = cupy_ndimage.binary_opening(binary_mask_gpu, structure=footprint)
            vol_curr = float(cp.sum(opened))
            spectrum.append(vol_prev - vol_curr)
            vol_prev = vol_curr
            
        # Catch anything larger than the maximum sieve
        if vol_prev > 0:
            spectrum.append(vol_prev)
            
        spectrum = np.array(spectrum)
        total = np.sum(spectrum)
        if total == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
        
        p_r = spectrum / total
        r_vals = np.arange(1, len(p_r) + 1)
        
        # Calculate Moments
        mean_r = np.sum(r_vals * p_r)
        var_r = np.sum((r_vals - mean_r)**2 * p_r)
        std_r = np.sqrt(var_r)
        
        if std_r > 1e-6:
            skew_r = np.sum(((r_vals - mean_r) / std_r)**3 * p_r)
            kurt_r = np.sum(((r_vals - mean_r) / std_r)**4 * p_r)
        else:
            skew_r, kurt_r = 0.0, 0.0
            
        entropy_r = -np.sum(p_r[p_r > 0] * np.log2(p_r[p_r > 0]))
        
        return float(mean_r), float(var_r), float(skew_r), float(kurt_r), float(entropy_r)

    # 1. Volume Granulometry (Diagonal)
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            mean_r, var_r, skew_r, kurt_r, ent_r = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            mean_r, var_r, skew_r, kurt_r, ent_r = get_granulometry_metrics_gpu(masks_gpu[i])
            
        granulometry_features[f'GLAM_VolumeGranulometryMean_{i}'] = mean_r
        granulometry_features[f'GLAM_VolumeGranulometryVariance_{i}'] = var_r
        granulometry_features[f'GLAM_VolumeGranulometrySkewness_{i}'] = skew_r
        granulometry_features[f'GLAM_VolumeGranulometryKurtosis_{i}'] = kurt_r
        granulometry_features[f'GLAM_VolumeGranulometryEntropy_{i}'] = ent_r

    # 2. Interface Granulometry (Off-Diagonal)
    for i in range(num_levels):
        if not cp.any(masks_gpu[i]):
            for j in range(num_levels):
                if i != j:
                    granulometry_features[f'GLAM_InterfaceGranulometryMean_{i}_{j}'] = 0.0
                    granulometry_features[f'GLAM_InterfaceGranulometryVariance_{i}_{j}'] = 0.0
                    granulometry_features[f'GLAM_InterfaceGranulometrySkewness_{i}_{j}'] = 0.0
                    granulometry_features[f'GLAM_InterfaceGranulometryKurtosis_{i}_{j}'] = 0.0
                    granulometry_features[f'GLAM_InterfaceGranulometryEntropy_{i}_{j}'] = 0.0
            continue
            
        dilated_i_gpu = cupy_ndimage.binary_dilation(masks_gpu[i])
        for j in range(num_levels):
            if i == j: continue
            
            if not cp.any(masks_gpu[j]):
                mean_r, var_r, skew_r, kurt_r, ent_r = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                interface_gpu = dilated_i_gpu & masks_gpu[j]
                if not cp.any(interface_gpu):
                    mean_r, var_r, skew_r, kurt_r, ent_r = 0.0, 0.0, 0.0, 0.0, 0.0
                else:
                    mean_r, var_r, skew_r, kurt_r, ent_r = get_granulometry_metrics_gpu(interface_gpu)
            
            granulometry_features[f'GLAM_InterfaceGranulometryMean_{i}_{j}'] = mean_r
            granulometry_features[f'GLAM_InterfaceGranulometryVariance_{i}_{j}'] = var_r
            granulometry_features[f'GLAM_InterfaceGranulometrySkewness_{i}_{j}'] = skew_r
            granulometry_features[f'GLAM_InterfaceGranulometryKurtosis_{i}_{j}'] = kurt_r
            granulometry_features[f'GLAM_InterfaceGranulometryEntropy_{i}_{j}'] = ent_r

    del image_gpu
    del masks_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return granulometry_features

def _calculate_glam_granulometry_cpu(image_3d, num_levels, max_radius=10):
    """Fallback CPU function for Granulometry Metrics."""
    print("  - Starting Granulometry analysis (CPU)...")
    from scipy import ndimage
    granulometry_features = {}
    
    spheres_cpu = []
    for r in range(1, max_radius + 1):
        L = 2 * r + 1
        z, y, x = np.ogrid[:L, :L, :L]
        spheres_cpu.append(((z - r)**2 + (y - r)**2 + (x - r)**2 <= r**2).astype(int))

    def get_granulometry_metrics(binary_mask):
        vol_prev = float(np.sum(binary_mask))
        if vol_prev == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
            
        spectrum = []
        for footprint in spheres_cpu:
            if vol_prev == 0: break
            opened = ndimage.binary_opening(binary_mask, structure=footprint)
            vol_curr = float(np.sum(opened))
            spectrum.append(vol_prev - vol_curr)
            vol_prev = vol_curr
            
        if vol_prev > 0: spectrum.append(vol_prev)
            
        spectrum = np.array(spectrum)
        total = np.sum(spectrum)
        if total == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
        
        p_r = spectrum / total
        r_vals = np.arange(1, len(p_r) + 1)
        mean_r = np.sum(r_vals * p_r)
        var_r = np.sum((r_vals - mean_r)**2 * p_r)
        std_r = np.sqrt(var_r)
        
        if std_r > 1e-6:
            skew_r = np.sum(((r_vals - mean_r) / std_r)**3 * p_r)
            kurt_r = np.sum(((r_vals - mean_r) / std_r)**4 * p_r)
        else:
            skew_r, kurt_r = 0.0, 0.0
            
        entropy_r = -np.sum(p_r[p_r > 0] * np.log2(p_r[p_r > 0]))
        return float(mean_r), float(var_r), float(skew_r), float(kurt_r), float(entropy_r)

    for i in range(num_levels):
        binary_image = (image_3d == i)
        mean_r, var_r, skew_r, kurt_r, ent_r = get_granulometry_metrics(binary_image)
        granulometry_features[f'GLAM_VolumeGranulometryMean_{i}'] = mean_r
        granulometry_features[f'GLAM_VolumeGranulometryVariance_{i}'] = var_r
        granulometry_features[f'GLAM_VolumeGranulometrySkewness_{i}'] = skew_r
        granulometry_features[f'GLAM_VolumeGranulometryKurtosis_{i}'] = kurt_r
        granulometry_features[f'GLAM_VolumeGranulometryEntropy_{i}'] = ent_r

    for i in range(num_levels):
        for j in range(num_levels):
            if i == j: continue
            mask_i, mask_j = (image_3d == i), (image_3d == j)
            if not np.any(mask_i) or not np.any(mask_j):
                mean_r, var_r, skew_r, kurt_r, ent_r = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                interface = ndimage.binary_dilation(mask_i) & mask_j
                mean_r, var_r, skew_r, kurt_r, ent_r = get_granulometry_metrics(interface)
            
            granulometry_features[f'GLAM_InterfaceGranulometryMean_{i}_{j}'] = mean_r
            granulometry_features[f'GLAM_InterfaceGranulometryVariance_{i}_{j}'] = var_r
            granulometry_features[f'GLAM_InterfaceGranulometrySkewness_{i}_{j}'] = skew_r
            granulometry_features[f'GLAM_InterfaceGranulometryKurtosis_{i}_{j}'] = kurt_r
            granulometry_features[f'GLAM_InterfaceGranulometryEntropy_{i}_{j}'] = ent_r

    return granulometry_features


def calculate_glam_frustration_index(spi_metrics, cdi_metrics, num_levels, epsilon=1e-6):
    """
    Calculates the Structural Frustration Index (F) from pre-computed SPI and CDI.
    Represents the ratio of structural stress to thermodynamic disorder (unjamming transition).
    """
    frustration_metrics = {}
    
    if not spi_metrics or not cdi_metrics:
        return frustration_metrics
        
    for alpha in range(num_levels):
        for beta in range(num_levels):
            spi_key = f'GLAM_StructuralPressureIndex_{alpha}_{beta}'
            cdi_key = f'GLAM_ConfigurationalDisorderIndex_{alpha}_{beta}'
            frust_key = f'GLAM_FrustrationIndex_{alpha}_{beta}'
            
            spi_val = spi_metrics.get(spi_key, np.nan)
            cdi_val = cdi_metrics.get(cdi_key, np.nan)
            
            if np.isnan(spi_val) or np.isnan(cdi_val):
                frustration_metrics[frust_key] = np.nan
            else:
                # Calculate ratio with regularization to prevent division by zero
                frustration_metrics[frust_key] = spi_val / (cdi_val + epsilon)
                
    return frustration_metrics

def calculate_glam_local_packing_fraction(rdf_df, num_levels, level_counts, total_roi_voxels):
    """Calculates the Local Packing Fraction (Phi) within the first coordination shell
    using cutoff-safe smoothed RDFs (including r=1 contact peaks).
    """
    savgol_window = get_config('SavgolWindow')
    savgol_poly = get_config('SavgolPoly')
    peak_prominence = get_config('PeakProminence')

    packing_fractions = {}
    if rdf_df.empty or total_roi_voxels <= 0:
        return packing_fractions

    from .rdf_smoothing import prepare_smoothed_rdf
    smoothed = prepare_smoothed_rdf(rdf_df, num_levels, savgol_window, savgol_poly)
    r = smoothed.radius
    n_r = r.size
    pairs = smoothed.pairs
    n_pairs = len(pairs)
    if not n_pairs or n_r == 0:
        return packing_fractions

    raw_mat = smoothed.raw
    cutoff_idx = smoothed.cutoff_idx
    col_grid = np.arange(n_r, dtype=np.int32)[None, :]
    in_sup = col_grid < cutoff_idx[:, None]
    has_pos = (cutoff_idx >= 2) & np.any(np.where(in_sup, raw_mat, 0.0) > 0, axis=1)

    r_min_indices, row_active = _find_first_shell_minima_cutoff_safe(
        smoothed.smooth, cutoff_idx, has_pos, 15, peak_prominence
    )
    row_active &= (r_min_indices >= 1)
    rho = np.asarray(level_counts, dtype=np.float64) / float(total_roi_voxels)

    for alpha in range(num_levels):
        for beta in range(num_levels):
            pack_key = f'GLAM_LocalPackingFraction_{alpha}_{beta}'
            row_i = smoothed.row_index.get((alpha, beta))
            if row_i is None or not row_active[row_i] or rho[beta] <= 0:
                packing_fractions[pack_key] = np.nan
                continue
            r_min_idx = int(r_min_indices[row_i])
            end = r_min_idx + 1
            integrand = raw_mat[row_i, :end] * (r[:end] ** 2)
            integral = float(np.trapezoid(integrand, r[:end]))
            Z = 4.0 * np.pi * rho[beta] * integral
            r_boundary = float(r[r_min_idx]) + 0.5
            shell_volume_voxels = (4.0 / 3.0) * np.pi * (r_boundary ** 3)
            packing_fractions[pack_key] = float(Z / shell_volume_voxels) if shell_volume_voxels > 0 else np.nan

    return packing_fractions


def _fit_multifractal_moments(processed_sizes, Z, q_values):
    """Fit D1 from sum(p log p); normalized mass has tau(1)=0, not D1."""
    x = np.log(np.asarray(processed_sizes, dtype=float))
    result, tau = {}, {}
    for q in q_values:
        z = np.asarray(Z[q], dtype=float)
        if q == 1:
            valid = np.isfinite(z) & np.isfinite(x)
            result[f"D_q_{q}"] = (
                float(stats.linregress(x[valid], z[valid]).slope)
                if valid.sum() >= 3 else np.nan)
            tau[q] = 0.0
        else:
            valid = (z > 0) & np.isfinite(z) & np.isfinite(x)
            slope = (float(stats.linregress(x[valid], np.log(z[valid])).slope)
                     if valid.sum() >= 3 else np.nan)
            tau[q] = slope
            result[f"D_q_{q}"] = slope / (q - 1)
    qs = sorted(q_values)
    alphas = [(tau[b] - tau[a]) / (b - a) for a, b in zip(qs[:-1], qs[1:])]
    if alphas and np.isfinite(alphas).all():
        result["Width"] = float(np.ptp(alphas))
        # Preserve the legacy summary definition; this is mean interval alpha,
        # not a newly introduced estimate of alpha(q=0).
        result["Alpha_0"] = float(np.mean(alphas))
    else:
        result["Width"] = result["Alpha_0"] = np.nan
    return result

# GLAM_RDF_FAST_PATCH_V2

# GLAM_BATCHED_SMOOTHING_INTEGRATION_V7

# GLAM_RDF_SHAPE_BATCH_V8

# GLAM_CPU_REDUCTION_ONLY_V1

# GLAM_PD_COMMON_DOMAIN_ANISO_INVALID_V1

# GLAM_GPU_RDF_OFFLOAD_V1

# GLAM_CDI_GUARD_VIZ_ROBUST_V1
