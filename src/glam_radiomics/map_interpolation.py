"""Conservative trilinear reconstruction of GLAM's regular sampled grid
with isolated single-node healing and pinhole-tolerant interior cells.
"""
import itertools
import os
import time
import numpy as np


def interpolate_grid(sampled, mask, strides):
    """Fill adjacent cells on the sampled grid.
    Heals isolated 1-node interior dropouts (e.g. from local single-bin saturation
    in OffDiagMean) using 6-connected coarse neighbors and tolerates minor mask
    pinholes inside interior cells while strictly masking out[~mask] = np.nan.
    """
    sampled = np.asarray(sampled)
    mask = np.asarray(mask) > 0
    if sampled.ndim != 3 or sampled.shape != mask.shape:
        raise ValueError('Interpolation requires matching 3D map/mask arrays')
    if len(strides) != 3 or any(int(s) != s or s < 1 for s in strides):
        raise ValueError('Interpolation strides must be three positive integers')
    strides = tuple(int(s) for s in strides)
    axes = [np.arange(0, d, s) for d, s in zip(mask.shape, strides)]
    out = np.full(sampled.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(sampled) & mask
    out[finite] = sampled[finite]
    if any(len(a) < 2 for a in axes):
        return out, 0

    coarse = sampled[np.ix_(*axes)].astype(np.float64, copy=True)
    coarse_mask = mask[np.ix_(*axes)]

    # Heal isolated interior NaN grid nodes that lie inside the mask and have >= 3
    # finite 6-connected coarse grid neighbors (up to 2 passes for 2-node pairs).
    for _ in range(2):
        missing = ~np.isfinite(coarse) & coarse_mask
        if not np.any(missing):
            break
        neighbor_sum = np.zeros_like(coarse, dtype=np.float64)
        neighbor_cnt = np.zeros(coarse.shape, dtype=np.int32)
        for axis in range(3):
            for delta in (-1, 1):
                sl_src = [slice(None)] * 3
                sl_dst = [slice(None)] * 3
                if delta == -1:
                    sl_src[axis] = slice(0, -1)
                    sl_dst[axis] = slice(1, None)
                else:
                    sl_src[axis] = slice(1, None)
                    sl_dst[axis] = slice(0, -1)
                val_src = coarse[tuple(sl_src)]
                fin_src = np.isfinite(val_src)
                neighbor_sum[tuple(sl_dst)] += np.where(fin_src, val_src, 0.0)
                neighbor_cnt[tuple(sl_dst)] += fin_src.astype(np.int32)
        can_heal = missing & (neighbor_cnt >= 3)
        if not np.any(can_heal):
            break
        coarse[can_heal] = neighbor_sum[can_heal] / neighbor_cnt[can_heal]

    corners_valid = np.ones(tuple(len(a) - 1 for a in axes), dtype=bool)
    for bits in itertools.product((0, 1), repeat=3):
        sl = tuple(slice(b, b + n) for b, n in zip(bits, corners_valid.shape))
        corners_valid &= np.isfinite(coarse[sl])

    integral = np.pad(mask.astype(np.int64), ((1, 0),) * 3)
    for axis in range(3):
        np.cumsum(integral, axis=axis, out=integral)

    def count_box(lo, hi):
        count = 0
        for bits in itertools.product((0, 1), repeat=3):
            idx = tuple(hi[a] + 1 if bits[a] else lo[a] for a in range(3))
            count += (-1) ** (3 - sum(bits)) * int(integral[idx])
        return count

    weights = []
    for s in strides:
        t = np.arange(s + 1, dtype=float) / s
        weights.append((1 - t, t))

    cells = 0
    for index in np.argwhere(corners_valid):
        lo = tuple(int(axes[a][index[a]]) for a in range(3))
        hi = tuple(int(axes[a][index[a] + 1]) for a in range(3))
        cell_vol = int(np.prod(np.subtract(hi, lo) + 1))
        # Allow interpolation if >= 75% of the 4x4x4 cell is in-mask (tolerating
        # 1-voxel mask pinholes); out[~mask] = np.nan below strictly clips to mask.
        if count_box(lo, hi) < max(1, int(0.75 * cell_vol)):
            continue
        block = np.zeros(tuple(s + 1 for s in strides), dtype=np.float64)
        for bits in itertools.product((0, 1), repeat=3):
            value = coarse[tuple(index[a] + bits[a] for a in range(3))]
            block += (
                value
                * weights[0][bits[0]][:, None, None]
                * weights[1][bits[1]][None, :, None]
                * weights[2][bits[2]][None, None, :]
            )
        sl = tuple(slice(l, h + 1) for l, h in zip(lo, hi))
        out[sl] = block.astype(np.float32)
        cells += 1

    # Restore original finite sampled nodes and strictly mask outside ROI
    out[finite] = sampled[finite]
    out[~mask] = np.nan
    return out, cells


def save_interpolated_map(sampled, mask, strides, reference, prefix, name,
                          output_dir, save_visualization=False):
    import SimpleITK as sitk
    start = time.perf_counter()
    dense, cells = interpolate_grid(sampled, mask, strides)
    finite = np.isfinite(dense)
    new_count = int(finite.sum()) - int((np.isfinite(sampled) & (np.asarray(mask) > 0)).sum())
    if new_count <= 0:
        print(f'  - WARNING: {name}: no supported interpolation cells; sampled map retained.')
        return
    image = sitk.GetImageFromArray(dense)
    image.CopyInformation(reference)
    stem = f'{prefix}_MAP_{name}_Interpolated'
    sitk.WriteImage(image, os.path.join(output_dir, stem + '.nii.gz'))
    if save_visualization:
        scaled = np.zeros(dense.shape, dtype=np.uint8)
        lo, hi = float(dense[finite].min()), float(dense[finite].max())
        scaled[finite] = (1 + 254 * (dense[finite] - lo) / (hi - lo)).astype(np.uint8) if hi - lo > 1e-6 else 128
        viz = sitk.GetImageFromArray(scaled)
        viz.CopyInformation(reference)
        sitk.WriteImage(viz, os.path.join(output_dir, stem + '_uint8.nii.gz'))
    print(f'  - Saved interpolated map: {stem}.nii.gz; {cells} supported cells, '
          f'{new_count} added voxels; elapsed {time.perf_counter()-start:.2f}s')


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

# GLAM_CDI_GUARD_VIZ_ROBUST_V1
