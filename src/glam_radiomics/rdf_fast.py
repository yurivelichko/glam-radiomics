"""CPU RDF acceleration for GLAM with volume-weighted reference pooling,
scale-aware shell support, explicit geometric support metadata propagation,
and focal reference sub-mask (`sample_mask`) inside a bounded target domain.
"""
import numpy as np
import pandas as pd
from scipy.fft import rfftn, irfftn, next_fast_len
from .rdf_smoothing import attach_geometric_support

_LAST_GEOMETRY = None
_CACHE_BUILDS = 0
_MAX_KERNEL_CACHE_BYTES = 128 * 1024 * 1024


def clear_geometry_cache():
    global _LAST_GEOMETRY, _CACHE_BUILDS
    _LAST_GEOMETRY = None
    _CACHE_BUILDS = 0


def cache_info():
    return {'geometry_builds': _CACHE_BUILDS,
            'cached': _LAST_GEOMETRY is not None}


def _shell_kernel(radius, shape):
    extent = np.minimum(radius, np.asarray(shape) - 1).astype(int)
    z, y, x = np.ogrid[tuple(slice(-e, e + 1) for e in extent)]
    d2 = z*z + y*y + x*x
    return ((d2 >= (radius - 0.5)**2) &
            (d2 < (radius + 0.5)**2)).astype(np.float64)


class _Geometry:
    def __init__(self, mask, radius):
        self.mask = mask.copy()
        self.radius = radius
        self.shape = mask.shape
        self.coords = np.argwhere(mask)
        self.indices = tuple(self.coords.T)
        self.extent = np.minimum(radius, np.asarray(self.shape) - 1).astype(int)
        self.kernel_shape = tuple(2*self.extent + 1)
        self.fft_shape = tuple(next_fast_len(d + k - 1)
                               for d, k in zip(self.shape, self.kernel_shape))
        self.crop = tuple(slice(int(e), int(e)+d)
                          for e, d in zip(self.extent, self.shape))
        self.kernels = {}
        self.cached_bytes = 0
        self.denominators = np.zeros((radius, len(self.coords)), dtype=np.float64)
        mask_fft = rfftn(mask.astype(np.float64), self.fft_shape, workers=1)
        for r in range(1, radius + 1):
            kfft = self.kernel_fft(r)
            counts = self.counts(mask_fft, kfft)
            self.denominators[r-1] = counts[self.indices]

    def kernel_fft(self, radius):
        if radius in self.kernels:
            return self.kernels[radius]
        kernel = _shell_kernel(radius, self.shape)
        pad = tuple((int(e)-(k-1)//2, int(e)-(k-1)//2)
                    for e, k in zip(self.extent, kernel.shape))
        kernel = np.pad(kernel, pad)
        result = rfftn(kernel, self.fft_shape, workers=1)
        if self.cached_bytes + result.nbytes <= _MAX_KERNEL_CACHE_BYTES:
            self.kernels[radius] = result
            self.cached_bytes += result.nbytes
        return result

    def counts(self, image_fft, kernel_fft):
        raw = irfftn(image_fft * kernel_fft, self.fft_shape, workers=1)[self.crop]
        rounded = np.rint(raw)
        if np.max(np.abs(raw-rounded), initial=0.0) > 1e-4:
            raise FloatingPointError('FFT shell counts exceed integer-rounding tolerance')
        return np.maximum(rounded, 0.0)


def _geometry(mask, radius):
    global _LAST_GEOMETRY, _CACHE_BUILDS
    old = _LAST_GEOMETRY
    if (old is not None and old.radius == radius and old.shape == mask.shape
            and np.array_equal(old.mask, mask)):
        return old
    _LAST_GEOMETRY = None
    old = None
    result = _Geometry(mask, radius)
    _LAST_GEOMETRY = result
    _CACHE_BUILDS += 1
    return result


def _ideal_shell_volumes(max_radius):
    """Exact unclipped 3D lattice shell volumes V_ideal(r) for r = 1..max_radius."""
    r_max = int(max_radius)
    z, y, x = np.ogrid[-r_max:r_max + 1, -r_max:r_max + 1, -r_max:r_max + 1]
    d = np.rint(np.sqrt(z * z + y * y + x * x)).astype(np.int32)
    counts = np.bincount(d.ravel(), minlength=r_max + 1)
    return counts[1:r_max + 1].astype(np.float64)


def _support_params():
    from . import config as cfg
    parsed = getattr(cfg, 'parsed_config', {}) or {}
    min_support_n = int(parsed.get('RdfMinSupportedReferences', 50))
    min_support_fraction = float(parsed.get('RdfMinSupportedFraction', 0.01))
    min_median_volume = float(parsed.get('RdfMinMedianShellVoxels', 20.0))
    min_ref_volume = max(1, int(parsed.get('RdfMinReferenceShellVoxels', 1)))
    min_shell_fraction = float(parsed.get('RdfMinShellVolumeFraction', 0.05))
    return min_support_n, min_support_fraction, min_median_volume, min_ref_volume, min_shell_fraction


def _required_median_volumes(max_radius, min_median_volume, min_shell_fraction):
    v_ideal = _ideal_shell_volumes(max_radius)
    capped_floor = np.minimum(float(min_median_volume), 0.5 * v_ideal)
    return np.maximum(capped_floor, float(min_shell_fraction) * v_ideal)


def _resolve_target_and_ref_masks(image, sample_mask, total_roi_voxels):
    """Resolve target support mask (`target_mask`, size == total_roi_voxels)
    and reference voxel mask (`ref_mask <= target_mask`).
    """
    nonneg = image >= 0
    if sample_mask is None:
        return nonneg, nonneg
    sm = np.asarray(sample_mask) > 0
    if sm.shape != image.shape:
        raise ValueError('RDF mask/image shape mismatch')
    n_nonneg = int(nonneg.sum())
    if n_nonneg == int(total_roi_voxels) and np.all(sm <= nonneg):
        return nonneg, (nonneg & sm)
    return sm, sm


def calculate_rdf_3d(image_3d, num_levels, max_radius, level_counts,
                     total_roi_voxels, num_randomisations, rdf_sample_points,
                     sample_mask=None):
    """Drop-in CPU RDF with volume-weighted reference pooling, scale-aware shell
    support, explicit geometric support metadata, and focal reference sub-mask support.
    """
    image = np.asarray(image_3d)
    if image.ndim != 3:
        raise ValueError('RDF requires a 3D image')
    for name, value in [('num_levels', num_levels), ('max_radius', max_radius),
                        ('rdf_sample_points', rdf_sample_points)]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    levels, radius = int(num_levels), int(max_radius)
    target_mask, ref_mask = _resolve_target_and_ref_masks(image, sample_mask, total_roi_voxels)
    values = image[target_mask]
    if not values.size or values.size != total_roi_voxels:
        raise ValueError('Empty RDF ROI or total_roi_voxels mismatch')
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values >= levels) or np.any(values != np.floor(values))):
        raise ValueError('Invalid RDF gray levels')

    labels = values.astype(np.int64)
    counts = np.bincount(labels, minlength=levels)
    is_ref = ref_mask[target_mask]
    geom = _geometry(target_mask, radius)

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
            volumes = geom.denominators[r-1, alpha_ids]
            valid_mask = volumes >= min_ref_volume
            supported = volumes[valid_mask]
            total_shell_vol = float(np.sum(supported))
            valid = (
                supported.size >= min_support_n
                and supported.size / alpha_ids.size >= min_support_fraction
                and total_shell_vol > 0.0
                and float(np.median(supported)) >= req_median_by_r[r-1]
            )
            if valid:
                alpha_radius_valid[r-1, alpha] = True
                ref_weights[r-1, alpha_ids[valid_mask]] = 1.0 / total_shell_vol
            else:
                tail_failed = True

    any_valid_at_r = np.any(alpha_radius_valid, axis=1)
    result = np.zeros((radius, levels, levels), dtype=np.float64)
    for beta in range(levels):
        if not counts[beta]:
            continue
        beta_mask = target_mask & (image == beta)
        beta_fft = rfftn(beta_mask.astype(np.float64), geom.fft_shape, workers=1)
        rho = counts[beta] / values.size
        for r in range(1, radius + 1):
            if not any_valid_at_r[r-1]:
                continue
            neighbor_counts = geom.counts(beta_fft, geom.kernel_fft(r))[geom.indices]
            result[r-1, :, beta] = np.bincount(
                labels,
                weights=neighbor_counts * ref_weights[r-1],
                minlength=levels,
            ) / rho

    r_vec = np.arange(1, radius + 1, dtype=np.float64)
    cols = ['r'] + [f'g_{a}_{b}' for a in range(levels) for b in range(levels)]
    flat_pairs = result.reshape(radius, levels * levels)
    df = pd.DataFrame(np.hstack((r_vec[:, None], flat_pairs)), columns=cols, copy=False)
    df.attrs['_glam_fast_raw'] = (r_vec, np.ascontiguousarray(flat_pairs.T))
    return attach_geometric_support(df, alpha_radius_valid, counts)
