"""CPU RDF acceleration for GLAM priority-fix v1.

Centered lattice shells and per-reference boundary correction are unchanged.
No SimpleITK/CUDA imports. One window geometry is cached per process.
"""
import numpy as np
import pandas as pd
from scipy.fft import rfftn, irfftn, next_fast_len

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
    # Offsets beyond the image extent cannot connect two voxels. Trimming
    # these zero-contribution offsets reduces FFT size without changing counts.
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
        # Center all shells in the same support, so a common FFT shape/crop works.
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
        # These are convolutions of binary arrays: the exact answer is integer.
        # Fail rather than silently round a numerically unstable transform.
        if np.max(np.abs(raw-rounded), initial=0.0) > 1e-4:
            raise FloatingPointError('FFT shell counts exceed integer-rounding tolerance')
        return np.maximum(rounded, 0.0)


def _geometry(mask, radius):
    global _LAST_GEOMETRY, _CACHE_BUILDS
    old = _LAST_GEOMETRY
    if (old is not None and old.radius == radius and old.shape == mask.shape
            and np.array_equal(old.mask, mask)):
        return old
    # Release the old window before allocating the next one.
    _LAST_GEOMETRY = None
    old = None
    result = _Geometry(mask, radius)
    _LAST_GEOMETRY = result
    _CACHE_BUILDS += 1
    return result


def calculate_rdf_3d(image_3d, num_levels, max_radius, level_counts,
                     total_roi_voxels, num_randomisations, rdf_sample_points,
                     sample_mask=None):
    """Drop-in CPU RDF, retaining priority-fix v1 sampling and normalization.

    num_randomisations remains a compatibility argument; the caller generates
    shuffled images. Empty states and unsupported shells retain legacy zeros.
    """
    image = np.asarray(image_3d)
    if image.ndim != 3:
        raise ValueError('RDF requires a 3D image')
    for name, value in [('num_levels', num_levels), ('max_radius', max_radius),
                        ('rdf_sample_points', rdf_sample_points)]:
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    levels, radius, samples = int(num_levels), int(max_radius), int(rdf_sample_points)
    mask = image >= 0 if sample_mask is None else np.asarray(sample_mask) > 0
    if mask.shape != image.shape:
        raise ValueError('RDF mask/image shape mismatch')
    values = image[mask]
    if not values.size or values.size != total_roi_voxels:
        raise ValueError('Empty RDF ROI or total_roi_voxels mismatch')
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values >= levels) or np.any(values != np.floor(values))):
        raise ValueError('Invalid RDF gray levels')
    labels = values.astype(np.int64)
    counts = np.bincount(labels, minlength=levels)
    geom = _geometry(mask, radius)
    # Match v1 np.random.choice calls, in ascending alpha order. Coordinates
    # are lexicographic np.argwhere order, identical to the reference routine.
    refs = {}
    for alpha in range(levels):
        ids = np.flatnonzero(labels == alpha)
        if ids.size:
            selected = np.random.choice(ids.size, min(ids.size, samples), replace=False)
            refs[alpha] = ids[selected]
    result = np.zeros((radius, levels, levels), dtype=np.float64)
    for beta in range(levels):
        if not counts[beta]:
            continue
        beta_mask = mask & (image == beta)
        beta_fft = rfftn(beta_mask.astype(np.float64), geom.fft_shape, workers=1)
        rho = counts[beta] / values.size
        for r in range(1, radius + 1):
            neighbor_counts = geom.counts(beta_fft, geom.kernel_fft(r))[geom.indices]
            denominators = geom.denominators[r-1]
            for alpha, ids in refs.items():
                den = denominators[ids]
                valid = den > 0
                if np.any(valid):
                    result[r-1, alpha, beta] = np.mean(
                        neighbor_counts[ids[valid]] / den[valid]) / rho
    data = {'r': np.arange(1, radius+1)}
    for alpha in range(levels):
        for beta in range(levels):
            data[f'g_{alpha}_{beta}'] = result[:, alpha, beta]
    return pd.DataFrame(data)
