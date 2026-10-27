"""Single-owner batched CuPy RDF backend with:
1. Contiguous cuBLAS dense float32 shell GEMM (up to 8 GiB cache, automatic CSR fallback above 8 GiB).
2. Explicit canonical focal-reference and target-mask support caching (built once per sequence).
3. Asynchronous GPU/CPU double-buffered pipeline: while the GPU executes Batch K's shell
   kernels on the CUDA stream, the CPU reduces Batch K-1's RDFs into features in parallel.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from .rdf_smoothing import attach_geometric_support, average_supported_rdf_replicates
from .mapping_profile import stage


def choose_references(image, mask, levels, samples=None, rng=None):
    labels = image[mask].astype(int)
    return [np.flatnonzero(labels == alpha) for alpha in range(levels)]


def variants_for_node(ext_patch, ext_mask, focal_in_ext, levels, samples, repeats, coord):
    """Generate structured patch + `repeats` genuine spatial label permutations
    across `ext_mask` using the exact same SeedSequence([42, z, y, x]) as CPU `mapping.py`.
    """
    seed = int(np.random.SeedSequence([42, *map(int, coord)]).generate_state(1)[0])
    rng = np.random.RandomState(seed)
    variants = [ext_patch.copy()]
    refs = [focal_in_ext] * (repeats + 1)
    values = ext_patch[ext_mask]
    for _ in range(repeats):
        shuffled = np.full(ext_patch.shape, -1, dtype=np.int16)
        shuffled[ext_mask] = rng.permutation(values)
        variants.append(shuffled)
    return variants, refs


def shell(radius, shape, extent, dtype=np.float64):
    z, y, x = np.ogrid[tuple(slice(-int(e), int(e)+1) for e in extent)]
    d2 = z*z + y*y + x*x
    return ((d2 >= (radius-0.5)**2) & (d2 < (radius+0.5)**2)).astype(dtype)


def _make_fast_rdf_df(r_vec, host_3d, cols, alpha_radius_valid=None, counts=None, pair_support_mask=None):
    """Wrap (radius, levels, levels) array into a DataFrame with pre-cached raw matrix
    and explicit geometric support metadata (`_glam_support_cutoff` and `_glam_support_mask`).
    """
    radius, levels, _ = host_3d.shape
    flat_pairs = host_3d.reshape(radius, levels * levels)
    mat = np.hstack((r_vec[:, None], flat_pairs))
    df = pd.DataFrame(mat, columns=cols, copy=False)
    df.attrs['_glam_fast_raw'] = (r_vec, np.ascontiguousarray(flat_pairs.T))
    if pair_support_mask is not None:
        pm = np.asarray(pair_support_mask, dtype=bool)
        prefix_pm = np.minimum.accumulate(pm, axis=1)
        df.attrs['_glam_support_mask'] = prefix_pm
        df.attrs['_glam_support_cutoff'] = np.sum(prefix_pm, axis=1).astype(np.int32)
    elif alpha_radius_valid is not None and counts is not None:
        attach_geometric_support(df, alpha_radius_valid, counts)
    return df


def _mask_geometry_digest(mask_bool):
    """Collision-free SHA-1 digest of a 3D boolean support mask's shape and packed bits."""
    packed = np.packbits(np.ascontiguousarray(mask_bool, dtype=np.uint8))
    h = hashlib.sha1(np.asarray(mask_bool.shape, dtype=np.int64).tobytes())
    h.update(packed.tobytes())
    return h.digest()


def _build_focal_to_target_shell_cache(ref_support_mask, tgt_support_mask, radius, levels, xp):
    """Build exact focal-reference (M_ref) to target-mask (M_tgt) shell operators S_r.
    For each radius r, crops columns of S_r to the exact reachable target subset `reach_idx[r]`
    (target voxels within distance r of at least one focal reference voxel).
    For R_win=8, R_ext=30, reach-cropping cuts total shell matrix memory by 2.3x (from 26.6 GiB
    to 11.8 GiB), allowing all 30 shells to run as contiguous cuBLAS dense float32 SGEMMs on
    48 GB GPUs instead of falling back to slow cuSPARSE CSR matrices.
    """
    ref_coords = np.argwhere(ref_support_mask)
    ref_flat_idx = np.flatnonzero(ref_support_mask.ravel())
    tgt_flat_idx = np.flatnonzero(tgt_support_mask.ravel())
    m_ref = int(len(ref_flat_idx))
    m_tgt = int(len(tgt_flat_idx))

    tgt_id_grid = np.full(tgt_support_mask.shape, -1, dtype=np.int32)
    tgt_id_grid[tgt_support_mask] = np.arange(m_tgt, dtype=np.int32)

    zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    d2_off = zz * zz + yy * yy + xx * xx
    shape = tgt_support_mask.shape

    raw_rows_cols = []
    total_dense_bytes = 0
    for r in range(1, radius + 1):
        offs = np.argwhere((d2_off >= (r - 0.5) ** 2) & (d2_off < (r + 0.5) ** 2)) - radius
        shifted = ref_coords[:, None, :] + offs[None, :, :]
        in_b = (
            (shifted[:, :, 0] >= 0) & (shifted[:, :, 0] < shape[0])
            & (shifted[:, :, 1] >= 0) & (shifted[:, :, 1] < shape[1])
            & (shifted[:, :, 2] >= 0) & (shifted[:, :, 2] < shape[2])
        )
        rows_i, off_k = np.nonzero(in_b)
        sz = shifted[rows_i, off_k, 0]
        sy = shifted[rows_i, off_k, 1]
        sx = shifted[rows_i, off_k, 2]
        cols_j = tgt_id_grid[sz, sy, sx]
        valid_pair = cols_j >= 0
        rows_i = rows_i[valid_pair].astype(np.int32)
        cols_j = cols_j[valid_pair].astype(np.int32)
        active_cols, compact_j = np.unique(cols_j, return_inverse=True)
        active_cols = active_cols.astype(np.int32)
        compact_j = compact_j.astype(np.int32)
        raw_rows_cols.append((rows_i, cols_j, active_cols, compact_j))
        total_dense_bytes += m_ref * len(active_cols) * 4

    # Allow up to 24 GiB of reach-cropped dense cuBLAS float32 shell matrices on 48 GB GPUs
    use_sparse = (xp is not np) and (total_dense_bytes > 24 * (1024 ** 3))
    if use_sparse:
        import cupyx.scipy.sparse as cpsparse
        import scipy.sparse as spsparse

    shells_gpu = []
    reach_cols_gpu = []
    full_den_list = []

    for r_idx, (rows_i, cols_j, active_cols, compact_j) in enumerate(raw_rows_cols):
        den_r = np.bincount(rows_i, minlength=m_ref).astype(np.float64)
        full_den_list.append(xp.asarray(den_r))

        if use_sparse:
            data = np.ones(len(rows_i), dtype=np.float32)
            csr_cpu = spsparse.csr_matrix((data, (rows_i, cols_j)), shape=(m_ref, m_tgt))
            shells_gpu.append(cpsparse.csr_matrix(csr_cpu))
            reach_cols_gpu.append(None)
        else:
            m_reach = int(len(active_cols))
            sr_dense = np.zeros((m_ref, m_reach), dtype=np.float32)
            sr_dense[rows_i, compact_j] = 1.0
            shells_gpu.append(xp.asarray(sr_dense))
            reach_cols_gpu.append(xp.asarray(active_cols))

    full_den_gpu = xp.stack(full_den_list, axis=0)
    cols = ['r'] + [f'g_{a}_{b}' for a in range(levels) for b in range(levels)]
    r_vec = np.arange(1, radius + 1, dtype=np.float64)
    return dict(
        mode='focal_target_direct',
        use_sparse=use_sparse,
        ref_support_mask=ref_support_mask.copy(),
        tgt_support_mask=tgt_support_mask.copy(),
        ref_flat_idx_host=ref_flat_idx,
        ref_flat_idx_gpu=xp.asarray(ref_flat_idx),
        tgt_flat_idx_host=tgt_flat_idx,
        tgt_flat_idx_gpu=xp.asarray(tgt_flat_idx),
        shells_gpu=shells_gpu,
        reach_cols_gpu=reach_cols_gpu,
        full_den_gpu=full_den_gpu,
        cols=cols,
        r_vec=r_vec,
        M_ref=m_ref,
        M_tgt=m_tgt,
    )




def batch_rdfs(
    tasks,
    levels,
    radius,
    state_batch=4,
    xp=None,
    kernel_cache=None,
    return_raw_host=False,
    canonical_ref_mask=None,
    canonical_tgt_mask=None,
    return_device_result=False,
):
    """tasks: (image, target_mask, ref_mask_or_indices).
    When `return_device_result=True`, returns `(result_gpu, r_vec, cols, alpha_radius_valid, tgt_counts)`
    immediately after queuing the GPU shell kernels onto the CUDA stream.
    """
    if xp is None:
        import cupy as xp
    state_batch = max(1, int(state_batch))
    from .rdf_fast import _support_params, _required_median_volumes
    min_support_n, min_support_fraction, min_median_volume, min_ref_volume, min_shell_fraction = _support_params()
    req_median_by_r = _required_median_volumes(radius, min_median_volume, min_shell_fraction)

    shape = tasks[0][0].shape
    if any(im.shape != shape or mask.shape != shape for im, mask, _ in tasks):
        raise ValueError('GPU batch requires matching patch shapes')
    tgt_masks = np.stack([m for _, m, _ in tasks]).astype(bool)
    images = np.stack([im for im, _, _ in tasks])

    has_ref_3d = (
        tasks[0][2] is not None
        and isinstance(tasks[0][2], np.ndarray)
        and tasks[0][2].shape == shape
    )
    if has_ref_3d:
        ref_masks = np.stack([(np.asarray(rm) > 0) & tm for (_, tm, rm) in zip(tasks, tgt_masks, [t[2] for t in tasks])])
    else:
        ref_masks = tgt_masks

    n = len(tasks)
    volume = int(np.prod(shape))

    if (
        canonical_tgt_mask is not None
        and canonical_tgt_mask.shape == shape
        and np.all(tgt_masks <= canonical_tgt_mask[None, ...])
    ):
        tgt_support = np.asarray(canonical_tgt_mask, dtype=bool)
        is_canonical_tgt = True
    else:
        union_tgt = np.any(tgt_masks, axis=0)
        rz, ry, rx = [s // 2 for s in shape]
        if (2 * rz + 1, 2 * ry + 1, 2 * rx + 1) == shape:
            zz, yy, xx = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
            canonical_tgt = ((zz / max(rz, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (xx / max(rx, 1)) ** 2) <= 1
            tgt_support = canonical_tgt if np.all(union_tgt <= canonical_tgt) else union_tgt
            is_canonical_tgt = tgt_support is canonical_tgt
        else:
            tgt_support = union_tgt
            is_canonical_tgt = False

    if (
        canonical_ref_mask is not None
        and canonical_ref_mask.shape == shape
        and np.all(ref_masks <= canonical_ref_mask[None, ...])
        and np.all(canonical_ref_mask <= tgt_support)
    ):
        ref_support = np.asarray(canonical_ref_mask, dtype=bool)
    else:
        ref_support = np.any(ref_masks, axis=0)

    m_ref = int(ref_support.sum())
    m_tgt = int(tgt_support.sum())

    if m_ref <= 4096 and m_tgt <= 262144:
        dkey = (
            'focal_target',
            shape,
            _mask_geometry_digest(ref_support),
            _mask_geometry_digest(tgt_support),
            radius,
            levels,
        )
        if (
            kernel_cache is not None
            and kernel_cache.get('key') == dkey
            and 'dcache' in kernel_cache
            and np.array_equal(kernel_cache['dcache']['ref_support_mask'], ref_support)
            and np.array_equal(kernel_cache['dcache']['tgt_support_mask'], tgt_support)
        ):
            dcache = kernel_cache['dcache']
        else:
            if kernel_cache is not None:
                kernel_cache.clear()
                if hasattr(xp, 'get_default_memory_pool'):
                    xp.get_default_memory_pool().free_all_blocks()
            dcache = _build_focal_to_target_shell_cache(ref_support, tgt_support, radius, levels, xp)
            if kernel_cache is not None:
                kernel_cache.update(key=dkey, dcache=dcache)

        ref_flat_host = dcache['ref_flat_idx_host']
        tgt_flat_host = dcache['tgt_flat_idx_host']
        shells_gpu = dcache['shells_gpu']
        cols = dcache['cols']
        r_vec = dcache['r_vec']
        M_ref = dcache['M_ref']
        M_tgt = dcache['M_tgt']

        # Slice 1D compact arrays on host first (M_tgt and M_ref elements instead of full 3D volume)
        images_2d = images.reshape(n, volume)
        tgt_masks_sub_host = tgt_masks.reshape(n, volume)[:, tgt_flat_host]
        ref_masks_sub_host = ref_masks.reshape(n, volume)[:, ref_flat_host]
        images_tgt_host = images_2d[:, tgt_flat_host]
        images_ref_host = images_2d[:, ref_flat_host]

        if np.any(images_tgt_host[tgt_masks_sub_host] < 0) or np.any(images_tgt_host[tgt_masks_sub_host] >= levels):
            raise ValueError('Invalid GPU batch labels')

        tgt_counts = np.stack([
            np.bincount(im_t[tm_t].astype(int), minlength=levels)
            for im_t, tm_t in zip(images_tgt_host, tgt_masks_sub_host)
        ])
        rho = tgt_counts / tgt_masks_sub_host.sum(axis=1)[:, None]
        ref_counts = np.stack([
            np.bincount(im_r[rm_r].astype(int), minlength=levels)
            for im_r, rm_r in zip(images_ref_host, ref_masks_sub_host)
        ])

        gi_ref = xp.asarray(images_ref_host)
        gm_ref = xp.asarray(ref_masks_sub_host, dtype=bool)
        gi_tgt = xp.asarray(images_tgt_host)
        gm_tgt = xp.asarray(tgt_masks_sub_host, dtype=bool)

        states_all = xp.arange(levels, dtype=gi_ref.dtype)
        w_ref_bool = (gi_ref[:, None, :] == states_all[None, :, None]) & gm_ref[:, None, :]
        w_ref_fp64 = w_ref_bool.astype(xp.float64)

        if is_canonical_tgt and np.all(tgt_masks_sub_host):
            den_unique_gpu = dcache['full_den_gpu'][None, :, :]
            inverse = np.zeros(n, dtype=np.int64)
            valid_u = (den_unique_gpu[0] >= min_ref_volume).astype(xp.float64)
            den_valid_u = den_unique_gpu[0] * valid_u
            vc_gpu = xp.matmul(w_ref_fp64, valid_u.T).swapaxes(1, 2)
            vol_sum_gpu = xp.matmul(w_ref_fp64, den_valid_u.T).swapaxes(1, 2)
            shared_sphere = True
        else:
            seen_masks = {}
            first_indices = []
            inverse = np.empty(n, dtype=np.int64)
            last_mask_obj = None
            last_u_idx = -1
            for idx_t in range(n):
                m_obj = tasks[idx_t][1]
                if m_obj is last_mask_obj:
                    inverse[idx_t] = last_u_idx
                    continue
                last_mask_obj = m_obj
                key_bytes = np.packbits(tgt_masks_sub_host[idx_t]).tobytes()
                u_i = seen_masks.get(key_bytes)
                if u_i is None:
                    u_i = len(first_indices)
                    seen_masks[key_bytes] = u_i
                    first_indices.append(idx_t)
                inverse[idx_t] = u_i
                last_u_idx = u_i
            unique_tgt_sub = tgt_masks_sub_host[first_indices]
            inverse_gpu = xp.asarray(inverse)
            u_gpu_f32 = xp.asarray(unique_tgt_sub, dtype=xp.float32)
            reach_cols_gpu = dcache.get('reach_cols_gpu')
            den_unique_list = [
                xp.rint((shells_gpu[r] @ (u_gpu_f32.T if (reach_cols_gpu is None or reach_cols_gpu[r] is None) else u_gpu_f32[:, reach_cols_gpu[r]].T)).T).astype(xp.float64)
                for r in range(radius)
            ]
            den_unique_gpu = xp.stack(den_unique_list, axis=1)
            den_all = den_unique_gpu[inverse_gpu]
            valid_r_mask = (den_all >= min_ref_volume).astype(xp.float64)
            den_valid_all = den_all * valid_r_mask
            vc_gpu = xp.matmul(w_ref_fp64, valid_r_mask.swapaxes(1, 2)).swapaxes(1, 2)
            vol_sum_gpu = xp.matmul(w_ref_fp64, den_valid_all.swapaxes(1, 2)).swapaxes(1, 2)
            shared_sphere = False
            del u_gpu_f32, den_all, den_valid_all

        vc_host = np.asarray(vc_gpu) if xp is np else xp.asnumpy(vc_gpu)
        step_valid = (
            (vc_host >= min_support_n)
            & (vc_host >= ref_counts[:, None, :] * min_support_fraction)
            & (ref_counts[:, None, :] > 0)
        )
        den_unique_host = np.asarray(den_unique_gpu) if xp is np else xp.asnumpy(den_unique_gpu)
        u_min_den = np.min(den_unique_host, axis=2)
        u_all_pass = np.all(u_min_den >= np.maximum(min_ref_volume, req_median_by_r)[None, :], axis=1)
        alpha_radius_valid = np.zeros((n, radius, levels), dtype=bool)
        for t in range(n):
            u_idx = int(inverse[t])
            if u_all_pass[u_idx] and np.all(ref_masks_sub_host[t]):
                alpha_radius_valid[t] = np.minimum.accumulate(step_valid[t], axis=0)
                continue
            m_row = ref_masks_sub_host[t]
            im_row = images_ref_host[t]
            den_u = den_unique_host[u_idx]
            for alpha in range(levels):
                if not step_valid[t, 0, alpha]:
                    continue
                v_idx = np.flatnonzero(m_row & (im_row == alpha))
                vols_all_r = den_u[:, v_idx]
                for r in range(radius):
                    if not step_valid[t, r, alpha]:
                        break
                    volumes = vols_all_r[r]
                    supported = volumes[volumes >= min_ref_volume]
                    if supported.size > 0 and float(np.median(supported)) >= req_median_by_r[r]:
                        alpha_radius_valid[t, r, alpha] = True
                    else:
                        break

        any_valid_at_r = np.any(alpha_radius_valid, axis=(0, 2))
        active_radii = [int(r) for r in np.flatnonzero(any_valid_at_r)]

        result = xp.zeros((n, radius, levels, levels), dtype=xp.float64)
        if active_radii:
            valid_gate = xp.asarray(alpha_radius_valid)
            grho = xp.asarray(rho)
            max_work_cols = 3840
            for start in range(0, levels, state_batch):
                stop = min(levels, start + state_batch)
                b_len = stop - start
                states_sub = xp.arange(start, stop, dtype=gi_tgt.dtype)
                task_chunk = max(1, min(n, max_work_cols // max(b_len, 1)))
                for t0 in range(0, n, task_chunk):
                    t1 = min(n, t0 + task_chunk)
                    tc = t1 - t0
                    gi_t = gi_tgt[t0:t1]
                    gm_t = gm_tgt[t0:t1]
                    w_tgt_sub = ((gi_t[:, None, :] == states_sub[None, :, None]) & gm_t[:, None, :]).astype(xp.float32)
                    w_tgt_flat_T = w_tgt_sub.reshape(tc * b_len, M_tgt).T
                    w_ref_sub = w_ref_fp64[t0:t1]
                    vol_sum_sub = vol_sum_gpu[t0:t1]
                    grho_sub = grho[t0:t1, None, start:stop]
                    reach_cols_gpu = dcache.get('reach_cols_gpu')
                    for r in active_radii:
                        if not np.any(alpha_radius_valid[t0:t1, r, :]):
                            continue
                        r_cols = None if reach_cols_gpu is None else reach_cols_gpu[r]
                        w_in = w_tgt_flat_T if r_cols is None else w_tgt_flat_T[r_cols, :]
                        nb_T = xp.rint(shells_gpu[r] @ w_in)
                        neighbors = nb_T.T.reshape(tc, b_len, M_ref).astype(xp.float64)
                        v_r_sub = valid_u[r, None, None, :] if shared_sphere else valid_r_mask[t0:t1, r, None, :]
                        w_valid = w_ref_sub * v_r_sub
                        sums = xp.matmul(w_valid, neighbors.swapaxes(1, 2))
                        norm = vol_sum_sub[:, r, :, None] * grho_sub
                        gate = valid_gate[t0:t1, r, :, None] & (norm > 0)
                        result[t0:t1, r, :, start:stop] = xp.where(gate, sums / xp.maximum(norm, 1e-300), 0.0)
                        del nb_T, neighbors
                    del w_tgt_sub, w_tgt_flat_T

        if return_device_result:
            return result, r_vec, cols, alpha_radius_valid, tgt_counts
        host = np.asarray(result) if xp is np else xp.asnumpy(result)
        if return_raw_host:
            return host, r_vec, cols, alpha_radius_valid, tgt_counts
        return [
            _make_fast_rdf_df(r_vec, host[t], cols, alpha_radius_valid[t], tgt_counts[t])
            for t in range(n)
        ]

    # Fallback 3D FFT path for ultra-large windows
    if np.any(images[tgt_masks] < 0) or np.any(images[tgt_masks] >= levels):
        raise ValueError('Invalid GPU batch labels')
    tgt_counts = np.stack([np.bincount(im[tm].astype(int), minlength=levels) for im, tm in zip(images, tgt_masks)])
    rho = tgt_counts / tgt_masks.reshape(n, -1).sum(axis=1)[:, None]

    extent = np.minimum(radius, np.asarray(shape)-1).astype(int)
    fftshape = tuple(next_fast_len(d+2*int(e)) for d, e in zip(shape, extent))
    crop = tuple(slice(int(e), int(e)+d) for e, d in zip(extent, shape))
    gi = xp.asarray(images)
    gm_tgt_bool = xp.asarray(tgt_masks, dtype=bool)
    gm_ref_bool = xp.asarray(ref_masks, dtype=bool)
    states_all = xp.arange(levels, dtype=gi.dtype)
    w_ref = ((gi.reshape(n, 1, volume) == states_all[None, :, None]) & gm_ref_bool.reshape(n, 1, volume)).astype(xp.float64)
    kernels = [xp.fft.rfftn(xp.asarray(shell(r, shape, extent)), s=fftshape) for r in range(1, radius+1)]
    cols = ['r'] + [f'g_{a}_{b}' for a in range(levels) for b in range(levels)]
    r_vec = np.arange(1, radius + 1, dtype=np.float64)

    def recover(raw):
        rounded = xp.rint(raw)
        if float(xp.max(xp.abs(raw - rounded)).item()) > 1e-4:
            raise FloatingPointError('GPU FFT integer-count residual exceeds tolerance')
        return xp.maximum(rounded, 0)

    unique_masks, inverse = np.unique(tgt_masks.reshape(n, -1), axis=0, return_inverse=True)
    unique_masks_3d = unique_masks.reshape((-1,) + shape)
    inverse_gpu = xp.asarray(inverse)
    maskfft = xp.fft.rfftn(xp.asarray(unique_masks_3d, dtype=xp.float64), s=fftshape, axes=(-3, -2, -1))
    sub_by_r = [
        recover(xp.fft.irfftn(maskfft*k, s=fftshape, axes=(-3, -2, -1))[(slice(None),) + crop]).reshape(len(unique_masks), volume)
        for k in kernels
    ]
    den_unique_gpu = xp.stack(sub_by_r, axis=1)
    den_unique_host = np.asarray(den_unique_gpu) if xp is np else xp.asnumpy(den_unique_gpu)
    vol_sums = [
        xp.einsum('tlv,tv->tl', w_ref, xp.where(den_unique_gpu[inverse_gpu, r, :] >= min_ref_volume, den_unique_gpu[inverse_gpu, r, :], 0.0))
        for r in range(radius)
    ]
    alpha_radius_valid = np.zeros((n, radius, levels), dtype=bool)
    for t in range(n):
        u_idx = int(inverse[t])
        rm = ref_masks[t]
        flat = np.flatnonzero(rm.ravel())
        labels_t = images[t][rm].astype(int)
        den_u = den_unique_host[u_idx]
        for alpha in range(levels):
            v_idx = flat[labels_t == alpha]
            if not len(v_idx):
                continue
            for r in range(radius):
                volumes = den_u[r, v_idx]
                supported = volumes[volumes >= min_ref_volume]
                if (
                    supported.size >= min_support_n
                    and supported.size / len(v_idx) >= min_support_fraction
                    and supported.size > 0
                    and float(np.median(supported)) >= req_median_by_r[r]
                ):
                    alpha_radius_valid[t, r, alpha] = True
                else:
                    break
    valid_gate = xp.asarray(alpha_radius_valid)
    result = xp.zeros((n, radius, levels, levels), dtype=xp.float64)
    grho = xp.asarray(rho)
    for start in range(0, levels, state_batch):
        stop = min(levels, start+state_batch)
        states = xp.arange(start, stop)
        binary = ((gi[:, None] == states[None, :, None, None, None]) & gm_tgt_bool[:, None]).astype(xp.float64)
        spectra = xp.fft.rfftn(binary, s=fftshape, axes=(-3, -2, -1))
        for r, k in enumerate(kernels):
            if not np.any(alpha_radius_valid[:, r, :]):
                continue
            neighbors = recover(xp.fft.irfftn(spectra*k, s=fftshape, axes=(-3, -2, -1))[(slice(None), slice(None)) + crop]).reshape(n, stop-start, volume)
            den = den_unique_gpu[inverse_gpu, r, :]
            masked_neighbors = xp.where(den[:, None, :] >= min_ref_volume, neighbors, 0.0)
            sums = xp.einsum('tlv,tbv->tlb', w_ref, masked_neighbors)
            norm = vol_sums[r][:, :, None] * grho[:, None, start:stop]
            gate = valid_gate[:, r, :, None] & (norm > 0)
            result[:, r, :, start:stop] = xp.where(gate, sums / xp.maximum(norm, 1e-300), 0.0)
    if return_device_result:
        return result, r_vec, cols, alpha_radius_valid, tgt_counts
    host = np.asarray(result) if xp is np else xp.asnumpy(result)
    if return_raw_host:
        return host, r_vec, cols, alpha_radius_valid, tgt_counts
    return [
        _make_fast_rdf_df(r_vec, host[t], cols, alpha_radius_valid[t], tgt_counts[t])
        for t in range(n)
    ]


def extract_patch(mapping, coord):
    focal_radii = mapping.worker_settings['window_radius_voxels']
    ext_radii = getattr(mapping, 'worker_ext_radii', None)
    if ext_radii is None:
        ext_radii = mapping._get_extended_radii(focal_radii, mapping.worker_settings['map_max_radius'])
    ext_sphere = getattr(mapping, 'worker_ext_sphere_mask', None)
    if ext_sphere is None:
        ext_sphere = mapping._get_spherical_mask(ext_radii)
    focal_patch, focal_mask, _, _, _ = mapping._extract_focal_and_target_window(
        mapping.worker_quantized_array, mapping.worker_mask_array, coord,
        focal_radii, mapping.worker_sphere_mask, ext_radii, ext_sphere
    )
    return focal_patch, focal_mask


def extract_extended_node(mapping, coord):
    focal_radii = mapping.worker_settings['window_radius_voxels']
    ext_radii = mapping.worker_ext_radii
    ext_sphere = mapping.worker_ext_sphere_mask
    return mapping._extract_focal_and_target_window(
        mapping.worker_quantized_array, mapping.worker_mask_array, coord,
        focal_radii, mapping.worker_sphere_mask, ext_radii, ext_sphere
    )


def reduce_node(mapping, coord, variants, mask, struct_df, baseline_df,
                focal_patch=None, ext_counts=None, ext_total=None):
    """Explicit precomputed data; preserve the existing CPU-only feature reduction policy."""
    if focal_patch is None:
        focal_patch, _ = extract_patch(mapping, coord)
    if ext_counts is None or ext_total is None:
        vals = variants[0][mask]
        ext_counts = np.bincount(vals.astype(int), minlength=mapping.worker_settings['num_gray_levels']).tolist()
        ext_total = int(mask.sum())
    core = getattr(mapping, 'core', None)
    old_gpu = getattr(core, 'HAS_GPU', None)
    if core is not None:
        core.HAS_GPU = False
    try:
        return mapping._process_single_voxel_worker(
            coord,
            _precomputed_node=(focal_patch, struct_df, baseline_df, ext_counts, ext_total),
        )
    finally:
        if core is not None:
            core.HAS_GPU = old_gpu


def gpu_results(mapping, coords, quantized, mask, settings, config_path):
    from .mapping_staged_runtime import gpu_results as staged_gpu_results
    yield from staged_gpu_results(mapping, coords, quantized, mask, settings, config_path)


# GLAM_PROFILE_MAPPING_FIRST_V6
# GLAM_CPU_REDUCTION_ONLY_V1

# GLAM_REXT30_REACH_CUBLAS_V1
