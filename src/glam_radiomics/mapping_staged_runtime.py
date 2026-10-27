"""Staged mapping runtime with batched GPU RDF reduction offload.
CUDA stays on the owner thread. Adjacent batched array operations on GPU RDF tensors
(support-aware baseline averaging, Savitzky-Golay smoothing, log transforms and PMF
integrands, cumulative integration for Wasserstein and PhenotypicDistance, and
finite-entry meta-feature reductions) execute on the GPU and are verified against
the isolated CPU reducer on the first batch.
"""
import os
import time
import types
import contextlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np


_GPU_OFFLOAD_BASES = {
    'PotentialEnergy',
    'EnergyDensity',
    'StructuralPressureIndex',
    'Compressibility',
    'B2',
    'Wasserstein',
    'PhenotypicDistance',
    'RDF_PeakPosition',
    'RDF_DispersionRatio',
    'LogRDF_Median',
    'LogRDF_Variance',
    'LogRDF_Skewness',
    'LogRDF_Kurtosis',
    'LogRDF_PeakHeight',
}


def _isolated_cpu_worker(mapping):
    original = mapping.core
    env = dict(vars(original))
    env['HAS_GPU'] = False
    for name, value in list(env.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is original.__dict__:
            clone = types.FunctionType(value.__code__, env, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            env[name] = clone
    cpu_core = types.SimpleNamespace(**env)
    worker = mapping._process_single_voxel_worker
    wg = dict(worker.__globals__)
    wg['core'] = cpu_core

    def forbidden(*args, **kwargs):
        raise RuntimeError('Unexpected RDF recomputation in parallel reduction')

    wg['_fast_calculate_rdf_3d'] = forbidden
    wg['_local_random_rdf'] = forbidden
    result = types.FunctionType(worker.__code__, wg, worker.__name__, worker.__defaults__, worker.__closure__)
    result.__kwdefaults__ = worker.__kwdefaults__
    return result


def _average_device(values, alpha_valid, counts, xp):
    # values: [node, variant, radius, alpha, beta], variant 0 is structured.
    n, per, radius, levels, _ = values.shape
    if per <= 1:
        return None, None
    ar = np.asarray(alpha_valid).reshape(n, per, radius, levels)
    cnt = np.asarray(counts).reshape(n, per, levels)
    sup = ar[:, 1:, :, :, None] & (cnt[:, 1:, None, None, :] > 0)
    sup = np.minimum.accumulate(sup, axis=2)
    gs = xp.asarray(sup)
    support_counts = xp.sum(gs, axis=1)
    valid = xp.cumprod((support_counts >= 1).astype(xp.int32), axis=1, dtype=xp.int32).astype(bool)
    sums = xp.sum(xp.where(gs, values[:, 1:], 0.0), axis=1)
    mean = xp.where(valid, sums / xp.maximum(support_counts, 1), 0.0)
    return mean, valid


def _gpu_smooth_rdf_batch(raw_3d, cut_2d, window, poly, xp):
    """Batched Savitzky-Golay smoothing on GPU over each row's supported prefix :L.
    raw_3d: (n, n_pairs, n_r) on device; cut_2d: (n, n_pairs) int32 on host.
    """
    from .rdf_smoothing import get_savgol_projection_matrix
    smooth = xp.zeros_like(raw_3d)
    flat_raw = raw_3d.reshape(-1, raw_3d.shape[-1])
    flat_smooth = smooth.reshape(-1, raw_3d.shape[-1])
    flat_cut = cut_2d.ravel()
    for L in np.unique(flat_cut):
        L = int(L)
        if L == 0:
            continue
        rows = np.flatnonzero(flat_cut == L)
        rows_gpu = xp.asarray(rows)
        sub = flat_raw[rows_gpu, :L]
        if L > window:
            proj_gpu = xp.asarray(get_savgol_projection_matrix(L, window, poly))
            flat_smooth[rows_gpu, :L] = sub @ proj_gpu
        else:
            flat_smooth[rows_gpu, :L] = sub
    return smooth


def _gpu_reduce_meta_batch(mat_3d, methods, xp):
    """Compute meta-feature reductions across a batch of (n, levels, levels) matrices on GPU."""
    n, levels, _ = mat_3d.shape
    out = {}
    eye_mask = xp.eye(levels, dtype=bool)
    off_mask = ~eye_mask

    for method in methods:
        if method == 'DiagMean':
            vals = xp.diagonal(mat_3d, axis1=1, axis2=2)
        elif method == 'OffDiagMean':
            vals = mat_3d[:, off_mask]
        else:
            vals = mat_3d.reshape(n, levels * levels)

        fin = xp.isfinite(vals)
        cnt = xp.sum(fin, axis=1)
        safe = xp.where(fin, vals, 0.0)
        mean_v = xp.where(cnt > 0, xp.sum(safe, axis=1) / xp.maximum(cnt, 1), xp.nan)

        if method in ('Mean', 'DiagMean', 'OffDiagMean'):
            res_gpu = mean_v
        elif method == 'Variance':
            diff = xp.where(fin, vals - mean_v[:, None], 0.0)
            res_gpu = xp.where(cnt > 0, xp.sum(diff * diff, axis=1) / xp.maximum(cnt, 1), xp.nan)
        elif method == 'Maximum':
            neg_inf = xp.where(fin, vals, -xp.inf)
            res_gpu = xp.where(cnt > 0, xp.max(neg_inf, axis=1), xp.nan)
        elif method == 'Energy':
            res_gpu = xp.where(cnt > 0, xp.sum(safe * safe, axis=1), xp.nan)
        else:
            raise ValueError(f'Unsupported mapping meta method: {method}')
        out[method] = xp.asnumpy(res_gpu)
    return out


def _gpu_compute_rdf_features_batch(
    mapping,
    structured_device,
    baseline_device,
    cut_s_host,
    cut_r_host,
    cnt_s_host,
    ext_totals_host,
    r_vec_host,
    settings,
    xp,
):
    """Compute adjacent batched RDF features and their meta-feature reductions on GPU.
    Returns a list of length `n` dicts mapping `f"{feature}_{method}" -> float`.
    """
    from .config import get_config
    n, radius, levels, _ = structured_device.shape
    n_pairs = levels * levels
    features_to_map = settings['features_to_map']
    methods = settings['meta_method']

    gpu_features = [
        f for f in features_to_map
        if mapping._strip_feature_transform(f)[0] in _GPU_OFFLOAD_BASES
    ]
    if not gpu_features:
        return [{} for _ in range(n)]

    bases = {mapping._strip_feature_transform(f)[0] for f in gpu_features}
    raw_s = xp.ascontiguousarray(structured_device.reshape(n, radius, n_pairs).swapaxes(1, 2))
    raw_r = (
        xp.ascontiguousarray(baseline_device.reshape(n, radius, n_pairs).swapaxes(1, 2))
        if baseline_device is not None else None
    )
    r_gpu = xp.asarray(r_vec_host, dtype=xp.float64)
    rho_host = cnt_s_host.astype(np.float64) / np.maximum(ext_totals_host[:, None], 1.0)
    rho_gpu = xp.asarray(rho_host, dtype=xp.float64)
    absent_host = cnt_s_host <= 0

    matrices_gpu = {}

    # 1. RDF Shape Matrices (7 matrices)
    rdf_shape_names = {
        'RDF_PeakPosition', 'RDF_DispersionRatio', 'LogRDF_Median',
        'LogRDF_Variance', 'LogRDF_Skewness', 'LogRDF_Kurtosis', 'LogRDF_PeakHeight',
    }
    if bool(bases & rdf_shape_names):
        for name in rdf_shape_names:
            matrices_gpu[name] = xp.full((n, n_pairs), xp.nan, dtype=xp.float64)
        flat_raw = raw_s.reshape(n * n_pairs, radius)
        flat_cut = cut_s_host.ravel()
        col_grid = xp.arange(radius, dtype=xp.int32)[None, :]
        sup_mask = col_grid < xp.asarray(flat_cut)[:, None]
        raw_sup = xp.where(sup_mask, flat_raw, 0.0)
        has_sig = (xp.asarray(flat_cut) > 0) & xp.any(raw_sup > 1e-6, axis=1)
        log_vals = xp.log1p(xp.maximum(flat_raw, 0.0))

        if bool(xp.any(has_sig)):
            raw_pk = xp.where(sup_mask, flat_raw, -xp.inf)
            log_pk = xp.where(sup_mask, log_vals, -xp.inf)
            pk_pos = xp.where(has_sig, r_gpu[xp.argmax(raw_pk, axis=1)], xp.nan)
            pk_hgt = xp.where(has_sig, xp.max(log_pk, axis=1), xp.nan)
            matrices_gpu['RDF_PeakPosition'] = pk_pos.reshape(n, n_pairs)
            matrices_gpu['LogRDF_PeakHeight'] = pk_hgt.reshape(n, n_pairs)

        eps = float(np.finfo(np.float64).eps)
        has_sig_host = xp.asnumpy(has_sig)
        valid_mom = has_sig_host & (flat_cut > 2)
        for L in np.unique(flat_cut[valid_mom]):
            L = int(L)
            ids = xp.asarray(np.flatnonzero(valid_mom & (flat_cut == L)))
            x = flat_raw[ids, :L]
            y = log_vals[ids, :L]
            mean_x = xp.mean(x, axis=1)
            var_x = xp.var(x, axis=1)
            ratio = xp.where(mean_x > 0, var_x / xp.maximum(mean_x, 1e-300), 0.0)

            mean_y = xp.mean(y, axis=1, keepdims=True)
            d = y - mean_y
            d2 = d * d
            m2 = xp.mean(d2, axis=1)
            m3 = xp.mean(d2 * d, axis=1)
            m4 = xp.mean(d2 * d2, axis=1)
            zero_var = m2 <= (eps * xp.abs(mean_y[:, 0])) ** 2
            safe_m2 = xp.where(zero_var, 1.0, m2)
            skew = xp.where(zero_var, xp.nan, m3 / (safe_m2 * xp.sqrt(safe_m2)))
            kurt = xp.where(zero_var, xp.nan, (m4 / (safe_m2 * safe_m2)) - 3.0)

            matrices_gpu['RDF_DispersionRatio'].ravel()[ids] = ratio
            matrices_gpu['LogRDF_Median'].ravel()[ids] = xp.median(y, axis=1)
            matrices_gpu['LogRDF_Variance'].ravel()[ids] = m2
            matrices_gpu['LogRDF_Skewness'].ravel()[ids] = skew
            matrices_gpu['LogRDF_Kurtosis'].ravel()[ids] = kurt

        for name in rdf_shape_names:
            matrices_gpu[name] = matrices_gpu[name].reshape(n, levels, levels)

    # 2. PotentialEnergy & EnergyDensity
    if 'PotentialEnergy' in bases or 'EnergyDensity' in bases:
        pe_flat = xp.full(n * n_pairs, xp.nan, dtype=xp.float64)
        flat_raw = raw_s.reshape(n * n_pairs, radius)
        flat_cut = cut_s_host.ravel()
        for L in np.unique(flat_cut):
            L = int(L)
            if L < 2:
                continue
            rows = xp.asarray(np.flatnonzero(flat_cut == L))
            r_sub = r_gpu[:L]
            g_sub = flat_raw[rows, :L]
            pmf = -xp.log(g_sub + 1e-9)
            integrand = pmf * g_sub * (r_sub[None, :] ** 2)
            pe_flat[rows] = 4.0 * np.pi * xp.trapezoid(integrand, r_sub, axis=-1)
        pe_mat = pe_flat.reshape(n, levels, levels)
        if 'PotentialEnergy' in bases:
            matrices_gpu['PotentialEnergy'] = pe_mat
        if 'EnergyDensity' in bases:
            ext_tot_gpu = xp.asarray(ext_totals_host, dtype=xp.float64)[:, None, None]
            matrices_gpu['EnergyDensity'] = xp.where(ext_tot_gpu > 0, pe_mat / ext_tot_gpu, xp.nan)

    # 3. StructuralPressureIndex (SPI)
    if 'StructuralPressureIndex' in bases:
        sw = int(get_config('SavgolWindow'))
        sp = int(get_config('SavgolPoly'))
        sm_s = _gpu_smooth_rdf_batch(raw_s, cut_s_host, sw, sp, xp)
        flat_raw = raw_s.reshape(n * n_pairs, radius)
        flat_sm = sm_s.reshape(n * n_pairs, radius)
        flat_cut = cut_s_host.ravel()
        integrals = xp.zeros(n * n_pairs, dtype=xp.float64)
        for L in np.unique(flat_cut):
            L = int(L)
            if L < 2:
                continue
            rows = xp.asarray(np.flatnonzero(flat_cut == L))
            r_sub = r_gpu[:L]
            raw_sub = flat_raw[rows, :L]
            pmf_sub = -xp.log(xp.maximum(flat_sm[rows, :L], 1e-9))
            edge_ord = 2 if L >= 3 else 1
            dW_dr = xp.gradient(pmf_sub, r_sub, axis=-1, edge_order=edge_ord)
            integrand = (r_sub[None, :] ** 3) * dW_dr * raw_sub
            integrals[rows] = xp.trapezoid(integrand, r_sub, axis=-1)
        int_3d = integrals.reshape(n, levels, levels)
        rho_ab = rho_gpu[:, :, None] * rho_gpu[:, None, :]
        valid_spi = (xp.asarray(cut_s_host.reshape(n, levels, levels)) >= 2) & (rho_ab > 0)
        matrices_gpu['StructuralPressureIndex'] = xp.where(
            valid_spi, -(rho_ab / 6.0) * 4.0 * np.pi * int_3d, xp.nan
        )

    # 4. Compressibility (diagonal only)
    if 'Compressibility' in bases:
        comp_mat = xp.full((n, levels, levels), xp.nan, dtype=xp.float64)
        diag_idx = np.arange(levels) * (levels + 1)
        diag_raw = raw_s[:, diag_idx, :]   # (n, levels, radius)
        diag_cut = cut_s_host[:, diag_idx] # (n, levels)
        comp_diag = xp.zeros((n, levels), dtype=xp.float64)
        flat_d_raw = diag_raw.reshape(n * levels, radius)
        flat_d_cut = diag_cut.ravel()
        for L in np.unique(flat_d_cut):
            L = int(L)
            if L < 2:
                continue
            rows = xp.asarray(np.flatnonzero(flat_d_cut == L))
            sub = flat_d_raw[rows, :L]
            has_p = xp.any(sub > 1e-6, axis=1)
            r_sub = r_gpu[:L]
            integrand = (sub - 1.0) * (r_sub[None, :] ** 2)
            vals = 4.0 * np.pi * xp.trapezoid(integrand, r_sub, axis=-1)
            comp_diag.ravel()[rows] = xp.where(has_p, vals, 0.0)
        comp_mat[:, np.arange(levels), np.arange(levels)] = comp_diag
        matrices_gpu['Compressibility'] = comp_mat

    # 5. B2
    if 'B2' in bases:
        common_cut = np.minimum(cut_s_host, cut_r_host).ravel()
        flat_s = raw_s.reshape(n * n_pairs, radius)
        flat_r = raw_r.reshape(n * n_pairs, radius)
        b2_flat = xp.full(n * n_pairs, xp.nan, dtype=xp.float64)
        for L in np.unique(common_cut):
            L = int(L)
            if L < 2:
                continue
            rows = xp.asarray(np.flatnonzero(common_cut == L))
            r_sub = r_gpu[:L]
            integrand = (flat_s[rows, :L] - flat_r[rows, :L]) * (r_sub[None, :] ** 2)
            b2_flat[rows] = -2.0 * np.pi * xp.trapezoid(integrand, r_sub, axis=-1)
        matrices_gpu['B2'] = b2_flat.reshape(n, levels, levels)

    # 6. Wasserstein
    if 'Wasserstein' in bases:
        common_cut = np.minimum(cut_s_host, cut_r_host)  # (n, n_pairs)
        rho_ab_pos = (rho_host[:, :, None] > 0) & (rho_host[:, None, :] > 0)
        valid_w = (common_cut >= 2) & rho_ab_pos.reshape(n, n_pairs)
        rho_beta_flat = np.repeat(rho_host, levels, axis=1)
        wait_flat = np.tile(rho_host, (1, levels)).ravel()
        flat_s = raw_s.reshape(n * n_pairs, radius)
        flat_r = raw_r.reshape(n * n_pairs, radius)
        flat_cut = common_cut.ravel()
        flat_valid = valid_w.ravel()
        w_flat = xp.full(n * n_pairs, xp.nan, dtype=xp.float64)
        rho_b_gpu = xp.asarray(wait_flat, dtype=xp.float64)
        for L in np.unique(flat_cut[flat_valid]):
            L = int(L)
            rows = xp.asarray(np.flatnonzero(flat_valid & (flat_cut == L)))
            r_sub = r_gpu[:L]
            dr = xp.diff(r_sub)[None, :]
            factor = (4.0 * np.pi * rho_b_gpu[rows, None]) * (r_sub[None, :] ** 2)
            diff_int = (flat_s[rows, :L] - flat_r[rows, :L]) * factor
            seg = 0.5 * (diff_int[:, :-1] + diff_int[:, 1:]) * dr
            n_diff = xp.hstack((xp.zeros((len(rows), 1), dtype=xp.float64), xp.cumsum(seg, axis=1)))
            w_flat[rows] = xp.trapezoid(xp.abs(n_diff), r_sub, axis=1)
        matrices_gpu['Wasserstein'] = w_flat.reshape(n, levels, levels)

    # 7. PhenotypicDistance (pairwise common-domain CDF normalization, diagonal masked to NaN)
    if 'PhenotypicDistance' in bases:
        diag_idx = np.arange(levels) * (levels + 1)
        diag_raw = raw_s[:, diag_idx, :]   # (n, levels, radius)
        diag_cut = np.where(rho_host > 0, cut_s_host[:, diag_idx], 0)
        r2_gpu = r_gpu ** 2
        dr = xp.diff(r_gpu)[None, None, :]
        integrand = (4.0 * np.pi * rho_gpu[:, :, None]) * diag_raw * r2_gpu[None, None, :]
        seg = 0.5 * (integrand[:, :, :-1] + integrand[:, :, 1:]) * dr
        cum_all = xp.concatenate((xp.zeros((n, levels, 1), dtype=xp.float64), xp.cumsum(seg, axis=2)), axis=2)

        pd_mat = xp.full((n, levels, levels), xp.nan, dtype=xp.float64)
        l_common = np.minimum(diag_cut[:, :, None], diag_cut[:, None, :])  # (n, levels, levels)
        # Mask diagonal to 0 so self-distance stays NaN for mapping
        for s_i in range(levels):
            l_common[:, s_i, s_i] = 0
        for L in np.unique(l_common):
            L = int(L)
            if L < 2:
                continue
            ni, ai, bi = np.nonzero(l_common == L)
            ni_g, ai_g, bi_g = xp.asarray(ni), xp.asarray(ai), xp.asarray(bi)
            ca = cum_all[ni_g, ai_g, :L]
            cb = cum_all[ni_g, bi_g, :L]
            ma = ca[:, L - 1]
            mb = cb[:, L - 1]
            ok = (ma > 1e-9) & (mb > 1e-9)
            diff = xp.abs(ca / xp.maximum(ma[:, None], 1e-300) - cb / xp.maximum(mb[:, None], 1e-300))
            vals = xp.where(ok, xp.trapezoid(diff, r_gpu[:L], axis=-1), xp.nan)
            pd_mat[ni_g, ai_g, bi_g] = vals
        matrices_gpu['PhenotypicDistance'] = pd_mat

    # Apply absent_ext mask, transforms, and meta-feature reductions on GPU
    absent_mask_3d = xp.asarray(absent_host[:, :, None] | absent_host[:, None, :])
    per_node_dicts = [{} for _ in range(n)]

    for feature in gpu_features:
        base, suffix = mapping._strip_feature_transform(feature)
        mat = xp.where(absent_mask_3d, xp.nan, matrices_gpu[base])
        if suffix == '_Symlog':
            mat = xp.sign(mat) * xp.log1p(xp.abs(mat))
        elif suffix == '_Log1p' or (suffix == '_Ln' and base in mapping._LOG1P_LN_BASES):
            mat = xp.log1p(xp.maximum(mat, 0.0))
            mat = xp.where(xp.isfinite(mat), mat, xp.nan)
        elif suffix == '_Ln':
            mat = xp.where(mat > 0, xp.log(xp.maximum(mat, 1e-300)), xp.nan)
            mat = xp.where(xp.isfinite(mat), mat, xp.nan)

        meta_arrays = _gpu_reduce_meta_batch(mat, methods, xp)
        for method, arr_host in meta_arrays.items():
            key = f'{feature}_{method}'
            for i in range(n):
                per_node_dicts[i][key] = float(arr_host[i])

    return per_node_dicts


def _assert_results(a, b):
    if len(a) != len(b):
        raise AssertionError('Reduction output count differs')
    for x, y in zip(a, b):
        if tuple(x[:3]) != tuple(y[:3]):
            raise AssertionError('Reduction coordinate order differs')
        if x[3] is None or y[3] is None:
            if x[3] is not y[3]:
                raise AssertionError('Reduction validity differs')
            continue
        if x[3].keys() != y[3].keys():
            raise AssertionError('Feature keys differ')
        np.testing.assert_allclose(
            list(x[3].values()), list(y[3].values()), rtol=1e-9, atol=1e-10, equal_nan=True
        )


class _Reducer:
    def __init__(self, mapping):
        self.worker = _isolated_cpu_worker(mapping)
        self.pool = None
        self.workers = 1
        self.ready = False
        self.mapping = mapping
        self.timings = {}

    def one(self, job):
        coord, fp, rdf, baseline, counts, total, pre_scalars = (
            job if len(job) == 7 else (*job, None)
        )
        return self.worker(
            coord,
            _precomputed_node=(fp, rdf, baseline, counts, total),
            _precomputed_scalars=pre_scalars,
        )

    def run(self, jobs):
        if self.pool is None:
            return [self.one(j) for j in jobs]
        return list(self.pool.map(self.one, jobs))

    def benchmark(self, jobs):
        from . import mapping_gpu as backend
        jobs = jobs[: min(32, len(jobs))]
        if not jobs:
            self.ready = True
            return
        self.one(jobs[0])
        reference = [self.one(j) for j in jobs]
        legacy = []
        for item in jobs:
            c, fp, s, b, cnt, total = item[:6]
            legacy.append(
                backend.reduce_node(
                    self.mapping, c, None, None,
                    s.copy(deep=True),
                    None if b is None else b.copy(deep=True),
                    focal_patch=fp, ext_counts=cnt, ext_total=total,
                )
            )
        _assert_results(reference, legacy)
        requested = os.environ.get('GLAM_REDUCE_WORKERS', '1').strip().lower()
        if requested == '1' and os.environ.get('GLAM_BENCHMARK_REDUCE', '0') != '1':
            self.workers = 1
            self.pool = None
            self.ready = True
            return
        candidates = [1, 2, 4, 8] if requested == 'auto' else [1, int(requested)]
        candidates = sorted(set(w for w in candidates if 1 <= w <= max(1, os.cpu_count() or 1)))
        if not candidates:
            candidates = [1]
        pools = {}
        try:
            for w in candidates:
                pool = None if w == 1 else ThreadPoolExecutor(max_workers=w, thread_name_prefix='glam-reduce')
                pools[w] = pool

                def execute():
                    return [self.one(j) for j in jobs] if pool is None else list(pool.map(self.one, jobs))

                _assert_results(reference, execute())
                times = []
                for _ in range(2):
                    start = time.perf_counter()
                    got = execute()
                    times.append(time.perf_counter() - start)
                    _assert_results(reference, got)
                self.timings[w] = float(np.median(times))
                print(f'  - Reduction benchmark: {w} worker(s), {len(jobs)} nodes, {self.timings[w]:.3f}s; parity PASS')
            best = min(self.timings, key=self.timings.get)
            if self.timings[best] >= 0.95 * self.timings.get(1, float('inf')):
                best = 1
            self.workers = best
            self.pool = pools.pop(best, None)
            self.ready = True
            print(f'  - Selected {best} CPU reduction worker(s); benchmark is sequence-specific.')
        finally:
            for pool in pools.values():
                if pool is not None:
                    pool.shutdown(wait=True)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def gpu_results(mapping, coords, quantized, mask, settings, config_path):
    import cupy as cp
    from .config import get_config
    from . import rdf_fast
    device = int(get_config('MapGPUDevice'))
    window_batch = int(get_config('MapGPUWindowBatch'))
    state_batch = int(get_config('MapGPUStateBatch'))
    if window_batch < 1 or state_batch < 1:
        raise ValueError('GPU batch sizes must be positive')
    old = (
        mapping.worker_quantized_array,
        mapping.worker_mask_array,
        mapping.worker_settings,
        mapping.worker_sphere_mask,
        getattr(mapping, 'worker_ext_sphere_mask', None),
        getattr(mapping, 'worker_ext_radii', None),
        getattr(mapping, 'worker_canonical_focal_in_ext', None),
    )
    rng_state = np.random.get_state()
    mapping.worker_quantized_array = quantized
    mapping.worker_mask_array = mask
    mapping.worker_settings = settings
    mapping.worker_sphere_mask = mapping._get_spherical_mask(settings['window_radius_voxels'])
    mapping.worker_ext_radii = mapping._get_extended_radii(settings['window_radius_voxels'], settings['map_max_radius'])
    mapping.worker_ext_sphere_mask = mapping._get_spherical_mask(mapping.worker_ext_radii)
    mapping.worker_canonical_focal_in_ext = mapping._get_canonical_focal_in_ext(
        settings['window_radius_voxels'],
        mapping.worker_sphere_mask,
        mapping.worker_ext_radii,
        mapping.worker_ext_sphere_mask,
    )
    levels = settings['num_gray_levels']
    radius = settings['map_max_radius']
    samples = settings['map_rdf_samples']
    bases = {mapping._strip_feature_transform(f)[0] for f in settings['features_to_map']}
    repeats = settings['randomizations'] if bool(bases & mapping._BASELINE_REQUIRED_BASES) else 0
    per = repeats + 1
    checked = False
    kernel_cache = {}

    from .mapping_gpu import batch_rdfs, variants_for_node, extract_extended_node, _make_fast_rdf_df
    from .rdf_smoothing import average_supported_rdf_replicates
    from .mapping_profile import stage

    reducer = _Reducer(mapping)
    use_gpu_baseline = os.environ.get('GLAM_GPU_BASELINE', '1') == '1'
    use_gpu_reduce = os.environ.get('GLAM_GPU_REDUCE', '1') == '1'
    baseline_verified = False
    gpu_reduce_verified = False
    pipeline_enabled = os.environ.get('GLAM_PIPELINE', '1') == '1'
    phase_seconds = {'launch_host': 0.0, 'download_and_host_setup': 0.0, 'cpu_reduce': 0.0}
    sequence_start = time.perf_counter()
    completed = 0

    try:
        from threadpoolctl import threadpool_limits
        native_limit = threadpool_limits(limits=1)
    except ImportError:
        native_limit = contextlib.nullcontext()
        print('  - threadpoolctl unavailable; native BLAS thread counts are not limited.')
    try:
        with cp.cuda.Device(device), native_limit:
            print(
                f'  - Staged mapping: download previous -> launch next -> CPU/GPU reduction; '
                f'GPU baseline={use_gpu_baseline}; GPU reduce={use_gpu_reduce}; overlap={pipeline_enabled}'
            )

            def launch_gpu(nodes):
                tasks = []
                records = []
                with stage('gpu_prepare_batch_variants', cuda=False):
                    for coord, node_focal_patch, ext_patch, ext_roi, focal_in_ext in nodes:
                        variants, refs = variants_for_node(ext_patch, ext_roi, focal_in_ext, levels, samples, repeats, coord)
                        records.append((coord, node_focal_patch, variants, ext_roi, focal_in_ext))
                        tasks.extend((im, ext_roi, ref) for im, ref in zip(variants, refs))
                with stage('gpu_batch_rdfs_launch', cuda=False):
                    res_gpu, r_vec, cols, ar_valid_all, tgt_counts_all = batch_rdfs(
                        tasks,
                        levels,
                        radius,
                        state_batch,
                        cp,
                        kernel_cache,
                        canonical_ref_mask=mapping.worker_canonical_focal_in_ext,
                        canonical_tgt_mask=mapping.worker_ext_sphere_mask,
                        return_device_result=True,
                    )
                return (records, res_gpu, r_vec, cols, ar_valid_all, tgt_counts_all)

            def download(flight):
                nonlocal checked, baseline_verified, gpu_reduce_verified
                records, device_values, r_vec, cols, ar_valid_all, tgt_counts_all = flight
                n = len(records)
                raw_host = None
                grouped_device = device_values.reshape(n, per, radius, levels, levels)
                ar = ar_valid_all.reshape(n, per, radius, levels)
                cnt = tgt_counts_all.reshape(n, per, levels)
                baseline_host = baseline_support = None
                mean_gpu = valid_gpu = None

                if (use_gpu_baseline or use_gpu_reduce) and repeats:
                    mean_gpu, valid_gpu = _average_device(grouped_device, ar, cnt, cp)

                if use_gpu_baseline and repeats:
                    if not baseline_verified or not checked:
                        raw_host = cp.asnumpy(device_values, blocking=True)
                    structured = cp.asnumpy(grouped_device[:, 0], blocking=True)
                    baseline_host = cp.asnumpy(mean_gpu, blocking=True)
                    baseline_support = cp.asnumpy(valid_gpu, blocking=True)
                else:
                    raw_host = cp.asnumpy(device_values, blocking=True)
                    structured = raw_host.reshape(n, per, radius, levels, levels)[:, 0]

                # Compute GPU-offloaded RDF features across the entire batch while tensors are on device
                pre_scalars_batch = None
                if use_gpu_reduce:
                    sup_s = np.minimum.accumulate(
                        ar[:, 0, :, :, None] & (cnt[:, 0, None, None, :] > 0), axis=1
                    )
                    cut_s_host = np.sum(sup_s, axis=1).reshape(n, levels * levels).astype(np.int32)
                    if repeats:
                        sup_b = (
                            baseline_support
                            if baseline_support is not None
                            else cp.asnumpy(valid_gpu, blocking=True)
                        )
                        cut_r_host = np.sum(sup_b, axis=1).reshape(n, levels * levels).astype(np.int32)
                    else:
                        cut_r_host = np.zeros_like(cut_s_host)
                    ext_totals_host = np.array([int(m.sum()) for _, _, _, m, _ in records], dtype=np.float64)
                    pre_scalars_batch = _gpu_compute_rdf_features_batch(
                        mapping,
                        grouped_device[:, 0],
                        mean_gpu,
                        cut_s_host,
                        cut_r_host,
                        cnt[:, 0],
                        ext_totals_host,
                        r_vec,
                        settings,
                        cp,
                    )

                del mean_gpu, valid_gpu

                if not checked:
                    coord, _, variants, ext_roi, focal_in_ext = records[0]
                    seed = int(np.random.SeedSequence([42, *map(int, coord)]).generate_state(1)[0])
                    np.random.seed(seed)
                    values = variants[0][ext_roi]
                    for i in range(per):
                        im = variants[0] if i == 0 else np.full(ext_roi.shape, -1, dtype=np.int16)
                        if i:
                            im[ext_roi] = np.random.permutation(values)
                        counts = np.bincount(im[ext_roi], minlength=levels).tolist()
                        cpu = rdf_fast.calculate_rdf_3d(
                            im, levels, radius, counts, int(ext_roi.sum()), 1, samples, sample_mask=focal_in_ext
                        )
                        frame_i = _make_fast_rdf_df(r_vec, raw_host[i], cols, ar_valid_all[i], tgt_counts_all[i])
                        np.testing.assert_allclose(frame_i.to_numpy(), cpu.to_numpy(), rtol=1e-9, atol=1e-9)
                    checked = True
                    print('  - First-window GPU/CPU RDF parity PASSED (structured and spatial-permutation baselines).')

                host_grouped = None if raw_host is None else raw_host.reshape(n, per, radius, levels, levels)
                jobs = []
                for i, (c, fp, variants, m, _) in enumerate(records):
                    s_df = _make_fast_rdf_df(r_vec, structured[i], cols, ar[i, 0], cnt[i, 0])
                    b_df = None
                    if repeats:
                        if use_gpu_baseline:
                            support = baseline_support[i].reshape(radius, levels * levels).T
                            b_df = _make_fast_rdf_df(r_vec, baseline_host[i], cols, pair_support_mask=support)
                        if not use_gpu_baseline or not baseline_verified:
                            vals = host_grouped[i, 1:].reshape(repeats, radius, levels * levels)
                            pair_sup = (
                                ar[i, 1:].transpose(0, 2, 1)[:, :, None, :]
                                & (cnt[i, 1:] > 0)[:, None, :, None]
                            ).reshape(repeats, levels * levels, radius)
                            cpu_b = average_supported_rdf_replicates(r_vec, cols, vals, pair_sup)
                            if use_gpu_baseline:
                                np.testing.assert_allclose(b_df.to_numpy(), cpu_b.to_numpy(), rtol=1e-9, atol=1e-10, equal_nan=True)
                                np.testing.assert_array_equal(b_df.attrs['_glam_support_mask'], cpu_b.attrs['_glam_support_mask'])
                                count_list = cnt[i, 0].tolist()
                                total = int(m.sum())
                                a = reducer.one((c, fp, s_df.copy(deep=True), cpu_b, count_list, total, None))
                                b = reducer.one((c, fp, s_df.copy(deep=True), b_df.copy(deep=True), count_list, total, None))
                                _assert_results([a], [b])
                            else:
                                b_df = cpu_b
                    pre_sc = pre_scalars_batch[i] if pre_scalars_batch is not None else None
                    if use_gpu_reduce and not gpu_reduce_verified and pre_sc:
                        count_list = cnt[i, 0].tolist()
                        total = int(m.sum())
                        cpu_only_res = reducer.one((c, fp, s_df.copy(deep=True), None if b_df is None else b_df.copy(deep=True), count_list, total, None))
                        gpu_off_res = reducer.one((c, fp, s_df.copy(deep=True), None if b_df is None else b_df.copy(deep=True), count_list, total, pre_sc))
                        _assert_results([cpu_only_res], [gpu_off_res])
                    jobs.append((c, fp, s_df, b_df, cnt[i, 0].tolist(), int(m.sum()), pre_sc))

                if use_gpu_baseline and repeats and not baseline_verified:
                    baseline_verified = True
                    print('  - GPU baseline values, support and final-feature parity PASS on first batch.')
                if use_gpu_reduce and not gpu_reduce_verified:
                    gpu_reduce_verified = True
                    print('  - Batched GPU RDF feature reduction parity PASS on first batch.')
                return jobs

            def batches():
                pending = []
                for coord in coords:
                    fp, roi, ep, em, fr = extract_extended_node(mapping, coord)
                    if int(roi.sum()) < settings['min_voxels']:
                        pending.append((coord, None, None, None, None))
                    else:
                        pending.append((coord, fp, ep, em, fr))
                    if len(pending) >= window_batch:
                        yield pending
                        pending = []
                if pending:
                    yield pending

            def launch(nodes):
                valid = [x for x in nodes if x[1] is not None]
                start = time.perf_counter()
                try:
                    flight = launch_gpu(valid) if valid else None
                    return nodes, flight
                except cp.cuda.memory.OutOfMemoryError:
                    if len(nodes) <= 1:
                        raise
                    cp.cuda.get_current_stream().synchronize()
                    kernel_cache.clear()
                    cp.get_default_memory_pool().free_all_blocks()
                    print(f'  - CUDA OOM: retrying {len(nodes)} nodes as smaller synchronous batches.')
                    mid = len(nodes) // 2
                    left = receive(launch(nodes[:mid]))
                    right = receive(launch(nodes[mid:]))
                    return nodes, ('host_jobs', left[1] + right[1])
                finally:
                    phase_seconds['launch_host'] += time.perf_counter() - start

            def receive(item):
                nodes, flight = item
                start = time.perf_counter()
                if isinstance(flight, tuple) and flight and isinstance(flight[0], str) and flight[0] == 'host_jobs':
                    jobs = flight[1]
                else:
                    jobs = download(flight) if flight is not None else []
                phase_seconds['download_and_host_setup'] += time.perf_counter() - start
                return nodes, jobs

            def reduce_received(received):
                nonlocal completed
                nodes, jobs = received
                if jobs and not reducer.ready:
                    reducer.benchmark(jobs)
                start = time.perf_counter()
                with stage('cpu_reduce_nodes', cuda=False):
                    out = reducer.run(jobs)
                phase_seconds['cpu_reduce'] += time.perf_counter() - start
                it = iter(out)
                for c, fp, *_ in nodes:
                    completed += 1
                    yield (*map(int, c), None) if fp is None else next(it)

            iterator = iter(batches())
            first = next(iterator, None)
            if first is not None:
                received = receive(launch(first))
                if received[1]:
                    reducer.benchmark(received[1])
                for nodes in iterator:
                    if pipeline_enabled:
                        next_flight = launch(nodes)
                        yield from reduce_received(received)
                        received = receive(next_flight)
                    else:
                        yield from reduce_received(received)
                        received = receive(launch(nodes))
                yield from reduce_received(received)
    finally:
        reducer.close()
        try:
            with cp.cuda.Device(device):
                cp.cuda.get_current_stream().synchronize()
                kernel_cache.clear()
                cp.get_default_memory_pool().free_all_blocks()
        finally:
            (
                mapping.worker_quantized_array,
                mapping.worker_mask_array,
                mapping.worker_settings,
                mapping.worker_sphere_mask,
                mapping.worker_ext_sphere_mask,
                mapping.worker_ext_radii,
                mapping.worker_canonical_focal_in_ext,
            ) = old
            np.random.set_state(rng_state)
        elapsed = time.perf_counter() - sequence_start
        print(
            f'  - Staged mapping finished/closed: {completed} nodes, {elapsed:.3f}s; '
            f'reduction timings={reducer.timings}; host phase seconds={phase_seconds}'
        )

# GLAM_GPU_RDF_OFFLOAD_V1

# GLAM_REXT30_REACH_CUBLAS_V1
