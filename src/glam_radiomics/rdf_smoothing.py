from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class SmoothedRDF:
    radius: np.ndarray
    pairs: tuple
    raw: np.ndarray        # [pair, radius]
    smooth: np.ndarray     # [pair, radius]
    row_index: dict
    cutoff_idx: np.ndarray # [pair] exclusive end index of active geometric support


_STANDARD_COLS_CACHE = {}


def _get_standard_meta(num_levels):
    meta = _STANDARD_COLS_CACHE.get(num_levels)
    if meta is None:
        pairs = tuple((a, b) for a in range(num_levels) for b in range(num_levels))
        cols = ['r'] + [f'g_{a}_{b}' for a, b in pairs]
        row_index = {pair: idx for idx, pair in enumerate(pairs)}
        meta = (pairs, cols, row_index)
        _STANDARD_COLS_CACHE[num_levels] = meta
    return meta


def attach_geometric_support(rdf_df, alpha_radius_valid, counts):
    """Attach explicit geometric support metadata to `rdf_df.attrs`.
    A shell r is geometrically supported for pair (alpha, beta) iff
    `alpha_radius_valid[r, alpha]` is True and `counts[beta] > 0`.
    """
    ar_valid = np.asarray(alpha_radius_valid, dtype=bool)  # (radius, levels)
    cnts = np.asarray(counts) > 0                          # (levels,)
    pair_mask = (ar_valid.T[:, None, :] & cnts[None, :, None]).reshape(-1, ar_valid.shape[0])
    # Enforce contiguous prefix support per pair
    prefix_mask = np.minimum.accumulate(pair_mask, axis=1)
    cutoff_idx = np.sum(prefix_mask, axis=1).astype(np.int32)
    if isinstance(getattr(rdf_df, 'attrs', None), dict):
        rdf_df.attrs['_glam_support_cutoff'] = cutoff_idx
        rdf_df.attrs['_glam_support_mask'] = prefix_mask
    return rdf_df


def average_supported_rdf_replicates(r_vec, cols, vals_stack, sup_masks_stack=None, min_replicates=1):
    """Average randomized RDF replicates strictly over supported replicates at each
    (pair, radius) so unsupported zero-padded replicates never dilute the baseline.

    Parameters
    ----------
    r_vec : np.ndarray of shape (n_r,)
    cols : list of str, length 1 + n_pairs ('r' followed by 'g_a_b')
    vals_stack : np.ndarray of shape (repeats, n_r, n_pairs)
    sup_masks_stack : optional np.ndarray of shape (repeats, n_pairs, n_r) bool
    min_replicates : int, minimum number of supported replicates required at (pair, r)
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    vals_stack = np.asarray(vals_stack, dtype=np.float64)
    repeats, n_r, n_pairs = vals_stack.shape
    min_rep = max(1, int(min_replicates))

    if sup_masks_stack is not None and len(sup_masks_stack) == repeats:
        sup_pair_r = np.asarray(sup_masks_stack, dtype=bool)  # (repeats, n_pairs, n_r)
    else:
        # Fallback if replicates lack explicit metadata: infer prefix from positive values per replicate
        pos = np.swapaxes(vals_stack > 0, 1, 2)               # (repeats, n_pairs, n_r)
        any_pos = np.any(pos, axis=2)
        last_pos = n_r - 1 - np.argmax(pos[:, :, ::-1], axis=2)
        cuts = np.where(any_pos, last_pos + 1, 0)
        grid = np.arange(n_r, dtype=np.int32)[None, None, :]
        sup_pair_r = grid < cuts[:, :, None]

    sup_r_pair = np.swapaxes(sup_pair_r, 1, 2)                # (repeats, n_r, n_pairs)
    sup_counts = np.sum(sup_r_pair, axis=0)                   # (n_r, n_pairs)
    valid_cell = sup_counts >= min_rep                        # (n_r, n_pairs)

    # Enforce contiguous supported prefix (0..L-1) for each pair
    prefix_valid_pair_r = np.minimum.accumulate(valid_cell.T, axis=1)  # (n_pairs, n_r)
    cutoff_idx = np.sum(prefix_valid_pair_r, axis=1).astype(np.int32)
    clean_mask_r_pair = prefix_valid_pair_r.T                 # (n_r, n_pairs)

    masked_vals = np.where(sup_r_pair, vals_stack, 0.0)
    avg_vals = np.where(
        clean_mask_r_pair & (sup_counts > 0),
        np.sum(masked_vals, axis=0) / np.maximum(sup_counts, 1),
        0.0,
    )

    mat = np.hstack((r_vec[:, None], avg_vals))
    df = pd.DataFrame(mat, columns=cols, copy=False)
    df.attrs['_glam_fast_raw'] = (r_vec, np.ascontiguousarray(avg_vals.T))
    df.attrs['_glam_support_mask'] = prefix_valid_pair_r
    df.attrs['_glam_support_cutoff'] = cutoff_idx
    return df


def extract_fast_raw_rdf(rdf_df, num_levels):
    """Fast O(1) extraction of (radius, raw_576xR, pairs, row_index) using attrs cache."""
    attrs = getattr(rdf_df, 'attrs', None)
    std_pairs, std_cols, std_row_index = _get_standard_meta(int(num_levels))
    if isinstance(attrs, dict) and '_glam_fast_raw' in attrs:
        r_vec, raw = attrs['_glam_fast_raw']
        if raw.shape[0] == len(std_pairs):
            return r_vec, raw, std_pairs, std_row_index

    if len(rdf_df.columns) == len(std_cols) and list(rdf_df.columns) == std_cols:
        arr = rdf_df.to_numpy(dtype=np.float64, copy=False)
        r_vec = arr[:, 0].copy()
        raw = np.ascontiguousarray(arr[:, 1:].T)
        if isinstance(attrs, dict):
            attrs['_glam_fast_raw'] = (r_vec, raw)
        return r_vec, raw, std_pairs, std_row_index

    pairs = tuple(
        (alpha, beta)
        for alpha in range(num_levels)
        for beta in range(num_levels)
        if f"g_{alpha}_{beta}" in rdf_df.columns
    )
    r_vec = rdf_df["r"].to_numpy(dtype=np.float64, copy=True)
    if pairs:
        columns = [f"g_{alpha}_{beta}" for alpha, beta in pairs]
        raw = np.ascontiguousarray(
            rdf_df.loc[:, columns].to_numpy(dtype=np.float64, copy=True).T
        )
    else:
        raw = np.empty((0, r_vec.size), dtype=np.float64)
    row_index = {pair: row for row, pair in enumerate(pairs)}
    return r_vec, raw, pairs, row_index


def get_row_cutoff_indices(raw_mat, rdf_df=None):
    """Return exclusive end index L for each row's geometrically supported domain (1..R_cutoff).
    Uses explicit `_glam_support_cutoff` metadata when present on `rdf_df.attrs`,
    preserving geometrically supported shells that have zero observed neighbors.
    Falls back to last positive index only for synthetic DataFrames without metadata.
    """
    n_pairs, n_r = raw_mat.shape
    if not n_pairs or not n_r:
        return np.zeros(n_pairs, dtype=np.int32)
    attrs = getattr(rdf_df, 'attrs', None) if rdf_df is not None else None
    if isinstance(attrs, dict) and '_glam_support_cutoff' in attrs:
        cut = np.asarray(attrs['_glam_support_cutoff'], dtype=np.int32)
        if cut.shape == (n_pairs,):
            return np.clip(cut, 0, n_r)
    pos = raw_mat > 0
    any_pos = np.any(pos, axis=1)
    last_pos = n_r - 1 - np.argmax(pos[:, ::-1], axis=1)
    return np.where(any_pos, last_pos + 1, 0).astype(np.int32)


_SAVGOL_PROJ_CACHE = {}


def get_savgol_projection_matrix(L, window, poly):
    """Return exact L x L Savitzky-Golay linear operator P_L such that
    `sub @ P_L` equals `savgol_filter(sub, window, poly, axis=-1, mode='interp')`.
    """
    key = (int(L), int(window), int(poly))
    mat = _SAVGOL_PROJ_CACHE.get(key)
    if mat is None:
        eye = np.eye(int(L), dtype=np.float64)
        mat = np.ascontiguousarray(
            savgol_filter(eye, window_length=int(window), polyorder=int(poly), axis=-1, mode="interp")
        )
        _SAVGOL_PROJ_CACHE[key] = mat
    return mat


def prepare_smoothed_rdf(rdf_df, num_levels, window, poly):
    """Smooth each RDF row strictly over its geometrically supported prefix (1..R_cutoff)
    using cached exact Savitzky-Golay projection matrices.
    """
    cache_key = ('_glam_smoothed_rdf', int(num_levels), int(window), int(poly))
    attrs = getattr(rdf_df, 'attrs', None)
    if isinstance(attrs, dict) and cache_key in attrs:
        return attrs[cache_key]

    radius, raw, pairs, row_index = extract_fast_raw_rdf(rdf_df, num_levels)
    cutoff_idx = get_row_cutoff_indices(raw, rdf_df=rdf_df)
    smooth = np.zeros_like(raw)

    for L in np.unique(cutoff_idx):
        L = int(L)
        if L == 0:
            continue
        rows = np.flatnonzero(cutoff_idx == L)
        sub = raw[rows, :L]
        if L > window:
            smooth[rows, :L] = sub @ get_savgol_projection_matrix(L, window, poly)
        else:
            smooth[rows, :L] = sub

    result = SmoothedRDF(
        radius=radius,
        pairs=pairs,
        raw=raw,
        smooth=smooth,
        row_index=row_index,
        cutoff_idx=cutoff_idx,
    )
    if isinstance(attrs, dict):
        attrs[cache_key] = result
    return result

# GLAM_GPU_RDF_OFFLOAD_V1
