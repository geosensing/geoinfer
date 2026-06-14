"""
Inference engine for geoinfer.

The ``estimate()`` function is the main entry point. It takes annotated
frame data plus a design object and returns point estimates, standard
errors, confidence intervals, and diagnostics.
"""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import spatial
from .designs import Design, PointDesign
from .types import CIResult, Diagnostics, InferenceResult, SEResult

# Pairwise dependence diagnostics are O(n^2); subsample above this many frames.
_MAX_DEPENDENCE_POINTS = 2500


# ─── Point estimators ────────────────────────────────────────────────


def _ratio_estimator(w: np.ndarray, h: np.ndarray) -> float:
    """R_hat = sum(w) / sum(h)."""
    total_h = h.sum()
    if total_h == 0:
        return float("nan")
    return float(w.sum() / total_h)


def _photo_mean_estimator(w: np.ndarray, h: np.ndarray) -> float:
    """theta_hat = mean(w_i / h_i) for h_i > 0."""
    mask = h > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(w[mask] / h[mask]))


def _ratio_bias_approx(w: np.ndarray, h: np.ndarray) -> float:
    """O(1/N) bias approximation for the ratio estimator."""
    n = len(w)
    if n < 2 or h.sum() == 0:
        return float("nan")
    r = w.sum() / h.sum()
    eh = np.mean(h)
    vh = np.var(h, ddof=1)
    cwh = np.cov(w, h, ddof=1)[0, 1]
    return float((r * vh - cwh) / (n * eh**2))


# ─── Variance estimators ─────────────────────────────────────────────


def _naive_se_ratio(w: np.ndarray, h: np.ndarray) -> float:
    """Naive SE for ratio estimator (delta method, iid assumption)."""
    n = len(w)
    if n < 2 or h.sum() == 0:
        return float("nan")
    r = w.sum() / h.sum()
    e = w - r * h
    v = np.sum(e**2) / (n - 1) / h.sum() ** 2 * n
    return float(np.sqrt(v))


def _naive_se_mean(w: np.ndarray, h: np.ndarray) -> float:
    """Naive SE for photo-level mean (iid assumption)."""
    mask = h > 0
    if mask.sum() < 2:
        return float("nan")
    p = w[mask] / h[mask]
    return float(np.std(p, ddof=1) / np.sqrt(len(p)))


def _cluster_robust_se_ratio(w: np.ndarray, h: np.ndarray, labels: np.ndarray) -> float:
    """Linearization-based cluster-robust SE for ratio estimator."""
    r = _ratio_estimator(w, h)
    if np.isnan(r) or h.sum() == 0:
        return float("nan")

    total_h = h.sum()
    e = w - r * h

    unique = np.unique(labels)
    g = len(unique)
    if g < 2:
        return float("nan")

    e_g = np.array([e[labels == c].sum() for c in unique])
    e_bar = e_g.mean()

    v = (g / (g - 1)) * np.sum((e_g - e_bar) ** 2) / total_h**2
    return float(np.sqrt(v))


def _cluster_robust_se_mean(w: np.ndarray, h: np.ndarray, labels: np.ndarray) -> float:
    """Cluster-robust SE for photo-level mean."""
    mask = h > 0
    if mask.sum() == 0:
        return float("nan")

    p = np.full(len(w), np.nan)
    p[mask] = w[mask] / h[mask]
    theta = float(np.nanmean(p[mask]))
    m = int(mask.sum())

    unique = np.unique(labels)
    g = len(unique)
    if g < 2:
        return float("nan")

    s_g = np.array(
        [
            np.sum(p[(labels == c) & mask] - theta) if np.any((labels == c) & mask) else 0.0
            for c in unique
        ]
    )

    v = (g / (g - 1)) * np.sum(s_g**2) / m**2
    return float(np.sqrt(v))


def wild_cluster_bootstrap_ci(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    reps: int = 999,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Wild cluster bootstrap (percentile-t) CI for the ratio estimator.

    The pairs/cluster bootstrap and the analytic cluster-robust t both
    under-cover when the number of itineraries G is small. The wild cluster
    bootstrap (Cameron, Gelbach & Miller 2008) perturbs each cluster's
    linearized score by a Rademacher (+/-1) weight and studentizes, which
    restores near-nominal coverage with few clusters.

    Returns:
        (cluster_robust_se, ci_lo, ci_hi). NaNs if G < 2.
    """
    r = _ratio_estimator(w, h)
    total_h = h.sum()
    if np.isnan(r) or total_h == 0:
        return float("nan"), float("nan"), float("nan")

    unique = np.unique(labels)
    g = len(unique)
    if g < 2:
        return float("nan"), float("nan"), float("nan")

    e = w - r * h
    a = np.array([e[labels == c].sum() for c in unique])
    a_c = a - a.mean()  # centered cluster scores
    v_obs = (g / (g - 1)) * np.sum(a_c**2) / total_h**2
    se = float(np.sqrt(v_obs))
    if se == 0:
        return se, r, r

    rng = np.random.default_rng(seed)
    t_star = np.empty(reps)
    valid = 0
    for _ in range(reps):
        weights = rng.choice(np.array([-1.0, 1.0]), size=g)
        a_b = weights * a_c
        delta = a_b.sum() / total_h
        a_bc = a_b - a_b.mean()
        v_b = (g / (g - 1)) * np.sum(a_bc**2) / total_h**2
        if v_b > 0:
            t_star[valid] = delta / np.sqrt(v_b)
            valid += 1

    if valid < reps * 0.5:
        return se, float("nan"), float("nan")

    t_star = t_star[:valid]
    alpha = 1 - ci_level
    q_lo = float(np.percentile(t_star, 100 * (alpha / 2)))
    q_hi = float(np.percentile(t_star, 100 * (1 - alpha / 2)))
    # Percentile-t: invert t = (r_hat - r) / se.
    return se, r - se * q_hi, r - se * q_lo


def _cluster_bootstrap(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    estimator_fn: Callable[[np.ndarray, np.ndarray], float],
    reps: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Cluster bootstrap: resample clusters, return (se, ci_lo, ci_hi)."""
    if rng is None:
        rng = np.random.default_rng(0)

    unique = np.unique(labels)
    g = len(unique)

    cluster_data = {}
    for c in unique:
        idx = labels == c
        cluster_data[c] = (w[idx], h[idx])

    estimates = np.empty(reps)
    for b in range(reps):
        resampled = rng.choice(unique, size=g, replace=True)
        w_b = np.concatenate([cluster_data[c][0] for c in resampled])
        h_b = np.concatenate([cluster_data[c][1] for c in resampled])
        estimates[b] = estimator_fn(w_b, h_b)

    valid = estimates[~np.isnan(estimates)]
    if len(valid) < reps * 0.5:
        return float("nan"), float("nan"), float("nan")

    se = float(np.std(valid, ddof=1))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    return se, lo, hi


# ─── ICC and design effect ───────────────────────────────────────────


def _compute_icc(values: np.ndarray, labels: np.ndarray) -> float:
    """Intraclass correlation within clusters."""
    unique = np.unique(labels)
    g = len(unique)
    n = len(values)

    if g < 2 or n < 3:
        return float("nan")

    grand_mean = values.mean()
    ssb = sum(np.sum(labels == c) * (values[labels == c].mean() - grand_mean) ** 2 for c in unique)
    ssw = sum(np.sum((values[labels == c] - values[labels == c].mean()) ** 2) for c in unique)

    msb = ssb / (g - 1)
    msw = ssw / (n - g) if n > g else 0.0

    m = np.array([np.sum(labels == c) for c in unique], dtype=float)
    m0 = (n - np.sum(m**2) / n) / (g - 1)

    denom = msb + (m0 - 1) * msw
    if denom == 0:
        return 0.0

    icc = (msb - msw) / denom
    return float(max(icc, 0.0))


# ─── Within-itinerary dependence diagnostics ─────────────────────────


def _require_columns(data: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError listing any of ``cols`` absent from ``data``."""
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. Available columns: {list(data.columns)}"
        )


def _axis_diagnostics(values: np.ndarray, dist: np.ndarray, seed: int) -> dict[str, float]:
    """Variogram range, correlation ratio, effective N, and Moran's I for one axis."""
    lags, gamma, counts = spatial.empirical_variogram(values, dist)
    c0, c1, rng_ = spatial.fit_variogram(lags, gamma, counts)
    corr_ratio = (c1 - c0) / c1 if np.isfinite(c1) and c1 > 0 else float("nan")
    n_eff = spatial.effective_n(values, dist, c0, c1, rng_)
    if np.isfinite(rng_) and rng_ > 0:
        cutoff = rng_
    else:
        pos = dist[dist > 0]
        cutoff = float(np.median(pos)) if pos.size else 0.0
    mi, mp = spatial.morans_i(values, dist, cutoff, seed=seed)
    return {
        "range": rng_,
        "corr_ratio": corr_ratio,
        "n_eff": n_eff,
        "morans_i": mi,
        "morans_i_p": mp,
    }


def _dependence_diagnostics(
    data: pd.DataFrame,
    mask: np.ndarray,
    p_obs: np.ndarray,
    labels_pos: np.ndarray,
    lon_var: str | None,
    lat_var: str | None,
    time_var: str | None,
    seed: int,
) -> dict[str, float]:
    """Spatial/temporal within-itinerary dependence on the h>0 frames."""
    out: dict[str, float] = {}
    n = len(p_obs)
    if n < 3:
        return out

    idx = np.arange(n)
    if n > _MAX_DEPENDENCE_POINTS:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=_MAX_DEPENDENCE_POINTS, replace=False))
        warnings.warn(
            f"Dependence diagnostics subsampled to {_MAX_DEPENDENCE_POINTS} of "
            f"{n} positive frames (pairwise cost is O(n^2)).",
            stacklevel=2,
        )

    vals = p_obs[idx]
    sub_labels = labels_pos[idx]

    if len(np.unique(sub_labels)) >= 2:
        wb = spatial.within_between_contrast(vals, sub_labels)
        out["within_between_ratio"] = wb["ratio"]

    if lon_var is not None and lat_var is not None:
        _require_columns(data, [lon_var, lat_var])
        lon = data[lon_var].to_numpy(dtype=float)[mask][idx]
        lat = data[lat_var].to_numpy(dtype=float)[mask][idx]
        sp = _axis_diagnostics(vals, spatial.haversine_matrix(lon, lat), seed)
        out["variogram_range_m"] = sp["range"]
        out["spatial_corr_ratio"] = sp["corr_ratio"]
        out["n_eff_space"] = sp["n_eff"]
        out["morans_i_space"] = sp["morans_i"]
        out["morans_i_space_p"] = sp["morans_i_p"]

    if time_var is not None:
        _require_columns(data, [time_var])
        ts = data[time_var].to_numpy()[mask][idx]
        tp = _axis_diagnostics(vals, spatial.time_gap_matrix(ts), seed)
        out["variogram_range_s"] = tp["range"]
        out["temporal_corr_ratio"] = tp["corr_ratio"]
        out["n_eff_time"] = tp["n_eff"]
        out["morans_i_time"] = tp["morans_i"]
        out["morans_i_time_p"] = tp["morans_i_p"]

    return out


# ─── Confidence interval construction ────────────────────────────────


def _build_ci(
    est: float,
    se: float,
    g: int,
    level: float = 0.95,
    boot_ci: tuple[float, float] | None = None,
    recommended_method: str = "naive",
) -> CIResult:
    """Construct CIs from multiple methods, mark recommended."""
    alpha = 1 - level
    z = -sp_stats.norm.ppf(alpha / 2)

    normal_ci = (est - z * se, est + z * se)

    if g >= 2:
        t_cv = -sp_stats.t.ppf(alpha / 2, df=g - 1)
        t_ci = (est - t_cv * se, est + t_cv * se)
    else:
        t_ci = (float("nan"), float("nan"))

    # Choose recommended
    if recommended_method == "cluster" and g < 30:
        recommended = t_ci
    else:
        recommended = normal_ci

    if boot_ci is not None and recommended_method == "bootstrap":
        recommended = boot_ci

    return CIResult(
        normal=normal_ci,
        t=t_ci,
        bootstrap=boot_ci,
        recommended=recommended,
        level=level,
    )


# ─── Main entry point ────────────────────────────────────────────────


def estimate(
    data: pd.DataFrame,
    women_var: str = "n_women",
    people_var: str = "n_people",
    design: Design | None = None,
    ci_level: float = 0.95,
    bootstrap: bool = True,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    lon_var: str | None = None,
    lat_var: str | None = None,
    time_var: str | None = None,
) -> InferenceResult:
    """Estimate population gender ratio with correct standard errors.

    This is the main inference function. It takes annotated frame-level
    data, a design specification, and produces point estimates, standard
    errors, confidence intervals, and diagnostics for two estimands:

    - **Ratio** (people-weighted): sum(women) / sum(people)
    - **Photo-level mean** (location-weighted): mean(w_i / h_i) for h_i > 0

    The design object determines which SE estimator is primary. Under
    SRS with post-hoc itinerary clustering, the naive SE is recommended
    (observations are approximately independent). Under walk designs,
    the between-walk SE is recommended. Under PPS/GRTS, the cluster-robust
    (Horvitz-Thompson linearization) SE is recommended.

    Args:
        data: DataFrame with one row per annotated frame.
        women_var: Column with women count per frame.
        people_var: Column with total people count per frame.
        design: Design object (PointDesign or WalkDesign). If None,
            defaults to PointDesign(sampling="srs") with no clustering.
        ci_level: Confidence level for intervals (default 0.95).
        bootstrap: Whether to compute cluster bootstrap CIs.
        bootstrap_reps: Number of bootstrap replications.
        seed: Random seed for bootstrap and the dependence diagnostics.
        lon_var: Column with longitude (decimal degrees). If given together
            with ``lat_var``, the spatial within-itinerary dependence
            diagnostics (variogram range, Moran's I, effective spatial N) are
            computed on the h>0 frames.
        lat_var: Column with latitude (decimal degrees). See ``lon_var``.
        time_var: Column with a per-frame timestamp (datetime or epoch
            seconds). If given, the temporal within-itinerary dependence
            diagnostics are computed. Independent of the spatial axis: supply
            either, both, or neither.

    Returns:
        InferenceResult with estimates, SEs, CIs, and diagnostics. The spatial
        and temporal dependence fields of ``diagnostics`` are NaN unless the
        corresponding coordinate/time columns are supplied.

    Example:
        >>> from geoinfer import PointDesign, estimate
        >>> design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        >>> result = estimate(df, "n_women", "n_people", design=design)
        >>> print(result.summary())
    """
    if design is None:
        design = PointDesign(sampling="srs")

    # ── Extract arrays ────────────────────────────────────────────
    w = data[women_var].to_numpy(dtype=float)
    h = data[people_var].to_numpy(dtype=float)
    n = len(w)

    # Cluster labels
    if design.has_clusters:
        cvar = design.cluster_var
        if cvar not in data.columns:
            raise ValueError(
                f"Cluster variable '{cvar}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        labels = data[cvar].to_numpy()
    else:
        # Each observation is its own cluster
        labels = np.arange(n)

    unique_clusters = np.unique(labels)
    g = len(unique_clusters)

    # ── Point estimates ───────────────────────────────────────────
    ratio = _ratio_estimator(w, h)
    photo_mean = _photo_mean_estimator(w, h)

    # ── Standard errors ───────────────────────────────────────────
    se_ratio_naive = _naive_se_ratio(w, h)
    se_ratio_cluster = _cluster_robust_se_ratio(w, h, labels)
    se_mean_naive = _naive_se_mean(w, h)
    se_mean_cluster = _cluster_robust_se_mean(w, h, labels)

    rng = np.random.default_rng(seed)
    se_ratio_boot, boot_ratio_lo, boot_ratio_hi = (None, None, None)
    se_mean_boot, boot_mean_lo, boot_mean_hi = (None, None, None)

    if bootstrap and g >= 3:
        se_ratio_boot, boot_ratio_lo, boot_ratio_hi = _cluster_bootstrap(
            w,
            h,
            labels,
            _ratio_estimator,
            reps=bootstrap_reps,
            rng=rng,
        )
        se_mean_boot, boot_mean_lo, boot_mean_hi = _cluster_bootstrap(
            w,
            h,
            labels,
            _photo_mean_estimator,
            reps=bootstrap_reps,
            rng=rng,
        )

    # ── Select recommended SE ─────────────────────────────────────
    rec_method = design.recommended_se_method

    def _pick_se(naive: float, cluster: float, boot: float | None, method: str) -> SEResult:
        if method == "naive":
            recommended = naive
        elif method == "cluster":
            recommended = cluster
        elif method == "bootstrap" and boot is not None:
            recommended = boot
        else:
            recommended = naive
        return SEResult(
            naive=naive,
            cluster=cluster,
            bootstrap=boot,
            recommended=recommended,
            method_used=method,
        )

    ratio_se = _pick_se(se_ratio_naive, se_ratio_cluster, se_ratio_boot, rec_method)
    mean_se = _pick_se(se_mean_naive, se_mean_cluster, se_mean_boot, rec_method)

    # ── Confidence intervals ──────────────────────────────────────
    boot_ratio_ci = (
        (boot_ratio_lo, boot_ratio_hi)
        if boot_ratio_lo is not None and boot_ratio_hi is not None
        else None
    )
    boot_mean_ci = (
        (boot_mean_lo, boot_mean_hi)
        if boot_mean_lo is not None and boot_mean_hi is not None
        else None
    )

    ratio_ci = _build_ci(
        ratio,
        ratio_se.recommended,
        g,
        ci_level,
        boot_ci=boot_ratio_ci,
        recommended_method=rec_method,
    )
    mean_ci = _build_ci(
        photo_mean,
        mean_se.recommended,
        g,
        ci_level,
        boot_ci=boot_mean_ci,
        recommended_method=rec_method,
    )

    # ── Diagnostics ───────────────────────────────────────────────
    mask = h > 0
    n_positive = int(mask.sum())
    n_empty = int((~mask).sum())

    p_obs = w[mask] / h[mask] if n_positive > 0 else np.array([])
    lab_pos = labels[mask] if n_positive > 0 else np.array([])

    icc = _compute_icc(p_obs, lab_pos) if n_positive > 2 else float("nan")
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_clusters])
    m_bar = float(cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0.0
    deff = 1 + (m_bar - 1) * icc if not np.isnan(icc) else float("nan")
    n_eff = n_positive / deff if deff > 0 and not np.isnan(deff) else float(n_positive)

    se_ratio_cn = (
        se_mean_cluster / se_mean_naive
        if se_mean_naive > 0 and not np.isnan(se_mean_cluster)
        else float("nan")
    )

    dep: dict[str, float] = {}
    if n_positive > 2 and (lon_var is not None or time_var is not None):
        dep = _dependence_diagnostics(
            data,
            mask,
            p_obs,
            lab_pos,
            lon_var,
            lat_var,
            time_var,
            seed,
        )

    def _dep(key: str) -> float:
        return dep.get(key, float("nan"))

    diagnostics = Diagnostics(
        n_obs=n,
        n_positive_frames=n_positive,
        n_empty_frames=n_empty,
        empty_frame_rate=n_empty / n if n > 0 else 0.0,
        n_clusters=g,
        cluster_sizes=cluster_sizes,
        cluster_size_mean=m_bar,
        cluster_size_cv=(float(cluster_sizes.std() / cluster_sizes.mean()) if m_bar > 0 else 0.0),
        icc=icc,
        deff=deff,
        n_eff=n_eff,
        se_ratio_cluster_to_naive=se_ratio_cn,
        ratio_bias_approx=_ratio_bias_approx(w, h),
        morans_i_space=_dep("morans_i_space"),
        morans_i_space_p=_dep("morans_i_space_p"),
        variogram_range_m=_dep("variogram_range_m"),
        spatial_corr_ratio=_dep("spatial_corr_ratio"),
        n_eff_space=_dep("n_eff_space"),
        morans_i_time=_dep("morans_i_time"),
        morans_i_time_p=_dep("morans_i_time_p"),
        variogram_range_s=_dep("variogram_range_s"),
        temporal_corr_ratio=_dep("temporal_corr_ratio"),
        n_eff_time=_dep("n_eff_time"),
        within_between_ratio=_dep("within_between_ratio"),
    )

    return InferenceResult(
        ratio=ratio,
        photo_mean=photo_mean,
        ratio_se=ratio_se,
        photo_mean_se=mean_se,
        ratio_ci=ratio_ci,
        photo_mean_ci=mean_ci,
        diagnostics=diagnostics,
        design_name=design.name,
        n_obs=n,
        n_clusters=g,
    )
