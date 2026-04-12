"""
Inference engine for geoinfer.

The ``estimate()`` function is the main entry point. It takes annotated
frame data plus a design object and returns point estimates, standard
errors, confidence intervals, and diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .designs import Design, PointDesign, WalkDesign
from .types import CIResult, Diagnostics, InferenceResult, SEResult


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
    v = np.sum(e**2) / (n - 1) / h.sum()**2 * n
    return float(np.sqrt(v))


def _naive_se_mean(w: np.ndarray, h: np.ndarray) -> float:
    """Naive SE for photo-level mean (iid assumption)."""
    mask = h > 0
    if mask.sum() < 2:
        return float("nan")
    p = w[mask] / h[mask]
    return float(np.std(p, ddof=1) / np.sqrt(len(p)))


def _cluster_robust_se_ratio(
    w: np.ndarray, h: np.ndarray, labels: np.ndarray
) -> float:
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


def _cluster_robust_se_mean(
    w: np.ndarray, h: np.ndarray, labels: np.ndarray
) -> float:
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

    s_g = np.array([
        np.sum(p[(labels == c) & mask] - theta) if np.any((labels == c) & mask) else 0.0
        for c in unique
    ])

    v = (g / (g - 1)) * np.sum(s_g**2) / m**2
    return float(np.sqrt(v))


def _cluster_bootstrap(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    estimator_fn,
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
    ssb = sum(
        np.sum(labels == c) * (values[labels == c].mean() - grand_mean) ** 2
        for c in unique
    )
    ssw = sum(
        np.sum((values[labels == c] - values[labels == c].mean()) ** 2)
        for c in unique
    )

    msb = ssb / (g - 1)
    msw = ssw / (n - g) if n > g else 0.0

    m = np.array([np.sum(labels == c) for c in unique], dtype=float)
    m0 = (n - np.sum(m**2) / n) / (g - 1)

    denom = msb + (m0 - 1) * msw
    if denom == 0:
        return 0.0

    icc = (msb - msw) / denom
    return float(max(icc, 0.0))


# ─── Confidence interval construction ────────────────────────────────

def _build_ci(
    est: float, se: float, g: int, level: float = 0.95,
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
        seed: Random seed for bootstrap.

    Returns:
        InferenceResult with estimates, SEs, CIs, and diagnostics.

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
            w, h, labels, _ratio_estimator, reps=bootstrap_reps, rng=rng,
        )
        se_mean_boot, boot_mean_lo, boot_mean_hi = _cluster_bootstrap(
            w, h, labels, _photo_mean_estimator, reps=bootstrap_reps, rng=rng,
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
            naive=naive, cluster=cluster, bootstrap=boot,
            recommended=recommended, method_used=method,
        )

    ratio_se = _pick_se(se_ratio_naive, se_ratio_cluster, se_ratio_boot, rec_method)
    mean_se = _pick_se(se_mean_naive, se_mean_cluster, se_mean_boot, rec_method)

    # ── Confidence intervals ──────────────────────────────────────
    boot_ratio_ci = (boot_ratio_lo, boot_ratio_hi) if boot_ratio_lo is not None else None
    boot_mean_ci = (boot_mean_lo, boot_mean_hi) if boot_mean_lo is not None else None

    ratio_ci = _build_ci(
        ratio, ratio_se.recommended, g, ci_level,
        boot_ci=boot_ratio_ci, recommended_method=rec_method,
    )
    mean_ci = _build_ci(
        photo_mean, mean_se.recommended, g, ci_level,
        boot_ci=boot_mean_ci, recommended_method=rec_method,
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

    diagnostics = Diagnostics(
        n_obs=n,
        n_positive_frames=n_positive,
        n_empty_frames=n_empty,
        empty_frame_rate=n_empty / n if n > 0 else 0.0,
        n_clusters=g,
        cluster_sizes=cluster_sizes,
        cluster_size_mean=m_bar,
        cluster_size_cv=(
            float(cluster_sizes.std() / cluster_sizes.mean())
            if m_bar > 0 else 0.0
        ),
        icc=icc,
        deff=deff,
        n_eff=n_eff,
        se_ratio_cluster_to_naive=se_ratio_cn,
        ratio_bias_approx=_ratio_bias_approx(w, h),
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
