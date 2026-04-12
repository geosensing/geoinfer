"""
Result types for geoinfer.

Structured dataclasses that carry estimates, standard errors,
confidence intervals, and diagnostics from the inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SEResult:
    """Standard error estimates from multiple methods.

    The ``recommended`` field holds the SE the design selects as primary.
    The others are reported for comparison and as robustness checks.
    """

    naive: float
    cluster: float
    bootstrap: float | None
    recommended: float
    method_used: str  # "naive", "cluster", "bootstrap"

    @property
    def ratio_cluster_to_naive(self) -> float:
        """Cluster SE / naive SE.  Near 1 ⇒ independence holds."""
        return self.cluster / self.naive if self.naive > 0 else float("nan")


@dataclass
class CIResult:
    """Confidence intervals from multiple methods."""

    normal: tuple[float, float]         # est ± z * se_recommended
    t: tuple[float, float]              # est ± t_{G-1} * se_recommended
    bootstrap: tuple[float, float] | None  # percentile CI
    recommended: tuple[float, float]
    level: float                        # e.g. 0.95


@dataclass
class Diagnostics:
    """Design and data quality diagnostics."""

    n_obs: int
    n_positive_frames: int
    n_empty_frames: int
    empty_frame_rate: float

    n_clusters: int
    cluster_sizes: np.ndarray
    cluster_size_mean: float
    cluster_size_cv: float

    icc: float
    deff: float
    n_eff: float

    se_ratio_cluster_to_naive: float   # for the photo-mean

    ratio_bias_approx: float           # O(1/N) bias estimate for ratio


@dataclass
class InferenceResult:
    """Full inference output from ``estimate()``.

    Carries point estimates, SEs, CIs, and diagnostics for both
    the ratio estimand and the photo-level mean.
    """

    # Point estimates
    ratio: float
    photo_mean: float

    # Standard errors
    ratio_se: SEResult
    photo_mean_se: SEResult

    # Confidence intervals
    ratio_ci: CIResult
    photo_mean_ci: CIResult

    # Diagnostics
    diagnostics: Diagnostics

    # Design metadata
    design_name: str
    n_obs: int
    n_clusters: int

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "geoinfer: Inference Result",
            "=" * 60,
            f"Design: {self.design_name}",
            f"Observations: {self.n_obs} ({self.diagnostics.n_positive_frames} with h>0, "
            f"{self.diagnostics.n_empty_frames} empty)",
            f"Clusters: {self.n_clusters}",
            "",
            "── Ratio estimand (people-weighted) ──",
            f"  Estimate:  {self.ratio:.4f}",
            f"  SE:        {self.ratio_se.recommended:.4f}  ({self.ratio_se.method_used})",
            f"  95% CI:    [{self.ratio_ci.recommended[0]:.4f}, "
            f"{self.ratio_ci.recommended[1]:.4f}]",
            "",
            "── Photo-level mean (location-weighted) ──",
            f"  Estimate:  {self.photo_mean:.4f}",
            f"  SE:        {self.photo_mean_se.recommended:.4f}  "
            f"({self.photo_mean_se.method_used})",
            f"  95% CI:    [{self.photo_mean_ci.recommended[0]:.4f}, "
            f"{self.photo_mean_ci.recommended[1]:.4f}]",
            "",
            "── Diagnostics ──",
            f"  ICC:                {self.diagnostics.icc:.4f}",
            f"  Design effect:      {self.diagnostics.deff:.2f}",
            f"  Effective N:        {self.diagnostics.n_eff:.1f}",
            f"  SE ratio (cl/nv):   {self.diagnostics.se_ratio_cluster_to_naive:.3f}",
            f"  Ratio bias O(1/N):  {self.diagnostics.ratio_bias_approx:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InferenceResult(ratio={self.ratio:.4f}, photo_mean={self.photo_mean:.4f}, "
            f"n={self.n_obs}, G={self.n_clusters}, design='{self.design_name}')"
        )
