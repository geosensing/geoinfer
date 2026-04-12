"""
Design specifications for geoinfer.

A design object encodes how the data was collected — the sampling
mechanism, the clustering structure, the weighting scheme — so that
the inference functions can choose the correct variance estimator.

The design is metadata about the data, not the data itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PointDesign:
    """Design for point-based spatial sampling (geo-sampling + allocator pipeline).

    The data consists of observations at discrete sampled locations,
    grouped into itineraries for field collection.

    Args:
        sampling: Sampling method used.
            - "srs": Simple random sampling (equal probability).
            - "grts": Generalized Random Tessellation Stratified.
            - "pps": Probability proportional to size.
        cluster_var: Column name identifying itineraries/clusters.
            If None, observations are treated as independent (no clustering).
        weight_var: Column name with inclusion probabilities or design weights.
            Required for "pps" and "grts" designs. Ignored for "srs".
        annotation_frac: Fraction of collected frames that were annotated.
            If < 1.0, inference accounts for the annotation subsampling.
            Can also be a column name for observation-level annotation probabilities.
        fpc: Finite population correction. If provided, the population size M
            from which N units were sampled. Reduces variance by factor (1 - N/M).
    """

    sampling: Literal["srs", "grts", "pps"] = "srs"
    cluster_var: str | None = None
    weight_var: str | None = None
    annotation_frac: float | str = 1.0
    fpc: int | None = None

    @property
    def name(self) -> str:
        parts = [f"point_{self.sampling}"]
        if self.cluster_var:
            parts.append("clustered")
        if self.weight_var:
            parts.append("weighted")
        return "_".join(parts)

    @property
    def has_clusters(self) -> bool:
        return self.cluster_var is not None

    @property
    def has_weights(self) -> bool:
        return self.weight_var is not None

    @property
    def recommended_se_method(self) -> str:
        """Which SE estimator the design recommends as primary.

        Under SRS with post-hoc itinerary clustering, observations are
        approximately independent (our simulation finding), so the naive
        SE is correct. The cluster-robust SE is reported as a robustness
        check. For PPS/GRTS, the Horvitz-Thompson linearization SE
        (which is cluster-robust by construction) is recommended.
        """
        if self.sampling == "srs" and self.cluster_var is not None:
            return "naive"
        elif self.sampling in ("pps", "grts"):
            return "cluster"
        elif self.cluster_var is None:
            return "naive"
        else:
            return "naive"

    def __post_init__(self) -> None:
        if self.sampling in ("pps", "grts") and self.weight_var is None:
            raise ValueError(
                f"Design '{self.sampling}' requires weight_var "
                f"(inclusion probabilities or design weights)."
            )
        if isinstance(self.annotation_frac, (int, float)):
            if not 0 < self.annotation_frac <= 1.0:
                raise ValueError("annotation_frac must be in (0, 1].")


@dataclass
class WalkDesign:
    """Design for random-walk transect sampling.

    The data consists of observations along independent random walks
    on the road network. Each walk is an independent Markov chain
    realization, and inference uses between-walk variance.

    Args:
        walk_var: Column name identifying independent walks.
        spacing_m: Annotation spacing along the walk in meters.
            Used to compute effective sample size per walk
            (n_eff ≈ walk_length / (2 * correlation_length)).
            If None, all annotated frames are used without
            effective-N adjustment.
    """

    walk_var: str = "walk_id"
    spacing_m: float | None = None

    @property
    def name(self) -> str:
        return "walk_transect"

    @property
    def has_clusters(self) -> bool:
        return True  # walks are always the clustering unit

    @property
    def cluster_var(self) -> str:
        return self.walk_var

    @property
    def has_weights(self) -> bool:
        return False  # self-weighting by construction

    @property
    def recommended_se_method(self) -> str:
        """Between-walk SE is always the correct estimator for walk designs."""
        return "cluster"

    def __post_init__(self) -> None:
        if self.spacing_m is not None and self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive.")


# Union type for design dispatch
Design = PointDesign | WalkDesign
