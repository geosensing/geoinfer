"""
Tests for geoinfer.

Unit tests for components plus simulation-based coverage tests
that verify the SEs are accurate and CIs achieve nominal coverage.
"""

import unittest

import numpy as np
import pandas as pd

from geoinfer import InferenceResult, PointDesign, WalkDesign, estimate


def _make_test_data(
    n: int = 200,
    g: int = 10,
    true_p: float = 0.3,
    lam: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate test data mimicking the geosensing pipeline."""
    rng = np.random.default_rng(seed)
    cluster_ids = np.repeat(np.arange(g), n // g)
    if len(cluster_ids) < n:
        cluster_ids = np.concatenate([cluster_ids, np.zeros(n - len(cluster_ids), dtype=int)])

    h = rng.poisson(lam, n)
    w = rng.binomial(h, true_p)

    return pd.DataFrame({
        "n_women": w,
        "n_people": h,
        "itinerary_id": cluster_ids,
        "walk_id": cluster_ids,
    })


class TestDesigns(unittest.TestCase):
    """Test design object construction and validation."""

    def test_point_design_srs(self):
        d = PointDesign(sampling="srs", cluster_var="itinerary_id")
        self.assertEqual(d.name, "point_srs_clustered")
        self.assertTrue(d.has_clusters)
        self.assertFalse(d.has_weights)
        self.assertEqual(d.recommended_se_method, "naive")

    def test_point_design_pps_requires_weights(self):
        with self.assertRaises(ValueError):
            PointDesign(sampling="pps")

    def test_point_design_pps_with_weights(self):
        d = PointDesign(sampling="pps", weight_var="inc_prob")
        self.assertTrue(d.has_weights)
        self.assertEqual(d.recommended_se_method, "cluster")

    def test_walk_design(self):
        d = WalkDesign(walk_var="walk_id")
        self.assertEqual(d.name, "walk_transect")
        self.assertTrue(d.has_clusters)
        self.assertEqual(d.cluster_var, "walk_id")
        self.assertEqual(d.recommended_se_method, "cluster")

    def test_walk_design_bad_spacing(self):
        with self.assertRaises(ValueError):
            WalkDesign(spacing_m=-10)

    def test_annotation_frac_validation(self):
        with self.assertRaises(ValueError):
            PointDesign(annotation_frac=0.0)
        with self.assertRaises(ValueError):
            PointDesign(annotation_frac=1.5)


class TestEstimateBasic(unittest.TestCase):
    """Test basic estimate() functionality."""

    def setUp(self):
        self.df = _make_test_data(n=200, g=10, true_p=0.3, seed=42)

    def test_returns_inference_result(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertIsInstance(result, InferenceResult)

    def test_ratio_in_range(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        self.assertGreater(result.ratio, 0.0)
        self.assertLess(result.ratio, 1.0)

    def test_photo_mean_in_range(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        self.assertGreater(result.photo_mean, 0.0)
        self.assertLess(result.photo_mean, 1.0)

    def test_se_positive(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertGreater(result.ratio_se.naive, 0)
        self.assertGreater(result.ratio_se.cluster, 0)
        self.assertGreater(result.photo_mean_se.naive, 0)
        self.assertGreater(result.photo_mean_se.cluster, 0)

    def test_ci_contains_estimate(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        lo, hi = result.ratio_ci.recommended
        self.assertLessEqual(lo, result.ratio)
        self.assertGreaterEqual(hi, result.ratio)

    def test_no_cluster_var(self):
        """Without cluster_var, each obs is its own cluster."""
        design = PointDesign(sampling="srs")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, len(self.df))

    def test_walk_design(self):
        design = WalkDesign(walk_var="walk_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.design_name, "walk_transect")
        self.assertEqual(result.ratio_se.method_used, "cluster")

    def test_missing_column_raises(self):
        design = PointDesign(cluster_var="nonexistent")
        with self.assertRaises(ValueError):
            estimate(self.df, "n_women", "n_people", design=design)

    def test_summary_string(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        s = result.summary()
        self.assertIn("Ratio estimand", s)
        self.assertIn("Photo-level mean", s)
        self.assertIn("Diagnostics", s)

    def test_diagnostics_populated(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        d = result.diagnostics
        self.assertEqual(d.n_obs, 200)
        self.assertGreater(d.n_positive_frames, 0)
        self.assertGreaterEqual(d.n_empty_frames, 0)
        self.assertEqual(d.n_clusters, 10)
        self.assertGreater(d.icc, -0.01)  # ICC can be slightly negative
        self.assertGreaterEqual(d.deff, 1.0)

    def test_bootstrap_produces_ci(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(
            self.df, "n_women", "n_people", design=design,
            bootstrap=True, bootstrap_reps=199, seed=99,
        )
        self.assertIsNotNone(result.ratio_ci.bootstrap)
        self.assertIsNotNone(result.ratio_se.bootstrap)


class TestEstimateEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def test_all_empty_frames(self):
        df = pd.DataFrame({"n_women": [0, 0, 0], "n_people": [0, 0, 0]})
        result = estimate(df, "n_women", "n_people", bootstrap=False)
        self.assertTrue(np.isnan(result.ratio))
        self.assertTrue(np.isnan(result.photo_mean))

    def test_single_observation(self):
        df = pd.DataFrame({"n_women": [3], "n_people": [10]})
        result = estimate(df, "n_women", "n_people", bootstrap=False)
        self.assertAlmostEqual(result.ratio, 0.3)
        self.assertAlmostEqual(result.photo_mean, 0.3)

    def test_two_clusters(self):
        df = pd.DataFrame({
            "n_women": [2, 3, 4, 5],
            "n_people": [10, 10, 10, 10],
            "cluster": [0, 0, 1, 1],
        })
        design = PointDesign(cluster_var="cluster")
        result = estimate(df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, 2)
        self.assertGreater(result.ratio_se.cluster, 0)

    def test_single_cluster(self):
        df = pd.DataFrame({
            "n_women": [2, 3, 4],
            "n_people": [10, 10, 10],
            "cluster": [0, 0, 0],
        })
        design = PointDesign(cluster_var="cluster")
        result = estimate(df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, 1)
        # Cluster SE should be NaN with one cluster
        self.assertTrue(np.isnan(result.ratio_se.cluster))


class TestCoverageSimulation(unittest.TestCase):
    """Simulation-based tests for SE accuracy and CI coverage.

    These tests verify the core statistical guarantees:
    1. Negligible bias
    2. SE estimator tracks true SD
    3. CI achieves nominal coverage
    """

    def _run_coverage(self, design_fn, n=200, g=10, true_p=0.3, lam=8.0,
                      n_sims=300, ci_key="recommended"):
        """Run coverage simulation and return metrics."""
        rng = np.random.default_rng(42)

        ratio_hats = []
        mean_hats = []
        ratio_covers = 0
        mean_covers = 0
        ratio_ses = []
        mean_ses = []
        valid = 0

        for sim in range(n_sims):
            df = _make_test_data(n=n, g=g, true_p=true_p, lam=lam, seed=rng.integers(2**31))
            design = design_fn()
            result = estimate(df, "n_women", "n_people", design=design, bootstrap=False)

            if np.isnan(result.ratio) or np.isnan(result.photo_mean):
                continue

            valid += 1
            ratio_hats.append(result.ratio)
            mean_hats.append(result.photo_mean)
            ratio_ses.append(result.ratio_se.recommended)
            mean_ses.append(result.photo_mean_se.recommended)

            lo_r, hi_r = getattr(result.ratio_ci, ci_key) if ci_key != "recommended" else result.ratio_ci.recommended
            lo_m, hi_m = getattr(result.photo_mean_ci, ci_key) if ci_key != "recommended" else result.photo_mean_ci.recommended
            ratio_covers += (lo_r <= true_p <= hi_r)
            mean_covers += (lo_m <= true_p <= hi_m)

        ratio_hats = np.array(ratio_hats)
        mean_hats = np.array(mean_hats)

        return {
            "ratio_bias": float(np.mean(ratio_hats) - true_p),
            "mean_bias": float(np.mean(mean_hats) - true_p),
            "ratio_sd": float(np.std(ratio_hats, ddof=1)),
            "mean_sd": float(np.std(mean_hats, ddof=1)),
            "ratio_mean_se": float(np.mean(ratio_ses)),
            "mean_mean_se": float(np.mean(mean_ses)),
            "ratio_coverage": ratio_covers / valid if valid > 0 else 0,
            "mean_coverage": mean_covers / valid if valid > 0 else 0,
            "n_valid": valid,
        }

    def test_point_srs_bias(self):
        """Bias should be negligible under SRS."""
        results = self._run_coverage(
            lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
        )
        self.assertAlmostEqual(results["ratio_bias"], 0, delta=0.01)
        self.assertAlmostEqual(results["mean_bias"], 0, delta=0.01)

    def test_point_srs_se_accuracy(self):
        """Recommended SE should track true SD (ratio near 1)."""
        results = self._run_coverage(
            lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
        )
        ratio_se_ratio = results["ratio_mean_se"] / results["ratio_sd"]
        mean_se_ratio = results["mean_mean_se"] / results["mean_sd"]
        # SE should be within 30% of true SD
        self.assertGreater(ratio_se_ratio, 0.7)
        self.assertLess(ratio_se_ratio, 1.3)
        self.assertGreater(mean_se_ratio, 0.7)
        self.assertLess(mean_se_ratio, 1.3)

    def test_point_srs_coverage(self):
        """95% CI should cover at least 90% (allowing for simulation noise)."""
        results = self._run_coverage(
            lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
        )
        self.assertGreater(results["ratio_coverage"], 0.90)
        self.assertGreater(results["mean_coverage"], 0.90)

    def test_walk_design_coverage(self):
        """Walk design should also achieve reasonable coverage."""
        results = self._run_coverage(
            lambda: WalkDesign(walk_var="walk_id"),
        )
        self.assertGreater(results["ratio_coverage"], 0.88)
        self.assertGreater(results["mean_coverage"], 0.88)

    def test_no_cluster_coverage(self):
        """Without clustering, naive SE should work well."""
        results = self._run_coverage(
            lambda: PointDesign(sampling="srs"),
        )
        self.assertGreater(results["ratio_coverage"], 0.90)
        self.assertGreater(results["mean_coverage"], 0.90)


if __name__ == "__main__":
    unittest.main()
