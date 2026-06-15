"""
Tests for the production / pipeline surface.

``estimate_from_csv`` is tested directly (no optional deps). The real
geo_sampling -> allocator path is tested only when the ``pipeline`` extra is
installed; otherwise those tests skip cleanly.
"""

import importlib.util
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from geoinference import estimate_from_csv
from geoinference.simulate import PopulationFactory, SimConfig, evaluate_scene

_HAS_ALLOCATOR = importlib.util.find_spec("allocator") is not None
_DELHI = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "allocator",
    "examples",
    "inputs",
    "delhi-roads-1k.csv",
)


def _write_frames(path: str, n: int = 300, g: int = 12, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pd.DataFrame(
        {
            "n_women": rng.binomial(8, 0.3, n),
            "n_people": rng.poisson(8, n) + 1,
            "itinerary_id": rng.integers(0, g, n),
            "longitude": rng.uniform(77.0, 77.05, n),
            "latitude": rng.uniform(28.6, 28.65, n),
            "timestamp": np.sort(rng.uniform(0, 1e6, n)),
        }
    ).to_csv(path, index=False)


class TestEstimateFromCsv(unittest.TestCase):
    def test_full_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "frames.csv")
            _write_frames(p)
            res = estimate_from_csv(p, bootstrap=False)
        self.assertGreater(res.ratio, 0.0)
        self.assertLess(res.ratio, 1.0)
        self.assertEqual(res.ratio_se.method_used, "cluster")
        self.assertFalse(np.isnan(res.diagnostics.n_eff_space))
        self.assertFalse(np.isnan(res.diagnostics.n_eff_time))

    def test_minimal_columns_no_optional(self):
        # Only the required counts present: optional vars are ignored, no error.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "frames.csv")
            pd.DataFrame({"n_women": [2, 3, 4], "n_people": [10, 10, 10]}).to_csv(p, index=False)
            res = estimate_from_csv(p, bootstrap=False)
        self.assertAlmostEqual(res.ratio, 0.3)
        self.assertTrue(np.isnan(res.diagnostics.n_eff_space))

    def test_missing_required_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "frames.csv")
            pd.DataFrame({"n_people": [10, 10]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                estimate_from_csv(p)


@unittest.skipUnless(
    _HAS_ALLOCATOR and os.path.exists(_DELHI),
    "pipeline extra (allocator) and the bundled Delhi roads CSV required",
)
class TestRealPipeline(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_subsample_scene_and_validate(self):
        from geoinference.pipeline import points_from_roads, subsample_scene

        uni = points_from_roads(_DELHI, per_segment=4)
        self.assertEqual(len(uni), 4000)
        sample_idx, scene = subsample_scene(
            uni, n_sample=200, method="random_partition", n_itineraries=40, seed=1
        )
        self.assertEqual(len(scene), len(sample_idx))
        self.assertEqual(scene.n_itineraries, 40)
        self.assertGreater(scene.day_span, 0.0)

        cfg = SimConfig(range_s_m=800.0, diurnal_amp=0.0, sd_t=0.0, n_sims=40)
        factory = PopulationFactory(cfg, uni["longitude"].to_numpy(), uni["latitude"].to_numpy())
        res = evaluate_scene(
            factory,
            sample_idx,
            scene.itinerary_id,
            scene.time_of_day_min,
            scene.timestamp_s,
            cfg,
            se_method="cluster",
            spatial_diag=False,
        )
        self.assertTrue(np.isfinite(res.bias))
        self.assertTrue(np.isfinite(res.coverage))
        self.assertGreater(res.coverage, 0.5)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()
