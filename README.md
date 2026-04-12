# geoinfer

Design-based inference for spatially distributed observation surveys.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## What this does

You collected street-level data using [geo-sampling](https://github.com/geosensing/geo-sampling) and [allocator](https://github.com/geosensing/allocator). You have a DataFrame of annotated frames with women counts and people counts. You want an unbiased estimate of the gender ratio with a correct standard error.

`geoinfer` takes the annotated data plus a description of how it was collected (the *design*) and produces point estimates, standard errors, confidence intervals, and diagnostics — with the right SE estimator chosen automatically.

## Install

```bash
pip install geoinfer
```

## Quick start

```python
from geoinfer import PointDesign, WalkDesign, estimate

# Point-based pipeline (geo-sampling → allocator → annotate)
design = PointDesign(sampling="srs", cluster_var="itinerary_id")
result = estimate(df, "n_women", "n_people", design=design)
print(result.summary())

# Walk-based pipeline (random walks with GoPro)
design = WalkDesign(walk_var="walk_id")
result = estimate(df, "n_women", "n_people", design=design)
print(result.summary())
```

## Two estimands

**Ratio** (people-weighted): `sum(women) / sum(people)` — what fraction of observed *people* are women.

**Photo-level mean** (location-weighted): `mean(w_i / h_i)` for frames with `h_i > 0` — at a random *location*, what fraction of people present are women.

## Two designs

**PointDesign**: Discrete locations sampled by geo-sampling, grouped into itineraries by allocator. Under SRS, observations are approximately independent (our simulation finding), so the naive SE is the primary estimator and the cluster-robust SE is a robustness check.

**WalkDesign**: Continuous random walks on the road network. Walks are independent by construction. The between-walk SE is the correct and only estimator.

## What you get back

```
============================================================
geoinfer: Inference Result
============================================================
Design: point_srs_clustered
Observations: 200 (195 with h>0, 5 empty)
Clusters: 10

── Ratio estimand (people-weighted) ──
  Estimate:  0.2987
  SE:        0.0109  (naive)
  95% CI:    [0.2773, 0.3201]

── Photo-level mean (location-weighted) ──
  Estimate:  0.3014
  SE:        0.0128  (naive)
  95% CI:    [0.2764, 0.3265]

── Diagnostics ──
  ICC:                0.0312
  Design effect:      1.59
  Effective N:        123.0
  SE ratio (cl/nv):   1.142
  Ratio bias O(1/N):  -0.000089
============================================================
```

## License

MIT
