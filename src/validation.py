"""Quantitative validation against established analog sites.

The question is simple: does a profile rank the sites scientists already
use for that environment above ordinary land? That is a ranking question,
so the measure is ROC AUC: the probability that a randomly chosen known
site outscores a randomly chosen land cell. 0.5 is chance, 1.0 is perfect.

Three things keep the number honest:

* **Sites are split by why they were chosen.** Sites picked for their
  environment should rank high. Sites picked for geology (an impact crater,
  an anorthosite body) or for operations (a habitat) need not, because
  climate and terrain cannot see geology. Pooling them would hide both the
  method's strength and its blind spot.
* **The nearest cell, not the best nearby.** Each site is scored at its
  nearest land cell, never the best cell in a neighbourhood.
* **A baseline.** The same AUC for aridity alone. If seven dimensions do no
  better than one, the extra dimensions are decoration.

With fifteen-odd sites per group the uncertainty is large, so every AUC is
reported with a bootstrap 95% interval over the sites.

Nothing is fitted to these sites, so there is no training set to leak into
and no need for cross-validation. Weights and targets were set from
planetary conditions before the expanded site list was assembled.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import analog
from known_sites import ENVIRONMENT, GEOLOGY, OPERATIONS, sites_for
from planet_profiles import PlanetProfile

BOOTSTRAP_ROUNDS = 2000
NEGATIVE_SAMPLE = 50000
RANDOM_SEED = 2026


def _auc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Mann-Whitney form of ROC AUC, ties counted as half."""
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    negatives = np.sort(negatives)
    below = np.searchsorted(negatives, positives, side="left")
    at_or_below = np.searchsorted(negatives, positives, side="right")
    wins = below + 0.5 * (at_or_below - below)
    return float(wins.sum() / (len(positives) * len(negatives)))


def _bootstrap(positives: np.ndarray, negatives: np.ndarray, rng) -> List[float]:
    if len(positives) < 2:
        return [float("nan"), float("nan")]
    stats = [
        _auc(rng.choice(positives, size=len(positives), replace=True), negatives)
        for _ in range(BOOTSTRAP_ROUNDS)
    ]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def site_scores(grid: pd.DataFrame, scores: np.ndarray, sites) -> np.ndarray:
    return np.array(
        [scores[analog.nearest_scored_cell(grid, scores, s.latitude, s.longitude)]
         for s in sites]
    )


def baseline_scores(grid: pd.DataFrame, percentiles: pd.DataFrame) -> np.ndarray:
    """Aridity alone: the driest land scores highest."""
    return 1.0 - percentiles["annual_precip_mm"].to_numpy()


def evaluate(
    grid: pd.DataFrame,
    percentiles: pd.DataFrame,
    profile: PlanetProfile,
    scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """AUC per site group for one profile, with the aridity baseline."""
    rng = np.random.default_rng(RANDOM_SEED)
    if scores is None:
        scores = analog.score(grid, percentiles, profile)["analog_score"].to_numpy()
    baseline = baseline_scores(grid, percentiles)

    valid = np.flatnonzero(np.isfinite(scores) & np.isfinite(baseline))
    sample = rng.choice(valid, size=min(NEGATIVE_SAMPLE, len(valid)), replace=False)

    rows: List[Dict] = []
    groups = [("environment", ENVIRONMENT), ("geology", GEOLOGY),
              ("operations", OPERATIONS), ("all", "")]
    for label, basis in groups:
        sites = sites_for(profile.site_key(), basis)
        if not sites:
            continue
        pos = site_scores(grid, scores, sites)
        pos_base = site_scores(grid, baseline, sites)
        rows.append(
            {
                "profile": profile.key,
                "sites": label,
                "n": len(sites),
                "auc": round(_auc(pos, scores[sample]), 3),
                "auc_ci": [round(x, 3) for x in _bootstrap(pos, scores[sample], rng)],
                "baseline_auc": round(_auc(pos_base, baseline[sample]), 3),
            }
        )
    return pd.DataFrame(rows)
