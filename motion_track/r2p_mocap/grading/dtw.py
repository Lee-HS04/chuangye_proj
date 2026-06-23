"""
Dynamic Time Warping for motion sequence comparison.

Uses scipy cdist for the distance matrix (vectorised C) and a
Sakoe-Chiba band to bound the DP so it stays O(n*w) not O(n²).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray,
                 window_ratio: float = 0.15,
                 metric: str = "euclidean") -> float:
    """
    Normalised DTW distance between two (T, D) sequences.
    Returns a value in [0, ∞) — lower means more similar.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    w   = max(int(max(n, m) * window_ratio), abs(n - m), 1)
    D   = cdist(seq_a, seq_b, metric=metric)   # (n, m) pairwise distances

    cost = np.full((n, m), np.inf)
    cost[0, 0] = D[0, 0]

    for i in range(n):
        j_lo = max(0, i - w)
        j_hi = min(m - 1, i + w)
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                continue
            p = np.inf
            if i > 0 and cost[i-1, j] < p:   p = cost[i-1, j]
            if j > 0 and cost[i, j-1] < p:   p = cost[i, j-1]
            if i > 0 and j > 0 and cost[i-1, j-1] < p: p = cost[i-1, j-1]
            cost[i, j] = D[i, j] + p

    return float(cost[n-1, m-1]) / (n + m)


def score_from_distance(dist: float,
                        perfect_thresh: float = 0.02,
                        fail_thresh: float    = 0.30) -> tuple[int, str]:
    """
    Convert a DTW distance into an NBA 2K-style grade.
    Returns (score 0–100, letter A/B/C/D/F).
    """
    t     = max(0.0, min(1.0, (dist - perfect_thresh) / (fail_thresh - perfect_thresh)))
    score = int(round((1.0 - t) * 100))
    if   score >= 90: grade = "A"
    elif score >= 80: grade = "B"
    elif score >= 70: grade = "C"
    elif score >= 60: grade = "D"
    else:             grade = "F"
    return score, grade
