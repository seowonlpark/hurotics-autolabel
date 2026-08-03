"""Hold the vectorized feature path to its scalar references.

    python -m stages.s2_ml.verify_features

`features.swap_counts` computes for every window at once what `rest.swap_count` computes
for one, via a prefix sum. That rewrite is where a silent error would hide: it is exact on
most windows and off by one on the rest, which no accuracy number would ever reveal as a
bug rather than as noise. DOMAIN_NOTES §11.1 is a list of tests that were themselves
broken, so this one compares against the reference implementation on random inputs
including the degenerate cases (flat signal, all-committed signal, window shorter than the
hysteresis) rather than against a remembered expectation.
"""

from __future__ import annotations

import numpy as np

from stages.s2_ml.features import WindowSpec, _min_over_parts, swap_counts
from stages.s2_ml.rest import swap_count


def check_swaps(n_signals: int = 400, seed: int = 0) -> int:
    """swap_counts vs rest.swap_count over random signals and random windows."""
    rng = np.random.default_rng(seed)
    bad = checked = 0
    for _ in range(n_signals):
        n = int(rng.integers(20, 400))
        kind = rng.random()
        if kind < 0.15:
            d = np.zeros(n)                                   # never commits
        elif kind < 0.30:
            d = np.full(n, 50.0)                              # commits once, never swaps
        elif kind < 0.60:
            d = 20 * np.sin(np.arange(n) * rng.uniform(0.02, 0.4)) + rng.normal(0, .5, n)
        else:
            d = rng.normal(0, rng.choice([0.2, 1.0, 5.0]), n) + rng.normal(0, 3)
        for _ in range(6):
            a = int(rng.integers(0, max(1, n - 5)))
            w = int(rng.integers(5, n - a + 1))
            got = int(swap_counts(d, np.array([a]), w)[0])
            want = swap_count(d[a:a + w])
            checked += 1
            bad += got != want
    print(f"[verify] swap_counts: {checked:,} windows, {bad} mismatches")
    return bad


def check_min_over_parts(seed: int = 0) -> int:
    """_min_over_parts vs an explicit loop, and its ordering property."""
    rng = np.random.default_rng(seed)
    D = rng.normal(0, 10, (200, 200))
    bad = 0
    for k in (2, 4, 6):
        want = np.array([min(np.ptp(row[i * (200 // k):(i + 1) * (200 // k)])
                             for i in range(k)) for row in D])
        bad += int(not np.allclose(_min_over_parts(D, k), want))
    # more parts can only tighten the minimum
    bad += int(not np.all(_min_over_parts(D, 4) <= _min_over_parts(D, 2) + 1e-12))
    print(f"[verify] _min_over_parts: {bad} mismatches")
    return bad


def check_spec() -> int:
    """WindowSpec converts seconds to samples without drift."""
    s = WindowSpec(window_s=2.0, stride_s=0.25, fs_hz=100.0)
    bad = int(s.n != 200) + int(s.step != 25)
    print(f"[verify] WindowSpec: {bad} mismatches")
    return bad


def main() -> None:
    bad = check_swaps() + check_min_over_parts() + check_spec()
    if bad:
        raise SystemExit(f"[verify] FAILED with {bad} mismatches")
    print("[verify] all feature checks passed")


if __name__ == "__main__":
    main()
