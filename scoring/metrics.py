"""Shared FAR / FRR / EER helpers for score evaluation.

Both the deep-only baseline in :mod:`training.evaluate` and the fusion fitter in
:mod:`training.fit_fusion` interpret scores the same way: a higher score means "more
likely the same signer", and a pair is accepted when ``score >= threshold``. Keeping
the metric code in one module means the two scripts cannot drift apart, and the deep
baseline numbers stay identical to the pre-fusion implementation.
"""

from __future__ import annotations

import numpy as np


def rates_at_threshold(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
    threshold: float,
) -> tuple[float, float]:
    """Return ``(FAR, FRR)`` at ``threshold``.

    A genuine pair is rejected when its score falls below the threshold, and a
    negative pair is falsely accepted when its score reaches it. This is the
    convention used throughout the repository.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    if genuine_array.size == 0 or negative_array.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")
    false_rejection_rate = float((genuine_array < threshold).mean())
    false_acceptance_rate = float((negative_array >= threshold).mean())
    return false_acceptance_rate, false_rejection_rate


def frr_at_threshold(
    genuine: np.ndarray | list[float],
    threshold: float,
) -> float:
    """Return the false-rejection rate for ``genuine`` at ``threshold``.

    Split out from :func:`rates_at_threshold` because an operating point is often
    chosen from the *negative* scores alone (see :func:`threshold_at_far`) and only
    the resulting FRR is needed.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    if genuine_array.size == 0:
        raise ValueError("genuine score array must be non-empty")
    return float((genuine_array < threshold).mean())


def threshold_at_far(
    negatives: np.ndarray | list[float],
    target_far: float,
) -> tuple[float, float]:
    """Return the largest threshold whose FAR on ``negatives`` is at most ``target_far``.

    EER is a summary statistic. A deployed system is usually specified as "accept at
    most ``target_far`` of the impostors, then reject as few genuine users as
    possible". Because FAR is non-increasing in the threshold and FRR is
    non-decreasing, the threshold that satisfies the FAR budget *and* minimises FRR is
    the largest one that still satisfies the budget. That is exactly what this returns,
    along with the FAR actually achieved.

    The decision rule everywhere in this repository is ``score >= threshold``, so the
    threshold is set to the ``allowed``-th highest negative score: exactly ``allowed``
    negatives are accepted, where ``allowed = floor(target_far * n_negatives)``.

    Args:
        negatives: Impostor scores, e.g. the validation split's skilled forgeries.
        target_far: FAR budget in ``[0, 1]``.

    Raises:
        ValueError: If ``negatives`` is empty or ``target_far`` is outside ``[0, 1]``.
    """
    negative_array = np.asarray(negatives, dtype=np.float64)
    if negative_array.size == 0:
        raise ValueError("negative score array must be non-empty")
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be in [0, 1]")

    descending = np.sort(negative_array)[::-1]
    allowed = int(np.floor(target_far * descending.size))
    if allowed <= 0:
        # No impostor may be accepted, so sit just above the highest negative score.
        return float(np.nextafter(descending[0], np.inf)), 0.0
    if allowed >= descending.size:
        return float(descending[-1]), 1.0
    threshold = float(descending[allowed - 1])
    far = float((negative_array >= threshold).mean())
    return threshold, far


def eer_threshold(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
) -> tuple[float, float]:
    """Return the threshold where FAR and FRR are closest, and their mean (the EER).

    The candidate set is every observed score. The first candidate with the smallest
    absolute FAR/FRR gap wins, matching the original single-pass implementation so
    baseline results are reproducible.

    The sweep is evaluated with sorted arrays and ``searchsorted`` rather than a Python
    loop over candidates. At a threshold ``t`` the rates are exactly
    ``count(negative >= t) / n_negative`` and ``count(genuine < t) / n_genuine``, which
    ``searchsorted`` reproduces in ``O(n log n)`` total instead of ``O(n^2)``. The
    returned threshold and EER are identical to the naive loop; the speed-up matters
    because a full validation split contributes ~40k random negative pairs and the
    fusion fitter scores several hundred thousand pairs.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    if genuine_array.size == 0 or negative_array.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")

    candidates = np.unique(np.concatenate([genuine_array, negative_array]))
    genuine_sorted = np.sort(genuine_array)
    negative_sorted = np.sort(negative_array)

    # count(genuine < t) and count(negative < t) for every candidate in one pass.
    genuine_below = np.searchsorted(genuine_sorted, candidates, side="left")
    negative_below = np.searchsorted(negative_sorted, candidates, side="left")
    false_rejection = genuine_below / genuine_array.size
    false_acceptance = (negative_array.size - negative_below) / negative_array.size
    gaps = np.abs(false_acceptance - false_rejection)

    # ``argmin`` returns the first minimum, matching the loop's strict ``<`` comparison.
    best_index = int(np.argmin(gaps))
    best_threshold = float(candidates[best_index])
    equal_error_rate = float(
        (false_acceptance[best_index] + false_rejection[best_index]) / 2.0
    )
    return best_threshold, equal_error_rate


def eer(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
) -> float:
    """Return only the equal error rate at the best threshold."""
    return eer_threshold(genuine, negatives)[1]
