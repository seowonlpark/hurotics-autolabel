# error taxonomy -- a faithful port of locoeval/diagnose.py bucketing
# ported (not reinvented) so our classifier is scored by the SAME yardstick as the
# incumbent; thresholds/precedence copied exactly (changing either breaks comparability).
# every row lands in one bucket by precedence:
#   correct > omission/swallowed/edge_omission > flicker > late > early > steady_confusion
# feed it row-level predictions from dense inference (predict.py), not window labels --
# the thresholds are ms-scale. runs per segment (Section 3.1). see README / DOMAIN_NOTES Section 7.

from __future__ import annotations

import numpy as np

# copied verbatim from diagnose.py -- do not tune, they define comparability
FLICKER_MAX_MS = 200.0 # pred run shorter than this, flanked by equal others => flicker
LAG_MAX_MS = 1000.0 # beyond this, a lag is not detection jitter
SUSTAINED_FRACTION = 0.5 # early/late over >= this fraction of the segment AND past
                         # LAG_MAX_MS is a sustained misclassification, not jitter

ERROR_BUCKETS = [
    "omission",
    "swallowed",
    "edge_omission",
    "flicker",
    "late",
    "early",
    "steady_confusion",
]
UNCLASSIFIED = "unclassified"
CORRECT = "correct"


# indices where the label changes, with the from/to pair
def transitions(labels: np.ndarray, times: np.ndarray) -> list[dict]:
    a = np.asarray(labels)
    t = np.asarray(times, dtype=float)
    idx = np.where(a[1:] != a[:-1])[0] + 1
    return [{"idx": int(i), "time": float(t[i]), "frm": int(a[i - 1]), "to": int(a[i])}
            for i in idx]


# contiguous runs of one label, with duration in ms
def segments(labels: np.ndarray, times: np.ndarray) -> list[dict]:
    a = np.asarray(labels)
    t = np.asarray(times, dtype=float)
    n = len(a)
    if n == 0:
        return []
    bounds = [0] + list(np.where(a[1:] != a[:-1])[0] + 1) + [n]
    last_end_t = t[-1] + (t[-1] - t[-2]) if n > 1 else t[-1]
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        end_t = t[e] if e < n else last_end_t
        out.append({"label": int(a[s]), "start": int(s), "end": int(e),
                    "start_time": float(t[s]), "end_time": float(end_t),
                    "dur_ms": float(end_t - t[s])})
    return out


# the labels flanking [s, e), or None at a recording boundary
def _flanks(arr: np.ndarray, s: int, e: int, n: int):
    return (arr[s - 1] if s > 0 else None), (arr[e] if e < n else None)


# claim helper: assign label to still-unclassified rows in [s, e); this is the precedence
# mechanism -- earlier buckets claim first, so nothing is ever overwritten
def _claim(bucket: np.ndarray, s: int, e: int, label: str) -> None:
    region = bucket[s:e]
    region[region == UNCLASSIFIED] = label


# one gt transition and how the prediction handled it
class TransitionResult:
    __slots__ = ("frm", "to", "offset_ms", "gt_idx", "full_segment_miss",
                 "sustained_early", "sustained_late", "late_end_idx",
                 "early_start_idx", "omission_kind")

    def __init__(self, frm, to, offset_ms, gt_idx, full_segment_miss, sustained_early,
                 sustained_late, late_end_idx, early_start_idx, omission_kind=None):
        self.frm, self.to, self.offset_ms = frm, to, offset_ms
        self.gt_idx, self.full_segment_miss = gt_idx, full_segment_miss
        self.sustained_early, self.sustained_late = sustained_early, sustained_late
        self.late_end_idx, self.early_start_idx = late_end_idx, early_start_idx
        self.omission_kind = omission_kind

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


# for each gt transition into class X, read forward until pred catches up to X
def transition_timing(gt: np.ndarray, pred: np.ndarray, t: np.ndarray) -> list[TransitionResult]:
    n = len(gt)
    if n == 0:
        return []
    gtr = transitions(gt, t)
    last_end_t = t[-1] + (t[-1] - t[-2]) if n > 1 else t[-1]
    results: list[TransitionResult] = []

    for k, tr in enumerate(gtr):
        i, to, frm = tr["idx"], tr["to"], tr["frm"]
        seg_end = gtr[k + 1]["idx"] if k + 1 < len(gtr) else n
        seg_start = gtr[k - 1]["idx"] if k > 0 else 0

        late_end = i
        while late_end < seg_end and pred[late_end] == frm:
            late_end += 1

        j = i
        while j < seg_end and pred[j] != to:
            j += 1

        if j < seg_end:
            if j == i and pred[i] == to:
                # pred was already in the target class: walk back to when it entered
                jj = i
                while jj > seg_start and pred[jj - 1] == to:
                    jj -= 1
                off = float(t[jj] - tr["time"]) # negative => early
                full_segment_miss = (jj == seg_start)
                seg_dur_ms = float(t[i] - t[seg_start])
                sustained_early = bool(seg_dur_ms > 0 and abs(off) >= LAG_MAX_MS
                                       and abs(off) / seg_dur_ms >= SUSTAINED_FRACTION)
                sustained_late, early_start = False, jj
            else:
                # late arrival; a stray class in between makes the offset unmeasurable
                off = float(t[j] - tr["time"]) if np.all(pred[i:j] == frm) else None
                full_segment_miss, sustained_early, early_start = False, False, i
                if off is not None:
                    seg_end_time = t[seg_end] if seg_end < n else last_end_t
                    seg_dur_ms = float(seg_end_time - t[i])
                    sustained_late = bool(seg_dur_ms > 0 and abs(off) >= LAG_MAX_MS
                                          and abs(off) / seg_dur_ms >= SUSTAINED_FRACTION)
                else:
                    sustained_late = False
            results.append(TransitionResult(frm, to, off, i, full_segment_miss,
                                            sustained_early, sustained_late,
                                            late_end, early_start))
        else:
            # pred never reaches the target class inside this gt segment
            left, right = _flanks(pred, i, seg_end, n)
            if left is None or right is None:
                kind = "edge_omission"
            elif left == right:
                kind = "swallowed"
            else:
                kind = "omission"
            results.append(TransitionResult(frm, to, None, i, False, False, False,
                                            late_end, i, omission_kind=kind))
    return results


# one bucket per row, assigned in strict precedence order
def row_buckets(gt: np.ndarray, pred: np.ndarray, t: np.ndarray,
                results: list[TransitionResult] | None = None) -> np.ndarray:
    n = len(gt)
    if n == 0:
        return np.full(0, UNCLASSIFIED, dtype=object)

    bucket = np.full(n, UNCLASSIFIED, dtype=object)
    bucket[gt == pred] = CORRECT

    if results is None:
        results = transition_timing(gt, pred, t)

    # (2) omission -- including the leading segment, which has no gt transition of its own
    first_end = results[0].gt_idx if results else n
    if not np.any(pred[0:first_end] == gt[0]):
        _claim(bucket, 0, first_end, "edge_omission")
    for k, r in enumerate(results):
        if r.omission_kind is not None:
            seg_end = results[k + 1].gt_idx if k + 1 < len(results) else n
            _claim(bucket, r.gt_idx, seg_end, r.omission_kind)

    # (3) flicker
    for seg in segments(pred, t):
        if seg["dur_ms"] < FLICKER_MAX_MS:
            s, e = seg["start"], seg["end"]
            left, right = _flanks(pred, s, e, n)
            if left is not None and right is not None and left == right and left != seg["label"]:
                _claim(bucket, s, e, "flicker")

    # (4) late / (5) early
    for r in results:
        _claim(bucket, r.gt_idx, r.late_end_idx, "late")
        _claim(bucket, r.early_start_idx, r.gt_idx, "early")

    # (6) steady_confusion absorbs the rest -- no catch-all bucket needed
    bucket[bucket == UNCLASSIFIED] = "steady_confusion"
    return bucket


# bucket counts/fractions over one gap-free segment
def bucket_errors(gt: np.ndarray, pred: np.ndarray, t: np.ndarray) -> dict:
    n = len(gt)
    if n == 0:
        return {"counts": {b: 0 for b in ERROR_BUCKETS},
                "fractions": {b: 0.0 for b in ERROR_BUCKETS},
                "dominant": None, "total_error_rows": 0, "correct_rows": 0}

    bucket = row_buckets(gt, pred, t)
    counts = {b: int(np.sum(bucket == b)) for b in ERROR_BUCKETS}
    total_err = sum(counts.values())
    return {
        "counts": counts,
        "fractions": {b: round(counts[b] / total_err, 3) if total_err else 0.0
                      for b in ERROR_BUCKETS},
        "dominant": max(ERROR_BUCKETS, key=counts.get) if total_err else None,
        "total_error_rows": total_err,
        "correct_rows": int(np.sum(bucket == CORRECT)),
    }


# sum bucket counts across segments/trials into one corpus-level result
def aggregate(per_segment: list[dict]) -> dict:
    counts = {b: 0 for b in ERROR_BUCKETS}
    correct = 0
    for r in per_segment:
        for b in ERROR_BUCKETS:
            counts[b] += r["counts"][b]
        correct += r["correct_rows"]
    total_err = sum(counts.values())
    return {
        "counts": counts,
        "fractions": {b: round(counts[b] / total_err, 3) if total_err else 0.0
                      for b in ERROR_BUCKETS},
        "dominant": max(ERROR_BUCKETS, key=counts.get) if total_err else None,
        "total_error_rows": total_err,
        "correct_rows": correct,
        "row_accuracy": round(correct / (correct + total_err), 4) if correct + total_err else 0.0,
    }
