# locoeval: score per-row predictions against ground-truth labels
# trimmed to the direct-scoring path (pred csv vs labels csv); the exe-runner and
# corpus-aggregation machinery was removed -- see score.py
from .core import load_aligned_direct, aligned_from_arrays, AlignedData, HealthReport, cname
from .measure import frame_metrics
from .diagnose import bucket_errors, build_findings, transition_timing, timing_summary, diagnose, Diagnosis

__all__ = ["load_aligned_direct", "aligned_from_arrays", "AlignedData", "HealthReport", "cname",
           "frame_metrics",
           "bucket_errors", "build_findings", "transition_timing", "timing_summary",
           "diagnose", "Diagnosis"]
