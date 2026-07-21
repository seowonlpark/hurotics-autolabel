# score a predictions csv directly against a ground-truth labels csv -- no algorithm .exe
# the two are joined by row index; UNKNOWN (255) is excluded from macro-F1 (the headline)
#
#   python score.py <pred.csv> <gt.csv>
#
# pred csv needs a prediction/label column, gt csv needs a label column; a time column
# in either file is used if present, else a synthetic 100 Hz timeline
import argparse
import json

from locoeval.core import load_aligned_direct
from locoeval.measure import frame_metrics


# (FrameMetrics, HealthReport) for one predictions-vs-labels pair
def score(pred_csv: str, gt_csv: str):
    aligned = load_aligned_direct(pred_csv, gt_csv)
    return frame_metrics(aligned), aligned.health


# print the headline metrics, per-class breakdown, and confusion matrix
def main() -> None:
    ap = argparse.ArgumentParser(description="score predictions against ground-truth labels")
    ap.add_argument("pred_csv", help="predictions csv (prediction/label column)")
    ap.add_argument("gt_csv", help="ground-truth csv (label column)")
    args = ap.parse_args()

    fm, health = score(args.pred_csv, args.gt_csv)
    print(f"macro-F1:          {fm.macro_f1}")
    print(f"balanced accuracy: {fm.balanced_accuracy}")
    print(f"overall accuracy:  {fm.overall_accuracy}")
    print(f"scored rows:       {fm.scored_rows:,}")
    for cls, m in fm.per_class.items():
        print(f"  {cls:10} P={m['precision']} R={m['recall']} F1={m['f1']} support={m['support_rows']:,}")
    print(f"confusion (truth->pred): {json.dumps(fm.confusion)}")
    if health.notes:
        print(f"notes: {health.notes}")


if __name__ == "__main__":
    main()
