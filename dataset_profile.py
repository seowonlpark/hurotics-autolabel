# dataset_profile.py: the single place to adapt this pipeline to a new dataset. encodes what OUR
# recordings look like (label codes, trial schema, naming, held-out subjects), not modelling choices.
# to bring in a new corpus, edit this file (plus config.py for raw device logs); ingest/ maps a sheet in.

from __future__ import annotations

# label encoding: the integer codes in a labeled trial's Label column. STAND/WALK are the two trained
# classes; the two "unknown" codes are opposite in kind and never merged.
STAND = 0                 # standing
WALK = 10                 # walking
HUMAN_UNKNOWN = -1        # a human looked and could not call it (a valid annotation)
MACHINE_UNKNOWN = 255     # data error: nothing was measured properly (not an annotation)
TRAIN_CLASSES = (STAND, WALK)

# labeled-trial schema: columns load_dataset requires in every annotated trial. a trial missing any is
# rejected at read time. header whitespace is stripped before matching.
TIME_COL = "Time"         # device uptime counter, milliseconds
LABEL_COL = "Label"       # the human ground-truth column, encoded with the codes above
# the four rotational features the classifier trains on (left/right sagittal angle and angular
# velocity, low-pass filtered).
FEATURES = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")

# file and subject naming. a labeled trial lives at data/labeled/<rev>/<TRIAL_GLOB match>; one rev is one
# subject on one day, one trial is one recording. the patterns pull rev id + trial number from the path.
TRIAL_GLOB = "annotated_loco_*_trial_*.csv"   # which files under data/labeled/ are trials
REV_PATTERN = r"(rev\d+)"                     # rev id anywhere in the path, e.g. rev13
TRIAL_PATTERN = r"trial_(\d+)"                # trial number in the filename

# subject splits. the lockbox is one or two whole revs held out of training until the single-use final
# test. set these to revs in YOUR data; on a new corpus they match nothing, so nothing is held out.
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")

# revs dropped from the corpus entirely (bad recordings). rev14 was confirmed anomalous; matches
# nothing on a different corpus.
EXCLUDED_REVS = ("rev14",)
