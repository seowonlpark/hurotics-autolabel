# dataset_profile.py -- THE single place to adapt this pipeline to a new dataset.
#
# Everything here encodes "what OUR recordings happen to look like": the label codes,
# the labeled-trial column schema, the file/subject naming, and which subjects are held
# out. Nothing here is a modelling choice -- those live in stages/s2_ml/champion_spec.json
# and the physics/fusion policy. This file is the data contract, and only the data contract.
#
# If you bring in a different corpus, this is the one file you edit (plus the raw-device
# column map in stages/s1_clean/config.py, if you also ingest raw device logs). The rest
# of the pipeline imports these names rather than hardcoding the literals, so adapting is a
# single deliberate edit here instead of a scavenger hunt across stages.
#
# The companion ingest layer (see ingest/) maps an arbitrary sheet INTO the contract defined
# below, so in practice you point the adapter at this profile rather than editing raw CSVs.

from __future__ import annotations

# --- Label encoding -------------------------------------------------------------------------
# The integer codes that appear in the Label column of a labeled trial. STAND/WALK are the
# two trained classes; the two "unknown" codes are opposite in kind and never merged.
STAND = 0                 # standing
WALK = 10                 # walking
HUMAN_UNKNOWN = -1        # a human looked and could not call it (a valid annotation)
MACHINE_UNKNOWN = 255     # data error: nothing was measured properly (not an annotation)
TRAIN_CLASSES = (STAND, WALK)

# --- Labeled-trial schema -------------------------------------------------------------------
# The columns load_dataset requires in every annotated trial. A trial missing any of these is
# rejected at read time (dataset._read_raw). Header whitespace is stripped before matching.
TIME_COL = "Time"         # device uptime counter, milliseconds
LABEL_COL = "Label"       # the human ground-truth column, encoded with the codes above
# The four rotational features the classifier trains on (the "rev2" derived view: left/right
# sagittal angle and angular velocity, low-pass filtered).
FEATURES = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")

# --- File and subject naming ----------------------------------------------------------------
# A labeled trial lives at data/labeled/<rev>/<TRIAL_GLOB match>. One "rev" is one subject
# recorded on one day; one trial is one recording within it. The patterns pull the rev id and
# trial number out of the path so the pipeline can group by subject.
TRIAL_GLOB = "annotated_loco_*_trial_*.csv"   # which files under data/labeled/ are trials
REV_PATTERN = r"(rev\d+)"                     # rev id anywhere in the path, e.g. rev13
TRIAL_PATTERN = r"trial_(\d+)"                # trial number in the filename

# --- Subject splits -------------------------------------------------------------------------
# The lockbox is one or two whole revs held out of training and never looked at until the
# single-use final test (README "The lockbox"). Set these to revs that exist in YOUR data;
# on a new corpus these names match nothing, so nothing is held out until you edit them.
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")

# Revs dropped from the corpus entirely (bad recordings). rev14 was confirmed anomalous on
# 2026-07-21 (low-amplitude gait, no raw source to audit). Harmlessly matches nothing on a
# different corpus.
EXCLUDED_REVS = ("rev14",)
