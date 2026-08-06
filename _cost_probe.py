# throwaway: where does S1+label wall-clock actually go, per file
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stages.s1_clean.census import read_header, resolve
from stages.s1_clean.channel_trust import detect_and_normalize
from stages.s1_clean.clean import select_columns
from stages.s1_clean.resample import resample_file
from stages.s2_ml.transform import (
    load_raw_frame, serve_dt, raw_to_features, load_trust, TIME_COL,
)

paths = sorted((ROOT / "data" / "raw").rglob("*.csv"))
rows = []
for i, p in enumerate(paths, 1):
    r = {"file": p.name, "bytes": p.stat().st_size}
    try:
        # ---- S1 path ----
        t0 = time.perf_counter()
        res = resolve(read_header(p))
        df = pd.read_csv(p, encoding="utf-8-sig", index_col=False)
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
        t1 = time.perf_counter()
        r["s1_parse_s"] = t1 - t0
        r["n_rows"] = int(len(df))
        r["n_cols_raw"] = int(len(df.columns))
        if len(df.columns) != len(res.index_by_name):
            r["skip"] = "colcount"; rows.append(r); continue
        df.columns = list(res.index_by_name)
        kept, missing = select_columns(list(df.columns))
        if missing:
            r["skip"] = "missing"; rows.append(r); continue
        df = df[kept]
        r["n_cols_kept"] = int(len(df.columns))
        t = df["Time"].to_numpy(float)
        if t.size < 2 or float(np.median(np.diff(t))) <= 0.0:
            r["skip"] = "clock"; rows.append(r); continue
        t2 = time.perf_counter()
        out, segs = resample_file(df, "Time")
        t3 = time.perf_counter()
        r["s1_resample_s"] = t3 - t2
        r["n_rows_grid"] = int(len(out))
        if out.empty:
            r["skip"] = "nosegs"; rows.append(r); continue
        t4 = time.perf_counter()
        out2, trust = detect_and_normalize(out)
        t5 = time.perf_counter()
        r["s1_trust_s"] = t5 - t4

        # ---- label/serve path on the same file ----
        t6 = time.perf_counter()
        raw, variant, family = load_raw_frame(p)
        t7 = time.perf_counter()
        r["lb_parse_s"] = t7 - t6
        if family != "raw_device":
            r["skip"] = f"family:{family}"; rows.append(r); continue
        trust_rec = load_trust(p)
        t8 = time.perf_counter()
        dt_s, _ = serve_dt(raw[TIME_COL].to_numpy(float))
        feat = raw_to_features(raw, variant, trust=trust_rec, dt_s=dt_s)
        t9 = time.perf_counter()
        r["lb_lpf_s"] = t9 - t8
        r["n_cols_view"] = int(len(feat.columns))
        t10 = time.perf_counter()
        frame, _ = resample_file(feat, TIME_COL)
        t11 = time.perf_counter()
        r["lb_resample_s"] = t11 - t10
    except Exception as exc:
        r["error"] = f"{type(exc).__name__}: {exc}"
    rows.append(r)
    print(f"[{i}/{len(paths)}] {p.name} {r.get('n_rows','?')} rows "
          f"s1_parse={r.get('s1_parse_s',0):.2f} s1_rs={r.get('s1_resample_s',0):.2f} "
          f"lb_parse={r.get('lb_parse_s',0):.2f} lb_lpf={r.get('lb_lpf_s',0):.3f} "
          f"lb_rs={r.get('lb_resample_s',0):.3f}", flush=True)

(ROOT / "_cost_probe.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
d = pd.DataFrame(rows)
num = [c for c in d.columns if c.endswith("_s")]
print("\n=== totals (s) ===")
print(d[num].sum().round(2).to_string())
print(f"\nrows raw total: {d['n_rows'].sum():,.0f}   grid total: {d.get('n_rows_grid', pd.Series([0])).sum():,.0f}")
print(f"bytes: {d['bytes'].sum()/1e9:.2f} GB")
