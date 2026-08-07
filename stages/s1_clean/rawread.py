# one parse per raw CSV: a verbatim transcode cache, so four stages stop re-tokenizing 2.7 GB
#
# `s1_census`, `s1_clean`, `s2_verify_serve` and `s2_label_all` each walked `data/raw` and parsed
# every file in full -- four passes, ~37 s each, and the widest consumer reads 13 of 67-91 columns.
# This is the one reader they all go through now.
#
# The cache is a TRANSCODE, not a computation, and that distinction is the whole design. It stores
# exactly what `pd.read_csv(index_col=False)` returns with the `Unnamed` columns dropped: no
# resampling, no filtering, no column selection, no unit normalization, no renaming. There is
# therefore no "which version of the code built this parquet" question to answer -- the only input
# is the CSV's bytes, so the source's own (size, mtime) is a complete key. `data/raw` stays
# source-of-truth; delete `data/cache` at any time and the next run rebuilds it.
#
# The trailing-comma guard gets STRONGER, not weaker. `index_col=False` runs once here, on the full
# width, and the three downstream readers inherit a frame that already passed it (see clean.py:111
# and DOMAIN_NOTES 1.3). Selecting columns with `usecols` instead would have suppressed exactly
# that guard at three call sites, which is why this reads whole files and caches them.

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

# set to 0 to bypass the cache entirely; verify_rawread reads both ways and compares
CACHE_ENV = "RAWREAD_CACHE"

# the source's identity, stored in the parquet footer- a stale entry is a miss, never a wrong answer
_KEY = b"rawread_source"

_OFF = {"0", "false", "no", "off"}


def cache_enabled() -> bool:
    return os.environ.get(CACHE_ENV, "1").strip().lower() not in _OFF


# identity of the SOURCE, never of this code; a re-export changes both fields
def source_key(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


# built as one name so a dotted stem cannot become two files, as data/clean does it
def cache_path(path: Path) -> Path:
    return CACHE_DIR / path.parent.name / f"{path.stem}.parquet"


# THE definition of a parsed raw body; nothing else in the repo calls read_csv on data/raw
def parse_csv(path: Path) -> pd.DataFrame:
    # index_col=False is load-bearing: a trailing comma otherwise shifts every column left by one
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False)
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]


# write to a pid-suffixed temp then rename: a killed run must not leave a truncated file reading as a hit
def _write(dest: Path, df: pd.DataFrame, key: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _KEY: key.encode()})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


# the frame every caller gets; a fresh object each time, so callers may rename and slice in place
def read_raw_body(path: Path) -> pd.DataFrame:
    if not cache_enabled():
        return parse_csv(path)

    dest = cache_path(path)
    key = source_key(path)
    if dest.exists():
        try:
            # the footer alone, not the data: a miss must not cost more than the parse it replaces
            meta = pq.read_schema(dest).metadata or {}
            if meta.get(_KEY, b"").decode() == key:
                return pq.read_table(dest).to_pandas()
        except Exception:
            pass  # an unreadable or half-written entry is a miss; the CSV is still source-of-truth

    # a file that will not parse raises here and caches nothing, so the caller sees the same error twice
    df = parse_csv(path)
    _write(dest, df, key)
    return df
