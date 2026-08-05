from __future__ import annotations

import sys


# never let an unencodable character turn output into a crash
def use_replacement_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
