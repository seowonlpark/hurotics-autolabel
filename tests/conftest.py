# make the repo root importable no matter how pytest is invoked, so `from stages...` /
# `from agents...` / `import runmeta` resolve the same as they do for the pipeline itself.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
