from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAMEWORK_SRC = ROOT / "framework" / "fuzzyxai"
if str(FRAMEWORK_SRC) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SRC))

H10_C3_SRC = ROOT / "experiments" / "h10_c3" / "src"
if str(H10_C3_SRC) not in sys.path:
    sys.path.insert(0, str(H10_C3_SRC))
