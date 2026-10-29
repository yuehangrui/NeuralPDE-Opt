import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in [ROOT / "src", ROOT / "experiments/ch03_varmion_v1", ROOT / "experiments/ch03_physics_v1"]:
    sys.path.append(str(p))
