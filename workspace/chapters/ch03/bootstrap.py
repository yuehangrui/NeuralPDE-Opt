"""Chapter 3 public entry points; historical numerical kernels stay immutable."""

import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.path.insert(0, str(ROOT / "src"))
for name in ["physics", "pod", "varmion", "varmion_v2"]:
    directory = "ch03_varmion_v2" if name == "varmion_v2" else f"ch03_{name}_v1"
    sys.path.append(str(ROOT / "experiments" / directory))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value
