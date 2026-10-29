from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json
import numpy as np
from chapters.ch05.common import Config as BaseConfig

METHODS = ("osnn", "admm", "cnp", "mimo", "pdol", "iuzawa_f", "iuzawa_s")
CORE = ("common.py", "data.py", "training.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    files = list((root / "chapters/ch05").glob("*.py")) + [
        root / "chapters/ch05_delivery" / n for n in CORE
    ]
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(files)}


def quadrature(n):
    t, w = np.polynomial.legendre.leggauss(n)
    t = (t + 1) / 2
    w = w / 2
    return np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2), np.outer(w, w).reshape(
        -1, 1
    )


@dataclass
class Config(BaseConfig):
    quadrature_order: int = 64
    validation_order: int = 80
    lbfgs_chunk: int = 100

    def validate(self):
        super().validate()
        for k in ("quadrature_order", "validation_order", "lbfgs_chunk"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k)
        return self


def configuration(method, seed, pilot=False):
    c = Config(method=method, seed=seed, device="cuda", train_points=4096)
    if pilot:
        c.adam_steps = 500
        c.lbfgs_steps = 100
        c.outer_steps = 3
        c.inner_steps = 200
        c.inner_lbfgs = 50
        c.epochs = 20
        c.evaluate_every = 250
    return c.validate()
