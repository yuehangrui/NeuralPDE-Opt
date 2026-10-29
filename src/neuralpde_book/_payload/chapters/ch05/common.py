from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, math
from .problem import Problem

HERE = Path(__file__).resolve().parent
INSTANCE = ("osnn", "admm", "cnp")
OPERATORS = ("mimo", "pdol", "iuzawa_f", "iuzawa_s")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


@dataclass
class Config:
    method: str = "osnn"
    alpha: float = 1e-3
    lower: float = 0.3
    upper: float = 1.0
    state_cap: float = 0.2
    seed: int = 0
    device: str = "cpu"
    width: int = 64
    depth: int = 6
    train_points: int = 10000
    val_points: int = 2048
    grid: int = 32
    adam_steps: int = 10000
    lbfgs_steps: int = 1000
    learning_rate: float = 1e-3
    evaluate_every: int = 250
    beta: float = 1e-3
    outer_steps: int = 30
    inner_steps: int = 2000
    inner_lbfgs: int = 500
    epochs: int = 300
    batch: int = 64
    modes: int = 8
    fno_width: int = 8
    fno_depth: int = 3
    projection_width: int = 64
    qb_width: int = 8
    uzawa_layers: int = 6
    pd_iterations: int = 500
    smoke: bool = False

    def problem(self):
        return Problem(self.alpha, self.lower, self.upper, self.state_cap)

    def validate(self):
        self.problem()
        if self.method not in INSTANCE + OPERATORS:
            raise ValueError("unknown method")
        for k, v in asdict(self).items():
            if k in ("method", "device", "smoke"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("invalid numeric config " + k)
        for k in ("seed", "adam_steps", "lbfgs_steps", "inner_lbfgs"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(k)
        for k in (
            "width",
            "depth",
            "train_points",
            "val_points",
            "grid",
            "evaluate_every",
            "outer_steps",
            "inner_steps",
            "epochs",
            "batch",
            "modes",
            "fno_width",
            "fno_depth",
            "projection_width",
            "qb_width",
            "uzawa_layers",
            "pd_iterations",
        ):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k)
        if self.learning_rate <= 0 or self.beta <= 0 or self.grid < 4:
            raise ValueError("invalid step/grid")
        if self.method in OPERATORS and 2 * self.modes + 1 > self.grid:
            raise ValueError("Fourier modes exceed padded grid")
        if self.method in ("osnn", "cnp") and self.adam_steps + self.lbfgs_steps == 0:
            raise ValueError("empty training budget")
        return self


def smoke_config(method, device="cpu"):
    return Config(
        method=method,
        device=device,
        smoke=True,
        width=12,
        depth=2,
        train_points=64,
        val_points=53,
        grid=8,
        adam_steps=4,
        lbfgs_steps=2,
        evaluate_every=2,
        outer_steps=2,
        inner_steps=3,
        inner_lbfgs=1,
        epochs=2,
        batch=4,
        modes=2,
        fno_width=4,
        fno_depth=2,
        projection_width=8,
        qb_width=3,
        uzawa_layers=2,
        pd_iterations=15,
    ).validate()
