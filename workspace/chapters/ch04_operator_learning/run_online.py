"""Post-selection derivative and neural-control validation, no model tuning."""

from pathlib import Path
from .online import run


def main():
    root = Path("results/ch04/operator_learning_v1")
    if not (root / "TRAIN_VAL_COMPLETE").exists():
        raise RuntimeError("training pilots incomplete")
    out = root / "online"
    out.mkdir(exist_ok=False)
    for architecture in ("fno", "deeponet", "fedonet"):
        run(
            root / (architecture + "_state") / "best.pt",
            "data/ch04/operator_learning_v1",
            out / architecture,
        )
    (out / "VALIDATION_COMPLETE").write_text(
        "Three derivative audits and six neural online control pilots complete; accuracy reported separately.\n"
    )


if __name__ == "__main__":
    main()
