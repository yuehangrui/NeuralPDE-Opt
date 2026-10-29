"""Reproducible fresh-directory train, full run, validation, and hash checks."""

from pathlib import Path
import argparse, hashlib, json
from .training import Config, train, identity
from .run_all import run as run_queue
from .validation import run as validate
from chapters.ch04_operator_learning.data import load as load_data
from chapters.ch04_operator_learning.training import write


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--data", required=True)
    v.add_argument("--result", required=True)
    q = sub.add_parser("verify")
    q.add_argument("--result")
    a = p.parse_args()
    if a.command == "train":
        train(Config(**json.loads(Path(a.config).read_text())), a.data, a.output)
    elif a.command == "run":
        load_data(a.data)
        root = Path(a.output)
        root.mkdir(parents=True, exist_ok=False)
        write(
            root / "protocol.json",
            dict(
                schema="ch04-operator-adaptation-v2",
                source_identity=identity(),
                epochs=1600,
                seeds=[0],
                data=str(Path(a.data).resolve()),
                dtype="float64",
                test_evaluations=0,
                online_validation=dict(
                    variants=["neural", "head_polished"],
                    tasks_per_variant=8,
                    derivative_inputs_per_variant=8,
                    adam_steps=100,
                    lbfgs_max_iter=300,
                    tolerance_grad=1e-12,
                    tolerance_change=1e-15,
                    selection_use=False,
                ),
            ),
        )
        run_queue(a.data, root / "training")
        validate(root, a.data)
    elif a.command == "validate":
        validate(a.result, a.data)
    else:
        code = Path(__file__).parent
        manifest = json.loads((code / "RUNTIME_MANIFEST.json").read_text())
        for name, h in manifest.items():
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != h:
                raise ValueError("source hash mismatch: " + name)
        count = 0
        dependency_count = 0
        if a.result:
            dependencies = json.loads((Path(a.result) / "DEPENDENCY_MANIFEST.json").read_text())
            for name, h in dependencies.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h:
                    raise ValueError("dependency hash mismatch: " + name)
            dependency_count = len(dependencies)
            artifacts = json.loads((Path(a.result) / "ARTIFACT_MANIFEST.json").read_text())
            for name, h in artifacts.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h:
                    raise ValueError("artifact hash mismatch: " + name)
            count = len(artifacts)
        print(
            json.dumps(
                dict(
                    status="verified",
                    sources=len(manifest),
                    dependencies=dependency_count,
                    artifacts=count,
                )
            )
        )


if __name__ == "__main__":
    main()
