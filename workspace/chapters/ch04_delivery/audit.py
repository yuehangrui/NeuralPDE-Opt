"""Independent delivery audit: recompute predictions, physical errors and provenance."""

from pathlib import Path
import json
import numpy as np
import torch
from chapters.ch04.problem import Problem
from chapters.ch04_adaptation.operator_data import generate, EIGENVALUES
from chapters.ch04_completion.common import SpectralProblem
from chapters.ch04_completion.ismo import Surrogate
from .protocol import verify_freeze, load_holdout, build_registry, sha, METHODS
from .assessment import load_model, coefficient_predict
from .metrics import control_metrics, state_metrics, aggregate
from .frozen_jobs import write


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-9, atol=1e-11)


def normalization(model, entry):
    tr = generate("train", 256, 0)
    if entry["kind"] == "grid_operator":
        if entry["method"].endswith("_state"):
            x, y = tr["rhs"], tr["state"][:, None, :]
        else:
            x, y = np.stack([tr["f"], tr["yd"]], 1).reshape(256, -1), np.stack(
                [tr["y"], tr["u"]], 1
            )
        for prefix, data in [("input", x), ("output", y)]:
            close(getattr(model, prefix + "_mean").cpu().numpy(), data.mean(0))
            close(getattr(model, prefix + "_scale").cpu().numpy(), data.std(0).clip(1e-10))
    elif entry["method"] == "mimo":
        for prefix, data in [
            ("input", np.stack([tr["f"], tr["yd"]], 1)),
            ("output", np.stack([tr["y"], tr["u"]], 1)),
        ]:
            close(getattr(model, prefix + "_mean").numpy(), data.mean(0))
            close(getattr(model, prefix + "_scale").numpy(), data.std(0))
    else:
        for prefix, key, bname in [("input", "rhs", "Psi"), ("output", "state", "Phi")]:
            b = getattr(model, bname).numpy()
            mean = getattr(model, prefix + "_mean").numpy()
            close(mean, tr[key].mean(0))
            close(b.T @ b, np.eye(17))
            close(
                getattr(model, prefix + "_scale").numpy(),
                np.sqrt(np.mean(((tr[key] - mean) @ b) ** 2, 0)),
            )


def check_hybrid(model, entry):
    if entry["method"] != "ismo":
        return
    path = Path(entry["directory"])
    metrics = json.loads((path / "metrics.json").read_text())
    problem = SpectralProblem()
    with np.load(path / "data.npz", allow_pickle=False) as a:
        for split in ("initial", "train", "validation"):
            close(a[split + "_states"], problem.state(a[split + "_controls"]))
        for k, weights in enumerate(model["state"]["round_models"]):
            net = Surrogate()
            net.load_state_dict(weights)
            for prefix, key in [("input", "initial_controls"), ("output", "initial_states")]:
                close(getattr(net, prefix + "_mean").numpy(), a[key].mean(0))
                close(getattr(net, prefix + "_scale").numpy(), a[key].std(0))
            with torch.no_grad():
                pred = net(torch.tensor(a["validation_controls"])).numpy()
            close(pred, a["validation_predictions"][k])
            score = float(
                np.mean(((pred - a["validation_states"]) / net.output_scale.numpy()) ** 2)
            )
            close(score, metrics["history"][k]["validation_normalized_mse"])
            expected = len(a["initial_controls"]) + 18 * k
            if metrics["history"][k]["training_samples"] != expected:
                raise ValueError("ISMO adaptive-training history mismatch")
        for candidate in a["candidates"].reshape(-1, 17):
            if not np.any(np.all(a["train_controls"] == candidate, axis=1)):
                raise ValueError("candidate not added to data")


def pretest(root):
    root = Path(root)
    if (root / "heldout").exists():
        raise RuntimeError("pretest audit must precede held-out data creation")
    torch.set_num_threads(2)
    entries = build_registry(root)
    for entry in entries:
        model = load_model(entry, "cpu")
        if entry["track"] == "problem_family":
            normalization(model, entry)
        elif entry["kind"] == "hybrid":
            check_hybrid(model, entry)
    result = dict(
        status="passed",
        selected_variants=len(entries),
        test_created=False,
        train_only_normalization_and_history_selection=True,
        checkpoint_hashes={e["key"]: e["sha256"] for e in entries},
    )
    write(root / "PRETEST_AUDIT.json", result)
    return result


def run(root):
    root = Path(root)
    if not (root / "INDEPENDENT_ASSESSMENT_COMPLETE").is_file():
        raise RuntimeError("assessment incomplete")
    frozen = verify_freeze(root)
    data = load_holdout(root)
    registry = build_registry(root)
    if registry != frozen["entries"]:
        raise ValueError("registry changed")
    summary = json.loads((root / "assessment/summary.json").read_text())
    if len(summary["rows"]) != len(registry) or {r["key"] for r in summary["rows"]} != {
        r["key"] for r in registry
    }:
        raise ValueError("missing delivered variants")
    torch.set_num_threads(2)
    problem = Problem()
    report = []
    for entry in registry:
        folder = root / "assessment" / entry["key"]
        row = json.loads((folder / "metrics.json").read_text())
        model = load_model(entry, "cpu")
        check = dict(key=entry["key"], checkpoint_identity=True)
        with np.load(folder / "fixed_fields.npz", allow_pickle=False) as a:
            x, w, u = a["points"], a["weights"], a["u"]
            close(w.sum(), 1.0)
            truth = problem.u(x)[:, 0]
            close(a["u_reference"], truth)
            error = float(np.sqrt(np.sum(w * (u - truth) ** 2) / np.sum(w * truth**2)))
            close(error, row["fixed_instance"]["control_relative_l2"])
            if "state_residual" in a:
                rel = float(
                    np.sqrt(
                        np.sum(w * a["state_residual"] ** 2) / np.sum(w * problem.f(x)[:, 0] ** 2)
                    )
                )
                close(rel, row["fixed_instance"]["state_residual_relative"])
        if entry["track"] == "problem_family":
            normalization(model, entry)
            state_role = entry["method"].endswith("_state") or entry["method"] in (
                "rb_state",
                "rb_dino",
            )
            inputs = data["rhs"] if state_role else np.stack([data["f"], data["yd"]], 1)
            with torch.no_grad():
                pred = coefficient_predict(
                    model, entry, torch.tensor(inputs, dtype=torch.float64)
                ).numpy()
            with np.load(folder / "heldout_predictions.npz", allow_pickle=False) as a:
                if not np.array_equal(a["ids"], data["ids"]):
                    raise ValueError("heldout ids mismatch")
                close(pred, a["prediction"])
                if state_role:
                    state = state_metrics(data["rhs"], pred)
                    for k, v in state.items():
                        close(v, a["state_" + k])
                    values = control_metrics(
                        data["f"][:8], data["yd"][:8], a["online_controls"], a["online_states"]
                    )
                    prefix = "online_"
                    section = "online"
                else:
                    values = control_metrics(
                        data["f"], data["yd"], pred[:, 1], pred[:, 0], -0.001 * pred[:, 1]
                    )
                    prefix = "direct_"
                    section = "direct_control"
                for k, v in values.items():
                    close(v, a[prefix + k])
                    close(aggregate({k: v})[k]["mean"], row["heldout"][section][k]["mean"])
            check.update(
                heldout_predictions_reloaded=True,
                physical_metrics_recomputed=True,
                normalization_train_only=True,
            )
        elif entry["kind"] == "hybrid":
            check_hybrid(model, entry)
            check["hybrid_protocol_checked"] = True
        report.append(check)
        print("audit", entry["key"], flush=True)
    old = []
    for name in (
        "ch04",
        "ch04_adaptation",
        "ch04_operator_learning",
        "ch04_operator_adaptation",
        "ch04_completion",
        "ch05",
    ):
        directory = Path("chapters") / name
        manifest = directory / "RUNTIME_MANIFEST.json"
        if manifest.exists():
            hashes = json.loads(manifest.read_text())
            for file, h in hashes.items():
                if sha(directory / file) != h:
                    raise ValueError("frozen source changed: " + str(directory / file))
            old.append(dict(module=name, verified_files=len(hashes)))
    for method in METHODS:
        seeds = {r["seed"] for r in registry if r["method"] == method}
        if seeds != {0, 1, 2}:
            raise ValueError("missing seed " + method)
    result = dict(
        status="passed",
        methods=len(METHODS),
        seeds=[0, 1, 2],
        training_experiments=len(METHODS) * 3,
        checkpoint_variants=len(registry),
        family_variants=sum(r["track"] == "problem_family" for r in registry),
        fresh_holdout_cases=len(data["ids"]),
        checks=report,
        old_source_protection=old,
        selections_unchanged=True,
        independent_metric_recomputation=True,
    )
    verify_freeze(root)
    write(root / "DELIVERY_AUDIT.json", result)
    (root / "DELIVERY_COMPLETE").write_text(
        "Complete declared training, independent assessments and audits; accuracy/convergence are separate reported outcomes.\n"
    )
    return result
