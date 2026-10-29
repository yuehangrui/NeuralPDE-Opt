"""Read-only artifact, checkpoint, data, selection and PDE audit."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .common import ALPHA, LAM, SpectralProblem, tensor, load_checkpoint, write_json
from .ismo import Surrogate
from .fem_nn import SourceIdentification


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("artifact mismatch: " + name)
    report = dict(artifact_hashes=len(manifest), methods={})
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "best.pt").exists():
            continue
        ck = load_checkpoint(directory / "best.pt")
        metrics = json.loads((directory / "metrics.json").read_text())
        config = ck["config"]
        if config["test_evaluations"] != 0 or metrics["test_evaluations"] != 0:
            raise ValueError("unexpected test evaluation")
        state, method = ck["state"], ck["method"]
        problem = SpectralProblem()
        u = state["control"].numpy()
        y = state["state_head"].numpy() / LAM if "state_head" in state else None
        diag = problem.diagnostics(u, y)
        for key, value in diag.items():
            np.testing.assert_allclose(value, metrics["diagnostics"][key], rtol=1e-11, atol=1e-13)
        row = dict(reloaded_diagnostics=True, true_gradient_relative=diag["true_gradient_relative"])
        if method in ("bilevel", "deep_uzawa"):
            scores = [r["validation_score"] for r in metrics["history"]]
            if int(np.argmin(scores)) != metrics["best_index"]:
                raise ValueError("selection is not earliest minimum validation score")
            row["validation_selection"] = True
        if method == "deep_uzawa":
            # The stored multiplier belongs to the same primal solve, not next k.
            q, z = state["state_head"].numpy(), state["multiplier"].numpy()
            gradq = (q / LAM - problem.yd) / LAM + ALPHA * (q - problem.f) / 2 + z
            gradu = ALPHA * u / 2 - z
            row["modified_stationarity_max"] = float(max(abs(gradq).max(), abs(gradu).max()))
            if row["modified_stationarity_max"] > 1e-8:
                raise ValueError("multiplier/primal checkpoint mismatch")
        if method == "ismo":
            with np.load(directory / "data.npz", allow_pickle=False) as data:
                for split in ("initial", "train", "validation"):
                    np.testing.assert_allclose(
                        data[split + "_states"],
                        problem.state(data[split + "_controls"]),
                        atol=1e-14,
                    )
                for k, weights in enumerate(state["round_models"]):
                    model = Surrogate()
                    model.load_state_dict(weights)
                    for prefix, values in (
                        ("input", data["initial_controls"]),
                        ("output", data["initial_states"]),
                    ):
                        np.testing.assert_allclose(
                            getattr(model, prefix + "_mean").numpy(), values.mean(0), atol=1e-14
                        )
                        np.testing.assert_allclose(
                            getattr(model, prefix + "_scale").numpy(), values.std(0), atol=1e-14
                        )
                    pred = model(tensor(data["validation_controls"])).detach().numpy()
                    np.testing.assert_allclose(pred, data["validation_predictions"][k], atol=1e-14)
                    record = metrics["history"][k]
                    score = np.mean(
                        ((pred - data["validation_states"]) / model.output_scale.numpy()) ** 2
                    )
                    np.testing.assert_allclose(
                        score, record["validation_normalized_mse"], rtol=1e-12
                    )
                    if record["selected_epoch"] != int(np.argmin(record["validation_history"])):
                        raise ValueError("invalid surrogate selection")
                    if len(record["validation_history"]) != record["updates"]:
                        raise ValueError("missing training updates")
                for candidate in data["candidates"].reshape(-1, len(u)):
                    if not np.any(np.all(data["train_controls"] == candidate, axis=1)):
                        raise ValueError("ISMO candidate was not added to training data")
                row["training_samples"] = len(data["train_controls"])
                row["train_only_normalization_and_active_samples"] = True
        if method == "fem_nn":
            val = SourceIdentification(config["audit_mesh"])
            v = val.relative_misfit(state["source_head"])
            np.testing.assert_allclose(
                v, metrics["selected_validation_observation_relative_l2"], rtol=1e-11
            )
            scores = [r["validation_observation_relative_l2"] for r in metrics["history"]]
            if int(np.argmin(scores)) != metrics["best_index"]:
                raise ValueError("invalid FEM-NN selection")
            row["source_network_reloaded_through_fem"] = True
        report["methods"][method] = row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = verify(args.root)
    if args.output:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
