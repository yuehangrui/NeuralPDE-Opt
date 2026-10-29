"""Read-only archive, normalization, selection, reload and online audits."""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from .data import load as load_data, arrays
from .training import load, predict, metrics, write
from .reference import Poisson
from .online import ControlNet
from .online_supported import SupportedControl


def read(p):
    return json.loads(Path(p).read_text())


def close(a, b, atol=1e-10, rtol=1e-8):
    if not np.isclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError((a, b))


def run():
    torch.set_num_threads(2)
    root = Path("results/ch04/operator_learning_v1")
    tr, va, manifest = load_data("data/ch04/operator_learning_v1")
    if (
        not (root / "TRAIN_VAL_COMPLETE").exists()
        or not (root / "online/VALIDATION_COMPLETE").exists()
    ):
        raise RuntimeError("unfinished experiment")
    ref = Poisson()
    report = dict(models={}, online={}, test_evaluations=0)
    for architecture in ("fno", "deeponet", "fedonet"):
        for role in ("state", "kkt"):
            name = architecture + "_" + role
            folder = root / name
            model, cfg, ck = load(folder / "best.pt")
            run_info = read(folder / "run.json")
            history = [
                json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()
            ]
            best = min(history, key=lambda row: row["validation_score"])
            saved = read(folder / "metrics.json")
            if (
                best != saved
                or ck["epoch"] != best["epoch"]
                or ck["data_hashes"] != manifest["hashes"]
            ):
                raise AssertionError("best/data identity mismatch")
            _, _, last = load(folder / "last.pt")
            expected = cfg.epochs * math.ceil(len(tr["ids"]) / cfg.batch)
            if (
                run_info["updates"] != expected
                or last["updates"] != expected
                or last["epoch"] != cfg.epochs
                or history[-1]["updates"] != expected
            ):
                raise AssertionError("update count mismatch")
            if len(history) != cfg.epochs + 1 or run_info["full_budget_completed"]:
                raise AssertionError("incorrect pilot budget statement")
            tx, ty = arrays(tr, role)
            vx, vy = arrays(va, role)
            ins = np.sqrt(np.mean(tx[:, :, 1:-1, 1:-1] ** 2, axis=(0, 2, 3), keepdims=True))
            outs = np.sqrt(np.mean(ty[:, :, 1:-1, 1:-1] ** 2, axis=(0, 2, 3), keepdims=True))
            np.testing.assert_allclose(model.input_scale.numpy(), ins, rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(model.output_scale.numpy(), outs, rtol=2e-6, atol=1e-7)
            archive = dict(np.load(folder / "validation_prediction.npz"))
            if not np.array_equal(archive["ids"], va["ids"]):
                raise AssertionError("prediction id mismatch")
            pred = archive["prediction"]
            fresh = metrics(pred, vy, vx, role, ref)
            for k, v in fresh.items():
                close(v, saved[k])
            # Independent direct L2 arithmetic from saved arrays, excluding boundary.
            errors = np.sqrt(
                np.sum((pred - vy)[:, :, 1:-1, 1:-1] ** 2, axis=(2, 3))
                / np.sum(vy[:, :, 1:-1, 1:-1] ** 2, axis=(2, 3))
            ).mean(0)
            close(errors[0], saved["state_relative_l2_mean"])
            if role == "kkt":
                close(errors[1], saved["control_relative_l2_mean"])
            again = predict(model, vx)
            diff = float(abs(again - pred).max())
            np.testing.assert_allclose(again, pred, atol=1e-5, rtol=2e-5)
            report["models"][name] = dict(
                epochs=cfg.epochs,
                updates=expected,
                best_epoch=ck["epoch"],
                metrics=fresh,
                cpu_reload_max_abs=diff,
                parameter_count=run_info["parameter_count"],
                train_only_normalization_verified=True,
                source_and_data_verified=True,
                selection_verified=True,
            )
        folder = root / "online" / architecture
        online = read(folder / "online.json")
        der = read(folder / "derivatives.json")
        if (
            online["selection_use"]
            or not online["fixed_surrogate_verified"]
            or len(online["rows"]) != 2
        ):
            raise AssertionError("online protocol mismatch")
        if (
            online["checkpoint_sha256"]
            != hashlib.sha256(
                (root / (architecture + "_state") / "best.pt").read_bytes()
            ).hexdigest()
        ):
            raise AssertionError("online checkpoint changed")
        controls = dict(np.load(folder / "controls.npz"))
        rows = []
        for i, row in enumerate(online["rows"]):
            u = controls[str(i)]
            net = ControlNet(ref)
            net.load_state_dict(
                torch.load(
                    folder / ("control_" + str(i) + ".pt"), map_location="cpu", weights_only=True
                )
            )
            np.testing.assert_allclose(net().detach().numpy(), u, atol=2e-10, rtol=1e-9)
            y, p, g, j = ref.recover(va["f"][i], va["yd"][i], u)
            _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(u))
            m = row["trajectory"][-1]
            close(j, m["J_true"])
            close(ref.norm(g) / ref.norm(g0), m["true_gradient_relative"])
            rel = float(ref.norm(u - va["u"][i]) / ref.norm(va["u"][i]))
            close(rel, m["control_relative_l2"])
            _, _, _, jstar = ref.recover(va["f"][i], va["yd"][i], va["u"][i])
            if j < jstar - 1e-12:
                raise AssertionError("true objective below reference optimum")
            rows.append(
                dict(
                    id=row["id"],
                    control_relative_l2=rel,
                    true_gradient_relative=m["true_gradient_relative"],
                    true_objective_gap=float(j - jstar),
                )
            )
        report["online"][architecture] = dict(
            derivatives=der,
            rows=rows,
            mean_control_relative_l2=float(np.mean([r["control_relative_l2"] for r in rows])),
        )
    if not (root / "online_supported/VALIDATION_COMPLETE").exists():
        raise RuntimeError("supported-space validation unfinished")
    report["online_supported"] = {}
    for architecture in ("fno", "deeponet", "fedonet"):
        folder = root / "online_supported" / architecture
        info = read(folder / "online.json")
        values = dict(np.load(folder / "controls.npz"))
        rows = []
        if (
            info["selection_use"]
            or not info["fixed_surrogate_verified"]
            or info["checkpoint_sha256"]
            != hashlib.sha256(
                (root / (architecture + "_state") / "best.pt").read_bytes()
            ).hexdigest()
        ):
            raise AssertionError("supported control protocol mismatch")
        for i, row in enumerate(info["rows"]):
            net = SupportedControl(ref)
            net.load_state_dict(
                torch.load(
                    folder / ("control_" + str(i) + ".pt"), map_location="cpu", weights_only=True
                )
            )
            u = values[str(i)]
            np.testing.assert_allclose(net().detach().numpy(), u, atol=2e-10, rtol=1e-9)
            np.testing.assert_allclose(
                net.project(torch.tensor(u)).detach().numpy(), u, atol=2e-10, rtol=1e-9
            )
            y, p, g, j = ref.recover(va["f"][i], va["yd"][i], u)
            _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(u))
            m = row["trajectory"][-1]
            close(j, m["J_true"])
            close(ref.norm(g) / ref.norm(g0), m["true_gradient_relative"])
            error = float(ref.norm(u - va["u"][i]) / ref.norm(va["u"][i]))
            close(error, m["control_relative_l2"])
            rows.append(
                dict(
                    id=row["id"],
                    control_relative_l2=error,
                    true_gradient_relative=m["true_gradient_relative"],
                )
            )
        report["online_supported"][architecture] = dict(
            rows=rows,
            mean_control_relative_l2=float(np.mean([r["control_relative_l2"] for r in rows])),
            supported_adjoint_action_relative_mean=float(
                np.mean(info["supported_adjoint_action_relative"])
            ),
        )
    counts = {}
    for chapter in ("ch04", "ch04_adaptation", "ch05"):
        folder = Path("chapters") / chapter
        manifest0 = read(folder / "RUNTIME_MANIFEST.json")
        for name, h in manifest0.items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != h:
                raise AssertionError("protected source changed")
        counts[chapter] = len(manifest0)
    for dirname in ("full_training_v1", "adaptation_v1"):
        m = read(Path("results/ch04") / dirname / "ARTIFACT_MANIFEST.json")
        for name, h in m.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h:
                raise AssertionError("protected result changed")
        counts[dirname] = len(m)
    report["protected"] = counts
    write(root / "audit.json", report)
    print(
        json.dumps(dict(audit="passed", models=6, online_controls=12, protected=counts)), flush=True
    )


if __name__ == "__main__":
    run()
