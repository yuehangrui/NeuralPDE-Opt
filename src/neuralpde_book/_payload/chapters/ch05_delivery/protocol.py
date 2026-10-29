"""Fail-closed training audit, immutable selection, then fresh held-out generation."""

from pathlib import Path
from dataclasses import asdict
import json, time
import numpy as np
import torch
from .common import METHODS, sha, write, quadrature, source_hashes
from .training import load as load_model, validation, instance_loss
from chapters.ch05.models import SplitState
from chapters.ch05.training import operator_loss_dataset
from .data import load as load_data, make, validate, fingerprints
from .queue import validate_config_data
from .common import Config


def check_split_identity(data, role):
    expected = np.array([f"ch05-delivery:{role}:{i}" for i in range(len(data["sample_id"]))])
    if data["sample_id"].dtype.kind != "U" or not np.array_equal(data["sample_id"], expected):
        raise ValueError("invalid split/sample identity")


def check_disjoint(a, b):
    import hashlib

    for keys in (("coeff_f", "coeff_yd"), ("coeff_v",), ("coeff_w",)):

        def hashes(d):
            return {
                hashlib.sha256(r.tobytes()).hexdigest()
                for r in np.concatenate([d[k] for k in keys], 1)
            }

        ah, bh = hashes(a), hashes(b)
        if len(ah) != len(a["sample_id"]) or len(bh) != len(b["sample_id"]) or ah & bh:
            raise ValueError("duplicate/overlapping problem or action inputs")
    if set(a["sample_id"]) & set(b["sample_id"]):
        raise ValueError("overlapping IDs")


def registry(root):
    root = Path(root)
    jobs = json.loads((root / "jobs.json").read_text())
    entries = []
    if {(t["method"], t["seed"]) for t in jobs} != {(m, s) for m in METHODS for s in (0, 1, 2)}:
        raise ValueError("expected seven methods, three seeds")
    if len(jobs) != 21 or any(t["pilot"] for t in jobs):
        raise ValueError("pilot/duplicate jobs")
    for t in jobs:
        name = t["method"] + "_s" + str(t["seed"])
        d = root / "training" / name
        c = json.loads((d / "config.json").read_text())
        status = json.loads((d / "status.json").read_text())
        h = json.loads((d / "history.json").read_text())
        if c != t["config"] or status["state"] != "complete" or status["parameter_movement"] <= 0:
            raise ValueError("incomplete/different config " + name)
        if c["method"] != t["method"] or c["seed"] != t["seed"]:
            raise ValueError("job/config identity mismatch")
        validate_config_data(Config(**c), json.loads((root / "data/manifest.json").read_text()))
        b = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        if b["config"] != c or last["config"] != c:
            raise ValueError("checkpoint/config identity " + name)
        if b["source_hashes"] != source_hashes() or last["source_hashes"] != source_hashes():
            raise ValueError("checkpoint source identity " + name)
        meta = json.loads((root / "data/manifest.json").read_text())
        expected_data = (
            None
            if c["method"] in ("osnn", "admm", "cnp")
            else {role: meta["splits"][role]["sha256"] for role in ("train", "val")}
        )
        if b["data_identity"] != expected_data or last["data_identity"] != expected_data:
            raise ValueError("checkpoint/data identity " + name)
        first = min(h, key=lambda r: r["selection_score"])
        if not all(np.isfinite(r["selection_score"]) for r in h):
            raise ValueError("nonfinite history")
        if b["counters"] != first["counters"] or last["counters"] != status["counters"]:
            raise ValueError("counter identity " + name)
        if (
            status["method"] != c["method"]
            or status["seed"] != c["seed"]
            or status["best_score"] != b["selection_score"]
            or status["best_updates"] != b["optimizer_updates"]
        ):
            raise ValueError("status/checkpoint identity " + name)
        if set(b["model"]) != set(last["model"]):
            raise ValueError("state keys " + name)
        for key, value in b["model"].items():
            other = last["model"][key]
            if (
                value.shape != other.shape
                or value.dtype != other.dtype
                or not torch.isfinite(value).all()
                or not torch.isfinite(other).all()
            ):
                raise ValueError("invalid selected/last tensors " + name)
        if (
            b["optimizer_updates"] != first["optimizer_updates"]
            or b["selection_score"] != first["selection_score"]
        ):
            raise ValueError("selection " + name)
        if (
            last["optimizer_updates"] != status["optimizer_updates"]
            or last["selection_score"] != h[-1]["selection_score"]
        ):
            raise ValueError("last checkpoint " + name)
        k = status["counters"]
        m = c["method"]
        if m == "admm":
            if (
                k["outer_steps"] != c["outer_steps"]
                or k["adam_updates"] != c["outer_steps"] * c["inner_steps"]
            ):
                raise ValueError("ADMM budget")
            if (
                k["lbfgs_iterations"] > c["outer_steps"] * c["inner_lbfgs"]
                or len(last["split_history"]) != c["outer_steps"]
            ):
                raise ValueError("ADMM state/budget")
            for checkpoint in (b, last):
                if len(checkpoint["split_history"]) != checkpoint["counters"]["outer_steps"]:
                    raise ValueError("ADMM selected history length")
                for snapshot in checkpoint["split_history"]:
                    if any(not torch.isfinite(v).all() for v in snapshot.values()):
                        raise ValueError("ADMM nonfinite snapshot")
                if checkpoint["split_history"]:
                    for key, value in checkpoint["split_history"][-1].items():
                        if not torch.equal(value, checkpoint["model"]["q_net." + key]):
                            raise ValueError("ADMM model/history mismatch")
            if [r["epoch"] for r in h] != list(range(c["outer_steps"] + 1)):
                raise ValueError("ADMM outer history")
        elif m in ("osnn", "cnp"):
            if k["adam_updates"] != c["adam_steps"] or k["lbfgs_iterations"] > c["lbfgs_steps"]:
                raise ValueError("instance budget")
        else:
            count = json.loads((root / "data/manifest.json").read_text())["splits"]["train"][
                "count"
            ]
            expected = c["epochs"] * ((count + c["batch"] - 1) // c["batch"])
            if (
                k["epochs"] != c["epochs"]
                or k["adam_updates"] != expected
                or len(h) != c["epochs"] + 1
            ):
                raise ValueError("operator budget")
            if [r["epoch"] for r in h] != list(range(c["epochs"] + 1)):
                raise ValueError("operator epoch history")
            if [r["optimizer_updates"] for r in h] != [
                i * ((count + c["batch"] - 1) // c["batch"]) for i in range(c["epochs"] + 1)
            ]:
                raise ValueError("operator update history")
        if status["optimizer_updates"] != k["adam_updates"] + k["lbfgs_iterations"]:
            raise ValueError("update accounting")
        if status["test_evaluations"] != 0:
            raise ValueError("early test usage")
        if any(
            h[i]["optimizer_updates"] > h[i + 1]["optimizer_updates"] for i in range(len(h) - 1)
        ):
            raise ValueError("nonmonotone update history")
        entries.append(
            dict(
                key=name,
                method=m,
                seed=c["seed"],
                directory=d.as_posix(),
                checkpoint=(d / "best.pt").as_posix(),
                sha256=sha(d / "best.pt"),
                initial_sha256=sha(d / "initial.pt"),
                config_sha256=sha(d / "config.json"),
                selection_score=b["selection_score"],
                best_updates=b["optimizer_updates"],
            )
        )
    for method in METHODS:
        if len({e["initial_sha256"] for e in entries if e["method"] == method}) != 3:
            raise ValueError("duplicate seed initializations")
    return entries


def normalization(model, tr, method):
    def check(name, arr):
        target = max(float(np.sqrt(np.mean(arr**2))), 1e-8)
        np.testing.assert_allclose(
            getattr(model, name).detach().cpu().numpy(), target, rtol=1e-6, atol=1e-9
        )

    np.testing.assert_allclose(
        model.input_scale.detach().cpu().numpy(),
        [np.sqrt(np.mean(tr[k] ** 2)) for k in ("f", "yd")],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.output_scale.detach().cpu().numpy(),
        [np.sqrt(np.mean(tr["y"] ** 2)), np.sqrt(np.mean((tr["p"] / 0.001) ** 2))],
        rtol=1e-6,
    )
    if method == "pdol":
        for name, key in [
            ("state_in", "v"),
            ("state_out", "sv"),
            ("adjoint_in", "w"),
            ("adjoint_out", "sw"),
        ]:
            check(name, tr[key])


def parameter_accounting(model, method):
    names = []
    count = 0
    total = 0
    for name, p in model.named_parameters():
        size = p.numel() * (2 if p.is_complex() else 1)
        total += size
        unused = (method == "pdol" and name.endswith(".projection.bias")) or (
            method.startswith("iuzawa_") and name.endswith("S_oper.projection.bias")
        )
        if method == "iuzawa_f" and name.startswith(
            f"unrolled.uzawa_layers.{model.config.uzawa_layers-1}.qb_sap_fourier_layers."
        ):
            unused = True
        if unused:
            names.append(name)
            count += size
    return dict(
        nominal_real_parameters=total,
        structurally_inactive_real_parameters=count,
        known_inactive_names=names,
        reason="Zero-preserving FNO subtraction cancels its output bias; the last free-layer dual update does not affect returned u. These are nominal, not effective-capacity counts.",
    )


def selected_training_loss(model, c, split, tr):
    if c.method not in ("osnn", "admm", "cnp"):
        return float(operator_loss_dataset(model, c, tr))
    xx, ww = quadrature(c.quadrature_order)
    x = torch.tensor(xx, dtype=torch.float64, device=c.device, requires_grad=True)
    w = torch.tensor(ww, dtype=torch.float64, device=c.device)
    values = None
    if c.method == "admm":
        previous = SplitState(c.problem(), c.beta)
        previous.snapshots = split.snapshots[:-1]
        with torch.no_grad():
            v = previous.at(x)
            values = (v["z"], v["lam"])
    return float(instance_loss(model, x, w, values, c.beta).detach())


def old_sources():
    answer = []
    for name in (
        "ch04",
        "ch04_adaptation",
        "ch04_operator_learning",
        "ch04_operator_adaptation",
        "ch04_completion",
        "ch04_delivery",
        "ch05",
    ):
        p = Path("chapters") / name
        m = json.loads((p / "RUNTIME_MANIFEST.json").read_text())
        for file, h in m.items():
            if sha(p / file) != h:
                raise ValueError("old frozen source changed: " + str(p / file))
        answer.append(dict(module=name, files=len(m)))
    return answer


def pretest(root):
    root = Path(root)
    if not (root / "TRAINING_COMPLETE").is_file() or (root / "heldout").exists():
        raise RuntimeError("pretest order")
    tr, _ = load_data(root / "data", "train")
    va, _ = load_data(root / "data", "val")
    check_split_identity(tr, "train")
    check_split_identity(va, "val")
    check_disjoint(tr, va)
    torch.set_num_threads(2)
    entries = registry(root)
    checks = []
    for e in entries:
        m, c, s, b = load_model(e["checkpoint"])
        v = validation(m, c, s, va)
        np.testing.assert_allclose(v["selection_score"], b["selection_score"], rtol=5e-4, atol=3e-8)
        device_check = {}
        if e["method"] not in ("osnn", "admm", "cnp"):
            normalization(m, tr, e["method"])
            if torch.cuda.is_available():
                from .assessment import predict

                gm, gc, _, _ = load_model(e["checkpoint"], "cuda")
                cpu, _ = predict(m, c, va["f"][:8], va["yd"][:8])
                gpu, _ = predict(gm, gc, va["f"][:8], va["yd"][:8])
                for key in cpu:
                    np.testing.assert_allclose(cpu[key], gpu[key], rtol=3e-4, atol=2e-6)
                    device_check[key] = float(np.max(np.abs(cpu[key] - gpu[key])))
                del gm
        checks.append(
            dict(
                key=e["key"],
                selection_reloaded=True,
                train_only_normalization=True,
                parameters=parameter_accounting(m, e["method"]),
                selected_training_loss=selected_training_loss(m, c, s, tr),
                validation_selection_score=v["selection_score"],
                validation_cross_device_max_abs=device_check,
            )
        )
        print("pretest", e["key"], flush=True)
    report = dict(
        status="passed",
        training_configurations=21,
        seeds=[0, 1, 2],
        checks=checks,
        old_source_protection=old_sources(),
        heldout_created=False,
    )
    write(root / "PRETEST_AUDIT.json", report)
    return report


def freeze(root):
    root = Path(root)
    if (root / "SELECTION_FROZEN.json").exists() or (root / "heldout").exists():
        raise RuntimeError("already frozen/test exists")
    if json.loads((root / "PRETEST_AUDIT.json").read_text())["status"] != "passed":
        raise RuntimeError("audit required")
    entries = registry(root)
    sources = [p for m in ("ch05", "ch05_delivery") for p in (Path("chapters") / m).rglob("*.py")]
    relevant = [
        root / "jobs.json",
        root / "protocol.json",
        root / "PRETEST_AUDIT.json",
        root / "tests.log",
    ] + list((root / "data").glob("*"))
    relevant += [p for e in entries for p in Path(e["directory"]).glob("*") if p.is_file()]
    record = dict(
        version="ch05-unified-delivery-v1",
        timestamp_unix=time.time(),
        entries=entries,
        source_hashes={p.as_posix(): sha(p) for p in sorted(sources)},
        training_hashes={p.as_posix(): sha(p) for p in sorted(relevant)},
        test_protocol=dict(
            count=128,
            grid=32,
            refinement_cases=8,
            refinement_grids=[64, 128],
            stream=[20261028, 5051, 20261029],
        ),
        instance_protocol=dict(gauss_order=96, boundary_points=129, fd_grids=[32, 64, 128]),
        limits="Fixed-instance spatial checks are not new-instance generalization; fixed-grid operator prolongation is not resolution transfer.",
    )
    write(root / "SELECTION_FROZEN.json", record)
    return record


def verify_freeze(root):
    f = json.loads((Path(root) / "SELECTION_FROZEN.json").read_text())
    for category in ("source_hashes", "training_hashes"):
        for p, h in f[category].items():
            if sha(p) != h:
                raise ValueError("post-freeze mutation " + p)
    if registry(root) != f["entries"]:
        raise ValueError("changed registry")
    return f


def generate_holdout(root):
    root = Path(root)
    f = verify_freeze(root)
    d = root / "heldout"
    if d.exists():
        raise FileExistsError(d)
    spec = f["test_protocol"]
    a, ref = make("test", spec["count"], spec["grid"])
    validate(a, spec["grid"])
    check_split_identity(a, "test")
    ids = fingerprints(a)
    if len(ids) != spec["count"]:
        raise ValueError("test duplicates")
    for role in ("train", "val"):
        other, _ = load_data(root / "data", role)
        check_disjoint(a, other)
    d.mkdir()
    np.savez_compressed(d / "test.npz", **a)
    write(
        d / "manifest.json",
        dict(
            role="test",
            count=spec["count"],
            grid=spec["grid"],
            created_unix=time.time(),
            sha256=sha(d / "test.npz"),
            selection_sha256=sha(root / "SELECTION_FROZEN.json"),
            reference=ref,
            no_train_val_overlap=True,
        ),
    )
    return a


def load_holdout(root):
    root = Path(root)
    verify_freeze(root)
    m = json.loads((root / "heldout/manifest.json").read_text())
    if m["selection_sha256"] != sha(root / "SELECTION_FROZEN.json") or m["sha256"] != sha(
        root / "heldout/test.npz"
    ):
        raise ValueError("test identity")
    with np.load(root / "heldout/test.npz", allow_pickle=False) as a:
        result = dict(a)
    validate(result, m["grid"])
    check_split_identity(result, "test")
    return result
