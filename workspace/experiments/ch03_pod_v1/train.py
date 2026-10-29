"""Resumable coefficient regression; validation uses the full reconstructed L2 error."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from pod import CoefficientNet, coefficient_loss, evaluate_coefficients, reconstruct
from neuralpde_opt.chapter3.operator_training import evaluate_arrays


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def validate_completed(path):
    path = Path(path)
    cert = json.loads((path / "COMPLETE").read_text())
    if cert["manifest_sha256"] != sha(path / "manifest.json"):
        raise ValueError("Run certificate changed")
    for name, digest in json.loads((path / "manifest.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(path / name) != digest:
            raise ValueError("Completed run changed: " + name)
    return json.loads((path / "metrics.json").read_text())


def train_one(path, config, arrays, scales_np, rank, seed, identity, resume=False, stop_after=None):
    path = Path(path)
    run_identity = {**identity, "config": config, "rank": rank, "seed": seed}
    if resume:
        if not path.is_dir():
            raise ValueError("No run to resume")
        if json.loads((path / "identity.json").read_text()) != run_identity:
            raise ValueError("Run identity changed")
        if (path / "COMPLETE").exists():
            validate_completed(path)
            return torch.load(path / "last.pt", weights_only=True)
    else:
        path.mkdir(parents=True, exist_ok=False)
        dump(path / "identity.json", run_identity)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = CoefficientNet(arrays["train"]["x"].shape[1], rank, config["width"], config["depth"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    generator = torch.Generator().manual_seed(seed + 10000)
    scales = torch.tensor(scales_np, dtype=torch.float64)
    train = arrays["train"]
    true_alpha = torch.tensor(train["alpha"], dtype=torch.float64)
    state = {
        "epoch": 0,
        "optimizer_steps": 0,
        "sample_visits": 0,
        "best_score": float("inf"),
        "best_epoch": 0,
        "best_model": None,
        "history": [],
        "elapsed_seconds": 0.0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "identity": run_identity,
    }
    if resume:
        state = torch.load(path / "last.pt", weights_only=True)
        if state["identity"] != run_identity:
            raise ValueError("Checkpoint identity changed")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        generator.set_state(state["generator_state"])
        torch.set_rng_state(state["torch_rng"])
    end = config["epochs"] if stop_after is None else stop_after
    if not state["epoch"] <= end <= config["epochs"]:
        raise ValueError("Invalid stopping epoch")
    elapsed = state["elapsed_seconds"]
    start = time.perf_counter()
    for epoch in range(state["epoch"] + 1, end + 1):
        model.train()
        order = torch.randperm(len(train["x"]), generator=generator)
        for begin in range(0, len(order), config["batch_size"]):
            ix = order[begin : begin + config["batch_size"]]
            opt.zero_grad(set_to_none=True)
            loss = coefficient_loss(
                model(train["x"][ix]), true_alpha[ix], scales, train["denominator"][ix]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite coefficient loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            state["optimizer_steps"] += 1
            state["sample_visits"] += len(ix)
        _, train_score = evaluate_coefficients(model, train, scales)
        _, val_score = evaluate_coefficients(model, arrays["val"], scales)
        row = {
            "epoch": epoch,
            "train_l2": train_score,
            "val_l2": val_score,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        state["history"].append(row)
        if val_score < state["best_score"]:
            state.update(
                best_score=val_score, best_epoch=epoch, best_model=copy.deepcopy(model.state_dict())
            )
        schedule.step()
        state["epoch"] = epoch
        if epoch % config["checkpoint_every"] == 0 or epoch == end:
            state.update(
                model=copy.deepcopy(model.state_dict()),
                optimizer=opt.state_dict(),
                scheduler=schedule.state_dict(),
                generator_state=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                elapsed_seconds=elapsed + time.perf_counter() - start,
            )
            save(path / "last.pt", state)
            save(
                path / "best.pt",
                {
                    "model": state["best_model"],
                    "identity": run_identity,
                    "best_epoch": state["best_epoch"],
                    "best_score": state["best_score"],
                },
            )
            dump(path / "history.json", state["history"])
            dump(
                path / "status.json",
                {
                    "status": "training" if epoch < config["epochs"] else "trained_pending_audit",
                    "epoch": epoch,
                    "best_epoch": state["best_epoch"],
                    "optimizer_steps": state["optimizer_steps"],
                },
            )
        if epoch == 1 or epoch % 400 == 0 or epoch == end:
            print("POD_PROGRESS", rank, seed, epoch, train_score, val_score, flush=True)
    return state


def finish_one(path, state, config, arrays, basis, rank, data, geometry):
    path = Path(path)
    expected = config["epochs"] * math.ceil(len(arrays["train"]["x"]) / config["batch_size"])
    if state["epoch"] != config["epochs"] or state["optimizer_steps"] != expected:
        raise RuntimeError("Incomplete optimizer budget")
    # Adam stores its per-parameter state under the "state" mapping.
    steps = sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
    if steps != [expected] or state["sample_visits"] != len(data["train"]["y"]) * config["epochs"]:
        raise RuntimeError("Actual optimizer or sample count mismatch")
    if min(state["history"], key=lambda row: row["val_l2"])["epoch"] != state["best_epoch"]:
        raise RuntimeError("Checkpoint selection mismatch")
    saved = torch.load(path / "best.pt", weights_only=True)
    if saved["identity"] != state["identity"] or saved["best_epoch"] != state["best_epoch"]:
        raise RuntimeError("Saved model identity mismatch")
    model = CoefficientNet(arrays["train"]["x"].shape[1], rank, config["width"], config["depth"])
    model.load_state_dict(saved["model"])
    reference = CoefficientNet(
        arrays["train"]["x"].shape[1], rank, config["width"], config["depth"]
    )
    reference.load_state_dict(state["best_model"])
    scales = torch.tensor(np.sqrt(basis["eigenvalues"][:rank]), dtype=torch.float64)
    metrics, checks = {}, {}
    for split, part in data.items():
        alpha, score = evaluate_coefficients(model, arrays[split], scales)
        alpha_ref, _ = evaluate_coefficients(reference, arrays[split], scales)
        if not np.array_equal(alpha, alpha_ref):
            raise RuntimeError("Checkpoint reload differed")
        prediction = reconstruct(alpha, basis, rank)
        metric = evaluate_arrays(prediction, part["y"], geometry, part["sample_id"])
        if metric["max"]["boundary_max"] > 1e-12:
            raise RuntimeError("Reconstruction violated zero boundary")
        projection_squared = arrays[split]["projection_squared"].numpy()
        coefficient_squared = np.sum((alpha - arrays[split]["alpha"]) ** 2, axis=1)
        total_squared = geometry.l2_squared(prediction - part["y"])
        pythagoras = float(np.max(abs(total_squared - projection_squared - coefficient_squared)))
        local_error = (prediction - part["y"]).reshape(len(prediction), -1)[:, geometry.elements]
        local_target = part["y"].reshape(len(prediction), -1)[:, geometry.elements]

        def norm(v):
            return np.sum((v.sum(-1) ** 2 + (v * v).sum(-1)) * geometry.areas / 12, axis=1)

        independent = np.sqrt(norm(local_error) / norm(local_target))
        computed = np.array([v["relative_l2"] for v in metric["per_sample"]])
        difference = float(np.max(abs(independent - computed)))
        if (
            pythagoras > 1e-10
            or difference > 1e-12
            or abs(score - metric["mean"]["relative_l2"]) > 1e-10
        ):
            raise RuntimeError("Independent L2/decomposition audit failed")
        metric["coefficient_mean_relative_l2"] = float(
            np.mean(np.sqrt(coefficient_squared / arrays[split]["denominator"].numpy()))
        )
        metric["projection_mean_relative_l2"] = float(
            np.mean(np.sqrt(projection_squared / arrays[split]["denominator"].numpy()))
        )
        metrics[split] = metric
        checks[split] = {
            "reload_max_difference": 0.0,
            "pythagorean_squared_max_defect": pythagoras,
            "independent_relative_l2_max_difference": difference,
        }
        np.savez_compressed(
            path / (split + "_predictions.npz"),
            prediction=prediction,
            coefficients=alpha,
            sample_id=part["sample_id"],
            total_squared=total_squared,
            projection_squared=projection_squared,
            coefficient_squared=coefficient_squared,
        )
    if abs(metrics["val"]["mean"]["relative_l2"] - state["best_score"]) > 1e-10:
        raise RuntimeError("Saved selection score mismatch")
    result = {
        "rank": rank,
        "seed": state["identity"]["seed"],
        "epochs": state["epoch"],
        "optimizer_steps": expected,
        "sample_visits": state["sample_visits"],
        "best_epoch": state["best_epoch"],
        "parameter_count": state["parameter_count"],
        "elapsed_seconds": state["elapsed_seconds"],
        "best": metrics,
        "test_evaluations": 0,
    }
    dump(path / "metrics.json", result)
    dump(
        path / "verification.json",
        {
            "passed": True,
            "checks": checks,
            "optimizer_state_steps": steps,
            "selection_verified": True,
            "test_evaluations": 0,
        },
    )
    dump(
        path / "status.json",
        {"status": "complete", "epoch": state["epoch"], "optimizer_steps": expected},
    )
    dump(
        path / "manifest.json",
        {
            p.name: sha(p)
            for p in path.iterdir()
            if p.is_file() and p.name not in {"manifest.json", "COMPLETE"}
        },
    )
    dump(
        path / "COMPLETE",
        {"schema_version": "ch03-pod-v1", "manifest_sha256": sha(path / "manifest.json")},
    )
    print(
        "POD_RUN_COMPLETE",
        rank,
        state["identity"]["seed"],
        metrics["train"]["mean"]["relative_l2"],
        metrics["val"]["mean"]["relative_l2"],
        flush=True,
    )
    return result
