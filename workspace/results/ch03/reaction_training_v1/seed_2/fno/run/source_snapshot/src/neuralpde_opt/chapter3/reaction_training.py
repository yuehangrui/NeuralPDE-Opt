"""Certified reaction-only epoch training with explicit resumable checkpoints.

Only c/y train and validation arrays are opened. Every validation sample has
equal weight for best-model selection by P1 relative L2; no test is generated.

Book/code notation:
Train/validation only: c is input, y is the reference PDE state, and predictions are y_theta. Normalization is fitted on train. Serialized field and checkpoint names are retained.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, hashlib, json, random, shutil, time, sys, platform
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from . import operator_training as t
from .reaction_data import verify_dataset
from .reaction_operators import ReactionOperator, fit_reaction_normalization
from .upstream_operators import DeepXDEOperator, GalerkinOperator

SCHEMA = "ch03-reaction-training-v1"
ROOT = Path(__file__).resolve().parents[3]


def source_identity():
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/neuralpde_opt/chapter3/reaction.py",
        ROOT / "src/neuralpde_opt/chapter3/reaction_data.py",
        ROOT / "src/neuralpde_opt/chapter3/reaction_operators.py",
        ROOT / "src/neuralpde_opt/chapter3/operator_training.py",
        ROOT / "src/neuralpde_opt/chapter3/upstream_operators.py",
        ROOT / "src/neuralpde_opt/chapter3/fno.py",
        ROOT / "src/neuralpde_opt/chapter3/gno.py",
    ]
    paths += list((ROOT / "src/neuralpde_opt/chapter3/upstream_vendor").glob("*.py"))
    return {str(p.relative_to(ROOT)): t.sha256(p) for p in sorted(paths)}


def validate_config(c):
    if c.get("schema_version") != SCHEMA:
        raise ValueError("Expected reaction training schema")
    if c.get("purpose") not in ["formal_trainval", "preflight"]:
        raise ValueError("Invalid purpose")
    for k in [
        "epochs",
        "batch_size",
        "microbatch_size",
        "eval_batch_size",
        "checkpoint_every",
        "train_eval_every",
        "cpu_threads",
    ]:
        if type(c.get(k)) is not int or c[k] < 1:
            raise ValueError("Invalid " + k)
    if c["microbatch_size"] > c["batch_size"]:
        raise ValueError("microbatch exceeds logical batch")
    if (
        not c["seeds"]
        or len(set(c["seeds"])) != len(c["seeds"])
        or any(type(s) is not int or not 0 <= s < 2**32 for s in c["seeds"])
    ):
        raise ValueError("Invalid seeds")
    if (
        c.get("objective") != "relative_p1_l2_squared"
        or c.get("selection") != "earliest_min_val_p1_relative_l2"
    ):
        raise ValueError("Invalid loss/selection")
    if c.get("test_evaluations") != 0:
        raise ValueError("Test evaluation forbidden")
    names = []
    for m in c["models"]:
        names.append(m["name"])
        if m["name"] not in ["fno", "gno", "deepxde_fourier4", "galerkin"]:
            raise ValueError("Unsupported model")
        p = m["training"]
        if p["optimizer"] not in ["Adam", "AdamW"] or p["scheduler"] not in ["cosine", "plateau"]:
            raise ValueError("Invalid optimizer/scheduler")
        for k in ["learning_rate", "weight_decay", "min_lr", "grad_clip"]:
            if type(p[k]) not in [int, float] or not np.isfinite(p[k]) or p[k] < 0:
                raise ValueError("Invalid " + k)
        if p["learning_rate"] <= 0 or p["min_lr"] > p["learning_rate"] or p["grad_clip"] <= 0:
            raise ValueError("Invalid scales")
        if p["scheduler"] == "plateau" and (
            type(p.get("patience")) is not int or p["patience"] < 0 or not 0 < p["factor"] < 1
        ):
            raise ValueError("Invalid plateau")
    if not names or len(set(names)) != len(names):
        raise ValueError("Repeated/no model names")
    if c["purpose"] == "formal_trainval" and ("train_limit" in c or "val_limit" in c):
        raise ValueError("Formal training must use all samples")
    return c


def seed_everything(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def public_data_only(directory):
    """Deny opening reference labels, old splits, or any test through NumPy."""
    directory = Path(directory).resolve()
    original = np.load
    opened = []
    allowed = {directory / n for n in ["train.npz", "val.npz", "observation_geometry.npz"]}

    def guarded(path, *args, **kwargs):
        p = Path(path).resolve()
        if p not in allowed or kwargs.get("allow_pickle", False):
            raise ValueError("Forbidden array read: " + str(p))
        opened.append(str(p))
        return original(p, *args, **kwargs)

    np.load = guarded
    try:
        yield opened
    finally:
        np.load = original


def model_for(spec, norm, epsilon):
    name = spec["name"]
    kwargs = spec["kwargs"]
    if name in ["fno", "gno"]:
        core = t.make_model(spec)
    elif name == "deepxde_fourier4":
        core = DeepXDEOperator(**kwargs)
    elif name == "galerkin":
        core = GalerkinOperator(**kwargs)
    else:
        raise ValueError(name)
    return ReactionOperator(core, norm, epsilon)


def scheduler_for(opt, p, epochs):
    if p["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=p["min_lr"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=p["patience"], factor=p["factor"], min_lr=p["min_lr"]
    )


def batch_update(model, c, y, indices, microbatch, opt, norm2, clip):
    """The last short microbatch has weight n_micro/n_logical, not 1/n_chunks."""
    opt.zero_grad(set_to_none=True)
    value = 0.0
    for start in range(0, len(indices), microbatch):
        ix = torch.as_tensor(indices[start : start + microbatch], device=c.device, dtype=torch.long)
        target = y[ix]
        pred = model(c[ix])
        denominator = norm2(target)
        if (denominator <= 0).any():
            raise ValueError("Zero energy target")
        loss = (norm2(pred - target) / denominator).sum() / len(indices)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        value += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    return value


def predict(model, data, device, batch_size):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(data["c"]), batch_size):
            c = torch.as_tensor(
                data["c"][start : start + batch_size], device=device, dtype=torch.float32
            )
            p = model(c)
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite prediction")
            result.append(p.cpu().numpy().astype(np.float64))
    return np.concatenate(result)


def best_payload(state):
    return {
        k: state[k]
        for k in [
            "schema_version",
            "config",
            "spec",
            "seed",
            "normalization",
            "identity",
            "best_epoch",
            "best_score",
            "best_model_state",
        ]
    }


def load_model(output, device="cpu"):
    output = Path(output)
    ck = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if ck.get("schema_version") != SCHEMA or ck["identity"]["sources"] != source_identity():
        raise ValueError("Checkpoint source/schema mismatch")
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["best.pt"] != t.sha256(output / "best.pt"):
            raise ValueError("Checkpoint hash mismatch")
    model = model_for(ck["spec"], ck["normalization"], ck["identity"]["epsilon"]).to(device)
    model.load_state_dict(ck["best_model_state"])
    return model.eval()


def run(
    config, data_directory, output, name, seed, device="cuda", resume=False, stop_after_epoch=None
):
    config = validate_config(config)
    data_directory = Path(data_directory).resolve()
    out = Path(output).resolve()
    if seed not in config["seeds"]:
        raise ValueError("Undeclared seed")
    spec = next((s for s in config["models"] if s["name"] == name), None)
    if spec is None:
        raise ValueError("Undeclared model")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("Invalid stopping epoch")
    if resume:
        if not out.is_dir() or (out / "COMPLETE").exists():
            raise ValueError("Resume requires unfinished existing run")
    else:
        out.mkdir(parents=True, exist_ok=False)
    with t.exclusive_run_lock(out), public_data_only(data_directory) as opened:
        data = verify_dataset(data_directory)
        dc = json.loads((data_directory / "config.json").read_text())
        if config["purpose"] == "preflight":
            for split in ["train", "val"]:
                size = config.get(split + "_limit", len(data[split]["c"]))
                if type(size) is not int or not 1 <= size <= len(data[split]["c"]):
                    raise ValueError("Invalid preflight subset")
                data[split] = {k: v[:size] for k, v in data[split].items()}
        seed_everything(seed, config["cpu_threads"])
        geom = t.P1Geometry(data_directory / "observation_geometry.npz")
        norm = fit_reaction_normalization(data["train"], geom)
        identity = {
            "sources": source_identity(),
            "dataset_manifest_sha256": t.sha256(data_directory / "manifest.json"),
            "dataset_directory": str(data_directory),
            "epsilon": dc["epsilon"],
            "train_sample_ids": data["train"]["sample_id"].tolist(),
            "val_sample_ids": data["val"]["sample_id"].tolist(),
            "runtime": {
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "python": platform.python_version(),
                "device": device,
                "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        }
        model = model_for(spec, norm, dc["epsilon"]).to(device)
        p = spec["training"]
        opt = getattr(torch.optim, p["optimizer"])(
            model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
        )
        scheduler = scheduler_for(opt, p, config["epochs"])
        rng = np.random.default_rng(seed)
        if resume:
            state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
            for k, v in [
                ("schema_version", SCHEMA),
                ("config", config),
                ("spec", spec),
                ("seed", seed),
                ("identity", identity),
                ("normalization", norm),
            ]:
                if state.get(k) != v:
                    raise ValueError("Resume mismatch: " + k)
            model.load_state_dict(state["model_state"])
            opt.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            t.restore_rng(state["rng"], rng)
            if (out / "history.json").exists():
                shutil.copyfile(
                    out / "history.json", out / f"history_before_resume_{time.time_ns()}.json"
                )
            t.dump_json(out / "history.json", state["history"])
            if state["best_epoch"] > 0:
                t.atomic_write(out / "best.pt", best_payload(state))
        else:
            state = {
                "schema_version": SCHEMA,
                "config": config,
                "spec": spec,
                "seed": seed,
                "normalization": norm,
                "identity": identity,
                "epoch": 0,
                "optimizer_steps": 0,
                "sample_visits": 0,
                "history": [],
                "best_epoch": 0,
                "best_score": float("inf"),
                "best_model_state": None,
                "elapsed_seconds": 0.0,
            }
            for key, value in [
                ("config", config),
                ("normalization", norm),
                ("identity", identity),
                ("model_spec", spec),
            ]:
                t.dump_json(out / (key + ".json"), value)
            # Frozen source snapshot includes licenses for the copied upstream implementation.
            for rel in identity["sources"]:
                dest = out / "source_snapshot" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dest)
            for base in ["third_party/ch03_upstream"]:
                for src in (ROOT / base).rglob("*"):
                    if src.is_file() and "__pycache__" not in src.parts:
                        dest = out / "source_snapshot" / src.relative_to(ROOT)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dest)
            t.dump_json(
                out / "environment.json",
                {
                    **identity["runtime"],
                    "parameters_real": sum(
                        x.numel() * (2 if x.is_complex() else 1) for x in model.parameters()
                    ),
                    "deterministic": True,
                    "tf32": False,
                    "test_evaluations": 0,
                },
            )
        # Device transfer does not draw random numbers; preserve the restored RNG state.
        c = torch.as_tensor(data["train"]["c"], dtype=torch.float32, device=device)
        y = torch.as_tensor(data["train"]["y"], dtype=torch.float32, device=device)
        norm2 = geom.torch_loss(device)
        count = [state["optimizer_steps"]]
        hook = opt.register_step_post_hook(lambda *args: count.__setitem__(0, count[0] + 1))
        elapsed_before = state["elapsed_seconds"]
        start = time.monotonic()
        stop = config["epochs"] if stop_after_epoch is None else stop_after_epoch

        def checkpoint():
            state.update(
                model_state=t._cpu_copy(model.state_dict()),
                optimizer_state=t._cpu_copy(opt.state_dict()),
                scheduler_state=scheduler.state_dict(),
                rng=t.capture_rng(rng),
            )
            t.atomic_write(out / "last.pt", state)

        if not resume:
            checkpoint()
        for epoch in range(state["epoch"] + 1, stop + 1):
            model.train()
            order = rng.permutation(len(c))
            objective = 0.0
            used_lr = float(opt.param_groups[0]["lr"])
            for start_index in range(0, len(c), config["batch_size"]):
                ix = order[start_index : start_index + config["batch_size"]]
                objective += (
                    batch_update(
                        model, c, y, ix, config["microbatch_size"], opt, norm2, p["grad_clip"]
                    )
                    * len(ix)
                    / len(c)
                )
                state["sample_visits"] += len(ix)
            pred = predict(model, data["val"], device, config["eval_batch_size"])
            val = t.evaluate_arrays(pred, data["val"]["y"], geom, data["val"]["sample_id"])["mean"]
            score = val["relative_l2"]
            train_l2 = None
            if epoch == 1 or epoch % config["train_eval_every"] == 0 or epoch == config["epochs"]:
                tp = predict(model, data["train"], device, config["eval_batch_size"])
                train_l2 = t.evaluate_arrays(tp, data["train"]["y"], geom)["mean"]["relative_l2"]
            scheduler.step(score) if p["scheduler"] == "plateau" else scheduler.step()
            if score < state["best_score"]:
                state.update(
                    best_score=score,
                    best_epoch=epoch,
                    best_model_state=t._cpu_copy(model.state_dict()),
                )
                t.atomic_write(out / "best.pt", best_payload(state))
            state.update(
                epoch=epoch,
                optimizer_steps=count[0],
                elapsed_seconds=elapsed_before + time.monotonic() - start,
            )
            row = {
                "epoch": epoch,
                "optimizer_steps": count[0],
                "sample_visits": state["sample_visits"],
                "objective": objective,
                "train_l2": train_l2,
                "val_l2": score,
                "val_h1": val["relative_h1_seminorm"],
                "learning_rate_used": used_lr,
                "next_learning_rate": float(opt.param_groups[0]["lr"]),
                "best_epoch": state["best_epoch"],
                "best_val_l2": state["best_score"],
                "elapsed_seconds": state["elapsed_seconds"],
            }
            state["history"].append(row)
            t.dump_json(out / "history.json", state["history"])
            t.dump_json(
                out / "status.json",
                {"status": "running", "model": name, "seed": seed, **row, "test_evaluations": 0},
            )
            if epoch % config["checkpoint_every"] == 0 or epoch == stop:
                checkpoint()
            if epoch == 1 or epoch % 5 == 0 or epoch == stop:
                print(json.dumps(row), flush=True)
        hook.remove()
        if stop < config["epochs"]:
            t.dump_json(
                out / "status.json",
                {
                    "status": "stopped_at_requested_epoch",
                    "epoch": state["epoch"],
                    "optimizer_steps": state["optimizer_steps"],
                },
            )
            return state
        expected = config["epochs"] * ((len(c) + config["batch_size"] - 1) // config["batch_size"])
        steps = sorted({int(s["step"]) for s in opt.state.values()})
        assert state["optimizer_steps"] == expected and steps == [expected]
        assert state["sample_visits"] == config["epochs"] * len(c)
        chosen = min(state["history"], key=lambda x: x["val_l2"])
        assert chosen["epoch"] == state["best_epoch"]
        reloaded = load_model(out, device)
        model.load_state_dict(state["best_model_state"])
        model.eval()
        metrics = {}
        reload_diff = {}
        independent_diff = {}
        for split, d in data.items():
            prediction = predict(reloaded, d, device, config["eval_batch_size"])
            same = predict(model, d, device, config["eval_batch_size"])
            reload_diff[split] = float(np.max(abs(prediction - same)))
            assert reload_diff[split] == 0
            metric = t.evaluate_arrays(prediction, d["y"], geom, d["sample_id"])
            metrics[split] = metric
            # Independent elementwise consistent-mass formula, not P1Geometry.l2_squared.
            e = (prediction - d["y"]).reshape(len(prediction), -1)[:, geom.elements]
            z = d["y"].reshape(len(prediction), -1)[:, geom.elements]

            def independent_l2(v):
                return np.sum((v.sum(-1) ** 2 + (v * v).sum(-1)) * geom.areas / 12, axis=1)

            independent = np.sqrt(independent_l2(e) / independent_l2(z))
            independent_diff[split] = float(
                np.max(
                    abs(independent - np.array([a["relative_l2"] for a in metric["per_sample"]]))
                )
            )
            assert independent_diff[split] < 1e-12
            np.savez_compressed(
                out / (split + "_predictions.npz"), prediction=prediction, sample_id=d["sample_id"]
            )
        assert abs(metrics["val"]["mean"]["relative_l2"] - state["best_score"]) < 1e-12
        assert (
            source_identity() == identity["sources"]
            and t.sha256(data_directory / "manifest.json") == identity["dataset_manifest_sha256"]
        )
        result = {
            "status": "complete",
            "model": name,
            "seed": seed,
            "epochs": state["epoch"],
            "optimizer_steps": expected,
            "sample_visits": state["sample_visits"],
            "best_epoch": state["best_epoch"],
            "best": metrics,
            "optimizer_state_steps": steps,
            "test_evaluations": 0,
            "opened_npz_files": sorted(set(opened)),
            "elapsed_seconds": state["elapsed_seconds"],
        }
        t.dump_json(out / "metrics.json", result)
        t.dump_json(
            out / "verification.json",
            {
                "passed": True,
                "checkpoint_reload_max_difference": reload_diff,
                "independent_relative_l2_max_difference": independent_diff,
                "selection_verified": True,
                "updates_verified": True,
                "test_evaluations": 0,
            },
        )
        t.dump_json(
            out / "status.json",
            {
                "status": "complete",
                "model": name,
                "seed": seed,
                "epoch": state["epoch"],
                "optimizer_steps": expected,
            },
        )
        t.dump_json(
            out / "manifest.json",
            {
                p.name: t.sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.name not in ["manifest.json", ".run.lock"]
            },
        )
        t.atomic_write(
            out / "COMPLETE",
            json.dumps(
                {"schema_version": SCHEMA, "manifest_sha256": t.sha256(out / "manifest.json")}
            ),
        )
        print("RUN_COMPLETE", name, seed, metrics["val"]["mean"]["relative_l2"], flush=True)
        return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epoch", type=int)
    a = p.parse_args()
    run(
        json.loads(Path(a.config).read_text()),
        a.data,
        a.output,
        a.model,
        a.seed,
        a.device,
        a.resume,
        a.stop_after_epoch,
    )


if __name__ == "__main__":
    main()
