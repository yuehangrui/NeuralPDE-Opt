"""Resumable coefficient regression shared by source and residual-map experiments."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import math, time
from pathlib import Path
import numpy as np
import torch
from common import dump, save, verify
from neuralpde_opt.chapter3.reaction_training import seed_everything
from neuralpde_opt.chapter3.operator_training import capture_rng, restore_rng, _cpu_copy


def tensor_data(arrays, device):
    out = {}
    for split, d in arrays.items():
        out[split] = {
            k: torch.tensor(
                v, device=device, dtype=torch.float32 if k in ["c", "z"] else torch.float64
            )
            for k, v in d.items()
        }
    return out


def evaluate(model, d):
    model.eval()
    with torch.no_grad():
        alpha = model(d["c"], d["z"]).double()
        coefficient = (alpha - d["alpha"]).square().sum(1)
        score = torch.sqrt((coefficient + d["projection_squared"]) / d["denominator"]).mean().item()
    return alpha.cpu().numpy(), score


def fit(
    path, config, seed, factory, arrays, identity, device="cuda", resume=False, stop_after=None
):
    path = Path(path)
    identity = {**identity, "config": config, "seed": seed, "device": device}
    if resume:
        import json

        if json.loads((path / "identity.json").read_text()) != identity:
            raise ValueError("Changed training identity")
    else:
        path.mkdir(parents=True, exist_ok=False)
        dump(path / "identity.json", identity)
    seed_everything(seed, 2)
    rng = np.random.default_rng(seed)
    model = factory().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    state = {
        "identity": identity,
        "epoch": 0,
        "steps": 0,
        "visits": 0,
        "best_epoch": 0,
        "best_score": float("inf"),
        "best_model": None,
        "history": [],
        "elapsed_seconds": 0.0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if resume:
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Changed checkpoint")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], rng)
    d = tensor_data(arrays, device)
    count = len(d["train"]["c"])
    before = state["elapsed_seconds"]
    start = time.perf_counter()

    def checkpoint():
        state.update(
            model=_cpu_copy(model.state_dict()),
            optimizer=_cpu_copy(opt.state_dict()),
            scheduler=schedule.state_dict(),
            rng=capture_rng(rng),
        )
        save(path / "last.pt", state)
        if state["best_model"] is not None:
            save(
                path / "best.pt",
                {
                    "identity": identity,
                    "model": state["best_model"],
                    "best_epoch": state["best_epoch"],
                    "best_score": state["best_score"],
                },
            )
        dump(path / "history.json", state["history"])

    if not resume:
        checkpoint()
    end = config["epochs"] if stop_after is None else stop_after
    if not state["epoch"] <= end <= config["epochs"]:
        raise ValueError("Invalid stopping epoch")
    for epoch in range(state["epoch"] + 1, end + 1):
        model.train()
        order = rng.permutation(count)
        for begin in range(0, count, config["batch_size"]):
            ix = torch.tensor(order[begin : begin + config["batch_size"]], device=device)
            opt.zero_grad(set_to_none=True)
            pred = model(d["train"]["c"][ix], d["train"]["z"][ix]).double()
            loss = (
                (pred - d["train"]["alpha"][ix]).square().sum(1) / d["train"]["denominator"][ix]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite coefficient loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            state["steps"] += 1
            state["visits"] += len(ix)
        _, val = evaluate(model, d["val"])
        _, train = evaluate(model, d["train"])
        if val < state["best_score"]:
            state.update(best_score=val, best_epoch=epoch, best_model=_cpu_copy(model.state_dict()))
        state["history"].append(
            {
                "epoch": epoch,
                "train_l2": train,
                "val_l2": val,
                "learning_rate": float(opt.param_groups[0]["lr"]),
            }
        )
        schedule.step()
        state.update(epoch=epoch, elapsed_seconds=before + time.perf_counter() - start)
        if epoch % config["checkpoint_every"] == 0 or epoch == end:
            checkpoint()
        if epoch == 1 or epoch % 200 == 0 or epoch == end:
            dump(
                path / "status.json",
                {
                    "status": "training" if epoch < config["epochs"] else "trained_pending_audit",
                    "epoch": epoch,
                    "best_epoch": state["best_epoch"],
                    "best_score": state["best_score"],
                },
            )
            print("COEFFICIENT_PROGRESS", path.name, seed, epoch, val, flush=True)
    if end < config["epochs"]:
        return state
    expected = config["epochs"] * math.ceil(count / config["batch_size"])
    if (
        state["steps"] != expected
        or state["visits"] != config["epochs"] * count
        or sorted({int(v["step"]) for v in state["optimizer"]["state"].values()}) != [expected]
    ):
        raise RuntimeError("Budget mismatch")
    if min(state["history"], key=lambda x: x["val_l2"])["epoch"] != state["best_epoch"]:
        raise RuntimeError("Selection mismatch")
    return state
