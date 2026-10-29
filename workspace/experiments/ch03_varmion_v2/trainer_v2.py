"""Deterministic, resumable, equal-update train/val ablation engine."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from model_v2 import operator_loss, supervised_loss
from common import dump, save
from neuralpde_opt.chapter3.reaction_training import seed_everything
from neuralpde_opt.chapter3.operator_training import capture_rng, restore_rng, _cpu_copy


def tensors(arrays, device):
    result = {}
    for split, d in arrays.items():
        result[split] = {
            k: torch.as_tensor(
                v, device=device, dtype=torch.float32 if k in {"sensors", "z"} else torch.float64
            )
            for k, v in d.items()
        }
    return result


@torch.no_grad()
def evaluate(model, d):
    model.eval()
    alpha = model(d["sensors"], d["z"]).double()
    score = torch.sqrt(
        ((alpha - d["alpha"]).square().sum(1) + d["projection_squared"]) / d["denominator"]
    ).mean()
    return alpha.cpu().numpy(), float(score)


def fit(
    path,
    config,
    seed,
    factory,
    arrays,
    operator_pool,
    identity,
    device="cuda",
    resume=False,
    stop_after=None,
):
    path = Path(path)
    identity = {**identity, "config": config, "seed": seed, "device": device}
    if resume:
        if json.loads((path / "identity.json").read_text()) != identity:
            raise ValueError("Training identity changed")
    else:
        path.mkdir(parents=True, exist_ok=False)
        dump(path / "identity.json", identity)
    seed_everything(seed, 2)
    rng = np.random.default_rng(seed)
    rng_operator = np.random.default_rng(seed + 100003)
    model = factory().to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    state = dict(
        identity=identity,
        epoch=0,
        steps=0,
        visits=0,
        operator_visits=0,
        best_epoch=0,
        best_score=float("inf"),
        best_model=None,
        history=[],
        elapsed_seconds=0.0,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    if resume:
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity changed")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], rng)
        rng_operator.bit_generator.state = state["operator_rng"]
    d = tensors(arrays, device)
    if operator_pool is not None:
        pool = {
            k: torch.as_tensor(v, device=device, dtype=torch.float32)
            for k, v in operator_pool.items()
        }
        if len(pool["sensors"]) != len(pool["B"]) or not len(pool["B"]):
            raise ValueError("Invalid operator pool")
    else:
        pool = None
    count = len(d["train"]["sensors"])
    start, before = time.perf_counter(), state["elapsed_seconds"]

    def checkpoint():
        state.update(
            model=_cpu_copy(model.state_dict()),
            optimizer=_cpu_copy(opt.state_dict()),
            scheduler=sched.state_dict(),
            rng=capture_rng(rng),
            operator_rng=rng_operator.bit_generator.state,
        )
        save(path / "last.pt", state)
        if state["best_model"] is not None:
            save(
                path / "best.pt",
                dict(
                    identity=identity,
                    model=state["best_model"],
                    best_epoch=state["best_epoch"],
                    best_score=state["best_score"],
                ),
            )
        dump(path / "history.json", state["history"])

    if not resume:
        checkpoint()
    end = config["epochs"] if stop_after is None else stop_after
    if not state["epoch"] <= end <= config["epochs"]:
        raise ValueError("Invalid endpoint")
    for epoch in range(state["epoch"] + 1, end + 1):
        model.train()
        order = rng.permutation(count)
        losses, constraints = [], []
        for begin in range(0, count, config["batch_size"]):
            ix = torch.as_tensor(order[begin : begin + config["batch_size"]], device=device)
            opt.zero_grad(set_to_none=True)
            prediction = model(d["train"]["sensors"][ix], d["train"]["z"][ix])
            data_loss = supervised_loss(
                prediction, d["train"]["alpha"][ix], d["train"]["denominator"][ix]
            )
            constraint = data_loss * 0
            if pool is not None:
                oi = torch.as_tensor(
                    rng_operator.integers(len(pool["B"]), size=config["operator_batch_size"]),
                    device=device,
                )
                k = model.matrix(pool["sensors"][oi])
                constraint = operator_loss(k, pool["B"][oi], model.inverse_l0)
                state["operator_visits"] += len(oi)
            loss = data_loss + config["operator_weight"] * constraint
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            state["steps"] += 1
            state["visits"] += len(ix)
            losses.append(float(data_loss.detach()))
            constraints.append(float(constraint.detach()))
        _, val = evaluate(model, d["val"])
        _, train = evaluate(model, d["train"])
        if val < state["best_score"]:
            state.update(best_score=val, best_epoch=epoch, best_model=_cpu_copy(model.state_dict()))
        state["history"].append(
            dict(
                epoch=epoch,
                train_l2=train,
                val_l2=val,
                minibatch_data_loss_mean=float(np.mean(losses)),
                operator_loss_mean=float(np.mean(constraints)),
                learning_rate=float(opt.param_groups[0]["lr"]),
            )
        )
        sched.step()
        state.update(epoch=epoch, elapsed_seconds=before + time.perf_counter() - start)
        if epoch % config["checkpoint_every"] == 0 or epoch == end:
            checkpoint()
            dump(
                path / "status.json",
                dict(
                    status="training" if epoch < config["epochs"] else "trained_pending_audit",
                    epoch=epoch,
                    best_epoch=state["best_epoch"],
                    best_score=state["best_score"],
                ),
            )
        if epoch == 1 or epoch % 100 == 0 or epoch == end:
            print(
                "V2_PROGRESS", path.parent.name, seed, epoch, val, state["best_score"], flush=True
            )
    if end == config["epochs"]:
        steps = config["epochs"] * math.ceil(count / config["batch_size"])
        if state["steps"] != steps or state["visits"] != count * config["epochs"]:
            raise RuntimeError("Budget mismatch")
        expected_op = steps * config["operator_batch_size"] if pool is not None else 0
        if state["operator_visits"] != expected_op:
            raise RuntimeError("Operator exposure mismatch")
        if {int(v["step"]) for v in state["optimizer"]["state"].values()} != {steps}:
            raise RuntimeError("Optimizer step mismatch")
        if min(state["history"], key=lambda x: x["val_l2"])["epoch"] != state["best_epoch"]:
            raise RuntimeError("Selection mismatch")
    return state
