"""Fresh-run train/val workflows. Checkpoints support inference, not exact resume."""

from dataclasses import asdict
from pathlib import Path
import copy, json, time
import numpy as np
import torch
from .common import Config, INSTANCE, write_json, source_hashes
from .models import Fields, SplitState, loss_terms
from .operators import Operators
from .evaluation import points, instance_metrics, operator_metrics
from .data import load_split


def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def save(path, model, c, step, score, split=None, data_identity=None):
    torch.save(
        dict(
            schema="ch05-inference-v1",
            config=asdict(c),
            model=cpu_state(model.state_dict()),
            optimizer_updates=step,
            selection_score=score,
            source_hashes=source_hashes(),
            split_history=[] if split is None else [cpu_state(s) for s in split.state_dict()],
            data_identity=data_identity,
            exact_resume_supported=False,
        ),
        path,
    )


def load(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("schema") != "ch05-inference-v1" or blob["source_hashes"] != source_hashes():
        raise ValueError("checkpoint schema/source identity mismatch")
    c = Config(**blob["config"])
    c.device = device
    c.validate()
    if c.method in INSTANCE:
        model = Fields(c.problem(), c.method, c.width, c.depth).to(
            device=device, dtype=torch.float64
        )
        split = SplitState(c.problem(), c.beta)
        split.load(blob["split_history"], model)
    else:
        model = Operators(c).to(device)
        split = None
    model.load_state_dict(blob["model"])
    return model, c, split, blob


def checked_step(optimizer, model, closure):
    before = [p.detach().clone() for p in model.parameters()]
    value = closure()
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite loss")
    value.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("invalid gradients")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("invalid parameters")
    return float(value.detach()), sum(
        float((a - b).abs().sum()) for a, b in zip(before, model.parameters())
    )


def train(config, out, data_root=None):
    c = config.validate()
    out = Path(out)
    # Validate all input data before creating a run; no test arrays are loaded.
    tr = va = identity = None
    if c.method not in INSTANCE:
        if data_root is None:
            raise ValueError("operator training requires --data")
        tr, meta = load_split(data_root, "train", c.problem(), c.grid)
        va, _ = load_split(data_root, "val", c.problem(), c.grid)
        if np.intersect1d(tr["sample_id"], va["sample_id"]).size:
            raise ValueError("train/val ID overlap")
        identity = {s: meta["splits"][s]["sha256"] for s in ("train", "val")}
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    torch.set_num_threads(2)
    started = time.time()
    history = []
    updates = 0
    movement = 0.0
    best = float("inf")
    write_json(out / "config.json", asdict(c))
    write_json(
        out / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            device=c.device,
            dtype="float64" if c.method in INSTANCE else "float32",
            source_hashes=source_hashes(),
        ),
    )
    split = None
    if c.method in INSTANCE:
        model = Fields(c.problem(), c.method, c.width, c.depth).to(
            device=c.device, dtype=torch.float64
        )
        split = SplitState(c.problem(), c.beta)
        tx = torch.tensor(points(c.seed + 1, c.train_points), dtype=torch.float64, device=c.device)
    else:
        model = Operators(c).to(c.device)
        model.set_scales(tr)

    def record(stage, epoch=None):
        nonlocal best
        if c.method in INSTANCE:
            metrics = instance_metrics(model, c, split)
            score = metrics["selection_score"]
        else:
            score = operator_loss_dataset(model, c, va)
            metrics = dict(selection_score=score)
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite validation score")
        item = dict(stage=stage, epoch=epoch, optimizer_updates=updates, **metrics)
        history.append(item)
        if score < best:  # earliest minimum
            best = score
            save(out / "best.pt", model, c, updates, score, split, identity)
        save(out / "last.pt", model, c, updates, score, split, identity)
        write_json(out / "history.json", history)
        write_json(
            out / "status.json",
            dict(state="running", updates=updates, best_score=best, stage=stage),
        )

    def inner(adam_steps, lbfgs_steps, split_values=None):
        nonlocal updates, movement
        opt = torch.optim.Adam(model.parameters(), lr=c.learning_rate)

        def loss():
            x = tx.detach().requires_grad_(True)
            return sum(loss_terms(model, x, split_values, c.beta).values())

        for step in range(adam_steps):
            opt.zero_grad(set_to_none=True)
            _, change = checked_step(opt, model, loss)
            updates += 1
            movement += change
            if c.method != "admm" and (step + 1) % c.evaluate_every == 0:
                record("adam")
        if lbfgs_steps:
            opt = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad(set_to_none=True)
                value = loss()
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite LBFGS loss")
                value.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite LBFGS gradient")
                return value

            opt.step(closure)
            updates += int(opt.state[next(iter(model.parameters()))].get("n_iter", 0))
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("invalid parameters")

    try:
        record("initial", 0)
        if c.method == "admm":
            for outer in range(c.outer_steps):
                with torch.no_grad():
                    prev = split.at(tx)
                    values = (prev["z"].detach(), prev["lam"].detach())
                inner(c.inner_steps, c.inner_lbfgs, values)
                split.append(model)
                record("admm", outer + 1)
        elif c.method in INSTANCE:
            inner(c.adam_steps, c.lbfgs_steps)
            record("final")
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.6)
            rng = np.random.default_rng(c.seed + 500)
            for epoch in range(1, c.epochs + 1):
                order = rng.permutation(len(tr["f"]))
                for start in range(0, len(order), c.batch):
                    batch = {
                        k: v[order[start : start + c.batch]]
                        for k, v in tr.items()
                        if k != "sample_id"
                    }
                    opt.zero_grad(set_to_none=True)
                    _, change = checked_step(opt, model, lambda: operator_loss(model, c, batch))
                    updates += 1
                    movement += change
                scheduler.step()
                record("epoch", epoch)
        model_best, bc, bs, blob = load(out / "best.pt", c.device)
        metrics = (
            instance_metrics(model_best, bc, bs)
            if c.method in INSTANCE
            else operator_metrics(model_best, bc, va)
        )
        write_json(out / "validation.json", metrics)
        result = dict(
            state="complete",
            method=c.method,
            smoke=c.smoke,
            optimizer_updates=updates,
            best_updates=blob["optimizer_updates"],
            best_score=best,
            elapsed_seconds=time.time() - started,
            adam_parameter_movement=movement,
            parameters_real=sum(
                p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
            ),
            test_evaluations=0,
            formal_convergence_claim=False,
            exact_resume_supported=False,
        )
        write_json(out / "status.json", result)
        return result
    except Exception as e:
        write_json(
            out / "status.json",
            dict(state="failed", error=str(e), updates=updates, test_evaluations=0),
        )
        raise


def relative_loss(pred, target):
    return (
        (pred - target).flatten(1).square().sum(1)
        / target.flatten(1).square().sum(1).clamp_min(1e-16)
    ).mean()


def operator_loss(model, c, data):
    t = lambda k: torch.as_tensor(data[k][..., None], dtype=torch.float32, device=c.device)
    if c.method == "pdol":
        return relative_loss(model.action(t("v"), "state"), t("sv")) + relative_loss(
            model.action(t("w"), "adjoint"), t("sw")
        )
    out = model(t("f"), t("yd"))
    if c.method == "mimo":
        u, y, p = out
        return relative_loss(u, t("u")) + relative_loss(y, t("y")) + relative_loss(p, t("p"))
    return relative_loss(out, t("u"))


def operator_loss_dataset(model, c, data):
    model.eval()
    total = 0.0
    count = len(data["f"])
    with torch.no_grad():
        for start in range(0, count, c.batch):
            batch = {k: v[start : start + c.batch] for k, v in data.items() if k != "sample_id"}
            total += float(operator_loss(model, c, batch)) * len(batch["f"])
    model.train()
    return total / count
