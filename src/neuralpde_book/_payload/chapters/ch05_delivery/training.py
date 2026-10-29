"""Full-budget training, common spatial validation, honest counters and source identities.

Book/code notation:
Book 5.2.1/5.2.2: instance_loss uses weighted quadrature. q_scaled=p/alpha is distinct from lambda_admm. CNP eliminates u=-Delta y for f=0 and enforces the state cap through a smooth ansatz.
"""

from dataclasses import asdict
from pathlib import Path
import json, time
import numpy as np
import torch
from chapters.ch05.common import INSTANCE
from chapters.ch05.models import Fields, SplitState, laplace
from chapters.ch05.operators import Operators
from chapters.ch05.training import cpu_state, operator_loss, operator_loss_dataset
from chapters.ch05.evaluation import instance_metrics, operator_metrics
from .common import Config, write, source_hashes, quadrature
from .data import load as load_data, fingerprints


def save(path, model, c, updates, score, split, identity, counters):
    torch.save(
        dict(
            schema="ch05-delivery-inference-v1",
            config=asdict(c),
            model=cpu_state(model.state_dict()),
            optimizer_updates=updates,
            selection_score=score,
            source_hashes=source_hashes(),
            split_history=[] if split is None else [cpu_state(s) for s in split.state_dict()],
            data_identity=identity,
            counters=dict(counters),
            exact_resume_supported=False,
        ),
        path,
    )


def load(path, device="cpu"):
    b = torch.load(path, map_location="cpu", weights_only=True)
    if b.get("schema") != "ch05-delivery-inference-v1" or b["source_hashes"] != source_hashes():
        raise ValueError("source/schema")
    c = Config(**b["config"])
    c.device = device
    c.validate()
    if c.method in INSTANCE:
        m = Fields(c.problem(), c.method, c.width, c.depth).to(device=device, dtype=torch.float64)
        s = SplitState(c.problem(), c.beta)
        s.load(b["split_history"], m)
    else:
        m = Operators(c).to(device)
        s = None
    m.load_state_dict(b["model"])
    return m, c, s, b


def instance_loss(model, x, w, split_values=None, beta=None):
    problem = model.problem
    y = model.y(x)
    if model.method == "cnp":
        u = -laplace(y, x)
        return (0.5 * ((y - problem.sine(x)) ** 2 + problem.alpha * u**2) * w).sum()
    q_scaled = model.q(x)
    if split_values is None:
        u = torch.clamp(-q_scaled, problem.lower, problem.upper)
    else:
        z, lambda_admm = split_values
        u = (beta * z - lambda_admm - problem.alpha * q_scaled) / (problem.alpha + beta)
    state = -laplace(y, x) - u - problem.f(x)
    adj = -problem.alpha * laplace(q_scaled, x) - y + problem.yd(x)
    return ((state**2 + adj**2 / problem.alpha) * w).sum()


def validation(model, c, split, va):
    if c.method not in INSTANCE:
        return dict(selection_score=operator_loss_dataset(model, c, va))
    metrics = instance_metrics(model, c, split, seed=10505)
    if c.method == "cnp":
        xx, ww = quadrature(c.validation_order)
        x = torch.tensor(xx, dtype=torch.float64, device=c.device, requires_grad=True)
        w = torch.tensor(ww, dtype=torch.float64, device=c.device)
        metrics["selection_score"] = float(instance_loss(model, x, w).detach())
        metrics["validation_gauss_objective"] = metrics["selection_score"]
    return metrics


def train(c, out, data_root):
    c.validate()
    out = Path(out)
    tr = va = identity = None
    if c.method not in INSTANCE:
        tr, meta = load_data(data_root, "train")
        va, _ = load_data(data_root, "val")
        if meta["grid"] != c.grid or fingerprints(tr) & fingerprints(va):
            raise ValueError("grid/overlap")
        identity = {k: meta["splits"][k]["sha256"] for k in ("train", "val")}
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    torch.set_num_threads(2)
    started = time.time()
    updates = 0
    best = float("inf")
    hist = []
    split = None
    counts = dict(
        adam_updates=0, lbfgs_iterations=0, lbfgs_function_evaluations=0, epochs=0, outer_steps=0
    )
    write(out / "config.json", asdict(c))
    write(out / "source_identity.json", source_hashes())
    if c.method in INSTANCE:
        model = Fields(c.problem(), c.method, c.width, c.depth).to(
            device=c.device, dtype=torch.float64
        )
        split = SplitState(c.problem(), c.beta)
        xx, ww = quadrature(c.quadrature_order)
        tx = torch.tensor(xx, dtype=torch.float64, device=c.device)
        tw = torch.tensor(ww, dtype=torch.float64, device=c.device)
    else:
        model = Operators(c).to(c.device)
        model.set_scales(tr)
    initial = cpu_state(model.state_dict())
    torch.save(initial, out / "initial.pt")

    def record(stage, epoch=None):
        nonlocal best
        metrics = validation(model, c, split, va)
        score = metrics["selection_score"]
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite validation")
        item = dict(
            stage=stage,
            epoch=epoch,
            optimizer_updates=updates,
            seconds=time.time() - started,
            counters=dict(counts),
            **metrics,
        )
        hist.append(item)
        if score < best:
            best = score
            save(out / "best.pt", model, c, updates, score, split, identity, counts)
        save(out / "last.pt", model, c, updates, score, split, identity, counts)
        write(out / "history.json", hist)
        write(
            out / "status.json",
            dict(state="training", updates=updates, best_score=best, stage=stage, counters=counts),
        )
        print(c.method, c.seed, stage, epoch, updates, score, flush=True)

    def finite_grad():
        gs = [p.grad for p in model.parameters() if p.grad is not None]
        if not gs or any(not torch.isfinite(g).all() for g in gs):
            raise FloatingPointError("gradient")

    def inner(adam, lbfgs, values=None):
        nonlocal updates
        opt = torch.optim.Adam(model.parameters(), lr=c.learning_rate)

        def objective():
            return instance_loss(model, tx.detach().requires_grad_(True), tw, values, c.beta)

        for k in range(adam):
            opt.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("loss")
            loss.backward()
            finite_grad()
            opt.step()
            updates += 1
            counts["adam_updates"] += 1
            if c.method != "admm" and (k + 1) % c.evaluate_every == 0:
                record("adam")
        remaining = lbfgs
        while remaining:
            limit = min(c.lbfgs_chunk, remaining)
            opt = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=limit,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
            )

            def closure():
                opt.zero_grad(set_to_none=True)
                v = objective()
                if not torch.isfinite(v):
                    raise FloatingPointError("LBFGS loss")
                v.backward()
                finite_grad()
                return v

            opt.step(closure)
            s = opt.state[next(iter(model.parameters()))]
            n = int(s.get("n_iter", 0))
            updates += n
            counts["lbfgs_iterations"] += n
            counts["lbfgs_function_evaluations"] += int(s.get("func_evals", 0))
            remaining -= limit
            if c.method != "admm":
                record("lbfgs")
            if n < limit:
                break
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("parameters")

    try:
        record("initial", 0)
        if c.method == "admm":
            for outer in range(c.outer_steps):
                with torch.no_grad():
                    v = split.at(tx)
                    values = (v["z"].detach(), v["lam"].detach())
                inner(c.inner_steps, c.inner_lbfgs, values)
                split.append(model)
                counts["outer_steps"] = outer + 1
                record("admm", outer + 1)
        elif c.method in INSTANCE:
            inner(c.adam_steps, c.lbfgs_steps)
            record("final")
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.6)
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
                    loss = operator_loss(model, c, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("operator loss")
                    loss.backward()
                    finite_grad()
                    opt.step()
                    updates += 1
                    counts["adam_updates"] += 1
                sched.step()
                counts["epochs"] = epoch
                record("epoch", epoch)
        m, bc, bs, b = load(out / "best.pt", c.device)
        metrics = validation(m, bc, bs, va)
        if c.method not in INSTANCE:
            metrics.update(operator_metrics(m, bc, va))
        write(out / "validation.json", metrics)
        movement = sum(
            float((v.cpu() - initial[k]).abs().sum())
            for k, v in model.state_dict().items()
            if k in dict(model.named_parameters())
        )
        result = dict(
            state="complete",
            method=c.method,
            seed=c.seed,
            optimizer_updates=updates,
            best_updates=b["optimizer_updates"],
            best_score=best,
            elapsed_seconds=time.time() - started,
            counters=counts,
            parameter_movement=movement,
            parameters_real=sum(
                p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
            ),
            test_evaluations=0,
            formal_convergence_claim=False,
            exact_resume_supported=False,
        )
        write(out / "status.json", result)
        return result
    except Exception as e:
        write(out / "status.json", dict(state="failed", error=str(e), counters=counts))
        raise
