"""Train/validation only. No analytic optimal-control or held-out evaluation.

Book/code notation:
Book 4.5.1/4.5.3: Uzawa learns y_theta, solves u=2*z/alpha analytically, and updates z with A_h*y-u-f. FEM-NN learns f_theta through the PDE and sparse observations, not optimal-control labels.
"""

from pathlib import Path
import copy
import time
import math
import numpy as np
import torch
from .common import (
    ALPHA,
    CoordinateNet,
    FDGrid,
    Mesh,
    SparseSolve,
    tensor,
    sp_tensor,
    smv,
    transfer_control,
    identity,
    write_json,
    digest,
)


def observations(train_count=256, val_count=128, noise=0.01, seed=20261026):
    if noise < 0 or not np.isfinite(noise) or min(train_count, val_count) < 8:
        raise ValueError("invalid observation protocol")
    rng = np.random.default_rng(seed)
    x = 0.025 + 0.95 * rng.random((train_count + val_count, 2))
    # Analytic *uncontrolled* observed state, not the optimal state/control.
    # Derived from the public f by solving -Delta y0=f, independently of FEM.
    y0 = np.sin(math.pi * x[:, 0]) * np.sin(math.pi * x[:, 1]) + np.sin(
        8 * math.pi * x[:, 0]
    ) * np.sin(8 * math.pi * x[:, 1]) / (ALPHA * (128 * math.pi**2) ** 2)
    scale = float(np.sqrt(np.mean(y0[:train_count] ** 2)))
    observed = y0 + noise * scale * rng.standard_normal(len(x))
    # Normalization fitted only from actual noisy training measurements.
    observation_scale = float(np.sqrt(np.mean(observed[:train_count] ** 2)))
    return dict(
        train_x=x[:train_count],
        val_x=x[train_count:],
        train_observed=observed[:train_count],
        val_observed=observed[train_count:],
        observation_scale=np.array(observation_scale),
        noise_sigma=np.array(noise * scale),
    )


class InverseProblem:
    def __init__(self, n, points, observed, scale, device="cpu"):
        self.mesh = Mesh(n)
        self.x = tensor(self.mesh.nodes, device)
        self.L = sp_tensor(self.mesh.L, device)
        self.M = sp_tensor(self.mesh.M, device)
        self.K = sp_tensor(self.mesh.K, device)
        self.H = sp_tensor(self.mesh.observe(points), device)
        self.observed = tensor(observed, device)
        self.scale = scale
        self.source_scale = 2 * math.pi**2 * scale

    def predict(self, net):
        f = self.source_scale * net(self.x)
        y = SparseSolve.apply(smv(self.L, f), self.mesh)
        return smv(self.H, y), f

    def loss(self, net, regularization):
        prediction, f = self.predict(net)
        misfit = 0.5 * ((prediction - self.observed) / self.scale).square().mean()
        prior = (
            0.5
            * (f @ smv(self.M, f) + (f @ smv(self.K, f)) / (2 * math.pi**2))
            / self.source_scale**2
        )
        return misfit + regularization * prior

    def score(self, net):
        with torch.no_grad():
            prediction, _ = self.predict(net)
            return float(torch.sqrt(((prediction - self.observed) / self.scale).square().mean()))


def checkpoint(net, config, best_index, score, initial, **extra):
    hidden_change = sum(
        float((value - initial[key]).square().sum())
        for key, value in net.state_dict().items()
        if key.startswith("net.") and not key.startswith(f"net.{2*net.depth}.")
    )
    return dict(
        model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        network=dict(width=net.width, depth=net.depth, boundary=net.boundary),
        config=config,
        selected_index=best_index,
        validation_score=score,
        hidden_parameter_change_l2=math.sqrt(hidden_change),
        source_identity=identity(),
        **extra,
    )


def train_fem(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    net = CoordinateNet(config["width"], config["depth"]).to(config["device"])
    initial = copy.deepcopy(net.state_dict())
    data = observations(config["train_observations"], config["val_observations"], config["noise"])
    np.savez_compressed(out / "observations.npz", **data)
    scale = float(data["observation_scale"])
    train = InverseProblem(
        config["mesh"], data["train_x"], data["train_observed"], scale, config["device"]
    )
    val = InverseProblem(
        config["validation_mesh"], data["val_x"], data["val_observed"], scale, config["device"]
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["adam_steps"], eta_min=config["lr"] * 0.05
    )
    history, best, calls = [], None, 0
    start = time.perf_counter()

    def record(step, phase):
        nonlocal best
        t, v = train.score(net), val.score(net)
        with torch.enable_grad():
            loss = train.loss(net, config["regularization"])
            gradient = torch.autograd.grad(loss, tuple(net.parameters()))
        gradnorm = float(torch.sqrt(sum(g.square().sum() for g in gradient)))
        history.append(
            dict(
                step=step,
                phase=phase,
                train_observation_relative=t,
                validation_score=v,
                training_gradient_l2=gradnorm,
                closure_calls=calls,
                elapsed=time.perf_counter() - start,
            )
        )
        if best is None or v < best["validation_score"]:
            best = checkpoint(
                net,
                config,
                len(history) - 1,
                v,
                initial,
                source_scale=train.source_scale,
                observation_sha256=digest(out / "observations.npz"),
            )
            torch.save(best, out / "best.pt")
        write_json(out / "history.json", history)

    record(0, "initial")
    for step in range(1, config["adam_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = train.loss(net, config["regularization"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite FEM-NN loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        calls += 1
        if step % config["record_every"] == 0 or step == config["adam_steps"]:
            record(step, "adam")
    optimizer = torch.optim.LBFGS(
        net.parameters(),
        max_iter=1,
        history_size=40,
        tolerance_grad=1e-12,
        tolerance_change=1e-20,
        line_search_fn="strong_wolfe",
    )
    for step in range(1, config["lbfgs_steps"] + 1):

        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            loss = train.loss(net, config["regularization"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite FEM-NN L-BFGS loss")
            loss.backward()
            calls += 1
            return loss

        optimizer.step(closure)
        if step % config["record_every"] == 0 or step == config["lbfgs_steps"]:
            record(config["adam_steps"] + step, "lbfgs")
    torch.save(
        checkpoint(
            net,
            config,
            len(history) - 1,
            history[-1]["validation_score"],
            initial,
            source_scale=train.source_scale,
            optimizer=optimizer.state_dict(),
            observation_sha256=digest(out / "observations.npz"),
        ),
        out / "last.pt",
    )
    write_json(
        out / "run.json",
        dict(
            config=config,
            seconds=time.perf_counter() - start,
            counts=dict(train=train.mesh.counts, val=val.mesh.counts, optimizer_closures=calls),
            selected_index=best["selected_index"],
            validation_score=best["validation_score"],
            parameter_count=sum(p.numel() for p in net.parameters()),
            test_evaluations=0,
        ),
    )
    (out / "TRAINING_COMPLETE").write_text("budget completed; convergence assessed separately\n")
    return best["validation_score"]


def train_uzawa(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    net = CoordinateNet(config["width"], config["depth"], boundary=True).to(config["device"])
    initial = copy.deepcopy(net.state_dict())
    train, val = FDGrid(config["mesh"]), FDGrid(config["validation_mesh"])
    x, xv = tensor(train.x, config["device"]), tensor(val.x, config["device"])
    f, yd = tensor(train.f, config["device"]), tensor(train.yd, config["device"])
    z = torch.zeros_like(f)
    if not 0 < config["rho"] < ALPHA / 2:
        raise ValueError("Uzawa requires 0 < rho < alpha/2 for the exact convex reference")
    history, best, calls = [], None, 0
    start = time.perf_counter()

    def loss_fn():
        y = net(x)
        source = train.apply(y) - f
        # Exact quadratic control block: u = 2 z / alpha. Its constant
        # contribution is omitted during minimization over state parameters.
        return (
            (y - yd).square() / (2 * ALPHA) + source.square() / 4 + (z / ALPHA) * source
        ).sum() / train.n**2

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    for _ in range(config["warmup_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Uzawa warmup")
        loss.backward()
        optimizer.step()
        calls += 1
    for outer in range(config["outer_steps"]):
        optimizer = torch.optim.LBFGS(
            net.parameters(),
            max_iter=config["inner_steps"],
            history_size=40,
            tolerance_grad=1e-10,
            tolerance_change=1e-20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Uzawa inner loss")
            loss.backward()
            calls += 1
            return loss

        optimizer.step(closure)
        gradients = torch.autograd.grad(loss_fn(), tuple(net.parameters()))
        with torch.no_grad():
            y, u = net(x), 2 * z / ALPHA
            yn, un = y.cpu().numpy(), u.cpu().numpy()
            yu = net(xv).cpu().numpy()
        uv = transfer_control(un, train.n, val.x)
        v = val.diagnostics(yu, uv)
        t = train.diagnostics(yn, un)
        history.append(
            dict(
                iteration=outer,
                **v,
                train=t,
                inner_parameter_gradient_l2=float(
                    torch.sqrt(sum(g.square().sum() for g in gradients))
                ),
                closure_calls=calls,
                elapsed=time.perf_counter() - start,
            )
        )
        if best is None or v["validation_score"] < best["validation_score"]:
            best = checkpoint(
                net,
                config,
                outer,
                v["validation_score"],
                initial,
                control=un.copy(),
                multiplier=z.detach().cpu().numpy().copy(),
            )
            torch.save(best, out / "best.pt")
        write_json(out / "history.json", history)
        # Function-space L2 ascent, without multiplying by h^2 again.
        with torch.no_grad():
            z = z + config["rho"] * (train.apply(y) - u - f)
    torch.save(
        checkpoint(
            net,
            config,
            len(history) - 1,
            history[-1]["validation_score"],
            initial,
            control=un.copy(),
            multiplier=(ALPHA / 2 * un),
            next_multiplier=z.detach().cpu().numpy(),
            optimizer=optimizer.state_dict(),
        ),
        out / "last.pt",
    )
    write_json(
        out / "run.json",
        dict(
            config=config,
            seconds=time.perf_counter() - start,
            optimizer_closures=calls,
            selected_index=best["selected_index"],
            validation_score=best["validation_score"],
            parameter_count=sum(p.numel() for p in net.parameters()),
            inner_exact=False,
            control_block="exact quadratic minimizer",
            test_evaluations=0,
        ),
    )
    (out / "TRAINING_COMPLETE").write_text("budget completed; no exact-inner convergence claim\n")
    return best["validation_score"]


def defaults(method, seed=0, pilot=False):
    if method == "uzawa":
        return dict(
            method=method,
            seed=seed,
            width=32,
            depth=2,
            device="cpu",
            mesh=32 if pilot else 64,
            validation_mesh=64 if pilot else 96,
            rho=0.2 * ALPHA,
            lr=0.003,
            warmup_steps=100 if pilot else 500,
            outer_steps=8 if pilot else 80,
            inner_steps=25 if pilot else 160,
        )
    if method not in ("fem_clean", "fem_noisy", "fem_regularized"):
        raise ValueError("unknown method")
    return dict(
        method=method,
        seed=seed,
        width=32,
        depth=2,
        device="cpu",
        mesh=32 if pilot else 64,
        validation_mesh=64 if pilot else 96,
        lr=0.003,
        train_observations=256,
        val_observations=128,
        noise=0.0 if method == "fem_clean" else 0.01,
        regularization=1e-4 if method == "fem_regularized" else 0.0,
        adam_steps=200 if pilot else 2000,
        lbfgs_steps=40 if pilot else 400,
        record_every=20,
    )


def validate_config(config):
    expected = defaults(config["method"])
    if set(config) != set(expected):
        raise ValueError("unknown or missing configuration fields")
    integer_keys = ["seed", "width", "depth", "mesh", "validation_mesh"]
    integer_keys += (
        ["warmup_steps", "outer_steps", "inner_steps"]
        if config["method"] == "uzawa"
        else ["adam_steps", "lbfgs_steps", "record_every", "train_observations", "val_observations"]
    )
    for key in integer_keys:
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid integer budget: " + key)
    if config["width"] < 2 or config["mesh"] < 16 or config["validation_mesh"] <= config["mesh"]:
        raise ValueError("require resolved training grid and finer independent validation grid")
    for key in ["lr"] + (["rho"] if config["method"] == "uzawa" else ["regularization", "noise"]):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("invalid scalar: " + key)
    if config["lr"] == 0 or (config["method"] == "uzawa" and not 0 < config["rho"] < ALPHA / 2):
        raise ValueError("invalid learning rate or Uzawa step")
