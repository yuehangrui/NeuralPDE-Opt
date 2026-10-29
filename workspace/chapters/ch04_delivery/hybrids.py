"""Declared complete budgets and initialization studies for Chapter 4 hybrids.

Primal/source networks retain their declared fixed sine features. Three actual
initializations are used; this is not a replication of general deep nonconvex
networks. ISMO retains the trainable nonlinear MLP and true-model feedback loop.

Book/code notation:
Book 4.4.1/4.5: shared delivery wrappers preserve each algorithm. In Uzawa, z is a multiplier and q=K*y-f; physical adjoint is p=-2*z at feasibility.
"""

import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
from chapters.ch04_completion.common import ALPHA, DIM, LAM, SpectralProblem, SineHead, tensor
from chapters.ch04_completion.bilevel import Bilevel
from chapters.ch04_completion.deep_uzawa import modified_loss
from chapters.ch04_completion.fem_nn import SourceIdentification
from chapters.ch04_completion import ismo, classical
from .frozen_jobs import write


def identity():
    paths = [Path(__file__)] + list(Path("chapters/ch04_completion").glob("*.py"))
    paths += [Path("chapters/ch04/problem.py"), Path("chapters/ch04/fem.py")]
    project = Path(__file__).resolve().parents[2]
    return {
        p.resolve().relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def bilevel_run(seed):
    torch.manual_seed(seed)
    solver = Bilevel()
    with torch.no_grad():
        solver.state_net.weight.normal_(0, 0.01)
    initial = solver.state_net.weight.detach().numpy().copy()
    history = []
    best = None

    def record(u):
        nonlocal best
        solver.fun(u)
        diag = solver.problem.diagnostics(u, solver.last["q"] / LAM)
        history.append(
            dict(
                **diag,
                inner=solver.last["inner"],
                hessian_solve_residual=solver.last["hessian_solve_residual"],
            )
        )
        if best is None or diag["validation_score"] < best["score"]:
            best = dict(
                score=diag["validation_score"],
                u=u.copy(),
                q=solver.last["q"].copy(),
                index=len(history) - 1,
            )

    record(np.zeros(DIM))
    opt = minimize(
        solver.fun,
        np.zeros(DIM),
        jac=True,
        method="BFGS",
        callback=record,
        options=dict(gtol=1e-12, maxiter=300),
    )
    record(opt.x)
    return dict(
        control=best["u"],
        state_head=best["q"],
        initial_state_head=initial,
        best_index=best["index"],
        history=history,
        diagnostics=history[best["index"]],
        iterations=int(opt.nit),
        optimizer_success=bool(opt.success),
        message=str(opt.message),
        counts=dict(outer_evaluations=solver.calls, inner_closures=solver.inner_calls),
    )


def uzawa_run(seed):
    torch.manual_seed(seed)
    problem = SpectralProblem()
    state, control = SineHead(), SineHead()
    with torch.no_grad():
        state.weight.normal_(0, 0.01)
        control.weight.normal_(0, 0.01)
    initial = np.stack(
        [state.weight.detach().numpy().copy(), control.weight.detach().numpy().copy()]
    )
    z = torch.zeros(DIM, dtype=torch.float64)
    rho = 0.2 * ALPHA
    diagonal = tensor(np.diag(problem.gram))
    hq = diagonal * (1 / tensor(LAM) ** 2 + ALPHA / 2) / ALPHA
    hu = diagonal / 2
    history = []
    best = None
    calls = 0
    for k in range(200):
        for inner in range(20):
            loss = modified_loss(state.weight, control.weight, z, problem) / ALPHA
            gq, gu = torch.autograd.grad(loss, (state.weight, control.weight))
            calls += 1
            grad = max(float(gq.abs().max()), float(gu.abs().max()))
            if grad < 1e-10:
                break
            with torch.no_grad():
                state.weight.sub_(gq / hq)
                control.weight.sub_(gu / hu)
        if grad > 1e-7:
            raise RuntimeError("unconverged inner solve")
        q, u = state.weight.detach().numpy().copy(), control.weight.detach().numpy().copy()
        diag = problem.diagnostics(u, q / LAM)
        history.append(dict(iteration=k, inner_gradient_max=grad, **diag))
        if best is None or diag["validation_score"] < best["score"]:
            best = dict(score=diag["validation_score"], u=u, q=q, z=z.numpy().copy(), index=k)
        if diag["validation_score"] < 1e-10:
            break
        z = z + rho * tensor(q - u - problem.f)
    return dict(
        control=best["u"],
        state_head=best["q"],
        multiplier=best["z"],
        initial_heads=initial,
        best_index=best["index"],
        history=history,
        diagnostics=history[best["index"]],
        iterations=len(history),
        counts=dict(inner_closures=calls),
        rho=rho,
    )


def fem_nn_run(seed, mesh=64, validation_mesh=128):
    torch.manual_seed(seed)
    training, validation = SourceIdentification(mesh), SourceIdentification(validation_mesh)
    model = SineHead()
    with torch.no_grad():
        model.weight.normal_(0, 0.01)
    initial = model.weight.detach().numpy().copy()
    history = []
    best = None
    calls = 0
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        tolerance_grad=1e-12,
        tolerance_change=1e-24,
        line_search_fn="strong_wolfe",
    )

    def record():
        nonlocal best
        tr = training.relative_misfit(model.weight)
        va = validation.relative_misfit(model.weight)
        history.append(
            dict(
                iteration=len(history),
                train_observation_relative_l2=tr,
                validation_observation_relative_l2=va,
            )
        )
        if best is None or va < best["score"]:
            best = dict(score=va, theta=model.weight.detach().clone(), index=len(history) - 1)

    record()
    for _ in range(200):

        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True)
            loss = training.loss(model.weight)
            loss.backward()
            calls += 1
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite inverse objective")
            return loss

        opt.step(closure)
        record()
        (gradient,) = torch.autograd.grad(training.loss(model.weight), model.weight)
        if float(gradient.abs().max()) < 1e-11:
            break
    theta = best["theta"].numpy()
    u, info = validation.control(theta)
    problem = SpectralProblem()
    delta = LAM * theta - problem.f
    # Same-mesh true-source comparison separates identification and FEM error.
    true_u, true_info = validation.control(problem.f / LAM)
    return dict(
        control=u,
        source_head=theta,
        initial_source_head=initial,
        best_index=best["index"],
        history=history,
        diagnostics=problem.diagnostics(u),
        iterations=len(history) - 1,
        optimizer=info,
        selected_validation_observation_relative_l2=best["score"],
        source_relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(problem.f)),
        true_source_same_mesh_control=true_u,
        true_source_same_mesh_diagnostics=problem.diagnostics(true_u),
        counts=dict(
            training=training.counts, validation_and_control=validation.counts, closures=calls
        ),
        data=dict(
            training_nodes=training.ref.nodes[training.ref.interior],
            training_observations=training.observed.numpy(),
            validation_nodes=validation.ref.nodes[validation.ref.interior],
            validation_observations=validation.observed.numpy(),
        ),
    )


def convert(value):
    if isinstance(value, np.ndarray):
        return tensor(value)
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert(v) for v in value]
    return value


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["schema"] != "ch04-delivery-hybrid-v1" or ck["source_identity"] != identity():
        raise ValueError("hybrid identity mismatch")
    return ck


def train(method, seed, output):
    if method not in ("bilevel", "deep_uzawa", "ismo", "fem_nn") or seed not in (0, 1, 2):
        raise ValueError("config")
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(
        method=method,
        seed=seed,
        dtype="float64",
        device="cpu",
        source_identity=identity(),
        test_evaluations=0,
        bilevel_outer_max=300,
        uzawa_outer_max=200,
        uzawa_rho=0.2 * ALPHA,
        ismo_rounds=12,
        ismo_epochs_per_round=1600,
        fem_training_mesh=64,
        fem_validation_mesh=128,
        representation="fixed sine-feature primal/source heads; trainable nonlinear ISMO surrogate",
        stopping="true PDE gradient/feasibility or explicit budget; status reported separately",
    )
    write(out / "config.json", config)
    start = time.perf_counter()
    if method == "bilevel":
        result = bilevel_run(seed)
    elif method == "deep_uzawa":
        result = uzawa_run(seed)
    elif method == "ismo":
        result = ismo.run(seed=seed, rounds=12, epochs=1600)
    else:
        result = fem_nn_run(seed)
    data = result.pop("data", None)
    if data is not None:
        np.savez_compressed(out / "data.npz", **data)
    keys = [
        "control",
        "state_head",
        "multiplier",
        "source_head",
        "initial_state_head",
        "initial_heads",
        "initial_source_head",
        "model",
        "round_models",
        "true_source_same_mesh_control",
    ]
    state = {k: convert(result.pop(k)) for k in keys if k in result}
    ck = dict(
        schema="ch04-delivery-hybrid-v1",
        method=method,
        config=config,
        source_identity=identity(),
        state=state,
    )
    torch.save(ck, out / "best.pt")
    check = load(out / "best.pt")
    problem = SpectralProblem()
    y = check["state"]["state_head"].numpy() / LAM if "state_head" in check["state"] else None
    diag = problem.diagnostics(check["state"]["control"].numpy(), y)
    for key in ("control_relative_l2", "true_objective", "validation_score"):
        np.testing.assert_allclose(diag[key], result["diagnostics"][key], rtol=1e-12, atol=1e-14)
    result.update(
        status="completed",
        optimization_converged=diag["validation_score"] < 1e-6,
        test_evaluations=0,
        seconds=time.perf_counter() - start,
        source_identity=identity(),
    )
    write(out / "metrics.json", result)
    write(
        out / "run.json",
        dict(
            status="completed",
            seconds=result["seconds"],
            optimization_converged=result["optimization_converged"],
            test_evaluations=0,
            source_identity=identity(),
        ),
    )
    print(method, seed, result["diagnostics"], flush=True)


def classical_study(output):
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    reference = classical.spectral()
    reference["control"] = reference["control"].tolist()
    rows = []
    for mesh in (32, 64, 128, 256):
        result = classical.fem(mesh, maxiter=500)
        np.savez_compressed(out / f"control_n{mesh}.npz", control=result.pop("control"))
        rows.append(result)
        write(out / f"mesh_{mesh}.json", result)
        print("classical", mesh, result["control_error_to_discrete_kkt"], flush=True)
    write(
        out / "metrics.json",
        dict(spectral=reference, meshes=rows, seconds=time.perf_counter() - start),
    )
    write(
        out / "run.json",
        dict(status="completed", seconds=time.perf_counter() - start, source_identity=identity()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method", choices=["bilevel", "deep_uzawa", "ismo", "fem_nn", "classical"], required=True
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.method == "classical":
        classical_study(a.output)
    else:
        train(a.method, a.seed, a.output)
