"""Section 4.5.2: train -> optimize -> true PDE -> enrich -> retrain.

The oracle is exact spectral Poisson on the declared 17-mode subspace. This
cheap instructional oracle cannot establish a speedup over a numerical solver.
Validation is fixed independently of adaptive training data; no test is opened.

Book/code notation:
Book 4.5.2: a nonlinear surrogate supports optimization and true-PDE enrichment. Surrogate gradients and true reduced gradients are recorded separately.
"""

import copy
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from .common import ALPHA, DIM, SpectralProblem, tensor


class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)
        self.residual = nn.Sequential(nn.Linear(DIM, 32), nn.Tanh(), nn.Linear(32, DIM))
        for key in ("input_mean", "output_mean"):
            self.register_buffer(key, torch.zeros(DIM))
        for key in ("input_scale", "output_scale"):
            self.register_buffer(key, torch.ones(DIM))
        self.double()

    def forward(self, u):
        z = (u - self.input_mean) / self.input_scale
        return self.output_mean + self.output_scale * (self.linear(z) + self.residual(z))


def train(model, controls, states, validation, epochs):
    x, y = tensor(controls), tensor(states)
    vx, vy = map(tensor, validation)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, score, best_epoch = None, float("inf"), -1
    validation_history = []
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - y) / model.output_scale).square().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = float(((model(vx) - vy) / model.output_scale).square().mean())
        if not np.isfinite(val):
            raise FloatingPointError("nonfinite surrogate loss")
        validation_history.append(val)
        if val < score:
            best, score, best_epoch = copy.deepcopy(model.state_dict()), val, epoch
    model.load_state_dict(best)
    return dict(
        validation_normalized_mse=score,
        selected_epoch=best_epoch,
        updates=epochs,
        validation_history=validation_history,
    )


def optimize(model, start, problem):
    def fun(u_np):
        u = tensor(u_np).requires_grad_(True)
        y = model(u)
        G = tensor(problem.gram)
        e = y - tensor(problem.yd)
        j = 0.5 * e @ G @ e + 0.5 * ALPHA * u @ G @ u
        (g,) = torch.autograd.grad(j, u)
        return float(j.detach()), g.detach().numpy()

    opt = minimize(fun, start, jac=True, method="BFGS", options=dict(gtol=1e-10, maxiter=300))
    _, gradient = fun(opt.x)
    return opt.x, dict(
        success=bool(opt.success),
        message=str(opt.message),
        iterations=int(opt.nit),
        surrogate_derivative_norm=float(np.linalg.norm(gradient)),
    )


def run(seed=0, rounds=4, epochs=600, initial_samples=64, enrichment=16):
    if min(rounds, epochs, enrichment) < 1 or initial_samples < DIM + 1:
        raise ValueError("invalid ISMO budget")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7101)
    vrng = np.random.default_rng(seed + 9101)
    problem = SpectralProblem()
    controls = rng.normal(size=(initial_samples, DIM))
    states = problem.state(controls)
    val_u = vrng.normal(size=(32, DIM))
    val_y = problem.state(val_u)
    initial_u, initial_y = controls.copy(), states.copy()
    model = Surrogate()
    with torch.no_grad():
        model.input_mean.copy_(tensor(controls.mean(0)))
        model.input_scale.copy_(tensor(controls.std(0)))
        model.output_mean.copy_(tensor(states.mean(0)))
        model.output_scale.copy_(tensor(states.std(0)))
    best_u = np.zeros(DIM)
    best_j = problem.fun(best_u)[0]
    history, round_models, candidates, validation_snapshots = [], [], [], []
    best_round = -1
    for k in range(rounds):
        fit = train(model, controls, states, (val_u, val_y), epochs)
        validation_snapshots.append(model(tensor(val_u)).detach().numpy())
        starts = [best_u.copy(), rng.normal(scale=0.25, size=DIM)]
        round_candidates, optimizers = [], []
        for start in starts:
            candidate, info = optimize(model, start, problem)
            # Mandatory high-fidelity evaluation at EACH candidate, before acceptance.
            true_y = problem.state(candidate)
            true_j = problem.objective(true_y, candidate)
            diag = problem.diagnostics(candidate)
            info.update(
                true_objective=true_j, true_gradient_relative=diag["true_gradient_relative"]
            )
            round_candidates.append(candidate)
            optimizers.append(info)
            if true_j < best_j:
                best_u, best_j, best_round = candidate.copy(), true_j, k
        candidates.append(np.stack(round_candidates))
        round_models.append(copy.deepcopy(model.state_dict()))
        # Add candidates themselves and local probes. Freeze original normalization.
        local = best_u + rng.normal(scale=0.25 / (k + 1), size=(enrichment, DIM))
        added = np.concatenate([np.stack(round_candidates), local])
        added_y = problem.state(added)
        before = len(controls)
        controls, states = np.concatenate([controls, added]), np.concatenate([states, added_y])
        best_diag = problem.diagnostics(best_u)
        history.append(
            dict(
                round=k,
                training_samples=before,
                added_samples=len(added),
                **fit,
                candidates=optimizers,
                best_true_objective=best_j,
                best_true_gradient_relative=best_diag["true_gradient_relative"],
            )
        )
        if best_diag["true_gradient_relative"] < 1e-6:
            break
    return dict(
        control=best_u,
        history=history,
        diagnostics=problem.diagnostics(best_u),
        best_round=best_round,
        iterations=len(history),
        counts=problem.counts,
        model=model.state_dict(),
        round_models=round_models,
        data=dict(
            initial_controls=initial_u,
            initial_states=initial_y,
            train_controls=controls,
            train_states=states,
            validation_controls=val_u,
            validation_states=val_y,
            validation_predictions=np.stack(validation_snapshots),
            candidates=np.stack(candidates),
        ),
        last_added_data_used_for_training=False,
    )
