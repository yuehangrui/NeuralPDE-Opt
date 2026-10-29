import numpy as np
import torch
from .reference import Grid
from .models import laplace


def points(seed, count):
    return np.random.default_rng(seed).uniform(0, 1, (count, 2))


def scalar(v):
    return float(np.asarray(v))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-14))


def array(t):
    return t.detach().cpu().numpy()


def active_sets(u, reference, p):
    result = {}
    for tol in (1e-4, 1e-3, 1e-2):
        classify = lambda v: np.where(v <= p.lower + tol, -1, np.where(v >= p.upper - tol, 1, 0))
        labels = classify(u)
        d = dict(
            lower_fraction=float(np.mean(labels == -1)), upper_fraction=float(np.mean(labels == 1))
        )
        if reference is not None:
            d["mismatch_fraction"] = float(np.mean(labels != classify(reference)))
        result[str(tol)] = d
    return result


def instance_metrics(model, config, split=None, seed=None):
    model.eval()
    problem = model.problem
    grid = Grid(config.grid)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    def tensor(a):
        return torch.tensor(a, dtype=dtype, device=device, requires_grad=True)

    x = tensor(points(config.seed + 10001 if seed is None else seed, config.val_points))
    y = model.y(x)
    if config.method == "cnp":
        u = -laplace(y, x)
        yd = problem.sine(x)
        score = float(
            (0.5 * (y - yd).square().mean() + 0.5 * problem.alpha * u.square().mean()).detach()
        )
        report = dict(
            selection_score=score,
            validation_objective=score,
            state_violation=float(torch.relu(y - problem.state_cap).max().detach()),
        )
    else:
        q_scaled = model.q(x)
        fields = split.at(x) if config.method == "admm" else None
        u = fields["u"] if fields is not None else model.u(x)
        rs = -laplace(y, x) - u - problem.f(x)
        ra = -problem.alpha * laplace(q_scaled, x) - y + problem.yd(x)
        report = dict(
            state_rms=float(rs.square().mean().sqrt().detach()),
            adjoint_rms=float(ra.square().mean().sqrt().detach()),
        )
        if fields is not None:
            report.update(
                admm_primal_rms=float(fields["primal"].square().mean().sqrt().detach()),
                admm_dual_rms=float(fields["dual"].square().mean().sqrt().detach()),
                raw_control_violation=float(
                    torch.maximum(torch.relu(problem.lower - u), torch.relu(u - problem.upper))
                    .max()
                    .detach()
                ),
            )
    gx = tensor(grid.points.reshape(-1, 2))
    gy = model.y(gx)
    yn = array(gy).reshape(config.grid - 1, config.grid - 1)
    if config.method == "cnp":
        un = array(-laplace(gy, gx)).reshape(yn.shape)
        f = np.zeros_like(un)
        yd = problem.sine(grid.points)[..., 0]
        recovered = grid.solve(un)
        gstate = yn - yd + problem.alpha * grid.apply(grid.apply(yn))
        report.update(
            grid_objective=scalar(
                0.5 * grid.norm(yn - yd) ** 2 + 0.5 * problem.alpha * grid.norm(un) ** 2
            ),
            recovered_state_violation=float(np.maximum(recovered - problem.state_cap, 0).max()),
            state_recovery_gap_l2=scalar(grid.norm(yn - recovered)),
            discrete_state_projected_residual=scalar(
                grid.norm(yn - np.minimum(yn - gstate, problem.state_cap))
            ),
            continuous_optimality_certified=False,
        )
    else:
        f = problem.f(grid.points)[..., 0]
        yd = problem.yd(grid.points)[..., 0]
        if config.method == "admm":
            fields = split.at(gx)
            un = array(fields["z"]).reshape(yn.shape)
            report["reported_control"] = "ADMM z (feasible); PDE residual above uses actual inner u"
        else:
            un = array(model.u(gx)).reshape(yn.shape)
        d = grid.diagnostics(un, f, yd, problem)
        report.update({"recovered_" + k: scalar(v) for k, v in d.items()})
        report.update(
            relative_y=rel(yn, problem.y(grid.points)[..., 0]),
            relative_u=rel(un, problem.u(grid.points)[..., 0]),
            relative_p=rel(
                array(problem.alpha * model.q(gx)).reshape(yn.shape), problem.p(grid.points)[..., 0]
            ),
            active_sets=active_sets(un, problem.u(grid.points)[..., 0], problem),
        )
        scale_f = max(float(np.sqrt(np.mean(f * f))), 1e-8)
        scale_yd = max(float(np.sqrt(np.mean(yd * yd))), 1e-8)
        report["selection_score"] = float(
            np.sqrt(
                (report["state_rms"] / scale_f) ** 2
                + (report["adjoint_rms"] / scale_yd) ** 2
                + (
                    report["recovered_projected_residual"]
                    / max(problem.upper - problem.lower, 1e-8)
                )
                ** 2
                + (report.get("admm_primal_rms", 0) / max(problem.upper - problem.lower, 1e-8)) ** 2
            )
        )
    # Boundary evaluation includes corners; independent of training points.
    t = np.linspace(0, 1, 31)
    b = np.concatenate(
        [
            np.stack((t, np.zeros_like(t)), 1),
            np.stack((t, np.ones_like(t)), 1),
            np.stack((np.zeros_like(t), t), 1),
            np.stack((np.ones_like(t), t), 1),
        ]
    )
    report["boundary_y_max"] = float(np.max(np.abs(array(model.y(tensor(b))))))
    report["grid_intervals"] = config.grid
    model.train()
    return report


def operator_metrics(model, config, data):
    model.eval()
    device = next(model.parameters()).device
    grid = Grid(config.grid)
    problem = config.problem()
    controls = []
    ys = []
    ps = []
    pd_info = []
    with torch.no_grad():
        for start in range(0, len(data["f"]), config.batch):
            f = torch.tensor(
                data["f"][start : start + config.batch, ..., None],
                dtype=torch.float32,
                device=device,
            )
            yd = torch.tensor(
                data["yd"][start : start + config.batch, ..., None],
                dtype=torch.float32,
                device=device,
            )
            out = model(f, yd)
            if config.method == "mimo":
                u, y, adj = out
                ys.append(array(y)[..., 0])
                ps.append(array(adj)[..., 0])
            elif config.method == "pdol":
                u, info = out
                pd_info.append(info)
            else:
                u = out
            controls.append(array(u)[..., 0])
    u = np.concatenate(controls)
    if not np.isfinite(u).all():
        raise FloatingPointError("nonfinite operator prediction")
    d = grid.diagnostics(u, data["f"], data["yd"], problem)
    report = dict(
        relative_u=rel(u, data["u"]),
        active_sets=active_sets(u, data["u"], problem),
        **{k + "_mean": float(np.mean(v)) for k, v in d.items()},
        **{k + "_max": float(np.max(v)) for k, v in d.items()},
    )
    if ys:
        y = np.concatenate(ys)
        adj = np.concatenate(ps)
        report.update(
            relative_y=rel(y, data["y"]),
            relative_p=rel(adj, data["p"]),
            predicted_state_residual_mean=float(np.mean(grid.norm(grid.apply(y) - u - data["f"]))),
            predicted_adjoint_residual_mean=float(
                np.mean(grid.norm(grid.apply(adj) - y + data["yd"]))
            ),
        )
    if pd_info:
        report["pd_iteration_diagnostics"] = pd_info
    # Pairing and zero tests concern the actual learned actions, not the true PDE.
    if config.method in ("pdol", "iuzawa_f", "iuzawa_s"):
        v = torch.tensor(data["v"][: config.batch, ..., None], dtype=torch.float32, device=device)
        w = torch.tensor(data["w"][: config.batch, ..., None], dtype=torch.float32, device=device)
        if config.method == "pdol":
            pairs = [(lambda x: model.action(x, "state"), lambda x: model.action(x, "adjoint"))]
        else:
            pairs = [
                (layer.state_action, layer.S_adj_oper) for layer in model.unrolled._iter_layers()
            ]
        defects = []
        with torch.no_grad():
            for s, a in pairs:
                sv, aw = s(v), a(w)
                lhs = (sv * w).sum()
                rhs = (v * aw).sum()
                denom = sv.norm() * w.norm() + v.norm() * aw.norm() + 1e-20
                defects.append(float((lhs - rhs).abs() / denom))
        report["learned_adjoint_pairing_defect"] = defects
        report["learned_adjoint_consistency_guaranteed"] = False
    model.train()
    return report
