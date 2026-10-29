"""Fixed-model independent spatial, distribution and learned-action assessment."""

from pathlib import Path
import json, time
import numpy as np
import torch
from chapters.ch05.problem import Problem
from chapters.ch05.reference import Grid
from chapters.ch05.models import laplace
from .common import write, quadrature, sha
from .training import load
from .protocol import verify_freeze, load_holdout
from .data import inputs
from .metrics import control, aggregate, prolong
from .reference import state_qp


def array(t):
    return t.detach().cpu().numpy()


def instance_fields(model, c, split, points, derivatives=True):
    storage = {}
    for start in range(0, len(points), 512):
        x = torch.tensor(
            points[start : start + 512], device=c.device, dtype=torch.float64, requires_grad=True
        )
        y = model.y(x)
        out = dict(y=y)
        if c.method == "cnp":
            out["u"] = -laplace(y, x)
        else:
            q_scaled = model.q(x)
            out["p"] = c.alpha * q_scaled
            if c.method == "admm":
                with torch.no_grad():
                    fields = split.at(x)
                out.update(
                    u=fields["z"],
                    raw_u=fields["u"],
                    lambda_split=fields["lam"],
                    primal=fields["primal"],
                    dual=fields["dual"],
                )
            else:
                out["u"] = model.u(x)
            if derivatives:
                out["state_residual"] = (
                    -laplace(y, x) - out.get("raw_u", out["u"]) - c.problem().f(x)
                )
                out["adjoint_residual"] = -c.alpha * laplace(q_scaled, x) - y + c.problem().yd(x)
        for k, v in out.items():
            storage.setdefault(k, []).append(array(v)[:, 0])
    return {k: np.concatenate(v) for k, v in storage.items()}


def predict(model, c, f, yd):
    outputs = {}
    information = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(f), 16):
            ft = torch.tensor(
                f[start : start + 16, ..., None], device=c.device, dtype=torch.float32
            )
            yt = torch.tensor(
                yd[start : start + 16, ..., None], device=c.device, dtype=torch.float32
            )
            o = model(ft, yt)
            if c.method == "mimo":
                u, y, p = o
                values = dict(u=u, y=y, p=p)
            elif c.method == "pdol":
                u, info = o
                y = model.action(u + ft, "state")
                p = model.action(y - yt, "adjoint")
                values = dict(u=u, y=y, p=p)
                information.append(info)
            else:
                values = dict(u=o)
            for k, v in values.items():
                outputs.setdefault(k, []).append(array(v)[..., 0])
    out = {k: np.concatenate(v) for k, v in outputs.items()}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise FloatingPointError("prediction")
    return out, information


def action_diagnostics(model, c, data):
    if c.method not in ("pdol", "iuzawa_f", "iuzawa_s"):
        return {}, {}
    v = torch.tensor(data["v"][:8, ..., None], device=c.device, dtype=torch.float32)
    w = torch.tensor(data["w"][:8, ..., None], device=c.device, dtype=torch.float32)
    grid = Grid(c.grid)
    if c.method == "pdol":
        pairs = [(lambda x: model.action(x, "state"), lambda x: model.action(x, "adjoint"), None)]
    else:
        layers = list(model.unrolled._iter_layers())
        if c.method == "iuzawa_s":
            layers = layers[:1]
        pairs = [
            (layer.state_action, layer.S_adj_oper, layer.qb_sap_fourier_layers) for layer in layers
        ]
    reports = []
    values = {}
    for i, (s, a, qb) in enumerate(pairs):
        with torch.no_grad():
            sv = s(v)
            aw = a(w)
            sw = s(w)
            zero = s(torch.zeros_like(v))
            mixed = s(v + w)
            double = s(2 * v)
        jvp = torch.autograd.functional.jvp(s, v, w, create_graph=False)[1]
        with torch.no_grad():
            fd = (s(v + 0.001 * w) - s(v - 0.001 * w)) / 0.002
        truth = torch.tensor(
            grid.solve(data["w"][:8])[..., None], device=c.device, dtype=torch.float32
        )
        norm = lambda t: t.flatten(1).norm(dim=1)
        lhs = (sv * w).flatten(1).sum(1)
        rhs = (v * aw).flatten(1).sum(1)
        metrics = dict(
            state_action_relative=norm(
                sv - torch.tensor(data["sv"][:8, ..., None], device=c.device, dtype=torch.float32)
            )
            / norm(
                torch.tensor(data["sv"][:8, ..., None], device=c.device, dtype=torch.float32)
            ).clamp_min(1e-12),
            adjoint_action_relative=norm(aw - truth) / norm(truth).clamp_min(1e-12),
            pairing_defect=(lhs - rhs).abs()
            / (norm(sv) * norm(w) + norm(v) * norm(aw)).clamp_min(1e-12),
            jvp_true_relative=norm(jvp - truth) / norm(truth).clamp_min(1e-12),
            jvp_self_fd_relative=norm(jvp - fd) / norm(jvp).clamp_min(1e-12),
            additivity_defect=norm(mixed - sv - sw)
            / (norm(mixed) + norm(sv) + norm(sw)).clamp_min(1e-12),
            homogeneity_defect=norm(double - 2 * sv)
            / (norm(double) + 2 * norm(sv)).clamp_min(1e-12),
            zero_norm=norm(zero),
        )
        if qb is not None:
            with torch.no_grad():
                qv = qb(v)
                qw = qb(w)
            metrics.update(
                preconditioner_pairing=((qv * w - v * qw).flatten(1).sum(1)).abs()
                / (norm(qv) * norm(w) + norm(v) * norm(qw)).clamp_min(1e-12),
                preconditioner_quadratic=(qv * v).flatten(1).sum(1)
                / (norm(v) ** 2).clamp_min(1e-12),
            )
        measures = {k: array(z) for k, z in metrics.items()}
        reports.append(dict(layer=i, metrics=aggregate(measures)))
        values.update({f"layer{i}_{k}": z for k, z in measures.items()})
        values.update(
            {
                f"layer{i}_state": array(sv)[..., 0],
                f"layer{i}_adjoint": array(aw)[..., 0],
                f"layer{i}_jvp": array(jvp)[..., 0],
            }
        )
    return (
        dict(
            sample_count=len(v),
            unique_layers=len(pairs),
            layers=reports,
            distinction="Empirical action/derivative diagnostics; nonlinear weight sharing is not an adjoint guarantee.",
        ),
        values,
    )


def classical(root):
    d = Path(root) / "classical"
    d.mkdir(parents=True, exist_ok=False)
    problem = Problem()
    rows = []
    for n in (16, 32, 64, 128):
        grid = Grid(n)
        f = problem.f(grid.points)[..., 0]
        yd = problem.yd(grid.points)[..., 0]
        started = time.time()
        box = grid.optimal_control(f, yd, problem)
        box_seconds = time.time() - started
        state, sinfo = state_qp(n)
        row = dict(
            grid=n,
            box=dict(
                iterations=box["iterations"],
                seconds=box_seconds,
                control_continuous_relative=float(
                    grid.norm(box["u"] - problem.u(grid.points)[..., 0])
                    / grid.norm(problem.u(grid.points)[..., 0])
                ),
                state_continuous_relative=float(
                    grid.norm(box["y"] - problem.y(grid.points)[..., 0])
                    / grid.norm(problem.y(grid.points)[..., 0])
                ),
                projected_residual=float(box["projected_residual"]),
            ),
            state_constraint=sinfo,
        )
        np.savez_compressed(
            d / f"grid{n}.npz",
            **{f"box_{k}": box[k] for k in ("u", "y", "p")},
            **{f"cnp_{k}": v for k, v in state.items()},
        )
        rows.append(row)
        print("classical", n, flush=True)
    write(d / "summary.json", rows)
    return rows


def fixed_instance(model, c, split, folder, root):
    problem = Problem()
    report = {}
    saved = {}
    is_instance = c.method in ("osnn", "admm", "cnp")
    if is_instance:
        x, w = quadrature(96)
        w = w[:, 0]
        fields = instance_fields(model, c, split, x)
        norm = lambda v: float(np.sqrt(np.sum(w * v * v)))
        if c.method == "cnp":
            j = (
                0.5 * norm(fields["y"] - problem.sine(x)[:, 0]) ** 2
                + 0.5 * problem.alpha * norm(fields["u"]) ** 2
            )
            report["quadrature"] = dict(
                objective=j,
                state_violation=float(np.maximum(fields["y"] - problem.state_cap, 0).max()),
                pde_control_elimination="u=-Laplace(y) by definition; recovered-state discrepancy checked independently",
            )
        else:
            report["quadrature"] = dict(
                control_relative_l2=norm(fields["u"] - problem.u(x)[:, 0])
                / norm(problem.u(x)[:, 0]),
                state_relative_l2=norm(fields["y"] - problem.y(x)[:, 0]) / norm(problem.y(x)[:, 0]),
                adjoint_relative_l2=norm(fields["p"] - problem.p(x)[:, 0])
                / norm(problem.p(x)[:, 0]),
                state_residual_l2=norm(fields["state_residual"]),
                adjoint_residual_l2=norm(fields["adjoint_residual"]),
                raw_objective=0.5 * norm(fields["y"] - problem.yd(x)[:, 0]) ** 2
                + 0.5 * problem.alpha * norm(fields["u"]) ** 2,
                bound_violation=float(
                    np.maximum(
                        np.maximum(problem.lower - fields["u"], fields["u"] - problem.upper), 0
                    ).max()
                ),
            )
            if c.method == "admm":
                report["quadrature"].update(
                    admm_primal_l2=norm(fields["primal"]),
                    admm_dual_l2=norm(fields["dual"]),
                    raw_control_violation=float(
                        np.maximum(
                            np.maximum(
                                problem.lower - fields["raw_u"], fields["raw_u"] - problem.upper
                            ),
                            0,
                        ).max()
                    ),
                )
        t = np.linspace(0, 1, 129)
        boundary = np.concatenate(
            [
                np.stack([t, t * 0], 1),
                np.stack([t, t * 0 + 1], 1),
                np.stack([t * 0, t], 1),
                np.stack([t * 0 + 1, t], 1),
            ]
        )
        with torch.no_grad():
            by = array(model.y(torch.tensor(boundary, dtype=torch.float64, device=c.device)))
        report["boundary_y_max"] = float(np.abs(by).max())
        np.savez_compressed(
            folder / "gauss_fields.npz",
            points=x,
            weights=w,
            **fields,
            boundary=boundary,
            boundary_y=by,
        )
    else:
        grid = Grid(c.grid)
        coarse, _ = predict(
            model, c, problem.f(grid.points)[None, ..., 0], problem.yd(grid.points)[None, ..., 0]
        )
        coarse = {k: v[0] for k, v in coarse.items()}
    for n in (32, 64, 128):
        grid = Grid(n)
        shape = (n - 1, n - 1)
        if is_instance:
            fields = {
                k: v.reshape(shape)
                for k, v in instance_fields(
                    model, c, split, grid.points.reshape(-1, 2), derivatives=False
                ).items()
            }
        else:
            fields = coarse if n == c.grid else dict(u=prolong(coarse["u"], n))
        with np.load(Path(root) / "classical" / f"grid{n}.npz", allow_pickle=False) as a:
            base = dict(a)
        if c.method == "cnp":
            y, u = fields["y"], fields["u"]
            yd = problem.sine(grid.points)[..., 0]
            recovered = grid.solve(u)
            ry, ru = base["cnp_y"], base["cnp_u"]
            grad = y - yd + problem.alpha * grid.apply(grid.apply(y))
            j = float(0.5 * grid.norm(y - yd) ** 2 + 0.5 * problem.alpha * grid.norm(u) ** 2)
            jr = float(0.5 * grid.norm(ry - yd) ** 2 + 0.5 * problem.alpha * grid.norm(ru) ** 2)
            detail = dict(
                raw_objective=j,
                reference_objective=jr,
                raw_objective_gap=j - jr,
                recovered_objective=float(
                    0.5 * grid.norm(recovered - yd) ** 2 + 0.5 * problem.alpha * grid.norm(u) ** 2
                ),
                control_relative_l2=float(grid.norm(u - ru) / grid.norm(ru)),
                raw_state_relative_l2=float(grid.norm(y - ry) / grid.norm(ry)),
                recovered_state_relative_l2=float(grid.norm(recovered - ry) / grid.norm(ry)),
                state_recovery_gap=float(grid.norm(y - recovered)),
                state_violation=float(np.maximum(y - problem.state_cap, 0).max()),
                recovered_state_violation=float(np.maximum(recovered - problem.state_cap, 0).max()),
                state_space_projected_residual=float(
                    grid.norm(y - np.minimum(y - grad, problem.state_cap))
                ),
            )
        else:
            ref = {k: base["box_" + k] for k in ("u", "y", "p")}
            detail = {
                k: float(v)
                for k, v in control(
                    grid,
                    fields["u"],
                    problem.f(grid.points)[..., 0],
                    problem.yd(grid.points)[..., 0],
                    ref,
                    fields.get("y"),
                    fields.get("p"),
                ).items()
            }
            detail["control_continuous_relative"] = float(
                grid.norm(fields["u"] - problem.u(grid.points)[..., 0])
                / grid.norm(problem.u(grid.points)[..., 0])
            )
        report["grid" + str(n)] = detail
        saved.update({f"grid{n}_{k}": v for k, v in fields.items()})
    np.savez_compressed(folder / "fixed_fields.npz", **saved)
    report["grid_semantics"] = (
        "FD state/adjoint recovery; operator controls use edge-extended bilinear prolongation to fine grids, not resolution transfer."
    )
    return report


def family(model, c, data, folder):
    if c.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.time()
    pred, info = predict(model, c, data["f"], data["yd"])
    if c.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.time() - start
    grid = Grid(c.grid)
    values = control(grid, pred["u"], data["f"], data["yd"], data, pred.get("y"), pred.get("p"))
    saved = {
        **pred,
        **{"metric_" + k: v for k, v in values.items()},
        "sample_id": data["sample_id"],
    }
    result = dict(
        count=len(data["sample_id"]),
        metrics=aggregate(values),
        prediction_seconds=seconds,
        prediction_device=c.device,
        pdol_iteration_diagnostics=info,
    )
    if c.method == "pdol":
        result["raw_field_semantics"] = (
            "y=S_theta(u+f); p=A_theta(y-yd) diagnostic reconstruction, not the observation-space dual iterate d."
        )
    for n in (64, 128):
        refined = inputs({k: v[:8] for k, v in data.items() if k.startswith("coeff_")}, n)
        fine = Grid(n)
        ref = fine.optimal_control(refined["f"], refined["yd"], c.problem())
        u = prolong(pred["u"][:8], n)
        fine_values = control(fine, u, refined["f"], refined["yd"], ref)
        result["refinement" + str(n)] = dict(
            count=8, metrics=aggregate(fine_values), reference_iterations=ref["iterations"]
        )
        saved.update(
            {
                f"grid{n}_u": u,
                **{f"grid{n}_reference_{k}": ref[k] for k in ("u", "y", "p")},
                **{f"grid{n}_metric_{k}": v for k, v in fine_values.items()},
            }
        )
    diagnostics, actions = action_diagnostics(model, c, data)
    result["learned_actions"] = diagnostics
    np.savez_compressed(folder / "heldout_predictions.npz", **saved)
    if actions:
        np.savez_compressed(folder / "action_diagnostics.npz", **actions)
    return result


def run(root, device="cuda", resume=False):
    root = Path(root)
    frozen = verify_freeze(root)
    data = load_holdout(root)
    torch.set_num_threads(2)
    started = time.time()
    if not (root / "classical/summary.json").exists():
        classical(root)
    rows = []
    for e in frozen["entries"]:
        folder = root / "assessment" / e["key"]
        if resume and (folder / "metrics.json").exists():
            row = json.loads((folder / "metrics.json").read_text())
            if row["checkpoint_sha256"] != e["sha256"]:
                raise ValueError("resume model identity")
            rows.append(row)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        model, c, split, _ = load(e["checkpoint"], device)
        row = dict(
            key=e["key"],
            method=e["method"],
            seed=e["seed"],
            checkpoint_sha256=e["sha256"],
            validation_selection_score=e["selection_score"],
            fixed=fixed_instance(model, c, split, folder, root),
        )
        if c.method not in ("osnn", "admm", "cnp"):
            row["heldout"] = family(model, c, data, folder)
        write(folder / "metrics.json", row)
        rows.append(row)
        print("assessed", e["key"], flush=True)
    verify_freeze(root)
    write(
        root / "assessment/summary.json", dict(rows=rows, assessment_seconds=time.time() - started)
    )
    (root / "INDEPENDENT_ASSESSMENT_COMPLETE").write_text(
        "Independent evaluation complete. Test has now been viewed; no tuning of this release.\n"
    )
    return rows
