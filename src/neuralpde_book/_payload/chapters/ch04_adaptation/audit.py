"""Read-only model/data/selection audits and validation-only PDE checks."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .instance import load as load_instance, quadrature, weighted_loss
from .train_operators import load as load_operator
from .operator_data import validate, basis, EIGENVALUES, ALPHA
from chapters.ch04.problem import Problem
from chapters.ch04.fem import Reference
from chapters.ch04.evaluation import evaluate, predict
from chapters.ch04.training import write_json


def read(path):
    return json.loads(Path(path).read_text())


def same(a, b, tol=2e-9):
    if not np.isclose(a, b, atol=tol, rtol=tol):
        raise AssertionError((a, b))


def audit_instances(root):
    results = {}
    problem = Problem()
    ref = Reference(problem, 128, 4)
    for group in ("instances", "refinement", "head_polish"):
        if not (root / group / "TRAIN_VAL_COMPLETE").exists():
            raise RuntimeError("unfinished " + group)
        for path in sorted((root / group).glob("*/metrics.json")):
            folder = path.parent
            metrics = read(path)
            run = read(folder / "run.json")
            if run["status"] != "completed":
                raise AssertionError("incomplete run")
            history = [
                json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()
            ]
            best = min(history, key=lambda row: row["validation_score"])
            if best["progress"] != metrics["progress"]:
                raise AssertionError("selection mismatch")
            model, cfg, ck = load_instance(folder / "best.pt", "cuda")
            if ck["progress"] != best["progress"]:
                raise AssertionError("checkpoint mismatch")
            _, _, last = load_instance(folder / "last.pt")
            if last["progress"] != history[-1]["progress"]:
                raise AssertionError("last checkpoint mismatch")
            for count_key in ("updates", "additional_updates"):
                if count_key in run and run[count_key] != history[-1][count_key]:
                    raise AssertionError("update accounting mismatch")
            if "head_solves" in run and run["head_solves"] != len(history) - 1:
                raise AssertionError("head solve accounting mismatch")
            data = np.load(folder / "sampling.npz")
            x, w, v = data["train"], data["weights"], data["validation"]
            for key, array in [("train", x), ("weights", w), ("validation", v)]:
                if not np.isfinite(array).all():
                    raise AssertionError("nonfinite data")
                if hashlib.sha256(array.tobytes()).hexdigest() != ck["sample_hashes"][key]:
                    raise AssertionError("sample hash mismatch")
            scale = float(np.sqrt(np.sum(w * problem.f(x) ** 2)))
            fresh = evaluate(model, problem, ref, v, scale)
            for key, value in fresh.items():
                same(value, metrics[key])
            results[str(folder.relative_to(root))] = dict(
                best_progress=best["progress"],
                validation=fresh,
                source_identity_verified=True,
                sample_hashes_verified=True,
                earliest_best_verified=True,
            )
            if group == "head_polish":
                checks = []
                for n in (48, 72):
                    q, wq = quadrature(n)
                    xt = torch.tensor(q, dtype=torch.float64, device="cuda", requires_grad=True)
                    wt = torch.tensor(wq, device="cuda")
                    _, j, r = weighted_loss(model, problem, xt, wt, 1.0, torch.zeros_like(wt))
                    checks.append(
                        dict(
                            order=n,
                            J=float(j.detach()),
                            PDE_L2=float((wt * r * r).sum().sqrt().detach()),
                        )
                    )
                results[str(folder.relative_to(root))]["quadrature_refinement"] = checks
                same(checks[0]["J"], checks[1]["J"], tol=1e-10)
                same(checks[0]["PDE_L2"], checks[1]["PDE_L2"], tol=1e-9)
    return results


class FamilyProblem:
    alpha = ALPHA
    control_frequency = 8

    def __init__(self, f, d):
        self.fc = f
        self.dc = d

    def f(self, x):
        return (basis(x.reshape(-1, 2)) @ self.fc).reshape(x.shape[:-1] + (1,))

    def yd(self, x):
        return (basis(x.reshape(-1, 2)) @ self.dc).reshape(x.shape[:-1] + (1,))


def audit_operators(root):
    if not (root / "operators/TRAIN_VAL_COMPLETE").exists():
        raise RuntimeError("unfinished operators")
    results = {}
    for method in ("mimo", "rb_state", "rb_dino"):
        folder = root / "operators" / method
        tr = dict(np.load(folder / "train.npz"))
        va = dict(np.load(folder / "validation.npz"))
        for split in (tr, va):
            for key in ("f", "yd", "rhs", "state", "y", "u", "p"):
                if split[key].shape != (len(split["ids"]), 17) or not np.isfinite(split[key]).all():
                    raise AssertionError("data schema/finite violation")
            np.testing.assert_allclose(split["state"] * EIGENVALUES, split["rhs"], atol=1e-12)
        hashes = validate(tr, va)
        model, ck = load_operator(folder / "best.pt")
        if hashes != ck["data_hashes"]:
            raise AssertionError("operator data identity mismatch")
        hist = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        best = min(hist, key=lambda row: row["validation_score"])
        metrics = read(folder / "metrics.json")
        if best["step"] != ck["step"] or best != metrics:
            raise AssertionError("operator selection mismatch")
        _, last = load_operator(folder / "last.pt")
        if last["step"] != hist[-1]["step"] or last["step"] != 2000:
            raise AssertionError("operator last checkpoint mismatch")
        if read(folder / "run.json")["updates"] != 2000:
            raise AssertionError("budget accounting")
        row = dict(
            best_step=best["step"], source_and_data_verified=True, earliest_best_verified=True
        )
        if method == "mimo":
            grid = np.linspace(0, 1, 17)
            xx, yy = np.meshgrid(grid, grid, indexing="ij")
            sensors = np.stack([xx.ravel(), yy.ravel()], 1)
            bs = basis(sensors)
            f = torch.tensor(va["f"] @ bs.T)
            d = torch.tensor(va["yd"] @ bs.T)
            x, w = quadrature(72)
            b = basis(x)
            with torch.no_grad():
                y, u, p = model(f, d, torch.tensor(x))
            rel = lambda a, t: float(
                np.mean(np.sqrt(np.sum(w.T * (a - t) ** 2, 1) / np.sum(w.T * t * t, 1)))
            )
            ry = rel(y.numpy(), va["y"] @ b.T)
            ru = rel(u.numpy(), va["u"] @ b.T)
            same(ry, metrics["state_relative_l2_mean"])
            same(ru, metrics["control_relative_l2_mean"])
            rp = np.random.default_rng(444).uniform(size=(256, 2))
            br = basis(rp)
            rs, ra = model.residuals(
                f[:4],
                d[:4],
                torch.tensor(rp),
                torch.tensor(va["f"][:4] @ br.T),
                torch.tensor(va["yd"][:4] @ br.T),
            )
            row.update(
                state_relative_l2_mean=ry,
                control_relative_l2_mean=ru,
                state_residual_relative=float(
                    rs.detach().square().mean().sqrt()
                    / torch.tensor(va["f"][:4] @ br.T).square().mean().sqrt()
                ),
                adjoint_residual_rms=float(ra.detach().square().mean().sqrt()),
            )
            # Independent P1 true-state recovery for two already-declared val tasks.
            recovered = []
            for i in range(2):
                for n in (64, 128):
                    problem = FamilyProblem(va["f"][i], va["yd"][i])
                    ref = Reference(problem, n, 4)
                    uq = []
                    with torch.no_grad():
                        points = ref.points.reshape(-1, 2)
                        for start in range(0, len(points), 4096):
                            uq.append(
                                model(
                                    f[i : i + 1],
                                    d[i : i + 1],
                                    torch.tensor(points[start : start + 4096]),
                                )[1]
                                .numpy()
                                .ravel()
                            )
                    uq = np.concatenate(uq).reshape(ref.weights.shape)
                    yr, pr, gr = ref.recover(uq)
                    recovered.append(
                        dict(
                            id=str(va["ids"][i]),
                            mesh=n,
                            J_true=ref.objective(ref.at_quad(yr), uq),
                            recovered_gradient_relative=ref.norm(gr) / ref.gradient_scale,
                        )
                    )
            row["P1_recovery"] = recovered
        else:
            x = torch.tensor(va["rhs"])
            truth = torch.tensor(va["state"])
            with torch.no_grad():
                pred = model(x)
            rel = float(((pred - truth).square().sum(1) / truth.square().sum(1)).sqrt().mean())
            same(rel, metrics["state_relative_l2_mean"])
            j = model.physical_jacobian(x).detach()
            jtrue = torch.diag(torch.tensor(1 / EIGENVALUES))
            je = float(((j - jtrue).square().sum((1, 2)) / jtrue.square().sum()).sqrt().mean())
            same(je, metrics["physical_jacobian_relative_mean"])
            online = read(folder / "online_validation.json")
            if len(online["rows"]) != 8 or any(
                r["true_objective_gap"] < -1e-12 for r in online["rows"]
            ):
                raise AssertionError("invalid online objective audit")
            row.update(
                state_relative_l2_mean=rel,
                physical_jacobian_relative_mean=je,
                online_control_relative_l2_mean=online["mean_control_relative_l2"],
                online_successes=sum(r["optimizer_success"] for r in online["rows"]),
            )
        # Purely linear supervised regression baseline over the same finite family.
        design = np.concatenate([tr["f"], tr["yd"], np.ones((len(tr["f"]), 1))], 1)
        test_design = np.concatenate([va["f"], va["yd"], np.ones((len(va["f"]), 1))], 1)
        coef = np.linalg.lstsq(design, tr["u"], rcond=None)[0]
        pred = test_design @ coef
        row["affine_control_baseline_relative_l2_mean"] = float(
            np.mean(np.linalg.norm(pred - va["u"], axis=1) / np.linalg.norm(va["u"], axis=1))
        )
        results[method] = row
    return results


def protected():
    counts = {}
    for chapter in ("ch04", "ch05"):
        root = Path("chapters") / chapter
        manifest = read(root / "RUNTIME_MANIFEST.json")
        for name, h in manifest.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != h:
                raise AssertionError("protected source changed: " + name)
        counts[chapter] = len(manifest)
    manifest = read("results/ch04/full_training_v1/ARTIFACT_MANIFEST.json")
    for name, h in manifest.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != h:
            raise AssertionError("old result changed: " + name)
    counts["ch04_full_training_v1"] = len(manifest)
    return counts


def main():
    torch.set_num_threads(2)
    root = Path("results/ch04/adaptation_v1")
    report = dict(
        instances=audit_instances(root),
        operators=audit_operators(root),
        protected=protected(),
        test_evaluations=0,
    )
    if not (root / "online_adaptation/VALIDATION_COMPLETE").exists():
        raise RuntimeError("scaled online validation incomplete")
    report["scaled_online"] = {}
    for method in ("rb_state", "rb_dino"):
        result = read(root / "online_adaptation" / (method + ".json"))
        if (
            result["checkpoint_sha256"]
            != hashlib.sha256((root / "operators" / method / "best.pt").read_bytes()).hexdigest()
        ):
            raise AssertionError("online model changed")
        if result["selection_use"] or len(result["rows"]) != 8:
            raise AssertionError("online protocol mismatch")
        if any(
            not r["optimizer_success"]
            or r["trajectory"][-1]["surrogate_gradient_relative_inf"] > 1e-6
            for r in result["rows"]
        ):
            raise AssertionError("online termination not verified")
        report["scaled_online"][method] = dict(
            tasks=8, successes=8, mean_control_relative_l2=result["mean_control_relative_l2"]
        )
    write_json(root / "audit.json", report)
    print(
        json.dumps(
            dict(
                audit="passed",
                instance_runs=len(report["instances"]),
                operator_runs=len(report["operators"]),
                protected=report["protected"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
