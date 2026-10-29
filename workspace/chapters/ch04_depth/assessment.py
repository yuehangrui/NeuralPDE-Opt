"""Freeze before new observations; independent PDE recovery and reload audit."""

from pathlib import Path
import datetime
import json
import math
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.fft import dstn, idstn
from chapters.ch04.problem import Problem
from chapters.ch04.fem import Reference
from .common import (
    ALPHA,
    FDGrid,
    Mesh,
    tensor,
    transfer_control,
    identity,
    digest,
    write_json,
    load_checkpoint,
)
from .training import InverseProblem


def files_under(root, sub):
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / sub).rglob("*"))
        if p.is_file()
    }


def audit_training(root):
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {f"{c['method']}_seed{c['seed']}" for c in protocol["jobs"]}
    if {p.name for p in (root / "training").iterdir()} != expected:
        raise ValueError("unexpected or missing training directories")
    rows = []
    for config in protocol["jobs"]:
        name = f"{config['method']}_seed{config['seed']}"
        path = root / "training" / name
        if not (path / "TRAINING_COMPLETE").exists():
            raise ValueError(f"incomplete training: {name}")
        net, ck = load_checkpoint(path / "best.pt")
        history = json.loads((path / "history.json").read_text())
        run = json.loads((path / "run.json").read_text())
        scores = np.array([r["validation_score"] for r in history])
        if not np.isfinite(scores).all() or ck["selected_index"] != int(scores.argmin()):
            raise ValueError("selection is not the earliest validation minimum")
        if ck["config"] != config or run["config"] != config or run["test_evaluations"] != 0:
            raise ValueError("protocol or training/test separation mismatch")
        if (
            abs(ck["validation_score"] - scores.min()) > 1e-12
            or ck["hidden_parameter_change_l2"] <= 1e-8
        ):
            raise ValueError("score mismatch or hidden layers not trained")
        if config["method"] == "uzawa":
            if len(history) != config["outer_steps"]:
                raise ValueError("incomplete Uzawa budget")
            validation_grid = FDGrid(config["validation_mesh"])
            with torch.no_grad():
                y = net(tensor(validation_grid.x)).numpy()
            u = transfer_control(ck["control"], config["mesh"], validation_grid.x)
            score = validation_grid.diagnostics(y, u)["validation_score"]
        else:
            if history[-1]["step"] != config["adam_steps"] + config["lbfgs_steps"]:
                raise ValueError("incomplete FEM-NN budget")
            if ck["observation_sha256"] != digest(path / "observations.npz"):
                raise ValueError("observation identity mismatch")
            with np.load(path / "observations.npz") as data:
                points = np.concatenate([data["train_x"], data["val_x"]])
                if len(set(map(tuple, points))) != len(points):
                    raise ValueError("observation overlap")
                scale = float(np.sqrt(np.mean(data["train_observed"] ** 2)))
                if abs(scale - float(data["observation_scale"])) > 1e-13:
                    raise ValueError("normalization not fitted on train observations")
                val = InverseProblem(
                    config["validation_mesh"], data["val_x"], data["val_observed"], scale
                )
                score = val.score(net)
        if not np.isclose(score, ck["validation_score"], rtol=1e-9, atol=1e-11):
            raise ValueError(f"CPU validation reload mismatch {name}")
        rows.append(
            dict(
                name=name,
                validation_score=score,
                selected_index=ck["selected_index"],
                hidden_parameter_change_l2=ck["hidden_parameter_change_l2"],
            )
        )
    return rows


def exact_uzawa(n, iterations=80, rho=0.2 * ALPHA):
    """Independent exact-inner discrete comparison; never initializes a network."""
    grid = FDGrid(n)
    eigenvalues = grid.eigenvalues
    fhat, ydhat = (dstn(a, type=1, norm="ortho") for a in (grid.f, grid.yd))
    optimal = idstn(
        (eigenvalues * ydhat - fhat) / (1 + ALPHA * eigenvalues**2), type=1, norm="ortho"
    )
    z = np.zeros_like(grid.f)
    history = []
    for _ in range(iterations):
        zhat = dstn(z, type=1, norm="ortho")
        y = idstn(
            (ydhat + 0.5 * ALPHA * eigenvalues * fhat - eigenvalues * zhat)
            / (1 + 0.5 * ALPHA * eigenvalues**2),
            type=1,
            norm="ortho",
        )
        u = 2 * z / ALPHA
        history.append(grid.diagnostics(y, u))
        z += rho * (grid.apply(y) - u - grid.f)
    return dict(
        mesh=n,
        history=history,
        control_error_to_discrete_optimum=grid.norm(u - optimal) / grid.norm(optimal),
    )


def predict(net, x, scale=1.0):
    shape = x.shape[:-1]
    parts = []
    with torch.no_grad():
        for points in np.array_split(x.reshape(-1, 2), max(1, math.ceil(np.prod(shape) / 8192))):
            parts.append(scale * net(tensor(points)).numpy())
    return np.concatenate(parts).reshape(shape)


def assess_uzawa(net, ck):
    result = dict(method="uzawa", seed=ck["config"]["seed"], meshes=[])
    problem = Problem()
    for n in (128, 256):
        grid = FDGrid(n)
        raw = predict(net, grid.x)
        u = transfer_control(ck["control"], ck["config"]["mesh"], grid.x)
        refu, refy = problem.u(grid.x)[..., 0], problem.y(grid.x)[..., 0]
        recovered = grid.solve(grid.f + u)
        result["meshes"].append(
            dict(
                mesh=n,
                **grid.diagnostics(raw, u),
                control_relative_l2=grid.norm(u - refu) / grid.norm(refu),
                raw_state_relative_l2=grid.norm(raw - refy) / grid.norm(refy),
                recovered_state_relative_l2=grid.norm(recovered - refy) / grid.norm(refy),
            )
        )
    result["exact_inner_baseline"] = exact_uzawa(ck["config"]["mesh"])
    t = np.linspace(0, 1, 129)
    boundary = np.concatenate(
        [
            np.stack([t, np.zeros_like(t)], 1),
            np.stack([t, np.ones_like(t)], 1),
            np.stack([np.zeros_like(t), t], 1),
            np.stack([np.ones_like(t), t], 1),
        ]
    )
    result["boundary_max"] = float(np.abs(predict(net, boundary)).max())
    return result


def assessment_mesh(n):
    ref = Reference(Problem(), n, 4)
    lu = splu(sparse.bmat([[ref.Ki, ref.Mi / ALPHA], [-ref.Mi, ref.Ki]], format="csc"))
    return ref, lu


def assess_fem(net, ck, meshes, heldout):
    cfg = ck["config"]
    scale = ck["source_scale"]
    result = dict(method=cfg["method"], seed=cfg["seed"], meshes=[])
    for n, (ref, lu) in meshes.items():
        f_nodes = predict(net, ref.nodes, scale)
        load = (ref.M @ f_nodes)[ref.interior]
        yp = lu.solve(np.concatenate([load, -ref.ydl[ref.interior]]))
        count = len(ref.interior)
        learned_y = ref.expand(yp[:count])
        u = ref.expand(-yp[count:] / ALPHA)
        uq = ref.at_quad(u)
        true_y, _, true_g = ref.recover(uq)
        state0 = ref.expand(ref.lu.solve(load))
        H = Mesh(n).observe(heldout["points"])
        predicted_obs = H @ state0[ref.interior]
        observation_scale = scale / (2 * math.pi**2)
        target = heldout["clean"] + (heldout["noise"] if cfg["noise"] else 0.0)
        result["meshes"].append(
            dict(
                mesh=n,
                control_relative_l2=ref.norm(uq - ref.problem.u(ref.points)[..., 0])
                / ref.norm(ref.problem.u(ref.points)[..., 0]),
                control_relative_to_fem=ref.norm(uq - ref.at_quad(ref.u))
                / ref.norm(ref.at_quad(ref.u)),
                true_gradient_relative=ref.norm(true_g) / ref.gradient_scale,
                learned_objective=ref.objective(ref.at_quad(learned_y), uq),
                true_objective=ref.objective(ref.at_quad(true_y), uq),
                reference_objective=ref.reference_objective,
                uncontrolled_test_observation_relative=float(
                    np.sqrt(np.mean((predicted_obs - target) ** 2)) / observation_scale
                ),
                uncontrolled_clean_observation_relative=float(
                    np.sqrt(np.mean((predicted_obs - heldout["clean"]) ** 2)) / observation_scale
                ),
            )
        )
    z, w = np.polynomial.legendre.leggauss(96)
    xx, yy = np.meshgrid((z + 1) / 2, (z + 1) / 2, indexing="ij")
    points = np.stack([xx, yy], -1)
    weights = np.outer(w / 2, w / 2)
    true_f = Problem().f(points)[..., 0]
    learned_f = predict(net, points, scale)
    norm = lambda v: float(np.sqrt(np.sum(weights * v * v)))
    result["source_relative_l2_gauss96"] = norm(learned_f - true_f) / norm(true_f)
    return result


def new_heldout():
    rng = np.random.default_rng(20261207)
    x = 0.025 + 0.95 * rng.random((512, 2))
    clean = np.sin(math.pi * x[:, 0]) * np.sin(math.pi * x[:, 1]) + np.sin(
        8 * math.pi * x[:, 0]
    ) * np.sin(8 * math.pi * x[:, 1]) / (ALPHA * (128 * math.pi**2) ** 2)
    # Separate physical observation protocol, 1% of fixed unit-sine L2 scale.
    # Actual sigma is reported; it does not alter train normalization.
    noise = 0.005 * rng.standard_normal(512)
    return dict(points=x, clean=clean, noise=noise)


def report(root, results):
    grouped = {}
    for name, row in results.items():
        grouped.setdefault(row["method"], []).append(row)
    rows = []
    for method, values in grouped.items():
        controls = np.array([v["meshes"][-1]["control_relative_l2"] for v in values]) * 100
        gradients = np.array([v["meshes"][-1]["true_gradient_relative"] for v in values])
        source = [v.get("source_relative_l2_gauss96") for v in values]
        rows.append(
            dict(
                method=method,
                count=len(values),
                control_percent_mean=float(controls.mean()),
                control_percent_sample_std=(
                    float(controls.std(ddof=1)) if len(controls) > 1 else None
                ),
                gradient_relative_mean=float(gradients.mean()),
                source_percent_mean=float(np.mean(source) * 100) if source[0] is not None else None,
            )
        )
    lines = [
        "# 第四章非线性表示与稀疏观测深化 v1",
        "",
        "四条流程、每条三个种子。Deep Uzawa使用可训练坐标状态网络和解析控制块；FEM-NN使用可训练坐标源项网络。所有隐藏层参与训练，输出不限制在原17模态空间。",
        "",
        "模型选择只用train/val。全部选择与源码冻结后才生成512个新观测位置，并进行独立空间/PDE复核；这是同一物理问题的独立诊断，不是新问题族泛化测试。",
        "",
        "| 流程 | 种子数 | 控制L2误差（%，均值 ± 样本标准差） | 真实PDE梯度相对量 | 源项L2误差（%） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        std = (
            "-"
            if row["control_percent_sample_std"] is None
            else f"{row['control_percent_sample_std']:.4g}"
        )
        source = "-" if row["source_percent_mean"] is None else f"{row['source_percent_mean']:.4g}"
        lines.append(
            f"| {row['method']} | {row['count']} | {row['control_percent_mean']:.4g} ± {std} | {row['gradient_relative_mean']:.4g} | {source} |"
        )
    lines += [
        "",
        "Deep Uzawa采用256网格诊断，另保留128网格结果；FEM-NN采用128网格诊断，另保留64网格结果。两类流程的离散化与信息条件不同，不作架构排名。",
        "",
        "FEM-NN三组分别为稀疏无噪声、稀疏1%噪声、同一噪声数据加H1/L2正则化；不能只凭观测误差认定源项被正确辨识。真实PDE重求后的控制目标和梯度与学习模型目标分别保存。",
        "",
        "完成预算不等于满足严格最优性。非线性Uzawa内层采用有限迭代，不能援引精确凸内层的收敛保证。控制从训练网格插值/外推，不宣称分辨率迁移。原固定特征、纯神经和输出头修正结果继续保留。",
        "",
        "本版本检查点支持推理重载；last保存部分优化器状态用于诊断，但CLI没有精确断点续训承诺。详细选模轨迹、预算、计时和逐种子指标见training与assessment。",
        "",
        "![控制误差](control_errors.png)",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "comparison.json", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [r["method"] for r in rows],
        [r["control_percent_mean"] for r in rows],
        yerr=[r["control_percent_sample_std"] or 0 for r in rows],
        capsize=4,
        color=["#3c6e9d", "#5c8d72", "#c7964a", "#8b65a6"][: len(rows)],
    )
    ax.set_ylabel("Control relative L2 error (%)")
    ax.set_title("Nonlinear Chapter 4 extensions: mean and sample SD")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "control_errors.png", dpi=160)
    plt.close(fig)


def freeze(root):
    audit = audit_training(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["pilot"]:
        raise ValueError("development pilots cannot be published as formal delivery")
    expected = {
        (m, s) for m in ["uzawa", "fem_clean", "fem_noisy", "fem_regularized"] for s in [0, 1, 2]
    }
    if {(c["method"], c["seed"]) for c in protocol["jobs"]} != expected or len(
        protocol["jobs"]
    ) != 12:
        raise ValueError("formal delivery requires exactly four methods x three seeds")
    write_json(root / "PRETEST_AUDIT.json", audit)
    frozen = dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_identity=identity(),
        files={
            **files_under(root, "training"),
            "protocol.json": digest(root / "protocol.json"),
            "environment.json": digest(root / "environment.json"),
        },
        protocol_sha256=digest(root / "protocol.json"),
    )
    write_json(root / "SELECTION_FROZEN.json", frozen)


def verify_frozen(root):
    frozen = json.loads((root / "SELECTION_FROZEN.json").read_text())
    if frozen["source_identity"] != identity():
        raise ValueError("frozen source changed")
    for path, sha in frozen["files"].items():
        if digest(root / path) != sha:
            raise ValueError("frozen training artifact changed: " + path)
    return frozen


def finish(root):
    root = Path(root)
    if (root / "DELIVERY_COMPLETE").exists():
        raise FileExistsError("already assessed; use verify")
    if (root / "SELECTION_FROZEN.json").exists():
        verify_frozen(root)
    else:
        if (root / "heldout.npz").exists() or (root / "HELDOUT_MANIFEST.json").exists():
            raise ValueError("held-out data exist before model selection freeze")
        freeze(root)
    test_path = root / "heldout.npz"
    if not test_path.exists():
        data = new_heldout()
        for p in (root / "training").glob("*/observations.npz"):
            with np.load(p) as old:
                if set(map(tuple, data["points"])) & set(
                    map(tuple, np.concatenate([old["train_x"], old["val_x"]]))
                ):
                    raise ValueError("held-out observations overlap train/val")
        np.savez_compressed(test_path, **data)
        write_json(
            root / "HELDOUT_MANIFEST.json",
            dict(
                created_after_freeze=True,
                sha256=digest(test_path),
                time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                selection_sha256=digest(root / "SELECTION_FROZEN.json"),
                seed=20261207,
                count=512,
                noise_sigma=0.005,
                scope="independent points, same physical problem",
            ),
        )
    with np.load(test_path) as data:
        heldout = dict(data)
    if digest(test_path) != json.loads((root / "HELDOUT_MANIFEST.json").read_text())["sha256"]:
        raise ValueError("heldout identity mismatch")
    meshes = {n: assessment_mesh(n) for n in (64, 128)}
    results = {}
    (root / "assessment").mkdir(exist_ok=True)
    probes = heldout["points"][:32]
    for path in sorted((root / "training").iterdir()):
        name = path.name
        target = root / "assessment" / f"{name}.json"
        if target.exists():
            results[name] = json.loads(target.read_text())
            continue
        net, ck = load_checkpoint(path / "best.pt")
        print("ASSESS", name, flush=True)
        result = (
            assess_uzawa(net, ck)
            if ck["config"]["method"] == "uzawa"
            else assess_fem(net, ck, meshes, heldout)
        )
        result["checkpoint_sha256"] = digest(path / "best.pt")
        result["reload_probe"] = predict(net, probes, ck.get("source_scale", 1.0)).tolist()
        write_json(target, result)
        results[name] = result
    report(root, results)
    audit = verify(root, require_complete=False)
    audit["artifact_hashes"] = {
        **files_under(root, "assessment"),
        **{
            p: digest(root / p)
            for p in [
                "heldout.npz",
                "HELDOUT_MANIFEST.json",
                "summary.md",
                "comparison.json",
                "control_errors.png",
            ]
        },
    }
    write_json(root / "DELIVERY_AUDIT.json", audit)
    (root / "DELIVERY_COMPLETE").write_text(
        "trained, frozen, assessed, and reloaded; not a convergence certificate\n"
    )


def verify(root, require_complete=True):
    root = Path(root)
    if require_complete and not (root / "DELIVERY_COMPLETE").exists():
        raise ValueError("not a complete delivery")
    verify_frozen(root)
    rows = audit_training(root)
    manifest = json.loads((root / "HELDOUT_MANIFEST.json").read_text())
    if digest(root / "heldout.npz") != manifest["sha256"]:
        raise ValueError("test identity mismatch")
    if digest(root / "SELECTION_FROZEN.json") != manifest["selection_sha256"]:
        raise ValueError("heldout does not reference the frozen selection")
    if require_complete:
        audit = json.loads((root / "DELIVERY_AUDIT.json").read_text())
        for path, sha in audit["artifact_hashes"].items():
            if digest(root / path) != sha:
                raise ValueError("assessed artifact changed: " + path)
    with np.load(root / "heldout.npz") as data:
        probes = data["points"][:32]
    for row in rows:
        name = row["name"]
        path = root / "training" / name / "best.pt"
        net, ck = load_checkpoint(path)
        result = json.loads((root / "assessment" / f"{name}.json").read_text())
        if result["checkpoint_sha256"] != digest(path):
            raise ValueError("assessment checkpoint mismatch")
        np.testing.assert_allclose(
            predict(net, probes, ck.get("source_scale", 1.0)),
            result["reload_probe"],
            rtol=1e-12,
            atol=1e-12,
        )
    return dict(
        passed=True,
        models=len(rows),
        training_selection_reloaded=True,
        prediction_reloaded=True,
        source_and_data_identity=True,
        test_exposed=True,
    )
