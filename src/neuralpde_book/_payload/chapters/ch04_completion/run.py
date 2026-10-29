"""Reproducible, bounded CPU experiments. Existing output directories are refused."""

import argparse
import hashlib
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import torch
from chapters.ch04.fem import Reference
from chapters.ch04.problem import Problem
from . import VERSION, classical, bilevel, deep_uzawa, ismo, fem_nn
from .common import (
    DIM,
    LAM,
    MODES,
    ALPHA,
    SpectralProblem,
    basis,
    tensor,
    identity,
    write_json,
    save_checkpoint,
    load_checkpoint,
)

METHODS = ("classical", "bilevel", "deep_uzawa", "ismo", "fem_nn")


def fem_audit(u, mesh=64):
    ref = Reference(Problem(), mesh, 4)
    uq = np.einsum("eqi,i->eq", basis(ref.points), u)
    y, p, g = ref.recover(uq)
    ix = ref.interior
    residual = ref.Ki @ y[ix] - (ref.fl + ref.load(uq))[ix]
    return dict(
        mesh=mesh,
        algebraic_state_residual=float(np.linalg.norm(residual)),
        recovered_objective=ref.objective(ref.at_quad(y), uq),
        recovered_gradient_l2=ref.norm(g),
        reference_p1=ref.report(),
        note="Sine controls and P1-reference controls have different discrete trial spaces.",
    )


def convert(value):
    if isinstance(value, np.ndarray):
        return tensor(value)
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert(v) for v in value]
    return value


def run_one(method, output, seed=0, epochs=600, rounds=4, fem_mesh=32, audit_mesh=64):
    if method not in METHODS or seed < 0 or audit_mesh <= fem_mesh:
        raise ValueError("invalid method, seed or validation mesh")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = dict(
        method=method,
        seed=seed,
        epochs=epochs,
        rounds=rounds,
        fem_mesh=fem_mesh,
        audit_mesh=audit_mesh,
        alpha=ALPHA,
        modes=MODES.tolist(),
        dtype="float64",
        device="cpu",
        experiment_role="bounded_method_validation",
        test_evaluations=0,
        network_scope="fixed sine features for bilevel/Uzawa/source; trainable MLP for ISMO",
    )
    write_json(out / "config.json", config)
    write_json(
        out / "run.json", dict(status="running", version=VERSION, source_identity=identity())
    )
    start = time.perf_counter()
    if method == "classical":
        result = classical.spectral()
        fe = classical.fem(fem_mesh)
        np.savez(out / "fem_reference.npz", control=fe.pop("control"))
        write_json(out / "fem_baseline.json", fe)
    elif method == "bilevel":
        result = bilevel.run()
    elif method == "deep_uzawa":
        result = deep_uzawa.run()
    elif method == "ismo":
        result = ismo.run(seed=seed, epochs=epochs, rounds=rounds)
    else:
        result = fem_nn.run(fem_mesh, audit_mesh)
    elapsed = time.perf_counter() - start
    data = result.pop("data", None)
    if data is not None:
        np.savez(out / "data.npz", **data)
    state_keys = ("control", "state_head", "source_head", "multiplier", "model", "round_models")
    state = {key: convert(result.pop(key)) for key in state_keys if key in result}
    save_checkpoint(out / "best.pt", method, state, config)
    ck = load_checkpoint(out / "best.pt")
    u = ck["state"]["control"].numpy()
    y = ck["state"]["state_head"].numpy() / LAM if "state_head" in ck["state"] else None
    reload_diag = SpectralProblem().diagnostics(u, y)
    for key in ("validation_score", "control_relative_l2", "true_objective"):
        if not np.isclose(result["diagnostics"][key], reload_diag[key], rtol=1e-12, atol=1e-14):
            raise RuntimeError("checkpoint reload changes diagnostics: " + key)
    result["elapsed_seconds"] = elapsed
    result["optimization_converged"] = bool(reload_diag["validation_score"] < 1e-6)
    result["status"] = "bounded_validation_complete"
    result["test_evaluations"] = 0
    write_json(out / "metrics.json", result)
    write_json(
        out / "validation.json",
        dict(
            checkpoint_reload_match=True,
            spectral=reload_diag,
            independent_fem=fem_audit(u, audit_mesh),
        ),
    )
    write_json(
        out / "run.json",
        dict(
            status="completed",
            version=VERSION,
            source_identity=identity(),
            elapsed_seconds=elapsed,
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                numpy=np.__version__,
                scipy=scipy.__version__,
            ),
        ),
    )
    print(
        method,
        result["status"],
        "control_relative_l2=",
        reload_diag["control_relative_l2"],
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--fem-mesh", type=int, default=32)
    parser.add_argument("--audit-mesh", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment: " + str(root))
    methods = METHODS if args.method == "all" else (args.method,)
    summaries = {}
    for method in methods:
        summaries[method] = run_one(
            method,
            root / method,
            args.seed,
            args.epochs,
            args.rounds,
            args.fem_mesh,
            args.audit_mesh,
        )
    write_json(root / "summary.json", summaries)
    files = [p for p in root.rglob("*") if p.is_file()]
    write_json(
        root / "artifact_manifest.json",
        {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    (root / "COMPLETE").write_text(
        "All requested bounded validations completed; not a claim of universal convergence.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
