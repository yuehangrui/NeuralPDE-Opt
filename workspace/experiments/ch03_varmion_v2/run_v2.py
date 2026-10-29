import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import subprocess
import sys
import numpy as np
import torch
from bootstrap import HERE, ROOT
from common import sha, dump, certify, verify
from model_v2 import MatrixMap
from trainer_v2 import fit, tensors, evaluate
from prepare_v2 import prepare, reload_prepared
from neuralpde_opt.chapter3.operator_training import evaluate_arrays, exclusive_run_lock


def sources():
    paths = list(HERE.glob("*.py")) + [HERE / "config.json"]
    paths += list((ROOT / "experiments/ch03_varmion_v1").glob("*.py"))
    paths += [ROOT / "experiments/ch03_physics_v1/physics.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def finish(path, kind, norm, k0, data, geometry, phi, arrays):
    last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != last["identity"] or saved["best_epoch"] != last["best_epoch"]:
        raise RuntimeError("Checkpoint selection mismatch")
    model = MatrixMap(kind, norm["c_mean"], norm["c_std"], k0).to("cuda", dtype=torch.float32)
    model.load_state_dict(saved["model"])
    model.eval()
    d = tensors(arrays, "cuda")
    metrics = {}
    matrix_diagnostics = {}
    for split, target in data.items():
        alpha, score = evaluate(model, d[split])
        prediction = (alpha @ phi.T).reshape(target["y"].shape)
        metrics[split] = evaluate_arrays(prediction, target["y"], geometry, target["sample_id"])
        if abs(metrics[split]["mean"]["relative_l2"] - score) > 1e-10:
            raise RuntimeError("Reconstruction metric mismatch")
        if metrics[split]["max"]["boundary_max"] > 1e-12:
            raise RuntimeError("Boundary failure")
        np.savez_compressed(
            path / (split + "_predictions.npz"),
            prediction=prediction,
            coefficients=alpha,
            sample_id=target["sample_id"],
        )
        with torch.no_grad():
            k = model.matrix(d[split]["sensors"]).double().cpu().numpy()
        sym = (k + k.transpose(0, 2, 1)) / 2
        matrix_diagnostics[split] = dict(
            relative_skew_mean=float(
                np.mean(
                    np.linalg.norm(k - k.transpose(0, 2, 1), axis=(1, 2))
                    / np.linalg.norm(k, axis=(1, 2))
                )
            ),
            sym_min_eigenvalue=float(np.linalg.eigvalsh(sym).min()),
        )
        if kind == "spd" and matrix_diagnostics[split]["sym_min_eigenvalue"] <= 0:
            raise RuntimeError("Numerically nonpositive SPD prediction")
    with torch.no_grad():
        sensors = d["val"]["sensors"]
        z = d["val"]["z"]
        z2 = z.roll(1, 0)
        y = model(sensors, 0.7 * z - 1.3 * z2)
        expected = 0.7 * model(sensors, z) - 1.3 * model(sensors, z2)
        structure = dict(
            zero_load_max=float(model(sensors, torch.zeros_like(z)).abs().max()),
            relative_superposition_max=float(
                (
                    torch.linalg.vector_norm(y - expected, dim=1)
                    / torch.linalg.vector_norm(expected, dim=1).clamp_min(1e-12)
                ).max()
            ),
        )
    if structure["zero_load_max"] != 0 or structure["relative_superposition_max"] > 2e-6:
        raise RuntimeError("Linearity failure")
    result = dict(
        kind=kind,
        seed=saved["identity"]["seed"],
        variant=saved["identity"]["variant"],
        epochs=last["epoch"],
        optimizer_steps=last["steps"],
        sample_visits=last["visits"],
        operator_visits=last["operator_visits"],
        parameter_count=last["parameter_count"],
        best_epoch=saved["best_epoch"],
        best_score=saved["best_score"],
        elapsed_seconds=last["elapsed_seconds"],
        metrics=metrics,
        matrix_diagnostics=matrix_diagnostics,
        structure=structure,
        test_evaluations=0,
    )
    dump(path / "metrics.json", result)
    dump(path / "status.json", dict(status="complete"))
    certify(path, "COMPLETE")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/ch03/varmion_v2"
    if config["test_evaluations"] != 0 or config["schema_version"] != "ch03-varmion-v2":
        raise ValueError("Protocol changed")
    identity = {
        "sources": sources(),
        "torch": str(torch.__version__),
        "config": config,
        "v1_prepared_manifest": sha(ROOT / "results/ch03/varmion_v1/prepared_manifest.json"),
        "data_manifest": sha(ROOT / "data/ch03/varmion_v1/manifest.json"),
    }
    frozen = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_frozen_release.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if frozen.returncode:
        raise RuntimeError("Frozen code changed")
    if args.resume:
        if json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("Run identity changed")
        if (out / "VARMION_V2_COMPLETE").exists():
            verify(out, "VARMION_V2_COMPLETE")
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "identity.json", identity)
        dump(out / "config.json", config)
        shutil.copytree(HERE, out / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    with exclusive_run_lock(out):
        prepared = (
            reload_prepared(out)
            if (out / "prepared_manifest.json").exists()
            else prepare(out, config)
        )
        data, geometry, phi, k0, norm, arrays, pool = prepared
        dump(out / "frozen_before.json", json.loads(frozen.stdout))
        if args.prepare_only:
            print("V2_PREPARED", flush=True)
            return
        if (
            not (out / "preflight.json").exists()
            or not json.loads((out / "preflight.json").read_text())["passed"]
        ):
            raise RuntimeError("Tests and independent operator audit must pass before training")
        rows = []

        def run(variant, seed):
            spec = config["variants"][variant]
            path = out / variant / f"seed_{seed}"
            run_identity = {
                **identity,
                "variant": variant,
                "prepared_manifest": sha(out / "prepared_manifest.json"),
            }
            factory = lambda: MatrixMap(spec["kind"], norm["c_mean"], norm["c_std"], k0)
            if (path / "COMPLETE").exists():
                verify(path, "COMPLETE")
                result = json.loads((path / "metrics.json").read_text())
            else:
                count = spec["operator_groups"]
                subset = {k: v[:count] for k, v in pool.items()} if count else None
                dump(
                    out / "status.json",
                    dict(status="training", variant=variant, seed=seed, completed_runs=len(rows)),
                )
                fit(path, config, seed, factory, arrays, subset, run_identity, resume=path.exists())
                result = finish(path, spec["kind"], norm, k0, data, geometry, phi, arrays)
            row = {
                k: result[k]
                for k in [
                    "variant",
                    "seed",
                    "parameter_count",
                    "best_epoch",
                    "optimizer_steps",
                    "operator_visits",
                    "elapsed_seconds",
                ]
            }
            row.update(
                train_l2=result["metrics"]["train"]["mean"]["relative_l2"],
                val_l2=result["metrics"]["val"]["mean"]["relative_l2"],
                val_h1=result["metrics"]["val"]["mean"]["relative_h1_seminorm"],
                val_skew=result["matrix_diagnostics"]["val"]["relative_skew_mean"],
            )
            rows.append(row)
            dump(out / "progress.json", rows)
            print("V2_RUN_COMPLETE", json.dumps(row), flush=True)
            return row

        screen = [run(name, config["screen_seed"]) for name in config["variants"]]
        candidates = [row for row in screen if row["variant"] != "dense_data"]
        selected = min(candidates, key=lambda row: row["val_l2"])["variant"]
        selection = dict(
            screen_seed=config["screen_seed"],
            selected=selected,
            screen=screen,
            note="Exploratory validation selection; seed0 participates in selection. No independent test claim.",
        )
        if (out / "selection.json").exists() and json.loads(
            (out / "selection.json").read_text()
        ) != selection:
            raise RuntimeError("Selection changed")
        dump(out / "selection.json", selection)
        for seed in config["confirmation_seeds"]:
            run("dense_data", seed)
            run(selected, seed)
        if sources() != identity["sources"]:
            raise RuntimeError("Source drift")
        groups = {}
        for name in config["variants"]:
            subset = [r for r in rows if r["variant"] == name]
            groups[name] = {
                "count": len(subset),
                **{
                    key: {
                        "mean": float(np.mean([r[key] for r in subset])),
                        "std_population": float(np.std([r[key] for r in subset])),
                    }
                    for key in ["train_l2", "val_l2", "val_h1", "elapsed_seconds"]
                },
            }
        dump(
            out / "aggregate.json",
            dict(selected=selected, groups=groups, rows=rows, test_evaluations=0),
        )
        dump(
            out / "status.json",
            dict(status="trained_pending_final_audit", completed_runs=len(rows), selected=selected),
        )
        print("V2_ALL_TRAINED", selected, flush=True)


if __name__ == "__main__":
    main()
