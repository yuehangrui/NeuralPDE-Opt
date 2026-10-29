"""Train one declared case in a new directory, independent of historical queues."""

import copy
import numpy as np
import torch
from bootstrap import ROOT, module
from integrity import read, require, dump, sha, verify, check_manifest
from registry import runs, source_manifest
from predictors import array_file
from validation import validate_fields
from neuralpde_opt.chapter3.operator_training import P1Geometry, exclusive_run_lock


def run(identifier, output, resume=False, smoke_epochs=None):
    row = next((r for r in runs() if r["id"] == identifier), None)
    require(row is not None, "Unknown training case")
    output = output.resolve()
    base = (ROOT / row["path"]).resolve().parent
    require(
        output == base,
        "Use the declared independent workspace run directory",
    )
    require(
        smoke_epochs is None or type(smoke_epochs) is int and 1 <= smoke_epochs < row["epochs"],
        "Invalid smoke budget",
    )
    epochs = smoke_epochs or row["epochs"]
    identity = dict(
        case=row, epochs=epochs, smoke_only=smoke_epochs is not None, sources=source_manifest()
    )
    for name in ["reaction_v1", "varmion_v1", "hints_v1"]:
        identity[name] = sha(ROOT / f"data/ch03/{name}/manifest.json")
        check_manifest(ROOT / f"data/ch03/{name}", read(ROOT / f"data/ch03/{name}/manifest.json"))
    if resume:
        require(read(output / "job.json") == identity, "Training job identity changed")
        if (output / "run/COMPLETE").exists():
            verify(output / "run", strict=row["family"] != "baseline")
            print("VERIFIED_COMPLETE_NO_TRAINING")
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        dump(output / "job.json", identity)
    path = output / "run"
    with exclusive_run_lock(output):
        family = row["family"]
        if family == "baseline":
            from neuralpde_opt.chapter3 import reaction_training

            config = read(ROOT / "configs/ch03_reaction_training_v1.json")
            config["epochs"] = epochs
            reaction_training.run(
                config,
                ROOT / "data/ch03/reaction_v1",
                path,
                row["variant"],
                row["seed"],
                resume=resume,
            )
            verify(path, strict=False)
            return
        dataset = "reaction_v1" if row["dataset"] == "fixed" else "varmion_v1"
        geometry = P1Geometry(ROOT / f"data/ch03/{dataset}/observation_geometry.npz")
        data = {s: array_file(ROOT / f"data/ch03/{dataset}/{s}.npz") for s in ["train", "val"]}
        for d in data.values():
            validate_fields(d)
        config_path = "ch03_varmion_v2" if family == "varmion_v2" else f"ch03_{family}_v1"
        config = read(ROOT / f"experiments/{config_path}/config.json")
        config["epochs"] = epochs
        if family == "pod":
            from pod import prepare_arrays

            engine = module("ch03_pod_train", "experiments/ch03_pod_v1/train.py")
            basis = array_file(ROOT / "results/ch03/pod_v1/basis.npz")
            norm = read(ROOT / "results/ch03/pod_v1/normalization.json")
            rank = int(row["variant"].split("_")[1])
            arrays = prepare_arrays(data, basis, geometry, norm, rank, config["sensor_stride"])
            state = engine.train_one(
                path,
                config,
                arrays,
                np.sqrt(basis["eigenvalues"][:rank]),
                rank,
                row["seed"],
                identity,
                resume=resume,
            )
            engine.finish_one(path, state, config, arrays, basis, rank, data, geometry)
        elif family == "physics":
            from physics import P1System

            engine = module("ch03_physics_run", "experiments/ch03_physics_v1/run.py")
            case = next(c for c in config["cases"] if c["name"] == row["variant"])
            system = P1System(64, 0.01)
            norm = engine.normalize_inputs(data, geometry)
            state = engine.train_one(
                path,
                config,
                case,
                row["seed"],
                data,
                geometry,
                system,
                norm,
                identity,
                resume=resume,
            )
            engine.finish_one(path, state, config, case, data, geometry, system, "cuda")
        elif family == "varmion_v2":
            from prepare_v2 import reload_prepared
            from model_v2 import MatrixMap
            from trainer_v2 import fit

            engine = module("ch03_varmion2_run", "experiments/ch03_varmion_v2/run_v2.py")
            data, geometry, phi, k0, norm, arrays, pool = reload_prepared(
                ROOT / "results/ch03/varmion_v2"
            )
            spec = config["variants"][row["variant"]]
            count = spec["operator_groups"]
            subset = {k: v[:count] for k, v in pool.items()} if count else None
            factory = lambda: MatrixMap(spec["kind"], norm["c_mean"], norm["c_std"], k0)
            fit(
                path,
                config,
                row["seed"],
                factory,
                arrays,
                subset,
                {**identity, "variant": row["variant"]},
                resume=resume,
            )
            engine.finish(path, spec["kind"], norm, k0, data, geometry, phi, arrays)
        else:
            from model import model_for
            from train_coeff import fit

            prepared = ROOT / "results/ch03/varmion_v1"
            check_manifest(prepared, read(prepared / "prepared_manifest.json"))
            basis = array_file(prepared / "basis.npz")
            phi, k0 = basis["basis"], basis["k0"]
            norm = read(prepared / "normalization.json")
            if family == "varmion":
                engine = module("ch03_varmion_run", "experiments/ch03_varmion_v1/run.py")
                arrays = {
                    s: array_file(prepared / (s + "_coordinates.npz")) for s in ["train", "val"]
                }
                factory = lambda: model_for(row["variant"], norm, k0)
                state = fit(path, config, row["seed"], factory, arrays, identity, resume=resume)
                engine.finish(
                    path,
                    state,
                    row["variant"],
                    config,
                    data,
                    geometry,
                    phi,
                    k0,
                    norm,
                    arrays,
                    "cuda",
                )
            else:
                # Import the historical HINTS data/finish routines in their own namespace.
                import sys

                sys.path.insert(0, str(ROOT / "experiments/ch03_hints_v1"))
                engine = module("ch03_hints_run", "experiments/ch03_hints_v1/run.py")
                residual = module(
                    "ch03_residual_data", "experiments/ch03_hints_v1/residual_data.py"
                )
                data, arrays = residual.load(ROOT / "data/ch03/hints_v1")
                checkpoint = torch.load(
                    prepared / f'varmion/seed_{row["seed"]}/run/best.pt',
                    map_location="cpu",
                    weights_only=True,
                )

                def factory():
                    model = model_for("varmion", norm, k0)
                    model.load_state_dict(checkpoint["model"])
                    return model

                state = fit(path, config, row["seed"], factory, arrays, identity, resume=resume)
                engine.finish(path, state, factory, arrays, data, phi, geometry)
        verify(path)
        require(source_manifest() == identity["sources"], "Sources changed during training")
        dump(
            output / "finished.json",
            dict(passed=True, epochs=epochs, smoke_only=smoke_epochs is not None),
        )
