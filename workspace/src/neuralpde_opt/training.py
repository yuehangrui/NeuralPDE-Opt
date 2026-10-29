"""Checkpointed Adam/L-BFGS training with independent diagnostics.

Optimization structure adapted from ocpde_dl/pinn/train.py. Objectives are
scalars, so Ritz is minimized directly rather than squared as a residual.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
import torch
from .problems import Poisson
from .mesh import make_mesh
from .models import CoordinateMLP
from .objectives import Objective
from .metrics import evaluate


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def environment():
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                  cpu_threads=torch.get_num_threads())
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        result["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    root = Path(__file__).resolve().parents[2]
    result["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((root/"src").rglob("*.py"))}
    return result


def run_one(config, dim, method, seed, output, device):
    p = Poisson(dim)
    pc = config["problems"][str(dim)]
    out = Path(output)/f"{p.name}_{method}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(config=config, dim=dim, method=method, seed=seed)
    config_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if (out/"metrics.json").exists():
        previous = json.loads((out/"metrics.json").read_text())
        if previous["config_id"] != config_id:
            raise ValueError(f"Configuration differs from completed run: {out}")
        print(f"SKIP completed {out.name}", flush=True)
        return previous
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, config["dtype"])
    model = CoordinateMLP(p, pc["width"], pc["depth"]).to(device=device, dtype=dtype)
    adam = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    block = config["lbfgs_block"]
    bfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=block, max_eval=block*3,
                             tolerance_grad=1e-11, tolerance_change=1e-14,
                             history_size=50, line_search_fn="strong_wolfe")
    obj = Objective(p, make_mesh(dim, pc["train_n"]), config["quadrature_order"], method, device, dtype)
    checkpoint = out/"checkpoint.pt"
    state = dict(adam_done=0, lbfgs_done=0, training_seconds=0., objective_evaluations=0,
                 history=[], config_id=config_id)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["progress"]["config_id"] != config_id:
            raise ValueError(f"Configuration differs from checkpoint: {out}")
        model.load_state_dict(saved["model"])
        adam.load_state_dict(saved["adam"])
        bfgs.load_state_dict(saved["lbfgs"])
        state = saved["progress"]
        print(f"RESUME {out.name} Adam={state['adam_done']} LBFGS={state['lbfgs_done']}", flush=True)
    save_json(out/"config.json", signature)

    def save_checkpoint():
        torch.save(dict(model=model.state_dict(), adam=adam.state_dict(), lbfgs=bfgs.state_dict(),
                        progress=state), out/"checkpoint.tmp")
        os.replace(out/"checkpoint.tmp", checkpoint)
        save_json(out/"history.json", state["history"])

    def log(stage):
        val = evaluate(model, p, pc["validation_n"], config["validation_order"], strong=False)
        loss = float(obj(model).detach().cpu())
        row = dict(stage=stage, adam_step=state["adam_done"], lbfgs_budget=state["lbfgs_done"],
                   training_seconds=state["training_seconds"], objective_evaluations=state["objective_evaluations"],
                   loss=loss, l2_rel=val["l2_rel"], h1_semi_rel=val["h1_semi_rel"])
        if not all(math.isfinite(row[k]) for k in ("loss", "l2_rel", "h1_semi_rel")):
            raise FloatingPointError(f"Non-finite diagnostics: {row}")
        state["history"].append(row)
        print(f"{out.name} {stage} Adam={state['adam_done']} BFGS={state['lbfgs_done']} "
              f"loss={loss:.4e} L2={val['l2_rel']:.3e} H1={val['h1_semi_rel']:.3e} "
              f"train_s={state['training_seconds']:.1f}", flush=True)
        save_checkpoint()

    if not state["history"]:
        log("initial")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while state["adam_done"] < config["adam_steps"]:
        end = min(state["adam_done"]+config["log_every"], config["adam_steps"])
        sync(device)
        start = time.perf_counter()
        while state["adam_done"] < end:
            adam.zero_grad(set_to_none=True)
            loss = obj(model)
            loss.backward()
            adam.step()
            state["adam_done"] += 1
            state["objective_evaluations"] += 1
        sync(device)
        state["training_seconds"] += time.perf_counter()-start
        log("adam")
    while state["lbfgs_done"] < config["lbfgs_steps"]:
        count = min(block, config["lbfgs_steps"]-state["lbfgs_done"])
        bfgs.param_groups[0]["max_iter"] = count
        bfgs.param_groups[0]["max_eval"] = 3*count

        def closure():
            bfgs.zero_grad(set_to_none=True)
            loss = obj(model)
            loss.backward()
            state["objective_evaluations"] += 1
            return loss

        sync(device)
        start = time.perf_counter()
        bfgs.step(closure)
        sync(device)
        state["training_seconds"] += time.perf_counter()-start
        state["lbfgs_done"] += count
        log("lbfgs")
    sync(device)
    start = time.perf_counter()
    metrics = evaluate(model, p, pc["validation_n"], config["validation_order"],
                       weak_n=pc["weak_validation_n"])
    sync(device)
    metrics["evaluation_seconds"] = time.perf_counter()-start
    # A second, refined integration verifies final metrics, not just training loss.
    refined = evaluate(model, p, pc["validation_n"]+1, config["validation_order"]+1, strong=False)
    metrics["quadrature_l2_relative_change"] = abs(refined["l2_rel"]-metrics["l2_rel"])/max(metrics["l2_rel"], 1e-15)
    metrics["quadrature_h1_relative_change"] = abs(refined["h1_semi_rel"]-metrics["h1_semi_rel"])/max(metrics["h1_semi_rel"], 1e-15)
    first_param = next(iter(model.parameters()))
    metrics.update(problem=p.name, dim=dim, method=method, seed=seed, config_id=config_id,
                   width=pc["width"], depth=pc["depth"], n_params=sum(t.numel() for t in model.parameters()),
                   training_points=len(obj.x), training_test_dofs=len(obj.mesh.interior) if method == "vpinn" else None,
                   training_seconds=state["training_seconds"], objective_evaluations=state["objective_evaluations"],
                   lbfgs_actual_iterations=int(bfgs.state[first_param].get("n_iter", 0)),
                   final_training_loss=float(obj(model).detach().cpu()),
                   device=str(device), dtype=config["dtype"],
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0)
    # Final portable weights contain no device-specific closures.
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out/"model.pt")
    save_json(out/"metrics.json", metrics)
    print(f"COMPLETE {out.name}", flush=True)
    return metrics
