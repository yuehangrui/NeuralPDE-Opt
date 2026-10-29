"""Restricted Chapter 3 supervised-operator smoke workflow.

All norms refer to P1 reconstruction on the public observation triangulation,
not continuum error or the separately saved finer reference solutions. Only
train.npz, val.npz, test.npz (a/f/y/sample_id), observation_geometry.npz and
dataset metadata are read. This entry point never generates data.

Book/code notation:
P1Geometry defines consistent mass-weighted L2 norms and gradient seminorms. Backbone inputs a/f are generic features in reaction adapters; they do not turn c into diffusion.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
import torch
from torch import nn


def dump_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_write(path, value):
    """Commit one text/torch artifact with rename; remove partial temp files."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if isinstance(value, str):
                handle.write(value.encode("utf-8"))
            else:
                torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive_run_lock(output):
    """An advisory process lock held through training and finalization."""
    with (Path(output) / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process holds this operator run lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_data_manifest(data_dir):
    """Check serialized public splits and geometry before any training starts.

    Hashing test.npz verifies bytes only; its arrays remain unopened until
    validation checkpoint selection has finished.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name in ("train.npz", "val.npz", "test.npz", "observation_geometry.npz"):
        if name not in manifest or sha256(data_dir / name) != manifest[name]:
            raise ValueError(f"Dataset manifest checksum mismatch: {name}")
    # A current manifest alone does not prove that these bytes passed the data
    # quality gates. Bind it to the committed certificate, without loading test.
    try:
        marker = (data_dir / "COMPLETE").read_text()
        config = json.loads((data_dir / "config.json").read_text())
        quality = json.loads((data_dir / "quality.json").read_text())
        audit = json.loads((data_dir / "reload_audit.json").read_text())
        for name in ("config.json", "quality.json"):
            if manifest.get(name) != sha256(data_dir / name):
                raise ValueError("Dataset quality metadata checksum mismatch: " + name)
        if (
            quality.get("passed") is not True
            or audit.get("passed") is not True
            or audit.get("failures") != []
        ):
            raise ValueError("Dataset quality audit did not pass")
        counts = config["splits"]
        if set(counts) != {"train", "val", "test"} or any(
            type(n) is not int or n < 1 for n in counts.values()
        ):
            raise ValueError("Invalid certified split counts")
        if any(audit["split_statistics"][s]["count"] != n for s, n in counts.items()):
            raise ValueError("Quality audit split counts mismatch")
        legacy_marker = "Pilot quality and serialized-data audit passed. No models trained.\n"
        if marker == legacy_marker:
            if (
                config.get("purpose") != "pilot_only_no_training"
                or (data_dir / "identity.json").exists()
            ):
                raise ValueError("Legacy pilot marker cannot authorize a modern dataset")
            return {
                "workflow": "legacy_pilot",
                "certificate_binding": "legacy metadata and public hashes",
            }
        certificate = json.loads(marker)
        workflow = "ch03-resumable-fields-v1"
        config_hash = hashlib.sha256(
            json.dumps(
                config, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        identity = json.loads((data_dir / "identity.json").read_text())
        if (
            certificate.get("workflow") != workflow
            or identity.get("workflow") != workflow
            or certificate.get("config_sha256") != config_hash
            or identity.get("config_sha256") != config_hash
            or manifest.get("identity.json") != sha256(data_dir / "identity.json")
        ):
            raise ValueError("Dataset certificate configuration identity mismatch")
        if certificate.get("manifest_sha256") != sha256(data_dir / "manifest.json"):
            raise ValueError("Dataset COMPLETE manifest checksum mismatch")
        if certificate.get("reload_audit_sha256") != sha256(data_dir / "reload_audit.json"):
            raise ValueError("Dataset COMPLETE reload audit checksum mismatch")
        if audit.get("independent_weak_residual_samples") != sum(counts.values()):
            raise ValueError("Dataset quality audit sample coverage mismatch")
        return {
            "workflow": workflow,
            "certificate_binding": "COMPLETE manifest/config/reload audit verified",
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Missing or malformed dataset quality certificate") from error


def read_split(data_dir, split):
    """Load only public field arrays, without inspecting finer-grid labels."""
    with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as z:
        result = {k: z[k].copy() for k in ("a", "f", "y", "sample_id")}
    shape = result["a"].shape
    if len(shape) != 3 or shape[1] != shape[2] or shape[1] < 3:
        raise ValueError("Expected [sample, x-node, y-node] square public fields")
    if any(result[k].shape != shape or not np.isfinite(result[k]).all() for k in ("a", "f", "y")):
        raise ValueError("Inconsistent or non-finite public fields")
    if len(result["sample_id"]) != shape[0] or len(set(result["sample_id"])) != shape[0]:
        raise ValueError("Missing or repeated sample IDs")
    return result


class P1Geometry:
    """Exact L2 mass and piecewise constant P1-gradient integration."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.coordinates = z["coordinates"].copy()
            self.elements = z["elements"].copy()
            self.gradients = z["gradients"].copy()
            self.areas = z["determinants"].copy() / 2
            self.mass = sparse.csr_matrix(
                (z["mass_data"], z["mass_indices"], z["mass_indptr"]), shape=tuple(z["mass_shape"])
            )
            self.weights = np.asarray(self.mass.sum(axis=1)).ravel()
            self.boundary = np.ones(self.mass.shape[0], dtype=bool)
            self.boundary[z["interior"]] = False
        self.grid_shape = self.coordinates.shape[:2]
        if abs(self.weights.sum() - 1) > 1e-10 or np.min(self.areas) <= 0:
            raise ValueError("Expected a valid unit-square P1 geometry")

    def l2_squared(self, values):
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.mass.shape[0])
        return np.maximum(np.einsum("bi,bi->b", flat, (self.mass @ flat.T).T), 0.0)

    def h1_squared(self, values):
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.mass.shape[0])
        grad = np.einsum("bea,ead->bed", flat[:, self.elements], self.gradients)
        return np.maximum(np.einsum("bed,bed,e->b", grad, grad, self.areas), 0.0)

    def torch_loss(self, device, dtype=torch.float32):
        """Return exact consistent-mass P1 squared norm, per batch sample."""
        elements = torch.as_tensor(self.elements, dtype=torch.long, device=device)
        area_over_12 = torch.as_tensor(self.areas / 12, dtype=dtype, device=device)

        def norm_squared(values):
            local = values.flatten(1)[:, elements]
            return ((local.sum(-1).square() + local.square().sum(-1)) * area_over_12).sum(-1)

        return norm_squared


def fit_normalization(train, geometry):
    """Fit scalar input mean/std and output RMS from training fields only."""
    stats = {
        "fit_split": "train",
        "fit_sample_ids": [str(s) for s in train["sample_id"]],
        "integration": "consistent P1 mass; mean uses exact mass row sums",
        "output_centered": False,
    }
    for key in ("a", "f"):
        values = train[key].reshape(len(train[key]), -1)
        mean = float(np.mean(values @ geometry.weights))
        variance = float(np.mean(geometry.l2_squared(train[key] - mean)))
        if variance <= 1e-24:
            raise ValueError(f"{key} has degenerate training variance")
        stats[f"{key}_mean"], stats[f"{key}_std"] = mean, float(np.sqrt(variance))
    stats["y_rms"] = float(np.sqrt(np.mean(geometry.l2_squared(train["y"]))))
    if stats["y_rms"] <= 1e-12:
        raise ValueError("Training output RMS is too small")
    return stats


def evaluate_arrays(prediction, target, geometry, sample_ids=None):
    """Evaluate physical values using exact public-grid P1 L2/H1 integrals."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.shape[1:] != geometry.grid_shape:
        raise ValueError("Prediction/target/geometry shapes do not agree")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite prediction or target")
    error = prediction - target
    l2 = np.sqrt(geometry.l2_squared(error))
    h1 = np.sqrt(geometry.h1_squared(error))
    tiny = np.finfo(np.float64).tiny
    rel_l2 = l2 / np.maximum(np.sqrt(geometry.l2_squared(target)), tiny)
    rel_h1 = h1 / np.maximum(np.sqrt(geometry.h1_squared(target)), tiny)
    boundary = np.max(np.abs(prediction.reshape(len(prediction), -1)[:, geometry.boundary]), axis=1)
    arrays = {
        "absolute_l2": l2,
        "relative_l2": rel_l2,
        "absolute_h1_seminorm": h1,
        "relative_h1_seminorm": rel_h1,
        "boundary_max": boundary,
    }
    ids = (
        list(map(str, sample_ids)) if sample_ids is not None else list(map(str, range(len(target))))
    )
    if len(ids) != len(target):
        raise ValueError("Sample ID count differs from batch size")
    return {
        "count": len(target),
        "mean": {k: float(v.mean()) for k, v in arrays.items()},
        "max": {k: float(v.max()) for k, v in arrays.items()},
        "per_sample": [
            {"sample_id": sid, **{k: float(v[i]) for k, v in arrays.items()}}
            for i, sid in enumerate(ids)
        ],
    }


FACTORIES = {
    "fno": ("neuralpde_opt.chapter3.fno", "FNO2d"),
    "deeponet": ("neuralpde_opt.chapter3.deeponet", "DeepONet"),
    "gno": ("neuralpde_opt.chapter3.gno", "GNO2d"),
    "transformer": ("neuralpde_opt.chapter3.transformer", "TransformerOperator"),
}
DTYPES = {"float32": torch.float32, "float64": torch.float64}
SCHEMA_VERSION = 2
SMOKE_PURPOSE = "chapter3_smoke_only_no_method_ranking"
OVERFIT_PURPOSE = "chapter3_overfit_subset_diagnostic_no_method_ranking"
EXPERIMENT_PURPOSE = "chapter3_supervised_experiment_v1"
OVERFIT_EXPERIMENT_PURPOSE = "chapter3_overfit_experiment_v1"


def is_experiment(config):
    return config["purpose"] in (EXPERIMENT_PURPOSE, OVERFIT_EXPERIMENT_PURPOSE)


def is_overfit(config):
    return config["purpose"] in (OVERFIT_PURPOSE, OVERFIT_EXPERIMENT_PURPOSE)


def split_limits(config):
    return (1024, 256, 256) if is_experiment(config) else (64, 32, 32)


def objective_per_sample(prediction, target, norm_squared, y_rms, objective="absolute_l2_squared"):
    """Absolute scaled P1 L2 squared, or each sample's relative P1 L2 squared."""
    if objective == "absolute_l2_squared":
        return norm_squared((prediction - target) / y_rms)
    if objective == "relative_l2_squared":
        denominator = norm_squared(target)
        if (denominator <= 0).any():
            raise ValueError(
                "Relative L2 objective requires every training target to have positive norm"
            )
        return norm_squared(prediction - target) / denominator
    if objective in ("nodal_mse", "scaled_nodal_mse"):
        residual = prediction - target
        if objective == "scaled_nodal_mse":
            residual = residual / y_rms
        return residual.flatten(1).square().mean(1)
    raise ValueError("Unknown supervised objective: " + str(objective))


def epoch_mode(config):
    return config.get("legacy_full_epochs") is not None or config.get("epoch_training") is not None


def epoch_scheduler(optimizer, profile, epochs):
    if profile["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=profile.get("patience", 30),
            factor=profile.get("factor", 0.5),
            min_lr=profile.get("min_lr", 0.0),
            threshold=profile.get("threshold", 1e-4),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=profile.get("min_lr", 0.0)
    )


def batch_update(
    model, arrays, index, microbatch, optimizer, norm_squared, y_rms, objective, grad_clip=None
):
    """One logical update, including an uneven last microbatch.

    Only the logical batch defines the optimizer trajectory; microbatch is a
    memory limit. Clip once after all weighted gradients have accumulated.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = torch.zeros((), dtype=arrays["y"].dtype, device=arrays["y"].device)
    for start in range(0, len(index), microbatch):
        idx = index[start : start + microbatch]
        prediction = model(arrays["a"][idx], arrays["f"][idx])
        part = objective_per_sample(
            prediction, arrays["y"][idx], norm_squared, y_rms, objective
        ).mean() * (len(idx) / len(index))
        if not torch.isfinite(part):
            raise RuntimeError("Non-finite epoch loss")
        part.backward()
        loss += part.detach()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Non-finite epoch gradient")
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss


class OperatorScheduler:
    """Explicit optimizer-step warmup/cosine or selection-metric plateau LR."""

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.spec = config.get("scheduler", {"name": "constant"})
        self.name = self.spec["name"]
        self.total_steps = config["steps"]
        self.base_lr = config["learning_rate"]
        self.optimizer_steps = 0
        self.metric_updates = 0
        self.last_optimizer_learning_rate = None
        self.plateau = None
        if self.name == "plateau":
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.spec.get("factor", 0.5),
                patience=self.spec.get("patience", 5),
                min_lr=self.spec.get("min_lr", 0.0),
                threshold=self.spec.get("threshold", 1e-4),
            )

    def before_step(self, step):
        if step != self.optimizer_steps + 1:
            raise ValueError("Scheduler/optimizer step count mismatch")
        if self.name == "warmup_cosine":
            warmup = self.spec.get("warmup_steps", 0)
            minimum = self.spec.get("min_lr", 0.0)
            if warmup and step <= warmup:
                learning_rate = self.base_lr * step / warmup
            else:
                progress = (
                    (step - warmup) / (self.total_steps - warmup)
                    if warmup
                    else (step - 1) / max(self.total_steps - 1, 1)
                )
                learning_rate = minimum + (self.base_lr - minimum) * 0.5 * (
                    1 + np.cos(np.pi * progress)
                )
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate)
        self.optimizer_steps = step
        self.last_optimizer_learning_rate = float(self.optimizer.param_groups[0]["lr"])

    def after_validation(self, metric):
        self.metric_updates += 1
        if self.plateau is not None:
            self.plateau.step(metric)

    def state_dict(self):
        plateau = self.plateau.state_dict() if self.plateau is not None else None
        if plateau is not None:
            # ReduceLROnPlateau keeps a +inf sentinel even after its first
            # metric. Encode it explicitly so JSON remains standards compliant.
            plateau = {
                key: (
                    ("Infinity" if value > 0 else "-Infinity")
                    if isinstance(value, float) and not np.isfinite(value)
                    else value
                )
                for key, value in plateau.items()
            }
        return {
            "spec": self.spec,
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "optimizer_steps": self.optimizer_steps,
            "metric_updates": self.metric_updates,
            "last_optimizer_learning_rate": self.last_optimizer_learning_rate,
            "learning_rates": [float(group["lr"]) for group in self.optimizer.param_groups],
            "plateau_state": plateau,
        }

    def load_state_dict(self, state):
        if (
            state["spec"] != self.spec
            or state["total_steps"] != self.total_steps
            or state["base_lr"] != self.base_lr
        ):
            raise ValueError("Resume scheduler configuration mismatch")
        self.optimizer_steps = state["optimizer_steps"]
        self.metric_updates = state["metric_updates"]
        self.last_optimizer_learning_rate = state["last_optimizer_learning_rate"]
        if self.plateau is not None:
            plateau = {
                key: (
                    (float("inf") if value == "Infinity" else float("-inf"))
                    if isinstance(value, str) and value in ("Infinity", "-Infinity")
                    else value
                )
                for key, value in state["plateau_state"].items()
            }
            self.plateau.load_state_dict(plateau)
        actual = [float(group["lr"]) for group in self.optimizer.param_groups]
        if actual != state["learning_rates"]:
            raise ValueError("Resume scheduler/optimizer learning-rate mismatch")


def make_model(spec):
    if spec["name"] not in FACTORIES:
        raise ValueError("Supported operators: " + ", ".join(FACTORIES))
    module, name = FACTORIES[spec["name"]]
    return getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))


class PhysicalOperator(nn.Module):
    """Raw physical a/f -> physical y; no centering of zero-boundary output."""

    def __init__(self, model, normalization, dtype=torch.float32):
        super().__init__()
        self.model = model.to(dtype=dtype)
        for key in ("a_mean", "a_std", "f_mean", "f_std", "y_rms"):
            self.register_buffer(key, torch.tensor(normalization[key], dtype=dtype))

    @property
    def supports_queries(self):
        # The fallback preserves loading first-version DeepONet checkpoints.
        return getattr(
            self.model,
            "supports_queries",
            self.model.__class__.__name__ in ("DeepONet", "TransformerOperator"),
        )

    def representation_metadata(self):
        method = getattr(self.model, "representation_metadata", None)
        return method() if method is not None else {"model": self.model.__class__.__name__}

    def forward(self, a, f, query_points=None, **query_options):
        if (query_points is not None or query_options) and not self.supports_queries:
            raise ValueError(f"{self.model.__class__.__name__} does not support coordinate queries")
        a, f = (a - self.a_mean) / self.a_std, (f - self.f_mean) / self.f_std
        if query_points is None and not query_options:
            return self.model(a, f) * self.y_rms
        return self.model(a, f, query_points=query_points, **query_options) * self.y_rms


def load_trained_model(run_dir, device="cpu"):
    """Load best checkpoint; old run configs default to float32.

    Returns eval-mode PhysicalOperator accepting raw a/f and returning physical
    y, with optional independent queries for DeepONet and Transformer.
    """
    run_dir = Path(run_dir)
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        for name in ("best.pt", "config.json", "normalization.json"):
            if name not in manifest or sha256(run_dir / name) != manifest[name]:
                raise ValueError("Model artifact checksum mismatch: " + name)
    config = json.loads((run_dir / "config.json").read_text())
    norm = json.loads((run_dir / "normalization.json").read_text())
    dtype_name = config.get("dtype", "float32")
    result = PhysicalOperator(make_model(config["model"]), norm, DTYPES[dtype_name]).to(device)
    payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if payload.get("schema_version") == SCHEMA_VERSION:
        if (
            payload["model_spec"] != config["model"]
            or payload["normalization"] != norm
            or payload["dtype"] != dtype_name
        ):
            raise ValueError("Best checkpoint/config/normalization mismatch")
    result.model.load_state_dict(payload["model_state"])
    return result.eval()


def predict(model, split, device, batch_size):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(split["y"]), batch_size):
            a, f = [
                torch.as_tensor(
                    split[k][start : start + batch_size], dtype=model.a_mean.dtype, device=device
                )
                for k in ("a", "f")
            ]
            values.append(model(a, f).cpu().numpy())
    return np.concatenate(values)


def validate_config(config):
    if config.get("purpose") not in (
        SMOKE_PURPOSE,
        OVERFIT_PURPOSE,
        EXPERIMENT_PURPOSE,
        OVERFIT_EXPERIMENT_PURPOSE,
    ):
        raise ValueError("An explicit smoke, experiment, or subset-overfit purpose is required")
    if (
        not isinstance(config.get("models"), list)
        or not config["models"]
        or any(not isinstance(s, dict) for s in config["models"])
    ):
        raise ValueError("models must be a nonempty list of model specifications")
    adapted = config.get("epoch_training") is not None
    if adapted:
        if "learning_rate" in config:
            raise ValueError("Adapted learning_rate belongs only in each model training profile")
        if config.get("legacy_full_epochs") is not None:
            raise ValueError("epoch_training and legacy_full_epochs are mutually exclusive")
        if (
            not isinstance(config["epoch_training"], dict)
            or set(config["epoch_training"]) != {"epochs"}
            or type(config["epoch_training"]["epochs"]) is not int
            or config["epoch_training"]["epochs"] != config.get("steps")
            or not 1 <= config["steps"] <= 5000
        ):
            raise ValueError("epoch_training must contain integer epochs equal to steps in 1..5000")
        if not config.get("defer_test"):
            raise ValueError("Adapted training requires defer_test=True")
        if config.get("objective") not in (
            "relative_l2_squared",
            "absolute_l2_squared",
            "scaled_nodal_mse",
        ):
            raise ValueError("Adapted training requires an explicit dimensionless objective")
        for spec in config["models"]:
            p = spec.get("training")
            if "legacy_training" in spec or not isinstance(p, dict):
                raise ValueError("Adapted models require training, without legacy_training")
            allowed = {
                "optimizer",
                "learning_rate",
                "batch_size",
                "scheduler",
                "weight_decay",
                "grad_clip",
                "patience",
                "factor",
                "min_lr",
                "threshold",
            }
            if set(p) - allowed:
                raise ValueError("Unknown adapted training profile fields")
            if p.get("optimizer") not in ("Adam", "AdamW") or p.get("scheduler") not in (
                "plateau",
                "cosine",
            ):
                raise ValueError("Adapted profile requires Adam/AdamW and plateau/cosine")
            if type(p.get("batch_size")) is not int or not 1 <= p["batch_size"] <= 1024:
                raise ValueError("Logical batch_size must be an integer in 1..1024")
            for key, default in (
                ("learning_rate", None),
                ("weight_decay", 0.0),
                ("min_lr", 0.0),
                ("threshold", 1e-4),
            ):
                value = p.get(key, default)
                if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid adapted " + key)
            if p["learning_rate"] <= 0 or p.get("min_lr", 0.0) > p["learning_rate"]:
                raise ValueError("Invalid adapted learning-rate range")
            if (
                type(p.get("patience", 30)) is not int
                or p.get("patience", 30) < 0
                or type(p.get("factor", 0.5)) not in (int, float)
                or not 0 < p.get("factor", 0.5) < 1
            ):
                raise ValueError("Invalid adapted plateau parameters")
            clip = p.get("grad_clip")
            if clip is not None and (
                type(clip) not in (int, float) or not np.isfinite(clip) or clip <= 0
            ):
                raise ValueError("grad_clip must be positive and finite")
    elif any("training" in spec for spec in config["models"]):
        raise ValueError("Model training profiles require epoch_training")
    if config.get("legacy_full_epochs") is None and any(
        "legacy_training" in s for s in config["models"]
    ):
        raise ValueError("legacy_training requires legacy_full_epochs")
    if epoch_mode(config) and config.get("scheduler", {"name": "constant"}) != {"name": "constant"}:
        raise ValueError(
            "Epoch training uses only per-model schedulers; global scheduler must be constant"
        )
    if config.get("legacy_full_epochs") is not None:
        if config["legacy_full_epochs"] != config.get("steps") or not 1 <= config["steps"] <= 5000:
            raise ValueError("Full epochs must equal steps and lie in 1..5000")
        for spec in config.get("models", []):
            profile = spec.get("legacy_training", {})
            if (
                profile.get("optimizer") not in ("Adam", "AdamW")
                or profile.get("batch_size", 0) < 128
            ):
                raise ValueError("Full profiles require Adam/AdamW and batch >=128")
            if profile.get("scheduler") not in ("plateau", "cosine"):
                raise ValueError("Full profile scheduler must be plateau or cosine")
            for key in ("learning_rate", "weight_decay", "grad_clip"):
                value = profile.get(key, 0.0 if key == "weight_decay" else None)
                if value is None and key == "grad_clip":
                    continue
                if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid full profile " + key)
            if profile["learning_rate"] <= 0:
                raise ValueError("Full learning rate must be positive")
    for key, default in (
        ("seed", None),
        ("steps", None),
        ("batch_size", None),
        ("eval_every", None),
        ("checkpoint_every", 5),
        ("cpu_threads", 2),
    ):
        if type(config.get(key, default)) is not int:
            raise ValueError(key + " must be an integer (booleans are not accepted)")
    lower, upper = (
        (1, 5000) if epoch_mode(config) else ((500, 5000) if is_experiment(config) else (1, 200))
    )
    if not 0 <= config["seed"] < 2**32 or not lower <= config["steps"] <= upper:
        raise ValueError(f"A single seed and {lower}–{upper} optimizer steps are required")
    batch_limit = 128 if epoch_mode(config) else (32 if is_experiment(config) else 8)
    if (
        not 1 <= config["batch_size"] <= batch_limit
        or not 1 <= config["eval_every"] <= config["steps"]
    ):
        raise ValueError("Small batch and bounded validation interval required")
    if not 1 <= config.get("checkpoint_every", 5) <= upper:
        raise ValueError(f"checkpoint_every must be in 1–{upper}")
    if config.get("dtype", "float32") not in DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if not adapted and (
        type(config.get("learning_rate")) not in (int, float)
        or not np.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        raise ValueError("Positive finite learning rate required")
    if config.get("objective", "absolute_l2_squared") not in (
        "absolute_l2_squared",
        "relative_l2_squared",
        "scaled_nodal_mse",
    ):
        raise ValueError("Unknown or unscaled objective")
    if type(config.get("defer_test", False)) is not bool:
        raise ValueError("defer_test must be boolean")
    scheduler = config.get("scheduler", {"name": "constant"})
    if not isinstance(scheduler, dict) or scheduler.get("name") not in (
        "constant",
        "warmup_cosine",
        "plateau",
    ):
        raise ValueError("scheduler requires name constant, warmup_cosine, or plateau")
    minimum = scheduler.get("min_lr", 0.0)
    if (
        type(minimum) not in (int, float)
        or not np.isfinite(minimum)
        or not 0 <= minimum <= config.get("learning_rate", 0.0)
    ):
        raise ValueError("scheduler min_lr must lie between zero and learning_rate")
    if scheduler["name"] == "warmup_cosine":
        warmup = scheduler.get("warmup_steps", 0)
        if type(warmup) is not int or not 0 <= warmup < config["steps"]:
            raise ValueError("scheduler warmup_steps must be an integer below total steps")
    if scheduler["name"] == "plateau":
        factor, patience = scheduler.get("factor", 0.5), scheduler.get("patience", 5)
        threshold = scheduler.get("threshold", 1e-4)
        if (
            type(factor) not in (int, float)
            or not np.isfinite(factor)
            or not 0 < factor < 1
            or type(patience) is not int
            or patience < 0
            or type(threshold) not in (int, float)
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("Invalid plateau factor, patience, or threshold")
    if (
        not isinstance(config.get("models"), list)
        or not config["models"]
        or any(
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or not isinstance(spec.get("kwargs", {}), dict)
            for spec in config["models"]
        )
    ):
        raise ValueError("models must be a nonempty list of named model specifications")
    if len({s["name"] for s in config["models"]}) != len(config["models"]):
        raise ValueError("Exactly one configuration per named model is allowed")
    if any(s["name"] not in FACTORIES for s in config["models"]):
        raise ValueError("Only fno/deeponet/gno/transformer baselines are enabled")
    subset = config.get("train_subset_ids")
    if is_overfit(config):
        if (
            not isinstance(subset, list)
            or not 1 <= len(subset) <= 8
            or any(not isinstance(sid, str) for sid in subset)
            or len(set(subset)) != len(subset)
        ):
            raise ValueError("Subset-overfit requires 1–8 unique train_subset_ids")
    elif subset is not None:
        raise ValueError("train_subset_ids requires explicit subset-overfit diagnostic purpose")
    if not 1 <= config.get("cpu_threads", 2) <= 8:
        raise ValueError("cpu_threads must be in 1–8")


def source_fingerprints(specs):
    paths = {Path(__file__).resolve()}
    for spec in specs:
        module = importlib.import_module(FACTORIES[spec["name"]][0])
        paths.add(Path(module.__file__).resolve())
    return {path.name: sha256(path) for path in sorted(paths)}


def data_fingerprints(data_dir):
    names = (
        "train.npz",
        "val.npz",
        "test.npz",
        "observation_geometry.npz",
        "config.json",
        "manifest.json",
        "COMPLETE",
    )
    return {
        name: sha256(Path(data_dir) / name) for name in names if (Path(data_dir) / name).is_file()
    }


def resume_identity(data_dir, config, device):
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "data_sha256": data_fingerprints(data_dir),
        "source_sha256": source_fingerprints(config["models"]),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": str(device),
            "cpu_threads": config.get("cpu_threads", 2),
        },
    }


def assert_identity(expected, actual):
    if expected != actual:
        changed = [
            key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)
        ]
        raise ValueError("Resume identity mismatch: " + ", ".join(sorted(changed)))


def artifact_hashes(output):
    output = Path(output)
    return {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path not in (output / "manifest.json", output / "COMPLETE", output / "FROZEN")
        and path.name != ".run.lock"
        and not path.name.startswith(".atomic-")
    }


def verify_completed(output):
    output = Path(output)
    if not (output / "COMPLETE").is_file():
        raise ValueError("Run has no COMPLETE marker")
    expected = json.loads((output / "manifest.json").read_text())
    if artifact_hashes(output) != expected:
        raise ValueError("Completed run artifact checksum mismatch")


def frozen_hashes(output):
    """Hash immutable training artifacts, excluding later test finalization."""
    mutable = {
        "metrics.json",
        "summary.json",
        "manifest.json",
        "frozen_manifest.json",
        "FROZEN",
        "COMPLETE",
        "test_commit.pt",
        "test_predictions.npz",
    }
    return {
        name: digest
        for name, digest in artifact_hashes(output).items()
        if Path(name).name not in mutable
    }


def verify_frozen(output):
    output = Path(output)
    if not (output / "FROZEN").is_file():
        raise ValueError("All training must be FROZEN before test finalization")
    if frozen_hashes(output) != json.loads((output / "frozen_manifest.json").read_text()):
        raise ValueError("Frozen training artifact checksum mismatch")


def export_results(output, final):
    output = Path(output)
    for split_name, record in final["predictions"].items():
        descriptor, temporary = tempfile.mkstemp(prefix=".atomic-", dir=output)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle, sample_id=np.array(record["sample_id"]), y=record["y"].numpy()
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output / f"{split_name}_predictions.npz")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    dump_json(output / "metrics.json", final["metrics"])


def mark_frozen(output):
    output = Path(output)
    dump_json(output / "manifest.json", artifact_hashes(output))
    dump_json(output / "frozen_manifest.json", frozen_hashes(output))
    atomic_write(
        output / "FROZEN", "Training and checkpoint selection frozen; test evaluation deferred.\n"
    )


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def checkpoint_digest(state):
    """Canonical hash of saved values, independent of torch zip serialization."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                if key == "checkpoint_sha256":
                    continue
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                add(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")

    add(state)
    return digest.hexdigest()


def capture_rng(rng):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_generator_json": json.dumps(rng.bit_generator.state),
        "numpy_global": {
            "bit_generator": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state, rng):
    random.setstate(state["python"])
    rng.bit_generator.state = json.loads(state["numpy_generator_json"])
    n = state["numpy_global"]
    np.random.set_state(
        (
            n["bit_generator"],
            n["keys"].numpy().astype(np.uint32),
            n["position"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG/device count mismatch during resume")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def best_payload(state, spec, normalization, dtype_name):
    result = {
        "schema_version": SCHEMA_VERSION,
        "model_state": state["best_model_state"],
        "step": state["best_step"],
        "selection_mean_relative_l2": state["best_value"],
        "model_spec": spec,
        "normalization": normalization,
        "dtype": dtype_name,
    }
    if state["selection_split"] == "val":
        result["validation_mean_relative_l2"] = state["best_value"]
    if "legacy_training" in spec:
        result["selection_metric"] = (
            "training_mse" if state["selection_split"] == "train" else "validation_mse"
        )
        result["selection_value"] = state["best_value"]
        result.pop("selection_mean_relative_l2", None)
        result.pop("validation_mean_relative_l2", None)
    return result


def _select_train_subset(train, config):
    if not is_overfit(config):
        return train
    lookup = {str(sid): index for index, sid in enumerate(train["sample_id"])}
    requested = config["train_subset_ids"]
    if any(sid not in lookup for sid in requested):
        raise ValueError("Overfit subset IDs must all belong to the training split")
    index = [lookup[sid] for sid in requested]
    return {key: value[index].copy() for key, value in train.items()}


def run_one(data_dir, config, spec, output, device, identity, resume=False, stop_after=None):
    """Resume only a fully matched run. last.pt is the committed state journal."""
    output = Path(output)
    if (output / "COMPLETE").exists():
        verify_completed(output)
        return json.loads((output / "metrics.json").read_text())
    if (output / "FROZEN").exists():
        verify_frozen(output)
        return json.loads((output / "frozen_metrics.json").read_text())
    train_full, val = [read_split(data_dir, key) for key in ("train", "val")]
    train_limit, val_limit, test_limit = split_limits(config)
    if len(train_full["y"]) > train_limit or len(val["y"]) > val_limit:
        raise ValueError("Restricted workflow is capped at small pilot splits")
    if set(train_full["sample_id"]) & set(val["sample_id"]):
        raise ValueError("Training and validation IDs overlap")
    train = _select_train_subset(train_full, config)
    geometry = P1Geometry(Path(data_dir) / "observation_geometry.npz")
    if train["y"].shape[1:] != geometry.grid_shape or val["y"].shape[1:] != geometry.grid_shape:
        raise ValueError("Public field grid does not match geometry")
    normalization = fit_normalization(train, geometry)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(config["seed"])
    dtype_name = config.get("dtype", "float32")
    dtype = DTYPES[dtype_name]
    model = PhysicalOperator(make_model(spec), normalization, dtype).to(device)
    full = config.get("legacy_full_epochs") is not None
    adapted = config.get("epoch_training") is not None
    epochs = full or adapted
    profile = spec["training"] if adapted else spec.get("legacy_training", {}) if full else {}
    optimizer_class = getattr(torch.optim, profile.get("optimizer", "Adam"))
    optimizer = optimizer_class(
        model.parameters(),
        lr=profile.get("learning_rate", config.get("learning_rate")),
        weight_decay=profile.get("weight_decay", 0.0),
    )
    full_scheduler = None
    if epochs:
        if full and len(train["y"]) > profile["batch_size"]:
            raise ValueError(
                "This full workflow requires one logical batch per epoch; use a larger batch"
            )
        full_scheduler = epoch_scheduler(optimizer, profile, config["steps"])
    scheduler = OperatorScheduler(
        optimizer, {**config, "learning_rate": profile["learning_rate"]} if epochs else config
    )
    norm_squared = geometry.torch_loss(device, dtype)
    selection_split = "train" if is_overfit(config) else "val"
    selection = (
        f"lowest {selection_split} nodal MSE; earliest on ties"
        if full
        else f"lowest {selection_split} mean per-sample relative P1 L2; earliest on ties"
    )
    objective = config.get("objective", "absolute_l2_squared")
    run_config = {
        **config,
        "model": spec,
        "device": str(device),
        "dtype": dtype_name,
        "selection": selection,
        "objective": "nodal_mse" if full else objective,
        "legacy_training": profile,
        "objective_description": (
            "mean per-sample consistent-mass relative P1 L2 squared"
            if objective == "relative_l2_squared"
            else "mean consistent-mass absolute P1 L2 squared / training y_rms squared"
        ),
    }
    dataset_config = (
        json.loads((Path(data_dir) / "config.json").read_text())
        if (Path(data_dir) / "config.json").exists()
        else {}
    )
    label_semantics = {
        "public_grid_shape": list(geometry.grid_shape),
        "reference_mesh_intervals_per_axis": dataset_config.get("reference_n"),
        "audit_mesh_intervals_per_axis": dataset_config.get("audit_n"),
        "definition": "Public y is the reference P1 solution evaluated at observation nodes; metrics compare public-grid P1 reconstructions, not continuum error.",
        "precision_caveat": "Reference/audit differences are empirical; consult the data quality report for reference-label and observation-reconstruction errors.",
    }
    if full:
        run_config["objective_description"] = (
            "Unweighted nodal mean squared physical state error, as in legacy MSELoss; no physical or relative loss."
        )
    if adapted:
        run_config["legacy_training"] = None
        run_config.update(
            training=profile,
            step_unit="epoch",
            selection=selection,
            optimizer_updates_per_epoch=int(np.ceil(len(train["y"]) / profile["batch_size"])),
            microbatch_size=config["batch_size"],
            logical_batch_size=min(profile["batch_size"], len(train["y"])),
        )
        if objective == "scaled_nodal_mse":
            run_config["objective_description"] = (
                "Physical nodal MSE divided by training y_rms squared."
            )
    saved = None
    if output.exists() and not resume:
        raise FileExistsError(output)
    if output.exists() and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Older last.pt lacks a complete verified resume state; load best.pt for inference"
            )
        if checkpoint_digest(saved) != saved.get("checkpoint_sha256"):
            raise ValueError("Resume checkpoint checksum mismatch")
        assert_identity(saved["identity"], identity)
        if saved["normalization"] != normalization or saved["model_spec"] != spec:
            raise ValueError("Resume normalization/model specification mismatch")
    output.mkdir(exist_ok=resume)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    if saved is not None:
        if json.loads((output / "config.json").read_text()) != run_config:
            raise ValueError("Resume saved config artifact mismatch")
        if json.loads((output / "normalization.json").read_text()) != normalization:
            raise ValueError("Resume normalization artifact mismatch")
    for module in (sys.modules[__name__], sys.modules[model.model.__class__.__module__]):
        source = Path(module.__file__)
        if saved is not None and (
            not (snapshot / source.name).exists()
            or sha256(snapshot / source.name) != identity["source_sha256"][source.name]
        ):
            raise ValueError("Resume source snapshot mismatch: " + source.name)
        if not (snapshot / source.name).exists():
            shutil.copyfile(source, snapshot / source.name)
    dump_json(output / "config.json", run_config)
    dump_json(output / "normalization.json", normalization)
    if not (output / "provenance.json").exists():
        dump_json(
            output / "provenance.json",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "data_directory": str(Path(data_dir).resolve()),
                **identity,
                "label_semantics": label_semantics,
                "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "representation": model.representation_metadata(),
                "training_sample_ids": list(map(str, train["sample_id"])),
                "full_training_split_ids": list(map(str, train_full["sample_id"])),
            },
        )
    if saved is None:
        state = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "model_spec": spec,
            "normalization": normalization,
            "step": -1,
            "optimizer_steps": 0,
            "validation_evaluations": 0,
            "history": [],
            "best_value": float("inf"),
            "best_step": None,
            "best_model_state": None,
            "selection_split": selection_split,
            "loop_wall_seconds": 0.0,
            "final_results": None,
        }
    else:
        state = saved
        model.model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if epochs:
            full_scheduler.load_state_dict(
                {
                    k: (
                        (float(v) if v in ("Infinity", "-Infinity") else v)
                        if isinstance(v, str)
                        else v
                    )
                    for k, v in state["full_scheduler_state"].items()
                }
            )
        restore_rng(state["rng_state"], rng)
        # Repair any artifacts written after the last committed checkpoint.
        if state["best_model_state"] is not None:
            atomic_write(output / "best.pt", best_payload(state, spec, normalization, dtype_name))
        dump_json(output / "history.json", state["history"])
    all_train = {
        key: torch.as_tensor(train[key], dtype=dtype, device=device) for key in ("a", "f", "y")
    }
    segment_started = time.perf_counter()
    previous_wall = state["loop_wall_seconds"]

    def checkpoint():
        state["model_state"] = _cpu_copy(model.model.state_dict())
        state["optimizer_state"] = _cpu_copy(optimizer.state_dict())
        state["scheduler_state"] = _cpu_copy(scheduler.state_dict())
        if epochs:
            state["full_scheduler_state"] = {
                k: (
                    ("Infinity" if v > 0 else "-Infinity")
                    if isinstance(v, float) and not np.isfinite(v)
                    else v
                )
                for k, v in full_scheduler.state_dict().items()
            }
        state["rng_state"] = capture_rng(rng)
        state["loop_wall_seconds"] = previous_wall + time.perf_counter() - segment_started
        state["checkpoint_sha256"] = checkpoint_digest(state)
        atomic_write(output / "last.pt", state)

    # stop_after is a safe test hook: epoch in epoch mode, update otherwise.
    if stop_after is not None and state["step"] >= stop_after and state["step"] < config["steps"]:
        return None
    for step in range(max(1 if epochs else 0, state["step"] + 1), config["steps"] + 1):
        loss = None
        if step:
            model.train()
            index = (
                rng.permutation(len(train["y"]))
                if epochs
                else rng.choice(
                    len(train["y"]), min(config["batch_size"], len(train["y"])), replace=False
                )
            )
            if adapted:
                loss = torch.zeros((), device=device, dtype=dtype)
                for start in range(0, len(index), profile["batch_size"]):
                    logical = index[start : start + profile["batch_size"]]
                    scheduler.before_step(state["optimizer_steps"] + 1)
                    part = batch_update(
                        model,
                        all_train,
                        logical,
                        config["batch_size"],
                        optimizer,
                        norm_squared,
                        normalization["y_rms"],
                        objective,
                        profile.get("grad_clip"),
                    )
                    loss += part * (len(logical) / len(index))
                    state["optimizer_steps"] += 1
                state["training_sample_exposures"] = state.get(
                    "training_sample_exposures", 0
                ) + len(index)
            elif full:
                scheduler.before_step(step)
                loss = batch_update(
                    model,
                    all_train,
                    index,
                    config["batch_size"],
                    optimizer,
                    norm_squared,
                    normalization["y_rms"],
                    "nodal_mse",
                    profile.get("grad_clip"),
                )
                state["optimizer_steps"] += 1
            else:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(all_train["a"][index], all_train["f"][index])
                loss = objective_per_sample(
                    prediction,
                    all_train["y"][index],
                    norm_squared,
                    normalization["y_rms"],
                    objective,
                ).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError(f"Non-finite gradient at step {step}")
                scheduler.before_step(step)
                optimizer.step()
                state["optimizer_steps"] += 1
        if epochs and step:
            full_val_prediction = predict(model, val, device, config["batch_size"])
            mse = float(
                np.mean((full_val_prediction.astype(np.float64) - val["y"].astype(np.float64)) ** 2)
            )
            epoch_val_metrics = (
                evaluate_arrays(full_val_prediction, val["y"], geometry, val["sample_id"])
                if adapted
                else None
            )
            epoch_selection = (
                evaluate_arrays(
                    predict(model, train, device, config["batch_size"]),
                    train["y"],
                    geometry,
                    train["sample_id"],
                )
                if adapted and selection_split == "train"
                else epoch_val_metrics
            )
            training_mse = None
            if full and selection_split == "train":
                training_prediction = predict(model, train, device, config["batch_size"])
                training_mse = float(
                    np.mean(
                        (training_prediction.astype(np.float64) - train["y"].astype(np.float64))
                        ** 2
                    )
                )
            full_value = (
                epoch_selection["mean"]["relative_l2"]
                if adapted
                else training_mse if selection_split == "train" else mse
            )
            if not np.isfinite(full_value):
                raise RuntimeError("Non-finite full validation MSE")
            (
                full_scheduler.step(full_value)
                if profile["scheduler"] == "plateau"
                else full_scheduler.step()
            )
            selection_row = {"epoch": step, "validation_mse": mse}
            if training_mse is not None:
                selection_row["training_mse"] = training_mse
            if adapted:
                selection_row["validation_mean_relative_l2"] = epoch_val_metrics["mean"][
                    "relative_l2"
                ]
                if selection_split == "train":
                    selection_row["train_mean_relative_l2"] = full_value
            state.setdefault("selection_history", []).append(selection_row)
            if full_value < state["best_value"]:
                state["best_value"], state["best_step"] = full_value, step
                state["best_model_state"] = _cpu_copy(model.model.state_dict())
                atomic_write(
                    output / "best.pt", best_payload(state, spec, normalization, dtype_name)
                )
        evaluate = step % config["eval_every"] == 0 or step == config["steps"]
        if evaluate:
            train_metrics = evaluate_arrays(
                predict(model, train, device, config["batch_size"]),
                train["y"],
                geometry,
                train["sample_id"],
            )
            val_metrics = (
                epoch_val_metrics
                if adapted
                else evaluate_arrays(
                    predict(model, val, device, config["batch_size"]),
                    val["y"],
                    geometry,
                    val["sample_id"],
                )
            )
            state["validation_evaluations"] += 1
            value = (train_metrics if selection_split == "train" else val_metrics)["mean"][
                "relative_l2"
            ]
            if not epochs:
                scheduler.after_validation(value)
            row = {
                "step": step,
                "batch_loss": float(loss.detach().cpu()) if loss is not None else None,
                "train": train_metrics["mean"],
                "val": val_metrics["mean"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "last_optimizer_learning_rate": scheduler.last_optimizer_learning_rate,
                "scheduler_optimizer_steps": scheduler.optimizer_steps,
                "elapsed_seconds": previous_wall + time.perf_counter() - segment_started,
            }
            state["history"].append(row)
            if not epochs and value < state["best_value"]:
                state["best_value"], state["best_step"] = value, step
                state["best_model_state"] = _cpu_copy(model.model.state_dict())
                atomic_write(
                    output / "best.pt", best_payload(state, spec, normalization, dtype_name)
                )
            dump_json(output / "history.json", state["history"])
            print(
                f'{spec["name"]} step={step} train_L2={train_metrics["mean"]["relative_l2"]:.6f} '
                f'val_L2={val_metrics["mean"]["relative_l2"]:.6f} best_step={state["best_step"]}',
                flush=True,
            )
        state["step"] = step
        stopped = stop_after is not None and step >= stop_after and step < config["steps"]
        if evaluate or step % config.get("checkpoint_every", 5) == 0 or stopped:
            checkpoint()
        if stopped:
            return None
    if epochs:
        dump_json(output / "selection_history.json", state["selection_history"])
    loop_wall_seconds = previous_wall + time.perf_counter() - segment_started
    if state["final_results"] is None:
        selected = load_trained_model(output, device)
        splits = [("train", train), ("val", val)]
        deferred = config.get("defer_test", False)
        if not deferred:
            test = read_split(data_dir, "test")
            if len(test["y"]) > test_limit or set(test["sample_id"]) & (
                set(train_full["sample_id"]) | set(val["sample_id"])
            ):
                raise ValueError("Invalid test split or overlapping IDs")
            splits.append(("test", test))
        scope = (
            "Memorization diagnostic on explicitly listed training IDs; training subset alone selects best, with no test-based selection."
            if is_overfit(config)
            else (
                "Supervised operator experiment; validation alone selects the frozen checkpoint."
                if is_experiment(config)
                else "Pipeline acceptance only; this pilot cannot rank methods or establish generalization."
            )
        )
        metrics = {
            "scope": scope,
            "norm_semantics": "Public observation mesh P1 reconstruction.",
            "label_semantics": label_semantics,
            "selection": selection,
            "selected_step": state["best_step"],
            "test_evaluations": 0 if deferred else 1,
            "test_status": "deferred" if deferred else "finalized",
            "objective": "nodal_mse" if full else objective,
            "scheduler": scheduler.state_dict(),
            "epochs_completed": config["steps"] if epochs else None,
            "legacy_training": profile if full else {},
            "selection_metric": (
                ("training_mse" if selection_split == "train" else "validation_mse")
                if full
                else "mean_relative_l2"
            ),
            "selection_split": selection_split,
            "selection_value": state["best_value"],
            "effective_batch_size": (
                min(profile["batch_size"], len(train["y"]))
                if epochs
                else min(config["batch_size"], len(train["y"]))
            ),
            "training_sample_exposures": config["steps"] * len(train["y"]) if epochs else None,
            "full_scheduler_state": state.get("full_scheduler_state"),
            "optimizer_steps": state["optimizer_steps"],
            "validation_evaluations": (
                len(state["selection_history"]) if epochs else state["validation_evaluations"]
            ),
            "training_sample_ids": list(map(str, train["sample_id"])),
            "dtype": dtype_name,
            "loop_wall_seconds": loop_wall_seconds,
            "timing_scope": "Loop wall time including optimization, train/val inference, metrics, logging and checkpoints; excludes setup, resume setup, and final selected-model evaluation.",
            "resume_policy": "Resume from the last committed checkpoint; uncommitted steps replay deterministically. An interrupted uncommitted final evaluation may be recomputed, without affecting selection.",
        }
        if adapted:
            metrics.update(
                training=profile,
                step_unit="epoch",
                microbatch_size=config["batch_size"],
                optimizer_updates_per_epoch=int(np.ceil(len(train["y"]) / profile["batch_size"])),
            )
        final = {"metrics": metrics, "predictions": {}}
        for split_name, split in splits:
            prediction = predict(selected, split, device, config["batch_size"])
            metrics[split_name] = evaluate_arrays(
                prediction, split["y"], geometry, split["sample_id"]
            )
            final["predictions"][split_name] = {
                "sample_id": list(map(str, split["sample_id"])),
                "y": torch.from_numpy(prediction.copy()),
            }
        state["final_results"] = final
        # Commit final predictions before exporting artifacts. Interrupted
        # exports resume from these values without another test evaluation.
        checkpoint()
    final = state["final_results"]
    export_results(output, final)
    if config.get("defer_test", False):
        dump_json(output / "frozen_metrics.json", final["metrics"])
        mark_frozen(output)
    else:
        dump_json(output / "manifest.json", artifact_hashes(output))
        atomic_write(output / "COMPLETE", "Configured operator run complete.\n")
    return final["metrics"]


def run(
    data_dir, config_path, output, device="cpu", resume=False, stop_after=None, defer_test=None
):
    """Train a bounded suite, or verify/resume its exact saved identity.

    ``stop_after`` simulates interruption at a committed epoch (epoch modes) or
    optimizer step without writing
    COMPLETE; normal callers omit it. ``resume=True`` safely skips verified
    completed runs and refuses different config/data/source/runtime identities.
    """
    config = json.loads(Path(config_path).read_text())
    if defer_test is not None:
        config["defer_test"] = defer_test
    validate_config(config)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a non-negative step integer")
    data_dir, output = Path(data_dir), Path(output)
    if not (data_dir / "COMPLETE").is_file():
        raise ValueError("A quality-checked dataset with COMPLETE marker is required")
    verify_data_manifest(data_dir)
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(config.get("cpu_threads", 2))
    identity = resume_identity(data_dir, config, device)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("Resume output directory does not exist")
    else:
        output.mkdir(parents=True)  # Preserve no-overwrite behavior.
    with exclusive_run_lock(output):
        if resume:
            path = output / "resume_identity.json"
            if not path.exists():
                raise ValueError(
                    "This older output has no verifiable resume identity; best.pt remains loadable"
                )
            assert_identity(json.loads(path.read_text()), identity)
            if json.loads((output / "config.json").read_text()) != config:
                raise ValueError("Resume suite config artifact mismatch")
            if (output / "COMPLETE").exists():
                verify_completed(output)
                return json.loads((output / "summary.json").read_text())
            if (output / "FROZEN").exists():
                verify_frozen(output)
                return json.loads((output / "frozen_summary.json").read_text())
        else:
            dump_json(output / "config.json", config)
            dump_json(output / "resume_identity.json", identity)
        result = {}
        for spec in config["models"]:
            metrics = run_one(
                data_dir,
                config,
                spec,
                output / spec["name"],
                device,
                identity,
                resume=resume,
                stop_after=stop_after,
            )
            if metrics is None:
                return {**result, "_interrupted": spec["name"]}
            result[spec["name"]] = metrics
        dump_json(output / "summary.json", result)
        if config.get("defer_test", False):
            dump_json(output / "frozen_summary.json", result)
            mark_frozen(output)
        else:
            dump_json(output / "manifest.json", artifact_hashes(output))
            atomic_write(output / "COMPLETE", "All configured operator runs completed.\n")
        return result


def finalize(data_dir, output, device="cpu"):
    """Evaluate a fully FROZEN suite once, without changing its configuration.

    All models must freeze before the first test array is opened. The committed
    test result is atomic; interrupted exports reuse it. An interruption during
    uncommitted inference can require recomputation, never new model selection.
    """
    data_dir, output = Path(data_dir), Path(output)
    device = torch.device(device)
    if not output.is_dir():
        raise FileNotFoundError(output)
    with exclusive_run_lock(output):
        verify_frozen(output)
        config = json.loads((output / "config.json").read_text())
        if not config.get("defer_test", False):
            raise ValueError("finalize requires a deferred-test training configuration")
        validate_config(config)
        verify_data_manifest(data_dir)
        frozen_identity = json.loads((output / "resume_identity.json").read_text())
        for key, value in (
            ("config", config),
            ("data_sha256", data_fingerprints(data_dir)),
            ("source_sha256", source_fingerprints(config["models"])),
        ):
            if frozen_identity[key] != value:
                raise ValueError("Frozen finalization identity mismatch: " + key)
        for spec in config["models"]:
            verify_frozen(output / spec["name"])
        if (output / "COMPLETE").exists():
            verify_completed(output)
            return json.loads((output / "summary.json").read_text())
        if device.type == "cpu":
            torch.set_num_threads(config.get("cpu_threads", 2))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        geometry = P1Geometry(data_dir / "observation_geometry.npz")
        test, result = None, {}
        for spec in config["models"]:
            directory = output / spec["name"]
            if (directory / "COMPLETE").exists():
                verify_completed(directory)
                result[spec["name"]] = json.loads((directory / "metrics.json").read_text())
                continue
            model_digest = sha256(directory / "best.pt")
            test_digest = frozen_identity["data_sha256"]["test.npz"]
            commit_path = directory / "test_commit.pt"
            if commit_path.exists():
                commit = torch.load(commit_path, map_location="cpu", weights_only=True)
                if (
                    checkpoint_digest(commit) != commit.get("checkpoint_sha256")
                    or commit["model_sha256"] != model_digest
                    or commit["test_sha256"] != test_digest
                ):
                    raise ValueError("Committed test result checksum/identity mismatch")
            else:
                if test is None:
                    test = read_split(data_dir, "test")
                frozen_metrics = json.loads((directory / "frozen_metrics.json").read_text())
                provenance = json.loads((directory / "provenance.json").read_text())
                val_ids = {row["sample_id"] for row in frozen_metrics["val"]["per_sample"]}
                forbidden = set(provenance["full_training_split_ids"]) | val_ids
                if len(test["y"]) > split_limits(config)[2] or set(test["sample_id"]) & forbidden:
                    raise ValueError("Invalid test split or overlapping IDs")
                selected = load_trained_model(directory, device)
                started = time.perf_counter()
                prediction = predict(selected, test, device, config["batch_size"])
                metrics = evaluate_arrays(prediction, test["y"], geometry, test["sample_id"])
                commit = {
                    "model_sha256": model_digest,
                    "test_sha256": test_digest,
                    "metrics": metrics,
                    "sample_id": list(map(str, test["sample_id"])),
                    "y": torch.from_numpy(prediction.copy()),
                    "test_evaluations": 1,
                    "evaluation_device": str(device),
                    "test_inference_and_metrics_wall_seconds": time.perf_counter() - started,
                }
                commit["checkpoint_sha256"] = checkpoint_digest(commit)
                atomic_write(commit_path, commit)
            metrics = json.loads((directory / "frozen_metrics.json").read_text())
            metrics.update(
                test=commit["metrics"],
                test_evaluations=1,
                test_status="finalized",
                evaluation_device=commit["evaluation_device"],
                test_inference_and_metrics_wall_seconds=commit[
                    "test_inference_and_metrics_wall_seconds"
                ],
            )
            export_results(
                directory,
                {
                    "metrics": metrics,
                    "predictions": {"test": {"sample_id": commit["sample_id"], "y": commit["y"]}},
                },
            )
            dump_json(directory / "manifest.json", artifact_hashes(directory))
            atomic_write(directory / "COMPLETE", "Frozen checkpoint test evaluation finalized.\n")
            result[spec["name"]] = metrics
        dump_json(output / "summary.json", result)
        dump_json(output / "manifest.json", artifact_hashes(output))
        atomic_write(output / "COMPLETE", "All frozen checkpoints evaluated on test once.\n")
        return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", nargs="?", choices=("train", "finalize"), default="train")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify identity, resume last state, or skip verified completion",
    )
    parser.add_argument(
        "--defer-test",
        action="store_const",
        const=True,
        default=None,
        help="Freeze all checkpoints without opening test arrays",
    )
    args = parser.parse_args()
    if args.command == "finalize":
        finalize(args.data, args.output, args.device)
    else:
        if args.config is None:
            parser.error("--config is required for training")
        run(
            args.data,
            args.config,
            args.output,
            args.device,
            resume=args.resume,
            defer_test=args.defer_test,
        )


if __name__ == "__main__":
    main()
