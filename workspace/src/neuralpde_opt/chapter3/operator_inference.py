"""Batched prediction from raw a/f fields; target labels are never required.

Grid operators return values on their supplied grid. Query operators can use
separate shared or per-sample coordinates. Outputs are new immutable run
directories with predictions, provenance, checksums and a completion marker.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from .operator_training import dump_json, load_trained_model, sha256, source_fingerprints


def read_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"a", "f"} <= set(archive.files):
            raise ValueError("Prediction inputs require a and f arrays")
        a, f = archive["a"].copy(), archive["f"].copy()
        ids = archive["sample_id"].copy() if "sample_id" in archive.files else None
    if a.ndim != 3 or a.shape != f.shape or not len(a) or min(a.shape[1:]) < 2:
        raise ValueError("a and f must be nonempty matching [sample,H,W] grids")
    if (
        a.dtype.kind != "f"
        or f.dtype.kind != "f"
        or not np.isfinite(a).all()
        or not np.isfinite(f).all()
    ):
        raise ValueError("a and f must be finite floating-point fields")
    if np.min(a) <= 0:
        raise ValueError("Raw diffusion coefficients must be positive")
    if ids is None:
        ids = np.array([f"input-{i:06d}" for i in range(len(a))])
    if ids.ndim != 1 or len(ids) != len(a) or len(set(map(str, ids))) != len(ids):
        raise ValueError("sample_id must contain one unique identifier per input")
    return a, f, np.asarray(ids, dtype=str)


def read_queries(path, layout, sample_count):
    with np.load(path, allow_pickle=False) as archive:
        keys = [k for k in ("query_points", "coordinates", "points") if k in archive.files]
        if len(keys) != 1:
            raise ValueError("Queries need exactly one of query_points, coordinates or points")
        points = archive[keys[0]].copy()
    if points.ndim < 2 or points.shape[-1] != 2 or points.dtype.kind != "f":
        raise ValueError("Queries must be floating-point coordinates with final dimension 2")
    if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError("Query coordinates must be finite and lie in [0,1]^2")
    if layout == "shared":
        shape = points.shape[:-1]
        points = points.reshape(-1, 2)
    elif layout == "per_sample":
        if points.ndim < 3 or points.shape[0] != sample_count:
            raise ValueError("Per-sample queries need shape [sample,...,2]")
        shape = points.shape[1:-1]
        points = points.reshape(sample_count, -1, 2)
    else:
        raise ValueError("query_layout must be shared or per_sample")
    return points, tuple(shape)


def run_inference(
    run_dir,
    input_path,
    output,
    device="cpu",
    batch_size=4,
    queries=None,
    query_layout="shared",
    query_chunk_size=1024,
):
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError(
            "batch_size must be an integer from 1 to 8 (up to 128 for full-training checkpoints)"
        )
    if batch_size > 8:
        config_path = Path(run_dir) / "config.json"
        full_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        epochs = full_config.get("legacy_full_epochs")
        if isinstance(full_config.get("epoch_training"), dict):
            epochs = full_config["epoch_training"].get("epochs")
        if type(epochs) is not int or epochs != full_config.get("steps"):
            raise ValueError("batch_size must be an integer from 1 to 8 for legacy checkpoints")
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    output, run_dir, input_path = Path(output), Path(run_dir), Path(input_path)
    if output.exists():
        raise FileExistsError(output)
    a, f, ids = read_inputs(input_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    model = load_trained_model(run_dir, device)
    dtype = next(model.parameters()).dtype
    supports_queries = bool(getattr(model.model, "supports_queries", False))
    points, query_shape = None, None
    if queries is not None:
        if not supports_queries:
            raise ValueError("This model accepts grid inputs, not separate coordinate queries")
        points, query_shape = read_queries(queries, query_layout, len(a))
    output.mkdir(parents=True)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(a), batch_size):
            stop = min(start + batch_size, len(a))
            av = torch.as_tensor(a[start:stop], dtype=dtype, device=device)
            fv = torch.as_tensor(f[start:stop], dtype=dtype, device=device)
            options = {"query_chunk_size": query_chunk_size} if supports_queries else {}
            if points is not None:
                query = points[start:stop] if query_layout == "per_sample" else points
                options["query_points"] = torch.as_tensor(query, dtype=dtype, device=device)
            result = model(av, fv, **options).cpu().numpy()
            if points is not None:
                result = result.reshape((stop - start,) + query_shape)
            if not np.isfinite(result).all():
                raise ValueError("Model produced non-finite predictions")
            predictions.append(result)
    prediction = np.concatenate(predictions)
    payload = {"sample_id": ids, "y": prediction}
    if points is not None:
        payload["query_points"] = points
        payload["query_shape"] = np.array(query_shape)
    temporary = output / "predictions.npz.tmp"
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
    temporary.replace(output / "predictions.npz")
    metadata = {
        "run_directory": str(run_dir.resolve()),
        "input_sha256": sha256(input_path),
        "checkpoint_sha256": sha256(run_dir / "best.pt"),
        "device": str(device),
        "dtype": str(dtype),
        "sample_count": len(ids),
        "output_shape": list(prediction.shape),
        "batch_size": batch_size,
        "math_backend": {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "input_semantics": "Raw physical a/f; no y labels used.",
        "query_layout": query_layout if queries is not None else "model output grid",
        "query_sha256": sha256(queries) if queries is not None else None,
        "run_artifact_sha256": {
            name: sha256(run_dir / name)
            for name in ("best.pt", "config.json", "normalization.json")
        },
        "source_sha256": {
            **source_fingerprints([json.loads((run_dir / "config.json").read_text())["model"]]),
            "operator_inference.py": sha256(__file__),
        },
        "representation": (
            model.model.representation_metadata()
            if hasattr(model.model, "representation_metadata")
            else {}
        ),
    }
    dump_json(output / "metadata.json", metadata)
    manifest = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    dump_json(output / "manifest.json", manifest)
    (output / "COMPLETE").write_text(
        "Physical-field predictions completed. No training or label evaluation.\n"
    )
    return prediction


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--query-layout", choices=("shared", "per_sample"), default="shared")
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    args = parser.parse_args()
    values = run_inference(
        args.run,
        args.input,
        args.output,
        args.device,
        args.batch_size,
        args.queries,
        args.query_layout,
        args.query_chunk_size,
    )
    print(json.dumps({"output": str(args.output), "shape": list(values.shape)}))


if __name__ == "__main__":
    main()
