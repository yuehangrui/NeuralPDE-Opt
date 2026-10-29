"""Resumable, gated Chapter 3 shared-data generation; the original pilot is immutable.

Generate ``--stage gate`` first, then ``--stage complete``. Production uses the
same raw-field schema as the pilot. Only selected samples receive the full
three-grid/quadrature audit; every reference solve receives algebraic, boundary,
range and serialization checks. Mesh differences remain empirical estimates.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import scipy
from scipy import sparse

from ..mesh import make_mesh
from .data import manufactured_audit, sample_gates, validate_config as validate_pilot
from .fields import FieldPair, draw_parameters
from .reference import Solution, compare_p1, save_geometry, solve_fields

WORKFLOW = "ch03-resumable-fields-v1"
SPLITS = {"train": 0, "val": 1, "test": 2}
ARRAY_KEYS = (
    "a",
    "f",
    "y",
    "a_reference",
    "f_reference",
    "y_reference",
    "a_coefficients",
    "f_coefficients",
    "sample_id",
)
CONTROL_FILES = {"manifest.json", "COMPLETE", "reload_audit.json", ".generation.lock"}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, writer):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    _atomic(path, lambda stream: stream.write(_canonical(value) + b"\n"))


def _npz(path, values):
    _atomic(path, lambda stream: np.savez_compressed(stream, **values))


def _read(path):
    return json.loads(Path(path).read_text())


def validate_config(c):
    fixture = c.get("purpose") == "chapter3_supervised_experiments_fixture"
    full = c.get("purpose") == "chapter3_supervised_full_training_v2"
    if not fixture and not full and c.get("purpose") != "chapter3_supervised_experiments_v1":
        raise ValueError("expected explicit Chapter 3 supervised-experiment purpose")
    if set(c["splits"]) != set(SPLITS) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in c["splits"].values()
    ):
        raise ValueError("positive train/val/test counts required")
    if set(c.get("audit_samples", {})) != set(SPLITS):
        raise ValueError("audit_samples must explicitly specify all three splits")
    for split, count in c["audit_samples"].items():
        minimum = 1 if fixture else {"train": 16, "val": 8, "test": 8}[split]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not minimum <= count <= c["splits"][split]
        ):
            raise ValueError(f"invalid audit_samples/{split}; minimum is {minimum}")
    if fixture:
        if sum(c["splits"].values()) > 12 or c["audit_n"] > 32:
            raise ValueError("fixture is limited to 12 samples and audit_n <= 32")
    elif (c["observation_n"], c["reference_n"], c["audit_n"]) != (64, 128, 256):
        raise ValueError("production workflow requires the approved 64/128/256 interval grids")
    limit = 1280 if full else 384
    if not fixture and sum(c["splits"].values()) > limit:
        raise ValueError(f"this workflow is restricted to at most {limit} production samples")
    if full and any(c["splits"][s] > cap for s, cap in zip(SPLITS, (1024, 128, 128))):
        raise ValueError("full training split limits are 1024/128/128")
    if not isinstance(c["seed"], int) or isinstance(c["seed"], bool) or c["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    legacy = dict(c, purpose="pilot_only_no_training", splits={s: 1 for s in SPLITS})
    validate_pilot(legacy)
    if not fixture and (c["quadrature_order"], c["audit_quadrature_order"]) != (4, 5):
        raise ValueError("production requires the approved quadrature audit 4 -> 5")
    return fixture


def _source_files():
    folder = Path(__file__).parent
    return [
        folder / name for name in ("experimental_data.py", "data.py", "reference.py", "fields.py")
    ] + [folder.parent / "mesh.py"]


def _identity(c):
    return {
        "workflow": WORKFLOW,
        "config_sha256": hashlib.sha256(_canonical(c)).hexdigest(),
        "audit_indices": {s: list(range(c["audit_samples"][s])) for s in SPLITS},
        "source_sha256": {p.name: _sha(p) for p in _source_files()},
    }


def estimate(c):
    """Conservative uncompressed storage estimate; includes duplicate split files."""
    n0, n1, n2 = (c[k] + 1 for k in ("observation_n", "reference_n", "audit_n"))
    count, audited = sum(c["splits"].values()), sum(c["audit_samples"].values())
    per_sample = 8 * (
        3 * n0 * n0 + 3 * n1 * n1 + sum(c["fields"][s]["modes"] ** 2 for s in ("a", "f"))
    )
    retained = 2 * count * per_sample + audited * 8 * (n2 * n2 + n1 * n1 + n0 * n0)
    return {
        "samples": count,
        "full_audit_samples": audited,
        "uncompressed_sample_bytes": per_sample,
        "retained_data_upper_estimate_bytes": retained + 64 * 1024**2,
        "recommended_free_bytes": 3 * retained + 512 * 1024**2,
        "note": "Conservative raw-array estimate plus geometry allowance; compression is data-dependent.",
    }


def _sample_id(c, split, index):
    return f'ch03-e{c["seed"]}-{split}-{index:06d}'


def _selected(c):
    return [(s, i) for s in SPLITS for i in range(c["audit_samples"][s])]


def _initialize(c, output):
    identity = _identity(c)
    if (output / "identity.json").exists():
        if _read(output / "identity.json") != identity or (
            (output / "config.json").exists() and _read(output / "config.json") != c
        ):
            raise ValueError(
                "immutable configuration or generator identity differs; use a new output directory"
            )
        if (output / "COMPLETE").exists():
            return
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing an existing directory without matching workflow identity")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("samples", "records", "audit_solutions", "source_snapshot"):
        (output / name).mkdir(exist_ok=True)
    if not (output / "identity.json").exists():
        _json(output / "identity.json", identity)
    if not (output / "config.json").exists():
        _json(output / "config.json", c)
    for source in _source_files():
        destination = output / "source_snapshot" / source.name
        if destination.exists():
            if _sha(destination) != identity["source_sha256"][source.name]:
                raise ValueError("source snapshot checksum mismatch: " + source.name)
        else:
            _atomic(destination, lambda stream, p=source: stream.write(p.read_bytes()))
    for name, n in (("observation", c["observation_n"]), ("reference", c["reference_n"])):
        if not (output / f"{name}_geometry.npz").exists():
            _atomic(
                output / f"{name}_geometry.npz",
                lambda stream, n=n: _save_geometry_stream(stream, n),
            )
    if not (output / "environment.json").exists():
        _json(
            output / "environment.json",
            {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "solver": "SciPy float64 sparse direct spsolve",
                "estimate": estimate(c),
                "generator": "NumPy PCG64 / SeedSequence([seed,split,index,channel])",
            },
        )
    if not (output / "manufactured.json").exists():
        _json(output / "manufactured.json", manufactured_audit())


def _save_geometry_stream(stream, n):
    # Preserve the existing geometry writer and schema without changing it.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "geometry.npz"
        save_geometry(path, n)
        stream.write(path.read_bytes())


@contextmanager
def _locked(output):
    with (output / ".generation.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another generator owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _base_checks(row, c):
    limits = c["quality_gates"]
    solves = row["solves"]
    checks = {
        key: all(np.isfinite(s[key]) and s[key] <= limits[key] for s in solves)
        for key in (
            "relative_algebraic_residual",
            "relative_energy_identity_defect",
            "boundary_max",
        )
    }
    a = c["fields"]["a"]
    checks.update(
        ellipticity=all(
            a["lower"] <= s["a_quadrature_min"] <= s["a_quadrature_max"] <= a["upper"]
            for s in solves
        ),
        cross_grid_input=row["cross_grid_input_max"] <= 1e-13,
        finite_fields=row["finite_fields"],
        observed_range=a["lower"] <= row["a_min"] <= row["a_max"] <= a["upper"],
    )
    if row["full_audit"]:
        checks.update(sample_gates(row, c))
    return checks


def _build_sample(c, split, index, meshes):
    start = time.perf_counter()
    n0, n1, n2 = (c[k] for k in ("observation_n", "reference_n", "audit_n"))
    parameters = draw_parameters(c, SPLITS[split], index)
    fields = FieldPair(parameters, c)
    audited = index < c["audit_samples"][split]
    ref = solve_fields(fields, n1, c["quadrature_order"], meshes[n1])
    a = fields.coefficient(meshes[n0].nodes).reshape(n0 + 1, n0 + 1)
    f = fields.source(meshes[n0].nodes).reshape(n0 + 1, n0 + 1)
    ar = fields.coefficient(meshes[n1].nodes).reshape(n1 + 1, n1 + 1)
    fr = fields.source(meshes[n1].nodes).reshape(n1 + 1, n1 + 1)
    yr = ref.values.reshape(n1 + 1, n1 + 1)
    arrays = dict(
        a=a,
        f=f,
        y=yr[::2, ::2].copy(),
        a_reference=ar,
        f_reference=fr,
        y_reference=yr,
        sample_id=np.asarray(_sample_id(c, split, index)),
        **parameters,
    )
    row = {
        "sample_id": str(arrays["sample_id"]),
        "split": split,
        "index": index,
        "full_audit": audited,
        "solves": [ref.diagnostics],
        "cross_grid_input_max": float(
            max(np.max(np.abs(a - ar[::2, ::2])), np.max(np.abs(f - fr[::2, ::2])))
        ),
        "a_min": float(min(a.min(), ar.min())),
        "a_max": float(max(a.max(), ar.max())),
        "finite_fields": all(np.isfinite(v).all() for k, v in arrays.items() if k != "sample_id"),
    }
    audit_arrays = None
    if audited:
        coarse = solve_fields(fields, n0, c["quadrature_order"], meshes[n0])
        fine = solve_fields(fields, n2, c["quadrature_order"], meshes[n2])
        enriched = solve_fields(fields, n1, c["audit_quadrature_order"], meshes[n1])
        cr, rf = compare_p1(coarse.values, n0, ref), compare_p1(ref.values, n1, fine)
        row.update(
            solves=[s.diagnostics for s in (coarse, ref, fine, enriched)],
            coarse_vs_reference=cr,
            reference_vs_audit=rf,
            refinement_ratio={k: rf[k] / cr[k] for k in rf},
            quadrature_comparison=compare_p1(ref.values, n1, enriched),
            observation_vs_audit=compare_p1(arrays["y"].ravel(), n0, fine),
        )
        audit_arrays = {
            "y_coarse": coarse.values.reshape(n0 + 1, n0 + 1),
            "y_audit": fine.values.reshape(n2 + 1, n2 + 1),
            "y_quadrature": enriched.values.reshape(n1 + 1, n1 + 1),
        }
    row["checks"] = _base_checks(row, c)
    row["passed"] = all(row["checks"].values())
    row["total_seconds"] = time.perf_counter() - start
    return arrays, audit_arrays, row


def _load_records(output, c):
    records = {}
    expected = {_sample_id(c, s, i) for s in SPLITS for i in range(c["splits"][s])}
    for path in sorted((output / "records").glob("*.json")):
        row = _read(path)
        sid = row["sample_id"]
        if sid not in expected or sid in records:
            raise ValueError("unexpected or duplicate sample record: " + sid)
        if _sample_id(c, row["split"], row["index"]) != sid or row["full_audit"] != (
            row["index"] < c["audit_samples"][row["split"]]
        ):
            raise ValueError("sample identity or audit scope mismatch: " + sid)
        if row["config_sha256"] != _identity_config(c):
            raise ValueError("sample configuration identity mismatch: " + sid)
        sample = output / "samples" / f"{sid}.npz"
        if not sample.is_file() or _sha(sample) != row["sample_sha256"]:
            raise ValueError("sample checksum mismatch: " + sid)
        if row["full_audit"]:
            audit = output / "audit_solutions" / f"{sid}.npz"
            if not audit.is_file() or _sha(audit) != row["audit_sha256"]:
                raise ValueError("audit-solution checksum mismatch: " + sid)
        if (
            row["checks"] != _base_checks(row, c)
            or not row["passed"]
            or not all(row["checks"].values())
        ):
            raise ValueError("sample quality gate failed: " + sid)
        records[sid] = row
    return records


def _identity_config(c):
    return hashlib.sha256(_canonical(c)).hexdigest()


def _checkpoint(output, c, records, stage):
    _json(
        output / "checkpoint.json",
        {
            "workflow": WORKFLOW,
            "config_sha256": _identity_config(c),
            "stage": stage,
            "completed_samples": sorted(records),
            "counts": {s: sum(r["split"] == s for r in records.values()) for s in SPLITS},
        },
    )


def _gate_report(output, c, records):
    selected = [_sample_id(c, s, i) for s, i in _selected(c)]
    manufactured = _read(output / "manufactured.json")
    passed = all(
        sid in records and records[sid]["full_audit"] and records[sid]["passed"] for sid in selected
    )
    sample_hashes = {sid: records[sid]["sample_sha256"] for sid in selected if sid in records}
    audit_hashes = {sid: records[sid]["audit_sha256"] for sid in selected if sid in records}
    reload_path = output / "gate_reload_audit.json"
    reload_report = _read(reload_path) if reload_path.exists() else {}
    passed = (
        passed
        and all(row["passed"] for row in manufactured.values())
        and reload_report.get("passed", False)
        and reload_report.get("sample_sha256") == sample_hashes
        and reload_report.get("audit_sha256") == audit_hashes
    )
    return {
        "passed": passed,
        "config_sha256": _identity_config(c),
        "selected_samples": selected,
        "audit_indices": {s: list(range(c["audit_samples"][s])) for s in SPLITS},
        "sample_sha256": sample_hashes,
        "gate_reload_audit_sha256": _sha(reload_path) if reload_path.exists() else None,
        "scope": "selected samples only: full three-grid and enriched-quadrature audit; not a full-dataset grid audit",
    }


def _verify_gate_samples(output, c, records, meshes):
    """Reload the selected committed sample files before authorizing expansion."""
    failures, residuals = [], []
    n0, n1, n2 = (c[k] for k in ("observation_n", "reference_n", "audit_n"))
    selected = [_sample_id(c, s, i) for s, i in _selected(c)]

    def check(condition, message):
        if not condition:
            failures.append(message)

    for split, index in _selected(c):
        sid = _sample_id(c, split, index)
        row = records[sid]
        parameters = draw_parameters(c, SPLITS[split], index)
        field = FieldPair(parameters, c)
        with np.load(output / "samples" / f"{sid}.npz", allow_pickle=False) as serialized:
            sample = {key: serialized[key] for key in serialized.files}
        check(set(sample) == set(ARRAY_KEYS), sid + ": schema")
        check(str(sample["sample_id"]) == sid, sid + ": ID")
        for key, expected in parameters.items():
            check(
                np.array_equal(sample[key], expected) and sample[key].dtype == np.float64,
                sid + ": parameters",
            )
        for suffix, n in (("", n0), ("_reference", n1)):
            for key, fn in (("a", field.coefficient), ("f", field.source)):
                value = sample[key + suffix]
                check(
                    value.shape == (n + 1, n + 1)
                    and value.dtype == np.float64
                    and np.isfinite(value).all(),
                    sid + ": raw field shape/precision/finite",
                )
                check(
                    np.max(np.abs(value - fn(meshes[n].nodes).reshape(n + 1, n + 1))) < 1e-13,
                    sid + ": raw field reconstruction",
                )
            value = sample["y" + suffix]
            check(
                value.shape == (n + 1, n + 1)
                and value.dtype == np.float64
                and np.isfinite(value).all(),
                sid + ": label shape/precision/finite",
            )
            check(
                np.max(np.abs(np.concatenate((value[0], value[-1], value[:, 0], value[:, -1]))))
                <= c["quality_gates"]["boundary_max"],
                sid + ": boundary",
            )
        check(np.array_equal(sample["y"], sample["y_reference"][::2, ::2]), sid + ": nested labels")
        residual = _residual(
            field, sample["y_reference"].ravel(), n1, c["quadrature_order"], meshes[n1]
        )
        residuals.append(residual)
        check(
            residual <= c["quality_gates"]["relative_algebraic_residual"],
            sid + ": independent reference residual",
        )
        with np.load(output / "audit_solutions" / f"{sid}.npz", allow_pickle=False) as serialized:
            audit = {key: serialized[key] for key in serialized.files}
        for key, n in (("y_coarse", n0), ("y_audit", n2), ("y_quadrature", n1)):
            check(
                audit[key].shape == (n + 1, n + 1)
                and audit[key].dtype == np.float64
                and np.isfinite(audit[key]).all(),
                sid + ": audit solution precision/shape",
            )
        ref = Solution(meshes[n1], sample["y_reference"].ravel(), {})
        fine = Solution(meshes[n2], audit["y_audit"].ravel(), {})
        enriched = Solution(meshes[n1], audit["y_quadrature"].ravel(), {})
        metrics = {
            "coarse_vs_reference": compare_p1(audit["y_coarse"].ravel(), n0, ref),
            "reference_vs_audit": compare_p1(ref.values, n1, fine),
            "observation_vs_audit": compare_p1(sample["y"].ravel(), n0, fine),
            "quadrature_comparison": compare_p1(ref.values, n1, enriched),
        }
        for key, metric in metrics.items():
            check(
                all(abs(metric[k] - row[key][k]) < 1e-12 for k in metric),
                sid + ": serialized " + key,
            )
    return {
        "passed": not failures,
        "failures": failures,
        "selected_samples": selected,
        "independent_weak_residual_samples": len(residuals),
        "max_independent_weak_residual": max(residuals, default=None),
        "sample_sha256": {sid: records[sid]["sample_sha256"] for sid in selected},
        "audit_sha256": {sid: records[sid]["audit_sha256"] for sid in selected},
    }


def _require_gate(output, c, records):
    report = _gate_report(output, c, records)
    if (
        not report["passed"]
        or not (output / "GATE_PASSED").is_file()
        or _read(output / "gate_report.json") != report
        or _read(output / "GATE_PASSED").get("gate_report_sha256")
        != _sha(output / "gate_report.json")
    ):
        raise RuntimeError("the selected full-audit gate has not passed; run --stage gate first")


def generate(config_path, output, stage="gate", max_new_samples=None):
    c, output = _read(config_path), Path(output)
    validate_config(c)
    if stage not in ("gate", "complete"):
        raise ValueError("stage must be gate or complete")
    if max_new_samples is not None and (
        not isinstance(max_new_samples, int) or max_new_samples < 1
    ):
        raise ValueError("max_new_samples must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        not output.exists()
        and shutil.disk_usage(output.parent).free < estimate(c)["recommended_free_bytes"]
    ):
        raise RuntimeError("insufficient free space for atomic generation/consolidation reserve")
    _initialize(c, output)
    with _locked(output):
        if (output / "COMPLETE").exists():
            report = audit_dataset(output)
            if not report["passed"]:
                raise RuntimeError("existing COMPLETE dataset failed strict audit")
            return report
        records = _load_records(output, c)
        if stage == "complete":
            _require_gate(output, c, records)
        tasks = (
            _selected(c)
            if stage == "gate"
            else [(s, i) for s in SPLITS for i in range(c["splits"][s])]
        )
        meshes = {n: make_mesh(2, n) for n in (c["observation_n"], c["reference_n"], c["audit_n"])}
        created = 0
        for split, index in tasks:
            sid = _sample_id(c, split, index)
            if sid in records:
                continue
            if max_new_samples is not None and created >= max_new_samples:
                _checkpoint(output, c, records, "paused_" + stage)
                return {
                    "passed": False,
                    "paused": True,
                    "completed": len(records),
                    "created": created,
                }
            arrays, audit_arrays, row = _build_sample(c, split, index, meshes)
            _npz(output / "samples" / f"{sid}.npz", arrays)
            row.update(
                sample_sha256=_sha(output / "samples" / f"{sid}.npz"),
                config_sha256=_identity_config(c),
            )
            if audit_arrays is not None:
                _npz(output / "audit_solutions" / f"{sid}.npz", audit_arrays)
                row["audit_sha256"] = _sha(output / "audit_solutions" / f"{sid}.npz")
            _json(output / "records" / f"{sid}.json", row)
            records[sid] = row
            _checkpoint(output, c, records, stage)
            print(
                f'{sid}: full_audit={row["full_audit"]} passed={row["passed"]} seconds={row["total_seconds"]:.3f}',
                flush=True,
            )
            if not row["passed"]:
                raise RuntimeError("sample quality gate failed; expansion stopped: " + sid)
            created += 1
        if stage == "gate":
            _json(
                output / "gate_reload_audit.json", _verify_gate_samples(output, c, records, meshes)
            )
            report = _gate_report(output, c, records)
            _json(output / "gate_report.json", report)
            if not report["passed"]:
                raise RuntimeError("selected full-audit gate failed; expansion prohibited")
            _json(output / "GATE_PASSED", {"gate_report_sha256": _sha(output / "gate_report.json")})
            _checkpoint(output, c, records, "gate_passed")
            return report
        return _finalize(output, c, records)


def _finalize(output, c, records):
    _require_gate(output, c, records)
    if len(records) != sum(c["splits"].values()):
        raise RuntimeError("cannot finalize an incomplete dataset")
    for split, count in c["splits"].items():
        arrays = {key: [] for key in ARRAY_KEYS}
        for index in range(count):
            with np.load(
                output / "samples" / f"{_sample_id(c, split, index)}.npz", allow_pickle=False
            ) as sample:
                for key in ARRAY_KEYS:
                    arrays[key].append(sample[key])
        _npz(output / f"{split}.npz", {key: np.asarray(value) for key, value in arrays.items()})
    rows = [records[_sample_id(c, s, i)] for s in SPLITS for i in range(c["splits"][s])]
    quality = {
        "passed": all(r["passed"] for r in rows),
        "samples": rows,
        "manufactured": _read(output / "manufactured.json"),
        "full_audit_counts": c["audit_samples"],
        "all_sample_counts": c["splits"],
        "full_audit_indices": {s: list(range(c["audit_samples"][s])) for s in SPLITS},
        "scope": "All samples: reference solve, algebraic/boundary/range checks. Selected subset only: three-grid and 4->5 quadrature audit.",
    }
    _json(output / "quality.json", quality)
    audited = [r for r in rows if r["full_audit"]]
    text = (
        "# 第三章公共数据：可恢复生成与分层质量核查\n\n"
        f"划分：{c['splits']}；独立种子：{c['seed']}。\n\n"
        f"全样本 {len(rows)} 条均保存原始 a/f/y 和参考网格场，并检查参考解代数残差、边界和范围。\n"
        f"仅选定 {len(audited)} 条（{c['audit_samples']}）完成三网格及积分加密核查，不代表全部样本网格审计。\n\n"
        f"抽查最大参考/加密相对 L2：{max(r['reference_vs_audit']['relative_l2'] for r in audited):.6e}；"
        f"H1 半范数：{max(r['reference_vs_audit']['relative_h1_seminorm'] for r in audited):.6e}。\n\n"
        "网格差为经验精度估计，不是连续问题严格误差界。公共 y 是参考 P1 解在公共节点的采样；"
        "原始场、参考解和跨网格重建参数不依赖模型降阶维数。逐样本 NPZ/JSON 保留以支持恢复。\n"
    )
    _atomic(output / "summary.md", lambda stream: stream.write(text.encode()))
    _checkpoint(output, c, records, "all_samples_serialized")
    files = {
        str(p.relative_to(output)): _sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in CONTROL_FILES and not p.name.startswith(".pending-")
    }
    # Keep the pilot's flat path->SHA256 schema for existing trainer/transfer.
    _json(output / "manifest.json", files)
    report = audit_dataset(output, require_complete=False)
    _json(output / "reload_audit.json", report)
    if not report["passed"]:
        raise RuntimeError("strict serialized-data audit failed; no COMPLETE marker")
    _json(
        output / "COMPLETE",
        {
            "workflow": WORKFLOW,
            "config_sha256": _identity_config(c),
            "manifest_sha256": _sha(output / "manifest.json"),
            "reload_audit_sha256": _sha(output / "reload_audit.json"),
        },
    )
    return report


def _residual(fields, values, n, order, mesh):
    """Reassemble the weak residual from serialized labels without re-solving."""
    points, weights, phi = mesh.quadrature(order)
    coefficient, source = fields.coefficient(points), fields.source(points)
    local = np.einsum(
        "ead,ebd,e->eab", mesh.gradients, mesh.gradients, np.sum(weights * coefficient, axis=1)
    )
    rows = np.broadcast_to(mesh.elements[:, :, None], local.shape)
    columns = np.broadcast_to(mesh.elements[:, None, :], local.shape)
    matrix = sparse.coo_matrix(
        (local.ravel(), (rows.ravel(), columns.ravel())), shape=(len(mesh.nodes), len(mesh.nodes))
    ).tocsr()
    load = np.zeros(len(mesh.nodes))
    np.add.at(load, mesh.elements.ravel(), np.einsum("eq,eq,qa->ea", weights, source, phi).ravel())
    interior = mesh.interior
    return float(
        np.linalg.norm((matrix @ values - load)[interior])
        / max(np.linalg.norm(load[interior]), np.finfo(float).tiny)
    )


def audit_dataset(output, require_complete=True, recompute_residuals=True):
    """Strict, read-only audit, including independent reference weak residuals."""
    output = Path(output)
    failures, stats = [], {}

    def check(condition, message):
        if not condition:
            failures.append(message)

    try:
        c = _read(output / "config.json")
        validate_config(c)
        identity = _read(output / "identity.json")
        check(
            identity["workflow"] == WORKFLOW and identity["config_sha256"] == _identity_config(c),
            "configuration identity",
        )
        check(
            identity["audit_indices"] == {s: list(range(c["audit_samples"][s])) for s in SPLITS},
            "predeclared audit indices",
        )
        for name, digest in identity["source_sha256"].items():
            check(_sha(output / "source_snapshot" / name) == digest, "source snapshot: " + name)
        manifest = _read(output / "manifest.json")
        for relative, digest in manifest.items():
            path = Path(relative)
            check(not path.is_absolute() and ".." not in path.parts, "unsafe manifest path")
            if path.is_absolute() or ".." in path.parts:
                continue
            check(
                (output / path).is_file() and _sha(output / path) == digest, "checksum: " + relative
            )
        actual = {
            str(p.relative_to(output))
            for p in output.rglob("*")
            if p.is_file() and p.name not in CONTROL_FILES and not p.name.startswith(".pending-")
        }
        check(actual == set(manifest), "manifest file coverage")
        if require_complete:
            complete = _read(output / "COMPLETE")
            check(
                complete["manifest_sha256"] == _sha(output / "manifest.json"), "COMPLETE manifest"
            )
            check(
                complete["reload_audit_sha256"] == _sha(output / "reload_audit.json"),
                "COMPLETE reload audit",
            )
        records = _load_records(output, c)
        _require_gate(output, c, records)
        quality = _read(output / "quality.json")
        check(
            quality["samples"]
            == [records[_sample_id(c, s, i)] for s in SPLITS for i in range(c["splits"][s])],
            "quality rows match immutable records",
        )
        n0, n1, n2 = (c[k] for k in ("observation_n", "reference_n", "audit_n"))
        meshes = {n: make_mesh(2, n) for n in (n0, n1, n2)}
        for name, n in (("observation", n0), ("reference", n1)):
            with np.load(output / f"{name}_geometry.npz", allow_pickle=False) as geometry:
                check(
                    np.array_equal(geometry["coordinates"].reshape(-1, 2), meshes[n].nodes),
                    "geometry coordinates",
                )
                check(np.array_equal(geometry["elements"], meshes[n].elements), "geometry topology")
                check(
                    abs(geometry["lumped_mass_weights"].sum() - 1) < 1e-13
                    and np.min(geometry["lumped_mass_weights"]) > 0,
                    "positive unit mass",
                )
        seen_fields, seen_ids, residuals = set(), set(), []
        for split, count in c["splits"].items():
            with np.load(output / f"{split}.npz", allow_pickle=False) as serialized:
                data = {key: serialized[key] for key in serialized.files}
                check(set(data) == set(ARRAY_KEYS), split + ": schema")
                for key in ARRAY_KEYS:
                    check(data[key].shape[0] == count, split + "/" + key + ": count")
                for index in range(count):
                    sid = _sample_id(c, split, index)
                    check(
                        str(data["sample_id"][index]) == sid and sid not in seen_ids,
                        "sample ID: " + sid,
                    )
                    seen_ids.add(sid)
                    with np.load(output / "samples" / f"{sid}.npz", allow_pickle=False) as sample:
                        check(set(sample.files) == set(ARRAY_KEYS), sid + ": sample schema")
                        for key in ARRAY_KEYS:
                            check(
                                np.array_equal(sample[key], data[key][index]),
                                sid + ": consolidated equality " + key,
                            )
                    params = draw_parameters(c, SPLITS[split], index)
                    for key, value in params.items():
                        check(
                            np.array_equal(value, data[key][index]),
                            sid + ": deterministic parameters",
                        )
                    field = FieldPair(params, c)
                    for suffix, n in (("", n0), ("_reference", n1)):
                        for key, function in (("a", field.coefficient), ("f", field.source)):
                            values = data[key + suffix][index]
                            check(
                                values.shape == (n + 1, n + 1)
                                and values.dtype == np.float64
                                and np.isfinite(values).all(),
                                sid + ": raw shape/dtype/finite",
                            )
                            check(
                                np.max(
                                    np.abs(function(meshes[n].nodes).reshape(n + 1, n + 1) - values)
                                )
                                < 1e-13,
                                sid + ": function reconstruction",
                            )
                        label = data["y" + suffix][index]
                        check(
                            label.shape == (n + 1, n + 1)
                            and label.dtype == np.float64
                            and np.isfinite(label).all(),
                            sid + ": label shape/dtype/finite",
                        )
                        check(
                            np.max(
                                np.abs(
                                    np.concatenate((label[0], label[-1], label[:, 0], label[:, -1]))
                                )
                            )
                            <= c["quality_gates"]["boundary_max"],
                            sid + ": boundary",
                        )
                    check(
                        np.array_equal(data["y"][index], data["y_reference"][index, ::2, ::2]),
                        sid + ": nested labels",
                    )
                    digest = hashlib.sha256(
                        data["a"][index].tobytes() + data["f"][index].tobytes()
                    ).hexdigest()
                    check(digest not in seen_fields, sid + ": duplicate input pair")
                    seen_fields.add(digest)
                    if recompute_residuals:
                        residual = _residual(
                            field,
                            data["y_reference"][index].ravel(),
                            n1,
                            c["quadrature_order"],
                            meshes[n1],
                        )
                        residuals.append(residual)
                        check(
                            residual <= c["quality_gates"]["relative_algebraic_residual"],
                            sid + ": independent weak residual",
                        )
                    if records[sid]["full_audit"]:
                        with np.load(
                            output / "audit_solutions" / f"{sid}.npz", allow_pickle=False
                        ) as audit:
                            fine = Solution(meshes[n2], audit["y_audit"].ravel(), {})
                            observed = compare_p1(data["y"][index].ravel(), n0, fine)
                            reference = compare_p1(data["y_reference"][index].ravel(), n1, fine)
                            for key, metric in (
                                ("observation_vs_audit", observed),
                                ("reference_vs_audit", reference),
                            ):
                                check(
                                    all(
                                        abs(metric[k] - records[sid][key][k]) < 1e-12
                                        for k in metric
                                    ),
                                    sid + ": serialized mesh comparison",
                                )
                stats[split] = {
                    "count": count,
                    "a_min": float(data["a"].min()),
                    "a_max": float(data["a"].max()),
                    "f_min": float(data["f"].min()),
                    "f_max": float(data["f"].max()),
                }
        check(len(records) == sum(c["splits"].values()), "all sample records present")
        check(
            all(r["passed"] for r in _read(output / "manufactured.json").values()),
            "manufactured convergence",
        )
        return {
            "passed": not failures,
            "failures": failures,
            "split_statistics": stats,
            "unique_ids": len(seen_ids),
            "unique_input_pairs": len(seen_fields),
            "full_audit_counts": c["audit_samples"],
            "full_audit_scope": "selected subset, not all samples",
            "independent_weak_residual_samples": len(residuals),
            "max_independent_weak_residual": max(residuals, default=None),
        }
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        TypeError,
        zipfile.BadZipFile,
        EOFError,
    ) as error:
        failures.append(f"audit exception: {type(error).__name__}: {error}")
        return {"passed": False, "failures": failures}


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--config", required=True, type=Path)
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--stage", choices=("gate", "complete"), default="gate")
    gen.add_argument("--max-new-samples", type=int)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", required=True, type=Path)
    cost = commands.add_parser("estimate")
    cost.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        report = generate(args.config, args.output, args.stage, args.max_new_samples)
    elif args.command == "audit":
        report = audit_dataset(args.output)
    else:
        c = _read(args.config)
        validate_config(c)
        report = estimate(c)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if report.get("passed") is False and not report.get("paused"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
