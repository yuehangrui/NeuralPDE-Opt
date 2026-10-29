"""Fail-closed certificates, safe paths and atomic JSON without assert gates."""

import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def inside(root, name):
    root = Path(root).resolve()
    p = Path(name)
    require(bool(p.parts) and not p.is_absolute() and ".." not in p.parts, "Unsafe manifest path")
    path = root / p
    require(
        path.resolve().is_relative_to(root) and not path.is_symlink(),
        "Path escapes root or is symlink",
    )
    return path


def inventory(root, excluded=()):
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".pyo"}
        and p.name not in set(excluded) | {".run.lock"}
    }


def check_manifest(root, records):
    for name, record in records.items():
        p = inside(root, name)
        expected = record["sha256"] if isinstance(record, dict) else record
        require(p.is_file() and sha(p) == expected, "Missing or changed file: " + str(p))


def certify(root, marker="COMPLETE"):
    root = Path(root)
    dump(root / "manifest.json", inventory(root, ["manifest.json", marker]))
    dump(root / marker, {"schema": "ch03-final-v1", "manifest_sha256": sha(root / "manifest.json")})


def verify(root, marker="COMPLETE", strict=True):
    root = Path(root)
    cert = read(root / marker)
    require(cert["manifest_sha256"] == sha(root / "manifest.json"), "Certificate/manifest mismatch")
    records = read(root / "manifest.json")
    check_manifest(root, records)
    if strict:
        require(
            set(records) == set(inventory(root, ["manifest.json", marker])),
            "Unexpected or missing artifact",
        )
    return records
