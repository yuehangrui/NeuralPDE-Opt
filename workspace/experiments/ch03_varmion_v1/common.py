import hashlib, json, os
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def certify(path, marker):
    path = Path(path)
    dump(
        path / "manifest.json",
        {
            p.relative_to(path).as_posix(): sha(p)
            for p in path.rglob("*")
            if p.is_file() and p != path / "manifest.json" and p.name not in {marker, ".run.lock"}
        },
    )
    dump(path / marker, {"manifest_sha256": sha(path / "manifest.json")})


def verify(path, marker):
    path = Path(path)
    if json.loads((path / marker).read_text())["manifest_sha256"] != sha(path / "manifest.json"):
        raise ValueError("Manifest changed")
    for name, digest in json.loads((path / "manifest.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(path / name) != digest:
            raise ValueError("Artifact changed: " + name)
