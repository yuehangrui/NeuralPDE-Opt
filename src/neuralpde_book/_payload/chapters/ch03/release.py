"""Read-only release verification; includes old freeze and new strict manifests."""

import subprocess
import sys
from bootstrap import ROOT
from integrity import require, read, sha, check_manifest, inventory, verify
from registry import source_manifest

NAME = "ch03-complete-v2.0-20261023"


def verify_release(code_only=False, archives=False, history=False):
    folder = ROOT / "releases" / NAME
    require(
        (folder / "release.json.sha256").read_text().split()[0] == sha(folder / "release.json"),
        "Release descriptor changed",
    )
    descriptor = read(folder / "release.json")
    require(
        descriptor["name"] == NAME and descriptor["training_runs"] == 57,
        "Release identity mismatch",
    )
    groups = ["code"] if code_only else ["code", "artifacts"] + (["preserved"] if history else [])
    for group in groups:
        spec = descriptor["manifests"][group]
        require(sha(ROOT / spec["path"]) == spec["sha256"], "Manifest changed: " + group)
        check_manifest(ROOT, read(ROOT / spec["path"]))
    code = read(folder / "code_manifest.json")
    require(all(code.get(k) == v for k, v in source_manifest().items()), "Unfrozen active code")
    if code_only:
        print("VERIFIED_CODE_ONLY", NAME)
        return
    for base in descriptor["managed_artifact_directories"]:
        expected = {
            k.removeprefix(base + "/"): v
            for k, v in read(folder / "artifacts_manifest.json").items()
            if k.startswith(base + "/")
        }
        require(
            inventory(ROOT / base) == expected, "Unexpected artifact or content change in " + base
        )
    verify(ROOT / "results/ch03/final_test_v1")
    verify(ROOT / "data/ch03/final_test_v1")
    if history:
        previous = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/verify_frozen_release.py"),
                "--artifacts",
                "--archive",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        require(
            previous.returncode == 0,
            "Original frozen release no longer verifies: " + previous.stdout + previous.stderr,
        )
    if archives:
        require(
            (folder / "archives.json.sha256").read_text().split()[0]
            == sha(folder / "archives.json"),
            "Archive descriptor changed",
        )
        for archive in read(folder / "archives.json"):
            require(sha(ROOT / archive["path"]) == archive["sha256"], "Archive changed")
    print(
        "VERIFIED",
        NAME,
        "57 training runs; 54 model tests; 448 HINTS solves; history_checked=",
        history,
    )
