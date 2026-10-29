"""Isolated test processes avoid generic historical module-name collisions."""

import os
import re
import subprocess
import sys
from bootstrap import ROOT, HERE
from integrity import require, dump
from registry import EXPERIMENTS


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    suites = (
        [("core", ROOT / "tests")]
        + [(n, ROOT / "experiments" / n) for n in EXPERIMENTS]
        + [("release", HERE)]
    )
    reports = []
    for name, directory in suites:
        env = os.environ.copy()
        paths = [
            directory,
            ROOT / "src",
            ROOT / "experiments/ch03_varmion_v1",
            ROOT / "experiments/ch03_physics_v1",
        ]
        env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(directory),
            "-p",
            "test*.py",
            "-v",
        ]
        done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        text = done.stdout + done.stderr
        (output / (name + ".log")).write_text(text, encoding="utf-8")
        count = re.search(r"Ran (\d+) tests? in", text)
        reports.append(
            dict(suite=name, exit_code=done.returncode, tests=int(count.group(1)) if count else 0)
        )
        dump(
            output / "results.json",
            dict(
                passed=all(r["exit_code"] == 0 and r["tests"] > 0 for r in reports), suites=reports
            ),
        )
        print("SUITE", reports[-1], flush=True)
    require(
        all(r["exit_code"] == 0 and r["tests"] > 0 for r in reports),
        "One or more test suites failed",
    )
    # Explicitly exercise fail-closed gates when Python assertions are disabled.
    done = subprocess.run(
        [sys.executable, "-O", "-B", str(HERE / "test_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (output / "optimized_release.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    require(done.returncode == 0, "Optimized-mode regression failure")
    dump(
        output / "results.json",
        dict(
            passed=True,
            suites=reports,
            total_tests=sum(r["tests"] for r in reports),
            optimized_mode_passed=True,
        ),
    )
