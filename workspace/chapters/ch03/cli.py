"""Unified Chapter 3 commands. Default actions never train or overwrite results."""

import argparse
from pathlib import Path
from bootstrap import ROOT
from integrity import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    sub.add_parser("register-test")
    sub.add_parser("generate-test")
    sub.add_parser("evaluate-test")
    verification = sub.add_parser("verify")
    verification.add_argument("--code-only", action="store_true")
    verification.add_argument("--archives", action="store_true")
    verification.add_argument("--history", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--run", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--smoke-epochs", type=int)
    args = parser.parse_args()
    final = ROOT / "results/ch03/final_test_v1"
    if args.command == "inventory":
        import json
        from registry import runs

        print(json.dumps(runs(), indent=2))
    elif args.command == "audit":
        from audit import run

        require(not args.output.exists(), "Audit output already exists")
        run(args.output)
    elif args.command == "check":
        from checks import run

        run(args.output)
    elif args.command == "register-test":
        from heldout import register

        register(final)
    elif args.command == "generate-test":
        from heldout import generate

        generate(final)
    elif args.command == "evaluate-test":
        from evaluation import run

        run(final)
    elif args.command == "train":
        from training import run

        run(args.run, args.output, args.resume, args.smoke_epochs)
    elif args.command == "verify":
        from release import verify_release

        verify_release(args.code_only, args.archives, args.history)


if __name__ == "__main__":
    main()
