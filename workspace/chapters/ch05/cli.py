"""Run from repository root: python -B -m chapters.ch05.cli --help."""

import argparse, json, sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "chapters.ch05"
from .common import Config, INSTANCE, OPERATORS, smoke_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true")
    d.add_argument("--grid", type=int, default=32)
    d.add_argument("--counts", type=int, nargs=3, default=[1024, 128, 128])
    d.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--data")
    t.add_argument("--method", choices=INSTANCE + OPERATORS)
    t.add_argument("--config")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--device", default="cpu")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--data")
    e.add_argument("--split", choices=["val", "test"], default="val")
    e.add_argument("--device", default="cpu")
    r = sub.add_parser("reference")
    r.add_argument("--grid", type=int, nargs="+", default=[16, 32, 64])
    r.add_argument("--out", required=True)
    sub.add_parser("check")
    args = parser.parse_args()
    if args.command == "check":
        import unittest
        from . import test_ch05

        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(test_ch05)
        )
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.command == "data":
        from .data import generate

        result = generate(
            args.out,
            8 if args.smoke else args.grid,
            (12, 4, 4) if args.smoke else tuple(args.counts),
            args.seed,
        )
    elif args.command == "train":
        from .training import train

        if args.smoke:
            if not args.method or args.config:
                parser.error("--smoke requires --method and excludes --config")
            c = smoke_config(args.method, args.device)
        else:
            if not args.config:
                parser.error("formal training requires an explicit --config")
            c = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
            c.device = args.device
            if args.method and args.method != c.method:
                parser.error("--method differs from config")
        result = train(c, args.out, args.data)
    else:
        out = Path(args.out)
        if out.exists():
            raise FileExistsError(out)
        if args.command == "reference":
            from .reference import Grid
            from .problem import Problem
            from .evaluation import rel

            p = Problem()
            result = []
            for n in args.grid:
                g = Grid(n)
                ref = g.optimal_control(p.f(g.points)[..., 0], p.yd(g.points)[..., 0], p)
                result.append(
                    dict(
                        n=n,
                        iterations=ref["iterations"],
                        projected_residual=float(ref["projected_residual"]),
                        **{
                            "relative_" + k: rel(ref[k], getattr(p, k)(g.points)[..., 0])
                            for k in ("y", "u", "p")
                        },
                    )
                )
        else:
            from .training import load
            from .evaluation import instance_metrics, operator_metrics

            model, c, split, blob = load(args.checkpoint, args.device)
            if c.method in INSTANCE:
                result = instance_metrics(
                    model, c, split, seed=c.seed + (10001 if args.split == "val" else 20001)
                )
            else:
                if not args.data:
                    parser.error("operator evaluation requires --data")
                from .data import load_split

                data, meta = load_split(args.data, args.split, c.problem(), c.grid)
                if blob["data_identity"] != {
                    s: meta["splits"][s]["sha256"] for s in ("train", "val")
                }:
                    raise ValueError("evaluation dataset identity differs from training run")
                result = operator_metrics(model, c, data)
            result.update(
                split=args.split,
                checkpoint=str(Path(args.checkpoint).resolve()),
                test_exposed=args.split == "test",
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
