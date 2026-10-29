"""Explicit fresh-output entry points; no automatic test evaluation."""

import argparse
import json
from pathlib import Path
from .data import create
from .training import Config, train


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int)
    t.add_argument("--device")
    a = p.parse_args()
    if a.command == "data":
        print(json.dumps(create(a.output)))
        return
    cfg = Config(**json.loads(Path(a.config).read_text()))
    if a.epochs is not None:
        cfg.epochs = a.epochs
    if a.device is not None:
        cfg.device = a.device
    train(cfg, a.data, a.output)


if __name__ == "__main__":
    main()
