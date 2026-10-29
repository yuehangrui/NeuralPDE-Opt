"""Entry point with explicit config/output; see the chapter README examples."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from .training import run_one, save_json, environment
from .problems import Poisson
from .fem import solve
from .paths import output_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dims", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--methods", nargs="+", choices=["pinn", "ritz", "vpinn"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(Path(args.config).read_text())
    output = output_directory(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        env_file = output/"environment.json"
        current_env = environment()
        if not env_file.exists():
            save_json(env_file, current_env)
        else:
            old_env = json.loads(env_file.read_text())
            if old_env["source_sha256"] != current_env["source_sha256"]:
                raise RuntimeError("Source changed since this experiment began; use a new output directory")
        save_json(output/"suite_config.json", config)
        fem_dir = output/"fem"
        fem_dir.mkdir(exist_ok=True)
        for dim in args.dims:
            for n in config["fem_levels"][str(dim)]:
                path = fem_dir/f"poisson{dim}d_n{n}.json"
                if not path.exists():
                    mesh, values, row = solve(Poisson(dim), n)
                    save_json(path, row)
                    np.savez(fem_dir/f"poisson{dim}d_n{n}.npz", nodes=mesh.nodes, elements=mesh.elements, values=values)
                    print(f"FEM {dim}D n={n} L2={row['l2_rel']:.3e} H1={row['h1_semi_rel']:.3e}", flush=True)
        errors = []
        for dim in args.dims:
            for seed in (args.seeds if args.seeds is not None else config["seeds"]):
                for method in (args.methods or config["methods"]):
                    try:
                        run_one(config, dim, method, seed, output, args.device)
                    except Exception:
                        text = traceback.format_exc()
                        print(text, flush=True)
                        errors.append(dict(dim=dim, seed=seed, method=method, traceback=text))
                        save_json(output/"failures.json", errors)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        if errors:
            raise RuntimeError(f"{len(errors)} runs failed; see failures.json. Completed runs are retained.")
        if (output/"failures.json").exists():
            (output/"failures.json").unlink()
    from .report import generate
    generate(output)


if __name__ == "__main__":
    main()
