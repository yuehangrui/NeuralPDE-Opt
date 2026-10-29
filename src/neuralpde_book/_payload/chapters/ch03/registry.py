"""Explicit scope and all completed training cases, without a retired CG gate."""

from bootstrap import ROOT
from integrity import read

EXPERIMENTS = [
    "ch03_pod_v1",
    "ch03_physics_v1",
    "ch03_varmion_v1",
    "ch03_varmion_v2",
    "ch03_hints_v1",
]


def runs():
    rows = []

    def add(family, variant, seed, relative, epochs, samples, batch, dataset):
        rows.append(
            dict(
                id=f"{family}__{variant}__s{seed}",
                family=family,
                variant=variant,
                seed=seed,
                path="results/ch03/" + relative + "/run",
                epochs=epochs,
                samples=samples,
                batch=batch,
                dataset=dataset,
            )
        )

    for seed in [0, 1, 2]:
        for name in ["fno", "deepxde_fourier4", "gno", "galerkin"]:
            add(
                "baseline",
                name,
                seed,
                f"reaction_training_v1/seed_{seed}/{name}",
                1600,
                128,
                16,
                "fixed",
            )
        for rank in [4, 8, 16, 32]:
            add(
                "pod",
                f"rank_{rank}",
                seed,
                f"pod_v1/rank_{rank}/seed_{seed}",
                1600,
                128,
                16,
                "fixed",
            )
        for name in ["supervised_128", "supervised_32", "energy_0", "hybrid_32", "hybrid_128"]:
            add("physics", name, seed, f"physics_v1/{name}/seed_{seed}", 400, 128, 16, "fixed")
        for name in ["varmion", "unstructured"]:
            add("varmion", name, seed, f"varmion_v1/{name}/seed_{seed}", 1600, 256, 32, "cf")
        for name in ["dense_data", "spd_joint_aug128"]:
            add("varmion_v2", name, seed, f"varmion_v2/{name}/seed_{seed}", 800, 256, 32, "cf")
        add(
            "hints",
            "residual",
            seed,
            f"hints_v1/residual_models/seed_{seed}",
            400,
            640,
            32,
            "residual",
        )
    for name in ["dense_joint", "spd_data", "spd_joint"]:
        add("varmion_v2", name, 0, f"varmion_v2/{name}/seed_0", 800, 256, 32, "cf")
    return rows


def source_manifest():
    from integrity import sha

    paths = []
    for directory in ["src", "chapters/ch03", "third_party"] + [
        "experiments/" + n for n in EXPERIMENTS
    ]:
        paths += [
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
