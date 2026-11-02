# NeuralPDE-Opt: standalone book companion

Companion implementations for the agreed scope of Chapters 2-5. Version 1.0.0 completed 156 formal neural training runs, 8 FEM runs, new tests, and independent audits. Version 1.0.1 provides the English edition of that delivery. Original training/validation datasets are excluded; installation does not start training.

## Reading and downloads

- [Book-wide checklist and audit report](workspace/reports/publication/main.pdf), [LaTeX source](workspace/reports/publication/main.tex)
- [Manuscript-to-code mapping](workspace/reports/BOOK_CHECKLIST_ZH.md), [audit report](workspace/reports/AUDIT_REPORT_ZH.md), [review notes](workspace/reports/REVIEW_NOTES_ZH.md)
- [English release v1.0.1](https://github.com/yuehangrui/NeuralPDE-Opt/releases/tag/v1.0.1): approximately 998 MiB experiment archive, installation wheel, report, and SHA256 checksums. The archive contains 180 selected/stage models, final tests and predictions, model bases, scalar training histories, and audit evidence.

The Git repository contains source code, documentation, and textual/figure evidence. Models and test arrays are provided as Release attachments. Historical `_ZH` filename suffixes are retained for stable links; their current contents are English. See [English-edition provenance](release/ENGLISH_EDITION.md) for the distinction between translated files and original training identities.

## Installation and execution

```bash
git clone git@github.com:yuehangrui/NeuralPDE-Opt.git
cd NeuralPDE-Opt
python -m pip install .
neuralpde-book init /absolute/path/new-experiments
neuralpde-book status --workspace /absolute/path/new-experiments
```

`init` expands a new workspace and refuses to overwrite an existing directory. `src/neuralpde_book/_payload/` is the installable numerical source template; `workspace/` contains the corresponding source and experiment-evidence snapshot. The 317 manifest-listed source files match between the two copies. Start new training in a fresh workspace.

To reproduce the experiments from scratch, explicitly run the following commands in a new directory:

```bash
neuralpde-book check --workspace /absolute/path/new-experiments
neuralpde-book prepare --workspace /absolute/path/new-experiments
neuralpde-book smoke --workspace /absolute/path/new-experiments
neuralpde-book pipeline --workspace /absolute/path/new-experiments
```

These commands perform regression checks, generate data, run small-budget verification, and execute full training/evaluation, respectively. The complete suite requires CUDA; see the [tested dependencies](requirements-tested.txt). A clean-operating-system installation or fully offline dependency installation has not been verified. The package does not require the original project directory or upstream repositories, but still depends on Python, PyTorch, NumPy, SciPy, and Matplotlib.

## Completed experiments

| Part | Formal training | Models/results |
|---|---:|---|
| Chapter 2 | 18 neural runs, plus 8 FEM runs | PINN, Deep Ritz, weak form; 18 final models |
| Chapter 3 | 57 runs | Operator learning, physical/variational extensions, VarMiON, HINTS; 57 models |
| Chapter 4 baseline | 48 runs | Optimal-control baselines and operator learning; 72 model stages |
| Chapter 4 extensions | 12 runs | Nonlinear Deep Uzawa, sparse-observation FEM-NN; 12 models |
| Chapter 5 | 21 runs | Agreed constrained-control workflows; 21 models |

The original delivery passed 334 numerical regression tests and independent audits. Model and training counts differ because Chapter 4 retains multiple delivery stages. Completing all budgets does not imply uniform accuracy, strict optimality, or an architecture-family ranking. The test sets have been viewed; subsequent development must use train/val. State-splitting ADMM in Section 5.2.2, Sections 5.3-5.4, and other deferred branches are not counted as complete; see the full checklist for precise coverage.

## Verification and provenance

After extracting the Release archive, run `python verify_package.py` to verify its files. Training data and some recovery files are deliberately excluded from both the repository and the compact archive. The complete artifacts referenced by the historical FINAL_AUDIT can be verified only in the original full workspace. English translations have new hashes recorded separately; do not treat the translated snapshot as a rerun of the historical audit or as an exact-resumption backup.

The original Chapter 4 summary contains inherited template text claiming that 9 seed0 runs were reused. Use the [corrected summary](workspace/results/ch04/unified_delivery_v1/summary.reviewed.md) and review notes: all 48 runs in this standalone delivery were trained from scratch. Historical audit identities remain unchanged; original bytes remain available in v1.0.0.

Ported code retains its original licenses and provenance: [third-party notices](src/neuralpde_book/_payload/third_party/ch03_upstream/NOTICE.md), [source origins](workspace/SOURCE_ORIGINS.json). DeepXDE components retain LGPL-2.1, Galerkin Transformer components retain MIT, and ocpde_dl provenance is preserved. The repository is not relicensed uniformly as MIT and is not claimed to be entirely original.
