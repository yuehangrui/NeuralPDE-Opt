# NeuralPDE-Opt standalone book companion

Covers the existing methods from Chapters 2-5. Depends on Python, PyTorch, NumPy, SciPy, and Matplotlib, without requiring upstream repositories, the original NeuralPDE-Opt directory, or a fixed server path. Ported third-party sources, licenses, and provenance are included. The edition is not claimed to be entirely original, and different licenses are not replaced by one MIT license.

An installable control package provides a numerical workspace template. `init` expands the complete source, configurations, and data generators into a user-specified new directory. Training uses independent subprocesses and explicit import paths within that directory, preserving the verified numerical modules. Installation and output locations may differ. Models, data, and formal results do not depend on an external legacy project.

```bash
python -m pip install neuralpde_book-1.0.1-py3-none-any.whl
neuralpde-book init /absolute/path/my-experiments
neuralpde-book check --workspace /absolute/path/my-experiments
neuralpde-book prepare --workspace /absolute/path/my-experiments
neuralpde-book smoke --workspace /absolute/path/my-experiments
neuralpde-book pipeline --workspace /absolute/path/my-experiments
neuralpde-book status --workspace /absolute/path/my-experiments
```

No training starts by default. Full Chapter 3 budgets retain the CUDA configuration. Chapter 2 and some modules support CPU, but the complete suite currently requires CUDA. Run numerical regression and small-budget checks first. Old checkpoints are not used to initialize formal retraining; HINTS uses only the VarMiON initialization newly trained in the same workspace. POD bases, normalization, and all training data can be generated there.

Sources are fixed before training, and new models carry this version's source identity. Chapters 3-5 generate final tests from new predetermined random streams only after model selection is frozen. Chapter 2 retains the final iterate and uses independent quadrature and FEM validation. State-splitting ADMM, interface control, obstacle control, and other deferred methods are not started by this edition.

`PAYLOAD_MANIFEST.json` records the current source identity. Historical `PORT_MANIFEST.json` files retain provenance; new runs use `RUNTIME_MANIFEST.json`. `SOURCE_CHANGES.json` records packaging and protocol changes. Historical provenance must not be rewritten as though it were the identity of the translated source. English-edition changes are recorded in the release localization manifest.

Interrupted training retains artifacts. Chapters 2 and 3 support checkpoint recovery; other workflows skip only audited completed tasks and do not promise exact optimizer resumption. Failed runs are not automatically overwritten or rerun with reduced budgets. The final checklist and audit report are generated only after all training, independent evaluation, and audits finish.
