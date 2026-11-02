# NeuralPDE-Opt standalone edition: compact release archive

English edition 1.0.1, 2026-11-02. Scientific results are from standalone version 1.0.0: 156 formal neural runs, 8 FEM runs, new tests, and audits completed on 2026-10-29.

## Included material

- workspace/: 317 manifest-listed numerical source files, configurations, generators, instructions, and licenses. English presentation changes are recorded separately from the original training identities.
- 180 formally selected models: Chapter 2, 18 final models; Chapter 3, 57; Chapter 4 baseline, 72 model stages; Chapter 4 extensions, 12; Chapter 5, 21. One Chapter 4 training run may include multiple delivery stages, so model and training counts differ. See MODEL_INDEX.json for model identities and hashes.
- Training curves, scalar metrics, final test data/predictions, classical references, HINTS solutions, inference-time POD/VarMiON bases, and normalization statistics.
- workspace/reports/publication/main.pdf and main.tex: the English book-wide checklist and audit report. Other logs and audit evidence are retained.
- wheel/: installable package; package_source/: package source and tests.

## Excluded material

Original training/validation samples, sampling caches, training/validation prediction arrays, initial/last/recovery checkpoints, intermediate models absent from formal selection manifests, run locks, and development trial runs are excluded. Retained selected checkpoints preserve their original bytes and may still include optimizer states; models were not rewritten to reduce size.

EXCLUDED_FILES.json records exclusions from the original full download and the compact archive. FULL_PACKAGE_MANIFEST.json preserves the original full archive identity; PACKAGE_MANIFEST.json describes the current archive. Historical FINAL_AUDIT and selection-freeze manifests remain unchanged. Some referenced training samples and recovery checkpoints are intentionally absent, and translated files have new identities; the archive cannot rerun every historical audit directly.

## Reading and verification

Extract into a new independent directory. Start with workspace/reports/publication/main.pdf and workspace/reports/REVIEW_NOTES_ZH.md. The inherited Chapter 4 summary wording is corrected in summary.reviewed.md. Original audit records are retained.

Run from the extracted archive root:

```bash
python verify_package.py
```

This command verifies individual files using only the Python standard library and does not train models. The ZIP checksum is provided in the Release SHA256SUMS file.

To install the package:

```bash
python -m pip install wheel/neuralpde_book-1.0.1-py3-none-any.whl
neuralpde-book status --workspace /absolute/path/to/workspace
```

Scientific dependencies such as Python/PyTorch/NumPy/SciPy/Matplotlib are still required; see package_source/requirements-tested.txt. Full Chapter 3 training uses CUDA. A Python environment is not included. Rebuild data in a fresh workspace for retraining. This archive is not a complete training-data backup and cannot exactly resume every run.

## Release use

This ZIP is a release attachment. Installable sources are in package_source/, with licenses and provenance preserved. Models, predictions, and large results are included here rather than in Git. Original v1.0.0 artifacts remain available for historical verification.
