# English edition v1.0.1

Prepared on 2026-11-02 (Asia/Shanghai) from v1.0.0, commit `e275a39b25feb1e646e726cb6bf3ca6d2af708ea`.

All Chinese prose in the current source distribution has been translated: the repository README, installation and chapter guides, manuscript mapping, report generators, generated result summaries, review notes, and the LaTeX/PDF audit report. Historical `_ZH` filename suffixes remain to preserve links and entry points. New reports produced by the translated generators are English.

This is a language revision, not a retraining run. Numerical algorithms, mathematical expressions, formal budgets, random seeds, model architectures, parameter counts, datasets, and selected model weights are unchanged. The 156 formal training runs and their scientific conclusions still refer to the v1.0.0 delivery. Previously viewed tests were not used to select or tune models.

## Source and artifact identities

- `PAYLOAD_MANIFEST.json` and `RUNTIME_MANIFEST.json` now identify the English source distribution. The package version and fresh-workspace marker are 1.0.1.
- `ORIGINAL_SOURCE_IDENTITIES.json` preserves the v1.0.0 source/runtime manifests and workspace marker as structured values. Original source bytes and the Chinese report remain available under the v1.0.0 Git tag and Release.
- `LOCALIZATION_MANIFEST.json` records original and English hashes for changed tracked files. `ENGLISH_VERIFICATION.json` records language, source-equivalence, package, and regression checks.
- Historical scientific audit and checkpoint-selection records are retained unchanged. Their hashes refer to the original sources and reports, not the English renderings. A passing historical audit is not relabeled as a new audit of translated bytes.
- The English archive has its own `PACKAGE_MANIFEST.json`; Release assets have new SHA256 checksums. All 180 selected checkpoint files remain byte-identical to v1.0.0.

## Using the sources and results

Install the 1.0.1 wheel or the repository and initialize a fresh workspace for new runs. The compact experiment archive supports inspection and retained-model inference; training/validation samples and recovery checkpoints remain deliberately excluded. It is not an exact-resumption backup and cannot rerun all historical audit procedures without the omitted original assets.

To verify original training provenance, use the original v1.0.0 source and manifests. To verify the translated release, use the 1.0.1 localization, source, package, and asset manifests. No frozen server project was overwritten.

Third-party licenses and original attribution remain unchanged. The manuscript itself is not included in the repository and was not translated as part of this software localization.
