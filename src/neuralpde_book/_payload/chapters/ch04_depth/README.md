# ch04_depth: standalone module

See the [Chapter 4 guide](../../docs/CH04_READING_ZH.md) for notation, formulas, and reading order. This directory retains the verified numerical core; use the [standalone instructions](../../README.md) for the unified entry point.

Create a new workspace with the installed `neuralpde-book`, then run `check -> prepare -> smoke -> pipeline`. All training assets are generated there. Formal models train from scratch; HINTS uses only the VarMiON initialization from the same run.

Low-level CLIs in this directory are called by the unified driver. The standalone edition uses workspace-root `book_driver.py` to orchestrate the full workflow. Formal results go to the current workspace's `results/` without reading the original project. See the guide and configurations for numerical methods, budgets, and representation limits; root `PAYLOAD_MANIFEST.json` records the current source identity.
