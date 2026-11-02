# ch03_pod_v1: Chapter 3 extension

This directory provides model, preprocessing, objective, and training functions. See the [Chapter 3 guide](../../docs/CH03_READING_ZH.md) for notation and reading order, and the [standalone entry](../../README.md) for execution.

`book_ch03.py` calls these numerical functions and generates all training assets in the current workspace; `book_driver.py` manages the 57 declared formal runs. The historical `run.py` main function retains its original orchestration and gates and is not the public standalone entry point. Weights are not imported from the old project.
