# Standalone edition: audit report

Generated: 2026-10-29T01:47:54.176068+00:00.

Training assets were regenerated in the independent workspace and all 156 formal runs were completed. Previous checkpoints were not used for formal initialization. HINTS uses the newly trained VarMiON in the same workspace, as specified by the method. Chapter 2 saves the final iterate; Chapters 3-5 select models using validation data.

Numerical regression: 334 tests passed; optimized-mode checks: True. All 21 small-budget checks across Chapter 3 training variants passed.

Final tests for Chapters 3-5 use new random streams declared in advance for the standalone edition, and are generated only after training is complete and model selection is frozen. Chapter 2 uses independent quadrature, physical consistency checks, and FEM convergence checks.

| Part | Audit evidence |
|---|---|
| ch02 | [DELIVERY_AUDIT.json](../results/ch02/full/DELIVERY_AUDIT.json), [results report](../results/ch02/full/summary.md) |
| ch03 | [verification.json](../results/ch03/final_test_v1/verification.json), [results report](../results/ch03/final_test_v1/summary.md) |
| ch04 | [DELIVERY_AUDIT.json](../results/ch04/unified_delivery_v1/DELIVERY_AUDIT.json), [results report](../results/ch04/unified_delivery_v1/summary.md) |
| ch04-depth | [DELIVERY_AUDIT.json](../results/ch04/nonlinear_depth_v1/DELIVERY_AUDIT.json), [results report](../results/ch04/nonlinear_depth_v1/summary.md) |
| ch05 | [DELIVERY_AUDIT.json](../results/ch05/unified_delivery_v1/DELIVERY_AUDIT.json), [results report](../results/ch05/unified_delivery_v1/summary.md) |

Accuracy is interpreted in each chapter report. Completing a workflow does not establish convergence. Spatial validation on one fixed instance is not generalization to new instances; extending a fixed-grid prediction is not resolution transfer. Different budgets, data, and physical information do not establish an architecture-family ranking.

The wheel can initialize any new workspace without the original project or upstream repositories. Scientific computing still depends on PyTorch, NumPy, SciPy, and Matplotlib. Ported DeepXDE and Galerkin components retain their licenses and provenance; ocpde_dl provenance is also retained. Licenses are not replaced with one uniform license, and the code is not claimed to be entirely original.

Installation used an independent virtual environment sharing the scientific dependencies already available on the server. No clean-operating-system or fully offline dependency installation is claimed. See [INSTALLATION.json](../verification/INSTALLATION.json).

Source identity: [PAYLOAD_MANIFEST.json](../PAYLOAD_MANIFEST.json); provenance: [SOURCE_ORIGINS.json](../SOURCE_ORIGINS.json); packaging changes: [SOURCE_CHANGES.json](../SOURCE_CHANGES.json). Chapter manifests separately verify training, model selection, and final artifact identities.
