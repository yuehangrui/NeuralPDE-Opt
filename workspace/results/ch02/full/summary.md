# Chapter 2 experiment results

Completed neural runs: 18.

| Problem | Method | Seeds | Relative L2 (mean +/- std) | Relative H1 seminorm (mean +/- std) | Training seconds (mean) |
|---|---|---:|---:|---:|---:|
| 1D | PINN | 3 | 1.035e-06 +/- 4.82e-07 | 9.558e-06 +/- 4.39e-06 | 142.7 |
| 1D | Deep Ritz | 3 | 1.482e-05 +/- 1.15e-05 | 1.685e-04 +/- 1.24e-04 | 103.7 |
| 1D | VPINN | 3 | 1.247e-05 +/- 1.16e-05 | 1.444e-04 +/- 1.28e-04 | 141.2 |
| 2D | PINN | 3 | 3.804e-06 +/- 1.47e-06 | 1.472e-05 +/- 5.25e-06 | 204.1 |
| 2D | Deep Ritz | 3 | 8.491e-05 +/- 1.65e-05 | 3.610e-04 +/- 7.61e-05 | 121.6 |
| 2D | VPINN | 3 | 7.787e-05 +/- 1.28e-05 | 3.968e-04 +/- 1.02e-04 | 176.0 |

## FEM reference

| Problem | Intervals per axis | Interior DOFs | Relative L2 | Relative H1 seminorm | Assembly + solve seconds |
|---|---:|---:|---:|---:|---:|
| 1D | 16 | 15 | 2.051e-03 | 5.165e-02 | 0.0044 |
| 1D | 32 | 31 | 5.130e-04 | 2.584e-02 | 0.0017 |
| 1D | 64 | 63 | 1.283e-04 | 1.292e-02 | 0.0017 |
| 1D | 128 | 127 | 3.207e-05 | 6.461e-03 | 0.0017 |
| 2D | 8 | 49 | 4.227e-02 | 1.944e-01 | 0.0028 |
| 2D | 16 | 225 | 1.075e-02 | 9.793e-02 | 0.0058 |
| 2D | 32 | 961 | 2.701e-03 | 4.906e-02 | 0.0214 |
| 2D | 64 | 3969 | 6.760e-04 | 2.454e-02 | 0.0940 |

## Validation

Maximum boundary mismatch: 2.220e-16.
Maximum relative L2 change after quadrature refinement: 4.872e-12.
Maximum relative H1 change after quadrature refinement: 1.301e-12.
Maximum energy-identity defect: 1.226e-15.

## Interpretation

Neural runs use equal architectures, seeds, quadrature and nominal optimization budgets within each dimension. The cost of a step differs by formulation; actual objective evaluations and synchronized optimization wall time are recorded. Timing excludes validation, checkpoint I/O and setup. FEM timing includes assembly and solve on CPU; neural training uses the recorded device. This is not an equal-hardware speedup claim.

The VPINN objective is the finite-test dual norm r^T G_W^{-1} r with P1 hat test functions. It is a norm-aware weak-residual variant, not the unweighted original VPINN loss. Additional test functions and finer quadrature are used for validation. A small finite-test loss is not a standalone continuum error certificate.

Errors are computed against analytic solutions. H1 denotes the gradient seminorm. The energy-gap identity applies because all trial functions satisfy the same essential boundary data. The pointwise strong residual is reported for neural fields only; elementwise Laplacians are not a valid global strong-residual measure for P1 FEM.

Solution-field figures always use seed 0. Standard deviations use ddof=1 when multiple seeds exist. No claim of a global optimum, exact theoretical error decomposition or generic superiority is made.
