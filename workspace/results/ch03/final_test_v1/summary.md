# Chapter 3: final frozen test

All checkpoints, data-generation rules, and HINTS parameters were locked before generating the new test sets. Fixed forcing: 128 samples; variable forcing: 32 new reaction-coefficient groups x 4 forcing terms. No selection or tuning after testing.

| Group | Seeds | Test relative L2 | Test relative H1 seminorm |
|---|---:|---:|---:|
| baseline/fno | 3 | 0.4620% ± 0.0273% | 3.2897% |
| baseline/deepxde_fourier4 | 3 | 7.7055% ± 0.5446% | 11.8322% |
| baseline/gno | 3 | 1.5940% ± 0.0175% | 21.2400% |
| baseline/galerkin | 3 | 0.7062% ± 0.0968% | 3.5672% |
| pod/rank_4 | 3 | 4.6278% ± 0.0070% | 9.3134% |
| pod/rank_8 | 3 | 2.7149% ± 0.0509% | 5.9411% |
| pod/rank_16 | 3 | 2.5273% ± 0.0712% | 5.1668% |
| pod/rank_32 | 3 | 2.6174% ± 0.0390% | 5.2824% |
| physics/supervised_128 | 3 | 0.6849% ± 0.0565% | 5.7625% |
| physics/supervised_32 | 3 | 1.7867% ± 0.1352% | 8.8836% |
| physics/energy_0 | 3 | 0.4519% ± 0.0169% | 1.6847% |
| physics/hybrid_32 | 3 | 0.4727% ± 0.0222% | 1.8211% |
| physics/hybrid_128 | 3 | 0.4372% ± 0.0264% | 1.7720% |
| varmion/varmion | 3 | 7.3381% ± 0.0854% | 12.9603% |
| varmion/unstructured | 3 | 9.6642% ± 0.2176% | 11.4802% |
| varmion_v2/dense_data | 3 | 7.2831% ± 0.0575% | 12.9895% |
| varmion_v2/spd_joint_aug128 | 3 | 3.2762% ± 0.0858% | 4.9198% |
| varmion_v2/dense_joint | 1 | 4.1956% ± 0.0000% | 6.6877% |
| varmion_v2/spd_data | 1 | 6.4915% ± 0.0000% | 11.3480% |
| varmion_v2/spd_joint | 1 | 5.8365% ± 0.0000% | 7.7016% |

The +/- values are population standard deviations across seeds, not confidence intervals. Forcing terms within a coefficient group are correlated; the 128 variable-forcing samples are not 128 independent coefficients. Different data, budgets, physical information, and output spaces do not establish architecture-family rankings.

| HINTS mode | Converged/total | Mean iterations | Total time (ms) | Speedup over Jacobi | Maximum discrete L2 error |
|---|---:|---:|---:|---:|---:|
| jacobi | 64/64 | 4040.4 | 201.45 | 1.0 | 2.745e-08 |
| fixed | 192/192 | 422.4 | 34.95 | 5.76360118871051 | 3.356e-09 |
| safeguarded | 192/192 | 406.0 | 35.00 | 5.75537996719642 | 7.102e-09 |

HINTS discrete error is relative to the direct solve of the same 65x65 P1 system. Continuous-PDE reference errors are stored separately in hints_summary.json. Timings include assembly, conditional matrices, residual conversion, and acceptance/fallback, but exclude model loading. Each case uses the median of 3 repetitions and is compared only with Jacobi on the same machine.

POD projection and exact reduced Galerkin are numerical diagnostics only. Projection using reference labels is not a deployable prediction model. All reference solutions passed residual, serialization, and predefined grid-refinement checks.
