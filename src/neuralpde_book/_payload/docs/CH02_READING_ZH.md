# Chapter 2 reading guide: neural function approximation

Manuscript Sections 2.2-2.6 share 1D/2D Poisson problems. The state is consistently `y`; the network is `y_theta=G+D*N_theta`, and the FEM state is `y_h`. The symbol `u` is reserved for control in later chapters.

Read [problems.py](../src/neuralpde_ch02/problems.py) and [models.py](../src/neuralpde_ch02/models.py), then the strong-residual PINN, direct-energy Deep Ritz, and Gram-norm-aware VPINN objectives in [objectives.py](../src/neuralpde_ch02/objectives.py). The weak-form implementation is not WAN.

[training.py](../src/neuralpde_ch02/training.py) manages Adam/L-BFGS and checkpoint recovery; [metrics.py](../src/neuralpde_ch02/metrics.py) computes independent quadrature, L2/H1 gradient seminorms, boundary errors, and energy diagnostics; [fem.py](../src/neuralpde_ch02/fem.py) supplies the classical reference.

The formal [full.json](../chapters/ch02/configs/full.json) contains 2 dimensions x 3 methods x 3 seeds: 18 training runs and 8 FEM runs. The final iterate is saved; validation is diagnostic and does not select the best iterate. L-BFGS budgets are nominal, with actual iterations and objective calls recorded separately.

Use the [unified entry point](../README.md) to retrain in a new workspace; results go to `results/ch02/full`. The original frozen Chapter 2 package remains independent and is not used as the new output directory.
