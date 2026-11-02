# Chapter 4 reading guide: smooth optimal control

Manuscript Sections 4.3-4.6. Start with the [unified delivery entry](../chapters/ch04_delivery/README.md); completed results appear in the [baseline report](../results/ch04/unified_delivery_v1/summary.md) and [extension report](../results/ch04/nonlinear_depth_v1/summary.md). The base problem is linear-quadratic elliptic control:

\[
\min_u J=\tfrac12\|y-y_d\|^2+\tfrac\alpha2\|u\|^2,
\quad -\Delta y=u+f,\quad y|_{\partial\Omega}=0.
\]

The physical adjoint satisfies `-Delta p=y-yd`, giving the reduced gradient `alpha*u+p`, consistent with manuscript (4.2) and (4.5). Problem objects use `problem`, file locations use `path`, and predictions use `y_prediction`, reserving `p` for the physical adjoint.

## Methods and reading routes

| Method | From problem to implementation | Notes |
|---|---|---|
| Classical methods and PINN | [problem.py](../chapters/ch04/problem.py) -> [models.py](../chapters/ch04/models.py) -> [training.py](../chapters/ch04/training.py); [fem.py](../chapters/ch04/fem.py) provides references | Direct, KKT, and ALM have distinct objectives; spatial and parameter differentiation are separate |
| Frequency-enriched instance methods | [instance.py](../chapters/ch04_adaptation/instance.py) | Gauss quadrature and full frequency enrichment; neural and head-correction stages separate |
| FNO/DeepONet/FEDONet | [models.py](../chapters/ch04_operator_adaptation/models.py) -> [training.py](../chapters/ch04_operator_adaptation/training.py) | `state` learns the state map; `kkt` learns optimal state/control |
| MIMOONet and RB-DINO | [operator_models.py](../chapters/ch04_adaptation/operator_models.py) -> [operators.py](../chapters/ch04_delivery/operators.py) | Four multiplicative branches and shared trunk; RB-DINO adds derivative supervision |
| Bilevel, Deep Uzawa, ISMO, FEM-NN baselines | [ch04_completion](../chapters/ch04_completion/README.md) -> [hybrids.py](../chapters/ch04_delivery/hybrids.py) | Finite sine-feature baselines and nonlinear surrogates are explained separately |
| Deep Uzawa/FEM-NN extensions | [common.py](../chapters/ch04_depth/common.py) -> [training.py](../chapters/ch04_depth/training.py) -> [assessment.py](../chapters/ch04_depth/assessment.py) | Coordinate networks with trainable hidden layers; sparse/noisy source identification |
| Unified delivery | [cli.py](../chapters/ch04_delivery/cli.py) -> [protocol.py](../chapters/ch04_delivery/protocol.py) -> [assessment.py](../chapters/ch04_delivery/assessment.py) | Training, freezing, independent evaluation, and audits retain separate stage identities |

## Multipliers, scaling, and residual signs

| Object | Definition and convention |
|---|---|
| Physical adjoint `p` | Solves `-Delta p=y-yd`; used for the true reduced gradient |
| Scaled adjoint `q` / `q_scaled` | Means `p/alpha` only where explicitly defined; not the meaning of every legacy local `q` |
| ALM multiplier | The term `+<lambda,r>` acts on `r=-Delta y-u-f`; at a feasible optimum, `lambda=-p` |
| ALM penalty | Code uses `mu*||r||^2`; manuscript (4.25)-(4.26) uses `beta/2*||r||^2`, so `beta=2mu`; update `lambda+=2mu*r` |
| Deep Uzawa multiplier `z` | Modified objective includes `alpha/4*||u||^2 + alpha/4*||Ay-f||^2 + <z,Ay-u-f>`; control block `u=2z/alpha`; at a feasible optimum `z=-p/2` |
| Legacy spectral local `q` | If defined as `K@y-f`, it is control recovered from the state, not a scaled adjoint |

Raw neural states, states recomputed from controls by a PDE solve, surrogate gradients, and true reduced gradients have separate fields. A low objective does not replace feasibility and stationarity checks.

## Tensors, discretization, and implementation depth

Coordinates have shape `(N,2)`, scalar fields usually `(N,1)`. Problem families use 17 sine modes with vectors `(batch,17)` and MIMO inputs/outputs `(batch,2,17)`. Adapted grid operators take `(batch,2,33,33)`; KKT outputs two channels for state and control, with physical adjoint recovered as `p=-alpha*u`. This baseline FNO layout cannot be directly mixed with channels-last implementations in other chapters.

`head_polished` denotes output-head correction on training data and must not be labeled as native neural accuracy. Modal MIMO, RB-DINO, and the 17-mode output space retain their representation assumptions despite a shared entry point.

In `ch04_depth`, nonlinear means a trainable nonlinear network representation; the base PDE remains linear Poisson. FEM-NN learns the source, FEM solves the state, and quadrature uses the consistent mass matrix. The extended Deep Uzawa trains all hidden layers but retains an analytic control update. The finite-feature bilevel baseline does not thereby become a general deep nonconvex bilevel experiment.

The baseline delivery has 48 configurations and 72 stage checkpoints; extensions add 12 runs. Original budgets, training-only normalization, and selection of the earliest minimum validation score are retained. Accuracy is determined by full training and independent evaluation in the workspace. Semilinear/parabolic extensions remain deferred.
