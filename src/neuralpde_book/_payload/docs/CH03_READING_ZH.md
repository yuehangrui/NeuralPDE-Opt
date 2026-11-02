# Chapter 3 reading guide: from input functions to solutions

Manuscript Sections 3.3-3.5; the [results interpretation](../results/ch03/final_test_v1/summary.md) is generated after full training and evaluation. The formal problem on the unit square is

\[
-\varepsilon\Delta y+c(x)y=f(x),\qquad y|_{\partial\Omega}=0,\qquad\varepsilon=0.01.
\]

`c` is the reaction coefficient and `epsilon` the fixed diffusion coefficient. The four-operator comparison learns `c -> y` with fixed forcing. VarMiON and HINTS involve variable forcing/residuals; input information and task scope must be described separately.

## Where to start

| Step | Code | Reading focus |
|---|---|---|
| Problem and discretization | [reaction.py](../src/neuralpde_opt/chapter3/reaction.py) | Reaction enters the mass-type matrix; diffusion enters the stiffness matrix |
| Four representations | [reaction_operators.py](../src/neuralpde_opt/chapter3/reaction_operators.py), [upstream_operators.py](../src/neuralpde_opt/chapter3/upstream_operators.py) | Model construction, normalization, hard boundaries; constructors identify ported upstream components |
| Training and selection | [reaction_training.py](../src/neuralpde_opt/chapter3/reaction_training.py) | Training-only statistics, validation selection, best/last checkpoints, and resumption identities |
| POD | [pod.py](../experiments/ch03_pod_v1/pod.py) | Mass-weighted reduction, input encoding, and coefficient reconstruction |
| Physical FNO | [physics.py](../experiments/ch03_physics_v1/physics.py) | Distinctions between energy, weak residual, and supervision |
| VarMiON | [model.py](../experiments/ch03_varmion_v1/model.py), [model_v2.py](../experiments/ch03_varmion_v2/model_v2.py) | Load linearity, matrix branch, SPD parameterization |
| HINTS | [hybrid.py](../chapters/ch03/hybrid.py) | Jacobi steps, neural residual corrections, safeguards, and stopping rules |
| Complete workflow | [registry.py](../chapters/ch03/registry.py), [training.py](../chapters/ch03/training.py), [predictors.py](../chapters/ch03/predictors.py) | 57 formal configurations, unified training, and label-free prediction interfaces |

## Notation and arrays

| Mathematical object | Code name | Meaning |
|---|---|---|
| Predicted solution | `y_prediction` / `y` | Avoids the local `p` that could be confused with an adjoint |
| POD space | `Phi` / `basis` | Columns are basis vectors; the discrete mass inner product is retained |
| Reduced input load \(z_f=\Phi^T M f\) | `load_coordinates` | Manuscript (3.33)-(3.35), linear input branch |
| Reduced solution coefficients \(\beta\) | `beta_coefficients` | Historical array/file keys may still use `alpha`; this is not the control regularization coefficient |
| Reduced map \(K_\theta(c)\) | Matrix-branch output | Inverse-operator-type action on the load, not a stiffness matrix |
| Linear-system residual | `r = rhs - A @ y` | HINTS correction direction, manuscript (3.43) and Algorithm 2 |

The four operators share a physical interface with reaction coefficients and states of shape `(batch,65,65)`. GNO internally uses 17x17 nodes with P1 reconstruction; this is not the output resolution. POD and VarMiON v2 use 289 sensors; reduced coefficients have shape `(batch,rank)` and the basis matrix `(4225,rank)`. Adapters handle native FNO/Transformer axis conventions individually; a single global transpose is not a substitute.

HINTS receives an assembled load residual and encodes it as `Phi_I.T @ r`; do not multiply by the mass matrix again. Physical training may write the equation residual as `A@y-rhs`. Squared norms agree, but correction directions differ; read the residual together with its update rule.

Fourier-enriched DeepONet coordinates and FNO spectral convolution are different mechanisms. Galerkin integral attention also differs from standard softmax attention. Method IDs and historical weight keys are retained; constructors and formal reports define their meaning.

## Training assets and results

`neuralpde-book prepare` generates and verifies the following assets in a new workspace. Paths are relative to that workspace:

| Path | Purpose |
|---|---|
| `data/ch03/reaction_v1`, `data/ch03/varmion_v1`, `data/ch03/hints_v1` | Train/val data, geometry, and manifests for fixed forcing, variable forcing, and residual learning |
| `results/ch03/pod_v1/basis.npz`, `normalization.json` | POD basis and training statistics |
| `prepared_manifest.json` and its declared files in `results/ch03/varmion_v1` | Bases, normalization, train/val load coordinates; HINTS also needs the same-seed VarMiON initialization weights |
| Preprocessing artifacts in `results/ch03/varmion_v2` | Preprocessed data and operator pools for SPD/joint training |

Start formal training through the unified entry point. POD bases and normalization use training data only. HINTS uses VarMiON newly trained in the same workspace, as specified by the method. Each configuration writes a `run/` directory containing histories, selected weights, and integrity manifests.

New fixed tests are generated after selection is frozen; historical tests do not guide this round of model selection. Physical information, output spaces, and computational costs differ across methods and must be interpreted separately; no architecture-family ranking follows.
