# Chapter 5 reading guide: control and state constraints

Manuscript Section 5.2.1 and the implemented CNP part of Section 5.2.2; see the [report after formal completion](../results/ch05/unified_delivery_v1/summary.md). Control box constraints `a<=u<=b` and the state upper bound `y<=psi` are separate problems with different references and metrics.

## Seven workflows

| Method | Core files | Interpretation |
|---|---|---|
| OSNN | [models.py](../chapters/ch05/models.py) | Projected optimality `u=Pi_[a,b](-p/alpha)`; learns state and scaled adjoint |
| ADMM-PINN | `SplitState` above; [delivery/training.py](../chapters/ch05_delivery/training.py) | Control splitting `u=z`, separate inner updates and outer projection/multiplier updates |
| Projected KKT-FNO | [operators.py](../chapters/ch05/operators.py) | Historical method ID `mimo`; outputs state and scaled adjoint before projection; not Chapter 4 MIMOONet |
| PDOL | `action` above; [reference.py](../chapters/ch05/reference.py) | Two trained surrogates approximate state and adjoint operators inside primal-dual iterations |
| iUzawa-F / iUzawa-S | [operator_blocks.py](../chapters/ch05/operator_blocks.py) | Each unrolled architecture retains its layer/sharing scheme; names do not imply all layers share weights |
| CNP | `method=cnp` in [models.py](../chapters/ch05/models.py) | Smooth feasible state representation; recover control by `u=-Delta y` (`f=0` here) |

Start with [problem.py](../chapters/ch05/problem.py) for physical quantities and constraints, then the representations and [losses/training](../chapters/ch05_delivery/training.py), followed by [protocol.py](../chapters/ch05_delivery/protocol.py) and [assessment.py](../chapters/ch05_delivery/assessment.py). [common.py](../chapters/ch05_delivery/common.py) defines formal budgets; [data.py](../chapters/ch05_delivery/data.py) defines data identities.

## Four distinct quantities

| Code/manuscript symbol | Mathematical meaning |
|---|---|
| `p` | Physical PDE adjoint satisfying `-Delta p=y-yd` |
| `q_scaled` | `p/alpha`, used by OSNN and projected KKT; checkpoint key `q_net` retained |
| `lambda_admm` | Unscaled multiplier for control splitting `u-z`; historical key `lam` retained |
| `p_dual` | PDOL/iUzawa observation-space dual; also denoted `p` in manuscript (5.11)-(5.13), explicitly distinguished in code |

At an exact fixed point, `p_dual=y-yd`, whereas the physical adjoint is `p=S* p_dual`. They are not interchangeable. Surrogate operators and finite unrolling may not reach an exact fixed point; this is a definition and limiting relationship.

ADMM uses `+<lambda,u-z>`, with inner control and projection updates

\[
u=\frac{\beta z-\lambda-\alpha q}{\alpha+\beta},\quad
z^+=\Pi_{[a,b]}(u+\lambda/\beta),\quad
\lambda^+=\lambda+\beta(u-z^+).
\]

Both inner control `u` and feasible split control `z` are retained; primal and dual residuals are reported separately. `SplitState` saves network snapshots after each inner stage and before the outer stage, allowing multiplier-history reconstruction at arbitrary new coordinates. The last `q_net` alone does not represent the complete ADMM state.

Internal iUzawa arguments formerly named `p` are now `p_dual` in `_u_step`, `_p_step`, and `forward`. Positional calls remain compatible; external low-level callers using `p=` must use `p_dual=`. Layer names and state-dictionary keys are unchanged.

## Shapes and constraint checks

Instance coordinates are `(N,2)`, with fields and quadrature weights `(N,1)`. Operators use an interior 31x31 grid and channels-last input `(batch,31,31,1)`, unlike the boundary-inclusive 33x33 grid in Chapter 4. Axis order and resolution are not forced to match. Multiply `q_scaled` by `alpha` to recover the physical adjoint.

CNP uses `psi-(sqrt(psi)+D*N_theta)^2`, with a boundary-vanishing factor `D`, satisfying the state upper bound and homogeneous boundaries in this example. Its smooth feasible representation and higher-order spatial derivatives are retained; straight-through clipping is not inserted into gradients. CNP has different optimality conditions from box-constrained control and is not the Section 5.4 obstacle problem.

Formal delivery contains seven workflows x three seeds, totaling 21 selected models. Historical tests do not guide model selection; new tests use predetermined random streams after selection is frozen. State-splitting ADMM in 5.2.2, interface control in 5.3, and obstacle control in 5.4 remain deferred; no such experiments were fabricated.
