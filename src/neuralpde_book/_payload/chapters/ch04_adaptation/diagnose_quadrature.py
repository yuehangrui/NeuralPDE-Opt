"""Isolate quadrature bias with a convex, PDE-feasible spectral control problem.

Uses only f and yd in optimization. Analytic u is for reporting afterward.
This is a diagnostic discretization, not a neural training result.
"""

import argparse
import numpy as np
from .instance import quadrature
from chapters.ch04.problem import Problem
from chapters.ch04.evaluation import sample_points
from chapters.ch04.training import write_json


def solve(x, w, modes=12):
    problem = Problem()
    k = np.array([(i, j) for i in range(1, modes + 1) for j in range(1, modes + 1)]) * np.pi
    eigenvalues = (k * k).sum(1)
    b = np.sin(x[:, 0:1] * k[:, 0]) * np.sin(x[:, 1:2] * k[:, 1])
    fc = np.linalg.lstsq(b * np.sqrt(w), problem.f(x) * np.sqrt(w), rcond=None)[0]
    s = b / eigenvalues
    h = problem.alpha * b.T @ (w * b) + s.T @ (w * s)
    c = np.linalg.solve(h, s.T @ (w * (problem.yd(x) - s @ fc)))
    return k, c, fc


def run():
    xv, wv = quadrature(72)
    problem = Problem()
    rows = []
    for sampling in ("gauss48", "random5000", "random10000"):
        if sampling == "gauss48":
            x, w = quadrature(48)
        else:
            x = sample_points(0, 1, int(sampling[6:]))
            w = np.full((len(x), 1), 1 / len(x))
        k, c, fc = solve(x, w)
        b = np.sin(xv[:, 0:1] * k[:, 0]) * np.sin(xv[:, 1:2] * k[:, 1])
        u = b @ c
        y = (b / (k * k).sum(1)) @ (c + fc)
        err = lambda a, t: float(np.sqrt(np.sum(wv * (a - t) ** 2) / np.sum(wv * t * t)))
        rows.append(
            dict(
                sampling=sampling,
                control_relative_l2=err(u, problem.u(xv)),
                state_relative_l2=err(y, problem.y(xv)),
                J_validation=float(
                    np.sum(wv * (0.5 * (y - problem.yd(xv)) ** 2 + 0.5 * problem.alpha * u * u))
                ),
            )
        )
    return dict(
        modes=12,
        description="Exact reduced minimizer of empirical quadrature objective under the PDE; no neural approximation.",
        results=rows,
        test_evaluations=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = run()
    write_json(a.output, result)
    print(result)
