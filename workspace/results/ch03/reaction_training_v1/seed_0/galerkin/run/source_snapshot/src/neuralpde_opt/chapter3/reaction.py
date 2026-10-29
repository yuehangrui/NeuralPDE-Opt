"""Independent Chapter 3 case: -epsilon Delta y + c(x)y = 1, y|boundary=0.

Reaction is assembled as a consistent weighted mass matrix, never as diffusion.
The original diffusion case and shared Chapter 2 solver remain unchanged.

Book/code notation:
Book 3.2: -epsilon*Delta y + c*y = f, with epsilon=0.01 and zero Dirichlet data. c is the varying reaction field, not the diffusion coefficient.
"""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..mesh import make_mesh
from .fields import cosine_series, spectral_weights
from .reference import Solution

SCHEMA = "ch03-reaction-v1"
SPLIT_CODES = {"train": 731, "val": 947}


def validate_config(config):
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Expected reaction schema")
    e = config["epsilon"]
    if not np.isfinite(e) or e <= 0:
        raise ValueError("epsilon must be finite and positive")
    s = config["reaction"]
    if (
        not all(
            np.isfinite(s[k])
            for k in ["lower", "upper", "length_scale", "exponent", "latent_scale"]
        )
        or not 0 < s["lower"] < s["upper"]
        or s["length_scale"] <= 0
        or s["exponent"] <= 0
        or s["latent_scale"] <= 0
        or type(s["modes"]) is not int
        or s["modes"] < 2
        or type(s["include_constant"]) is not bool
    ):
        raise ValueError("Invalid reaction field law")
    if config.get("source") != 1.0 or config.get("boundary") != "zero_dirichlet":
        raise ValueError("This case fixes f=1 and homogeneous Dirichlet data")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Invalid seed")
    if set(config["splits"]) != set(SPLIT_CODES) or any(
        type(v) is not int or v < 1 for v in config["splits"].values()
    ):
        raise ValueError("Only positive train/val counts are supported")
    for k in [
        "observation_n",
        "reference_n",
        "audit_n",
        "quadrature_order",
        "audit_quadrature_order",
    ]:
        if type(config[k]) is not int or config[k] < 2:
            raise ValueError("Invalid grid/order: " + k)
    a, b, c = [config[k] for k in ["observation_n", "reference_n", "audit_n"]]
    if b % a or c % b or not a < b < c:
        raise ValueError("Require nested observation < reference < audit grids")
    if config["audit_quadrature_order"] <= config["quadrature_order"]:
        raise ValueError("Audit quadrature must be higher")
    if set(config["audit_samples"]) != set(SPLIT_CODES) or any(
        type(v) is not int or not 1 <= v <= config["splits"][s]
        for s, v in config["audit_samples"].items()
    ):
        raise ValueError("Invalid audit coverage")
    if any(not np.isfinite(v) or v <= 0 for v in config["quality_gates"].values()):
        raise ValueError("Invalid quality gates")
    return config


def draw_reaction(config, split, index):
    if split not in SPLIT_CODES or type(index) is not int or index < 0:
        raise ValueError("Invalid train/val sample index")
    spec = config["reaction"]
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([config["seed"], SPLIT_CODES[split], index, 23]))
    )
    return rng.standard_normal((spec["modes"], spec["modes"])) * spectral_weights(
        spec["modes"], spec["length_scale"], spec["exponent"], spec["include_constant"]
    )


class ReactionField:
    def __init__(self, coefficients, spec):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.spec = spec
        if (
            self.coefficients.shape != (spec["modes"], spec["modes"])
            or not np.isfinite(self.coefficients).all()
        ):
            raise ValueError("Invalid reaction coefficients")

    def reaction(self, points):
        s = self.spec
        return (s["lower"] + s["upper"]) / 2 + (s["upper"] - s["lower"]) / 2 * np.tanh(
            s["latent_scale"] * cosine_series(points, self.coefficients)
        )

    @staticmethod
    def source(points):
        return np.ones(points.shape[:-1])


class ReactionFEM:
    """Cached geometry for continuous-field P1 reference solves."""

    def __init__(self, n, epsilon=0.01, order=4):
        if type(n) is not int or n < 2 or type(order) is not int or order < 2:
            raise ValueError("n and quadrature order must be integers >= 2")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.n, self.epsilon, self.order = n, float(epsilon), order
        self.mesh = make_mesh(2, n)
        self.points, self.weights, self.phi = self.mesh.quadrature(order)
        self.stiffness = self.mesh.stiffness()

    def assemble(self, fields):
        c = np.asarray(fields.reaction(self.points))
        f = np.asarray(fields.source(self.points))
        if c.shape != self.weights.shape or f.shape != c.shape:
            raise ValueError("Fields must have one value per quadrature point")
        if not np.isfinite(c).all() or c.min() <= 0 or not np.isfinite(f).all():
            raise ValueError("Invalid fields: finite f and positive finite c required")
        local = np.einsum("eq,eq,qa,qb->eab", self.weights, c, self.phi, self.phi)
        rows = np.broadcast_to(self.mesh.elements[:, :, None], local.shape)
        cols = np.broadcast_to(self.mesh.elements[:, None, :], local.shape)
        mass = sparse.coo_matrix(
            (local.ravel(), (rows.ravel(), cols.ravel())), shape=self.stiffness.shape
        ).tocsr()
        load = np.zeros(len(self.mesh.nodes))
        local_load = np.einsum("eq,eq,qa->ea", self.weights, f, self.phi)
        np.add.at(load, self.mesh.elements.ravel(), local_load.ravel())
        return self.epsilon * self.stiffness + mass, load, c

    def solve(self, fields):
        start = time.perf_counter()
        matrix, load, c = self.assemble(fields)
        ii = self.mesh.interior
        y = np.zeros(len(self.mesh.nodes))
        y[ii] = spsolve(matrix[ii][:, ii], load[ii])
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite reference solution")
        work = float(y @ load)
        energy = float(y @ (matrix @ y))
        residual = np.linalg.norm((matrix @ y - load)[ii]) / max(
            np.linalg.norm(load[ii]), np.finfo(float).tiny
        )
        return Solution(
            self.mesh,
            y,
            {
                "n": self.n,
                "epsilon": self.epsilon,
                "quadrature_order": self.order,
                "relative_algebraic_residual": float(residual),
                "relative_energy_identity_defect": abs(energy - work)
                / max(abs(work), np.finfo(float).tiny),
                "boundary_max": float(np.abs(y[np.setdiff1d(np.arange(len(y)), ii)]).max()),
                "c_quadrature_min": float(c.min()),
                "c_quadrature_max": float(c.max()),
                "y_min": float(y.min()),
                "y_max": float(y.max()),
                "seconds": time.perf_counter() - start,
            },
        )

    def independent_weak_residual(self, fields, values):
        """Elementwise weak form, with no assembled system matrix or load reuse."""
        y = np.asarray(values)
        if y.shape != (len(self.mesh.nodes),) or not np.isfinite(y).all():
            raise ValueError("Invalid reference values")
        local_y = y[self.mesh.elements]
        grad_y = np.einsum("ea,ead->ed", local_y, self.mesh.gradients)
        yq = np.einsum("ea,qa->eq", local_y, self.phi)
        c, f = fields.reaction(self.points), fields.source(self.points)
        local = self.epsilon * np.einsum(
            "ed,ead,e->ea", grad_y, self.mesh.gradients, self.mesh.determinants / 2
        )
        local += np.einsum("eq,eq,qa->ea", self.weights, c * yq - f, self.phi)
        residual, load = np.zeros(len(y)), np.zeros(len(y))
        np.add.at(residual, self.mesh.elements.ravel(), local.ravel())
        np.add.at(
            load,
            self.mesh.elements.ravel(),
            np.einsum("eq,eq,qa->ea", self.weights, f, self.phi).ravel(),
        )
        ii = self.mesh.interior
        return float(np.linalg.norm(residual[ii]) / np.linalg.norm(load[ii]))


class ReactionManufactured:
    def __init__(self, epsilon=0.01, variable=True):
        self.epsilon, self.variable = epsilon, variable

    def reaction(self, p):
        return 1 + 0.25 * p[..., 0] + 0.125 * p[..., 1] if self.variable else np.ones(p.shape[:-1])

    @staticmethod
    def exact(p):
        return np.sin(np.pi * p[..., 0]) * np.sin(np.pi * p[..., 1])

    @staticmethod
    def gradient(p):
        x, y = p[..., 0], p[..., 1]
        return np.pi * np.stack(
            (np.cos(np.pi * x) * np.sin(np.pi * y), np.sin(np.pi * x) * np.cos(np.pi * y)), -1
        )

    def source(self, p):
        return (2 * self.epsilon * np.pi**2 + self.reaction(p)) * self.exact(p)
