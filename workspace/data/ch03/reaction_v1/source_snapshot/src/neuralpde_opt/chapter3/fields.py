"""Grid-independent, finite cosine series with explicit probability law.

The truncation controls the sampled function distribution, not a model rank.
No sample-wise normalization, clipping, or grid-dependent FFT scaling is used.
"""

import numpy as np


def spectral_weights(modes, length_scale, exponent, include_constant):
    k = np.arange(modes)
    weights = (1 + (np.pi * length_scale) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2)) ** (
        -exponent / 2
    )
    if not include_constant:
        weights[0, 0] = 0
    return weights / np.linalg.norm(weights)


def draw_parameters(config, split_code, index):
    """Independent, append-stable streams for each split/sample/channel."""
    root = [config["seed"], split_code, index]
    result = {}
    for channel, name in enumerate(("a", "f")):
        spec = config["fields"][name]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(root + [channel])))
        weights = spectral_weights(
            spec["modes"], spec["length_scale"], spec["exponent"], spec["include_constant"]
        )
        result[name + "_coefficients"] = rng.standard_normal(weights.shape) * weights
    return result


def cosine_series(points, coefficients, chunk=65536):
    shape = points.shape[:-1]
    points = np.asarray(points).reshape(-1, 2)
    k = np.arange(coefficients.shape[0])
    normalization = np.where(k == 0, 1.0, np.sqrt(2.0))
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        bx = np.cos(np.pi * p[:, 0, None] * k) * normalization
        by = np.cos(np.pi * p[:, 1, None] * k) * normalization
        result[start : start + chunk] = np.einsum(
            "pi,ij,pj->p", bx, coefficients, by, optimize=True
        )
    return result.reshape(shape)


class FieldPair:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config

    def coefficient(self, points):
        spec = self.config["fields"]["a"]
        g = cosine_series(points, self.parameters["a_coefficients"])
        return (spec["lower"] + spec["upper"]) / 2 + (spec["upper"] - spec["lower"]) / 2 * np.tanh(
            spec["latent_scale"] * g
        )

    def source(self, points):
        return self.config["fields"]["f"]["rms_in_expectation"] * cosine_series(
            points, self.parameters["f_coefficients"]
        )
