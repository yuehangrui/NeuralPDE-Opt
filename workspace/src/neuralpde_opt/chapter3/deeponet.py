"""A fixed-sensor, arbitrary-query DeepONet for the Chapter 3 pilot.

The branch jointly encodes sampled ``a`` and ``f``.  Its input sensor layout is
fixed at construction; changing output query points does not change that input
representation.  Coordinates use the project's ``indexing='ij'`` convention:
the first grid index is x and the second is y, both in [0, 1].

Book/code notation:
Book 3.3.1: branch features encode input functions and trunk features encode query coordinates. Their contraction reconstructs y_theta; Fourier input features are an explicit implementation choice.
"""

import math

import torch
from torch import nn


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mlp(input_dim, width, output_dim, depth):
    layers = []
    for _ in range(depth):
        layers.extend((nn.Linear(input_dim, width), nn.Tanh()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


class DeepONet(nn.Module):
    """Learn ``(a, f) -> y`` using a joint branch and a coordinate trunk.

    ``grid_size`` is the number of nodes on each axis, including endpoints.
    ``sensor_stride`` subsamples each axis and always includes the last node.
    ``sensor_indices`` records every observed (x-index, y-index), in branch
    flattening order; ``sensor_points`` records its coordinates in [0, 1]^2.
    The branch receives all a sensors followed by all f sensors, then applies a
    nonlinear MLP.  The first layer therefore learns interactions between a and
    f, without a reduced-basis rank or source-coefficient restriction.

    ``depth`` is the number of hidden layers in each MLP.  An inner product of
    the two latent vectors, scaled by sqrt(latent_dim), gives the scalar state.
    With ``hard_boundary``, multiplication by 16*x*(1-x)*y*(1-y) enforces zero
    Dirichlet data.  This class does not normalize fields or include a physics
    loss; field normalization belongs to the training-only data adapter.
    """

    supports_queries = True

    def __init__(
        self,
        grid_size=65,
        sensor_stride=1,
        latent_dim=64,
        branch_width=64,
        trunk_width=64,
        depth=2,
        hard_boundary=True,
        coordinate_modes=0,
    ):
        super().__init__()
        self.grid_size = _positive_integer("grid_size", grid_size, 2)
        self.sensor_stride = _positive_integer("sensor_stride", sensor_stride)
        self.latent_dim = _positive_integer("latent_dim", latent_dim)
        _positive_integer("branch_width", branch_width)
        _positive_integer("trunk_width", trunk_width)
        _positive_integer("depth", depth)
        if not isinstance(hard_boundary, bool):
            raise ValueError("hard_boundary must be a bool")
        self.hard_boundary = hard_boundary
        self.coordinate_modes = _positive_integer("coordinate_modes", coordinate_modes, 0)

        axis = list(range(0, grid_size, sensor_stride))
        if axis[-1] != grid_size - 1:
            axis.append(grid_size - 1)
        sensor_axis = torch.tensor(axis, dtype=torch.long)
        indices = torch.stack(
            torch.meshgrid(sensor_axis, sensor_axis, indexing="ij"), dim=-1
        ).reshape(-1, 2)
        # Keep the coordinate record at double precision until an explicit
        # module/forward cast, including for grids with non-dyadic spacing.
        sensor_points = indices.to(torch.float64) / (grid_size - 1)
        grid_axis = torch.arange(grid_size, dtype=torch.float64) / (grid_size - 1)
        grid_points = torch.stack(
            torch.meshgrid(grid_axis, grid_axis, indexing="ij"), dim=-1
        ).reshape(-1, 2)
        self.register_buffer("sensor_axis_indices", sensor_axis)
        self.register_buffer("sensor_indices", indices)
        self.register_buffer("sensor_points", sensor_points)
        self.register_buffer("grid_points", grid_points)
        self.num_sensors = len(indices)
        self.branch = _mlp(2 * self.num_sensors, branch_width, latent_dim, depth)
        self.trunk = _mlp(2 + 4 * self.coordinate_modes, trunk_width, latent_dim, depth)
        self.bias = nn.Parameter(torch.zeros(()))

    def _coordinate_features(self, points):
        scaled = 2 * points - 1
        if not self.coordinate_modes:
            return scaled
        frequencies = torch.arange(
            1, self.coordinate_modes + 1, device=points.device, dtype=points.dtype
        )
        phase = math.pi * points.unsqueeze(-1) * frequencies
        return torch.cat((scaled, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)

    def _validate_fields(self, a, f):
        if not isinstance(a, torch.Tensor) or not isinstance(f, torch.Tensor):
            raise TypeError("a and f must be torch tensors")
        expected = (self.grid_size, self.grid_size)
        if a.ndim != 3 or a.shape[0] == 0 or tuple(a.shape[1:]) != expected:
            raise ValueError(f"a must have shape [B, {expected[0]}, {expected[1]}], B > 0")
        if f.shape != a.shape:
            raise ValueError("f must have the same [B, H, W] shape as a")
        if a.dtype not in (torch.float32, torch.float64) or f.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError("a and f must have float32 or float64 dtype")
        if a.dtype != f.dtype or a.device != f.device:
            raise ValueError("a and f must have the same dtype and device")
        self._validate_model_match(a)
        if not torch.isfinite(a).all() or not torch.isfinite(f).all():
            raise ValueError("a and f must contain only finite values")

    def _validate_model_match(self, reference):
        if any(
            parameter.dtype != reference.dtype or parameter.device != reference.device
            for parameter in self.parameters()
        ):
            raise ValueError(
                "fields/encoded inputs and model parameters must have the same dtype and device"
            )
        if self.sensor_indices.device != reference.device:
            raise ValueError("model sensor buffers and inputs must be on the same device")

    def representation_metadata(self):
        """Return a JSON-serializable description of the finite representation."""
        axis = self.sensor_axis_indices.detach().cpu().tolist()
        return {
            "model": "DeepONet",
            "supports_queries": True,
            "coordinate_modes": self.coordinate_modes,
            "coordinate_features": "raw scaled coordinates plus axiswise sin/cos(pi*k*x), k=1..coordinate_modes",
            "input_fields": ["a", "f"],
            "input_grid": {
                "shape": [self.grid_size, self.grid_size],
                "domain": [[0.0, 1.0], [0.0, 1.0]],
                "axis_order": "ij: x then y",
                "coordinates": "uniform nodes including both endpoints",
            },
            "sensors": {
                "stride": self.sensor_stride,
                "axis_indices": axis,
                "axis_coordinates": [index / (self.grid_size - 1) for index in axis],
                "shape": [len(axis), len(axis)],
                "count": self.num_sensors,
                "layout": "Cartesian product in ij flattening order; final endpoint always included",
            },
            "input_encoding": "all sampled a values followed by all sampled f values; joint nonlinear MLP",
            "finite_input_limit": "Only recorded sensor values affect predictions; arbitrary output queries do not recover unobserved input information.",
            "query": {
                "domain": "[0,1]^2",
                "layouts": ["[Q,2]", "[B,Q,2]"],
                "default": "full input grid",
                "independent_queries": True,
                "chunking": True,
                "default_chunk_size": 1024,
                "trunk_coordinates": "2*coordinates-1",
            },
            "boundary": {
                "hard_zero_dirichlet": self.hard_boundary,
                "factor": "16*x*(1-x)*y*(1-y)" if self.hard_boundary else None,
            },
            "supported_field_dtypes": ["float32", "float64"],
        }

    def encode_inputs(self, a, f):
        """Encode fields once and return an explicit, caller-owned context.

        The module stores no context or autograd graph across calls.  Reuse a
        context for several query sets under ``torch.no_grad()`` for inference,
        or combine their losses before a single backward pass.  Re-encode after
        changing parameters; contexts are not checkpoints or persistent caches.
        """
        self._validate_fields(a, f)
        i, j = self.sensor_indices.unbind(-1)
        branch_input = torch.cat((a[:, i, j], f[:, i, j]), dim=-1)
        return {"_model_id": id(self), "branch_values": self.branch(branch_input)}

    def _validated_context(self, encoded):
        if not isinstance(encoded, dict) or encoded.get("_model_id") != id(self):
            raise ValueError("encoded inputs must come from this model's encode_inputs")
        branch_values = encoded.get("branch_values")
        if (
            not isinstance(branch_values, torch.Tensor)
            or branch_values.ndim != 2
            or branch_values.shape[0] == 0
            or branch_values.shape[1] != self.latent_dim
        ):
            raise ValueError("encoded branch_values must have shape [B, latent_dim], B > 0")
        if branch_values.dtype not in (torch.float32, torch.float64):
            raise TypeError("encoded branch_values must have float32 or float64 dtype")
        self._validate_model_match(branch_values)
        if not torch.isfinite(branch_values).all():
            raise ValueError("encoded branch_values must contain only finite values")
        return branch_values

    def decode_queries(self, encoded, query_points=None, *, query_chunk_size=1024):
        """Decode a context at independent queries, optionally in chunks.

        None selects the full configured grid and returns [B,H,W].  Shared
        [Q,2] or batched [B,Q,2] points return [B,Q], including empty queries.
        Float32/64 queries are explicitly cast to the context dtype/device for
        compatibility with the original API; other query dtypes are rejected.
        """
        branch_values = self._validated_context(encoded)
        if query_chunk_size is not None:
            _positive_integer("query_chunk_size", query_chunk_size)
        full_grid = query_points is None
        if full_grid:
            points = self.grid_points.to(device=branch_values.device, dtype=branch_values.dtype)
        else:
            if not isinstance(query_points, torch.Tensor):
                raise TypeError("query_points must be a torch tensor")
            if query_points.ndim not in (2, 3) or query_points.shape[-1] != 2:
                raise ValueError("query_points must have shape [Q, 2] or [B, Q, 2]")
            if query_points.ndim == 3 and query_points.shape[0] != branch_values.shape[0]:
                raise ValueError("batched query_points must have the same batch size as a")
            if query_points.dtype not in (torch.float32, torch.float64):
                raise TypeError("query_points must have float32 or float64 dtype")
            if (
                not torch.isfinite(query_points).all()
                or ((query_points < 0) | (query_points > 1)).any()
            ):
                raise ValueError("query_points must be finite and lie in [0, 1]^2")
            points = query_points.to(device=branch_values.device, dtype=branch_values.dtype)

        query_count = points.shape[-2]
        if query_count == 0:
            # Preserve the source and query autograd paths for a valid no-op.
            return branch_values.sum(-1, keepdim=True) * points[..., 0]
        chunk_size = query_count if query_chunk_size is None else query_chunk_size
        predictions = []
        for start in range(0, query_count, chunk_size):
            query = points[..., start : start + chunk_size, :]
            # Only coordinates, not field values, are mapped to [-1,1].
            trunk_values = self.trunk(self._coordinate_features(query))
            if query.ndim == 2:
                prediction = torch.einsum("bk,qk->bq", branch_values, trunk_values)
            else:
                prediction = torch.einsum("bk,bqk->bq", branch_values, trunk_values)
            prediction = prediction / math.sqrt(self.latent_dim) + self.bias
            if self.hard_boundary:
                x, y = query.unbind(-1)
                prediction = prediction * (16 * x * (1 - x) * y * (1 - y))
            predictions.append(prediction)
        prediction = torch.cat(predictions, dim=-1)
        if full_grid:
            return prediction.reshape(branch_values.shape[0], self.grid_size, self.grid_size)
        return prediction

    def forward(self, a, f, query_points=None, *, query_chunk_size=1024):
        """Encode fields once and predict a full grid or independent queries.

        Fields are [B,grid_size,grid_size] and must match the model dtype/device.
        None queries give [B,H,W]; [Q,2] or [B,Q,2] queries give [B,Q].  A
        positive chunk size bounds each trunk evaluation; None disables chunks.
        Chunking does not change predictions or their gradients mathematically.
        """
        return self.decode_queries(
            self.encode_inputs(a, f), query_points, query_chunk_size=query_chunk_size
        )
