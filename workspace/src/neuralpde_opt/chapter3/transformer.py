"""A teaching Transformer operator with weighted, independent query attention.

This is a small supervised baseline, not a reproduction of GNOT or Galerkin
Transformer.  Sparse input sensors encode [a, f, x, y]; a weighted encoder
mixes their tokens, and coordinate queries attend to the encoded sensors.
There is deliberately no attention between output queries.  Their ordering and
chunk size therefore do not change the mathematical prediction at a point.

Book/code notation:
Book 3.3.3: token features are latent vectors, while coordinates and PDE output y have physical meanings. The formal reaction model uses the separately documented Galerkin backbone.
"""

import math

import torch
from torch import nn


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def weighted_scaled_dot_product_attention(query, key, value, key_weights):
    """Return softmax(q*k/sqrt(d) + log(w)) * v, without dropout.

    q, k, v have shapes [..., Q, d], [..., S, d], [..., S, dv] with the
    same leading dimensions.  Positive ``key_weights`` must broadcast to
    [..., S].  For [B, heads, Q, d] tensors, per-batch weights may be supplied
    as [B, 1, S]; a common [S] vector works for any number of leading axes.

    Weights approximate the spatial integral in both numerator and denominator
    of normalized attention.  Multiplying all weights by a positive constant
    leaves the result unchanged.  Duplicating a token and splitting its weight
    also leaves the result unchanged, unlike an unweighted token average.
    """
    tensors = (query, key, value, key_weights)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("attention inputs and key_weights must be torch tensors")
    if any(tensor.dtype not in (torch.float32, torch.float64) for tensor in tensors):
        raise TypeError("attention inputs and key_weights must have float32 or float64 dtype")
    if query.ndim < 2 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key and value must have the same rank >= 2")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key and value must have the same leading dimensions")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] == 0:
        raise ValueError("query and key need the same positive feature dimension")
    if key.shape[-2] != value.shape[-2] or key.shape[-2] == 0:
        raise ValueError("key and value need the same positive token count")
    if any(tensor.dtype != query.dtype or tensor.device != query.device for tensor in (key, value)):
        raise ValueError("query, key and value must have the same dtype and device")
    if any(not torch.isfinite(tensor).all() for tensor in (query, key, value)):
        raise ValueError("query, key and value must contain only finite values")
    if key_weights.ndim == 0 or key_weights.shape[-1] != key.shape[-2]:
        raise ValueError("key_weights must end in the key token dimension")
    target_shape = key.shape[:-2] + (key.shape[-2],)
    try:
        broadcast_shape = torch.broadcast_shapes(key_weights.shape, target_shape)
    except RuntimeError as error:
        raise ValueError("key_weights do not broadcast to the key tokens") from error
    if broadcast_shape != target_shape:
        raise ValueError("key_weights must not introduce new batch dimensions")
    if not torch.isfinite(key_weights).all() or (key_weights <= 0).any():
        raise ValueError("key_weights must be finite and strictly positive")
    weights = key_weights.to(device=query.device, dtype=query.dtype)
    logits = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    attention = torch.softmax(logits + weights.log().unsqueeze(-2), dim=-1)
    return attention @ value


class WeightedMultiheadAttention(nn.Module):
    """Batch-first multihead attention with positive source quadrature weights."""

    def __init__(self, width, heads):
        super().__init__()
        _positive_integer("width", width)
        _positive_integer("heads", heads)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads = width, heads
        self.query_projection = nn.Linear(width, width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)

    def _validate_source(self, source):
        if (
            not isinstance(source, torch.Tensor)
            or source.ndim != 3
            or source.shape[-1] != self.width
            or source.shape[0] == 0
            or source.shape[1] == 0
        ):
            raise ValueError("source must have shape [B, S, width], B > 0, S > 0")
        if source.dtype not in (torch.float32, torch.float64):
            raise TypeError("source must have float32 or float64 dtype")
        if any(
            parameter.dtype != source.dtype or parameter.device != source.device
            for parameter in self.parameters()
        ):
            raise ValueError("attention source and parameters must have the same dtype and device")
        if not torch.isfinite(source).all():
            raise ValueError("source must contain only finite values")

    def project_source(self, source):
        """Project source K/V once; the returned tensors belong to the caller.

        No graph or cache is saved on the attention module.  These projections
        can be reused for query chunks while retaining ordinary autograd.
        """
        self._validate_source(source)
        batch, count = source.shape[:2]
        head_dim = self.width // self.heads
        key = (
            self.key_projection(source).reshape(batch, count, self.heads, head_dim).transpose(1, 2)
        )
        value = (
            self.value_projection(source)
            .reshape(batch, count, self.heads, head_dim)
            .transpose(1, 2)
        )
        return key, value

    def forward(self, query, source, source_weights, *, projected_key_value=None):
        """Attend to source, optionally reusing explicit projected K/V tensors."""
        if not isinstance(query, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise TypeError("query and source must be torch tensors")
        if query.ndim != 3 or source.ndim != 3:
            raise ValueError("query and source must have shape [B, tokens, width]")
        if (
            query.shape[0] != source.shape[0]
            or query.shape[-1] != self.width
            or source.shape[-1] != self.width
        ):
            raise ValueError("query and source must match batch size and attention width")
        if query.dtype not in (torch.float32, torch.float64) or source.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError("query and source must have float32 or float64 dtype")
        if query.dtype != source.dtype or query.device != source.device:
            raise ValueError("query and source must have the same dtype and device")
        self._validate_source(source)
        if not torch.isfinite(query).all():
            raise ValueError("query must contain only finite values")
        if not isinstance(source_weights, torch.Tensor):
            raise TypeError("source_weights must be a torch tensor")
        batch, query_count = query.shape[:2]
        source_count = source.shape[1]
        head_dim = self.width // self.heads

        def split_heads(tensor, count):
            return tensor.reshape(batch, count, self.heads, head_dim).transpose(1, 2)

        q = split_heads(self.query_projection(query), query_count)
        if projected_key_value is None:
            k, v = self.project_source(source)
        else:
            if not isinstance(projected_key_value, (tuple, list)) or len(projected_key_value) != 2:
                raise ValueError("projected_key_value must be an explicit (key, value) pair")
            k, v = projected_key_value
            expected = (batch, self.heads, source_count, head_dim)
            if any(
                not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected
                for tensor in (k, v)
            ):
                raise ValueError(
                    "projected K/V must match source batch, heads, tokens and head width"
                )
        if source_weights.ndim == 2:
            if source_weights.shape[0] != batch:
                raise ValueError("per-batch source_weights must match query batch size")
            source_weights = source_weights.unsqueeze(1)
        elif source_weights.ndim != 1:
            raise ValueError("source_weights must have shape [S] or [B, S]")
        attended = weighted_scaled_dot_product_attention(q, k, v, source_weights)
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.width)
        return self.output_projection(attended)


class _EncoderLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = WeightedMultiheadAttention(width, heads)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens, weights):
        normalized = self.norm_attention(tokens)
        tokens = tokens + self.attention(normalized, normalized, weights)
        return tokens + self.feedforward(self.norm_feedforward(tokens))


class _QueryLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm_query = nn.LayerNorm(width)
        self.norm_source = nn.LayerNorm(width)
        self.attention = WeightedMultiheadAttention(width, heads)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def prepare_source(self, source):
        normalized = self.norm_source(source)
        return {"source": normalized, "key_value": self.attention.project_source(normalized)}

    def forward(self, queries, source, weights, *, prepared_source=None):
        # Only source tokens supply keys/values: no query-query interaction.
        if prepared_source is None:
            normalized, projected = self.norm_source(source), None
        else:
            normalized = prepared_source["source"]
            projected = prepared_source["key_value"]
        queries = queries + self.attention(
            self.norm_query(queries), normalized, weights, projected_key_value=projected
        )
        return queries + self.feedforward(self.norm_feedforward(queries))


class TransformerOperator(nn.Module):
    """Sparse-sensor Transformer baseline for (a, f) -> scalar state y.

    ``grid_size`` counts nodes per axis including endpoints.  A stride samples
    both axes, always retaining the last endpoint.  ``sensor_indices`` and
    ``sensor_points`` record the exact token layout.  ``sensor_weights`` are
    the positive tensor-product trapezoid weights of that possibly nonuniform
    layout and sum to the unit-square area, one.

    Coordinates are recorded in [0, 1]^2 and mapped to [-1, 1] for embedding;
    field normalization belongs to the training-only data adapter.  Every
    encoder and cross-attention uses log quadrature weights in its logits.
    ``layers`` selects the number of source encoder blocks and query decoder
    blocks.  There is no dropout or attention across query locations.

    Default grid_size=65, sensor_stride=4 uses 17*17=289 source tokens, not
    65*65 tokens.  Source attention costs O(S^2); query attention costs O(Q*S)
    and is evaluated in chunks by default.  Unobserved input field values
    cannot affect the prediction.  Arbitrary query coordinates do not remove
    that input representation limit or guarantee resolution convergence.
    """

    supports_queries = True

    def __init__(
        self,
        grid_size=65,
        sensor_stride=4,
        width=32,
        heads=4,
        layers=2,
        hard_boundary=True,
        coordinate_modes=0,
    ):
        super().__init__()
        self.grid_size = _positive_integer("grid_size", grid_size, 2)
        self.sensor_stride = _positive_integer("sensor_stride", sensor_stride)
        self.width = _positive_integer("width", width)
        self.heads = _positive_integer("heads", heads)
        self.layers = _positive_integer("layers", layers)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(hard_boundary, bool):
            raise ValueError("hard_boundary must be a bool")
        self.hard_boundary = hard_boundary
        self.coordinate_modes = _positive_integer("coordinate_modes", coordinate_modes, 0)

        axis = list(range(0, grid_size, sensor_stride))
        if axis[-1] != grid_size - 1:
            axis.append(grid_size - 1)
        axis_indices = torch.tensor(axis, dtype=torch.long)
        axis_points = axis_indices.to(torch.float64) / (grid_size - 1)
        axis_weights = torch.empty_like(axis_points)
        axis_weights[0] = (axis_points[1] - axis_points[0]) / 2
        axis_weights[-1] = (axis_points[-1] - axis_points[-2]) / 2
        axis_weights[1:-1] = (axis_points[2:] - axis_points[:-2]) / 2
        sensor_indices = torch.stack(
            torch.meshgrid(axis_indices, axis_indices, indexing="ij"), dim=-1
        ).reshape(-1, 2)
        sensor_points = sensor_indices.to(torch.float64) / (grid_size - 1)
        sensor_weights = (axis_weights[:, None] * axis_weights[None, :]).reshape(-1)
        grid_axis = torch.arange(grid_size, dtype=torch.float64) / (grid_size - 1)
        grid_points = torch.stack(
            torch.meshgrid(grid_axis, grid_axis, indexing="ij"), dim=-1
        ).reshape(-1, 2)
        self.register_buffer("sensor_axis_indices", axis_indices)
        self.register_buffer("sensor_indices", sensor_indices)
        self.register_buffer("sensor_points", sensor_points)
        self.register_buffer("sensor_axis_weights", axis_weights)
        self.register_buffer("sensor_weights", sensor_weights)
        self.register_buffer("grid_points", grid_points)
        self.num_sensors = len(sensor_indices)
        self.input_embedding = nn.Sequential(
            nn.Linear(4 + 4 * self.coordinate_modes, width), nn.GELU(), nn.Linear(width, width)
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(2 + 4 * self.coordinate_modes, width), nn.GELU(), nn.Linear(width, width)
        )
        self.encoder = nn.ModuleList(_EncoderLayer(width, heads) for _ in range(layers))
        self.decoder = nn.ModuleList(_QueryLayer(width, heads) for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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
        if a.ndim != 3 or a.shape[0] == 0 or tuple(a.shape[1:]) != (self.grid_size, self.grid_size):
            raise ValueError(f"a must have shape [B, {self.grid_size}, {self.grid_size}], B > 0")
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
        """Return JSON-serializable representation and attention conventions."""
        axis = self.sensor_axis_indices.detach().cpu().tolist()
        return {
            "model": "TransformerOperator",
            "variant": "teaching weighted-attention baseline",
            "reproduction": "Not a GNOT or Galerkin Transformer reproduction",
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
                "axis_quadrature_weights": self.sensor_axis_weights.detach().cpu().tolist(),
                "quadrature": "positive tensor-product trapezoid weights; unit total area",
            },
            "input_encoding": "joint [a,f,2*x-1,2*y-1] token embedding and weighted source self-attention",
            "finite_input_limit": "Only recorded sparse sensor values affect predictions; arbitrary output queries do not recover unobserved input information.",
            "attention": {
                "normalization": "softmax(q*k/sqrt(head_dim) + log(source_weight))",
                "source_self_attention": True,
                "query_self_attention": False,
                "source_key_value_projection": "once per decoder layer per explicit input encoding",
            },
            "query": {
                "domain": "[0,1]^2",
                "layouts": ["[Q,2]", "[B,Q,2]"],
                "default": "full input grid",
                "independent_queries": True,
                "chunking": True,
                "default_chunk_size": 1024,
                "embedding_coordinates": "2*coordinates-1",
            },
            "boundary": {
                "hard_zero_dirichlet": self.hard_boundary,
                "factor": "16*x*(1-x)*y*(1-y)" if self.hard_boundary else None,
            },
            "supported_field_dtypes": ["float32", "float64"],
        }

    def encode_inputs(self, a, f):
        """Encode sparse inputs and project each decoder's source K/V once.

        The returned context is caller-owned; the module never caches a graph.
        Reuse under ``torch.no_grad()`` for inference, or combine query losses
        before one backward pass.  Re-encode after any parameter update.  A
        context is transient computation, not part of a checkpoint/state_dict.
        """
        self._validate_fields(a, f)
        i, j = self.sensor_indices.unbind(-1)
        coordinates = self.sensor_points.to(device=a.device, dtype=a.dtype)
        coordinates = (
            (self._coordinate_features(coordinates)).unsqueeze(0).expand(a.shape[0], -1, -1)
        )
        features = torch.cat((a[:, i, j, None], f[:, i, j, None], coordinates), dim=-1)
        tokens = self.input_embedding(features)
        weights = self.sensor_weights.to(device=a.device, dtype=a.dtype)
        for layer in self.encoder:
            tokens = layer(tokens, weights)
        prepared = tuple(layer.prepare_source(tokens) for layer in self.decoder)
        return {
            "_model_id": id(self),
            "source_tokens": tokens,
            "source_weights": weights,
            "decoder_sources": prepared,
        }

    def _validated_context(self, encoded):
        if not isinstance(encoded, dict) or encoded.get("_model_id") != id(self):
            raise ValueError("encoded inputs must come from this model's encode_inputs")
        tokens, weights = encoded.get("source_tokens"), encoded.get("source_weights")
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] == 0
            or tuple(tokens.shape[1:]) != (self.num_sensors, self.width)
        ):
            raise ValueError("encoded source_tokens must have shape [B, num_sensors, width], B > 0")
        if tokens.dtype not in (torch.float32, torch.float64):
            raise TypeError("encoded source_tokens must have float32 or float64 dtype")
        self._validate_model_match(tokens)
        if not torch.isfinite(tokens).all():
            raise ValueError("encoded source_tokens must contain only finite values")
        if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != (self.num_sensors,):
            raise ValueError("encoded source_weights must have shape [num_sensors]")
        if weights.dtype != tokens.dtype or weights.device != tokens.device:
            raise ValueError("encoded source_weights must match source_tokens dtype and device")
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("encoded source_weights must be finite and strictly positive")
        prepared = encoded.get("decoder_sources")
        if not isinstance(prepared, (tuple, list)) or len(prepared) != len(self.decoder):
            raise ValueError("encoded decoder_sources must match the decoder layers")
        expected = (tokens.shape[0], self.heads, self.num_sensors, self.width // self.heads)
        for entry in prepared:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("source"), torch.Tensor)
                or tuple(entry["source"].shape) != tuple(tokens.shape)
            ):
                raise ValueError("encoded decoder source must match source_tokens shape")
            key_value = entry.get("key_value")
            if not isinstance(key_value, (tuple, list)) or len(key_value) != 2:
                raise ValueError("encoded decoder source must contain a projected K/V pair")
            for tensor in (entry["source"], *key_value):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != tokens.dtype
                    or tensor.device != tokens.device
                ):
                    raise ValueError("encoded decoder tensors must match source dtype and device")
                if not torch.isfinite(tensor).all():
                    raise ValueError("encoded decoder tensors must contain only finite values")
            if any(tuple(tensor.shape) != expected for tensor in key_value):
                raise ValueError("encoded projected K/V dimensions do not match this model")
        return tokens, weights, prepared

    def decode_queries(self, encoded, query_points=None, *, query_chunk_size=1024):
        """Decode full-grid or independent shared/batched point queries.

        K/V projections in ``encoded`` are reused across all chunks and repeated
        query sets.  Float32/64 query tensors are explicitly cast to the context
        dtype/device, preserving the old API; other query dtypes are rejected.
        """
        tokens, weights, prepared = self._validated_context(encoded)
        if query_chunk_size is not None:
            _positive_integer("query_chunk_size", query_chunk_size)
        full_grid = query_points is None
        if full_grid:
            points = self.grid_points.to(device=tokens.device, dtype=tokens.dtype)
        else:
            if not isinstance(query_points, torch.Tensor):
                raise TypeError("query_points must be a torch tensor")
            if query_points.ndim not in (2, 3) or query_points.shape[-1] != 2:
                raise ValueError("query_points must have shape [Q, 2] or [B, Q, 2]")
            if query_points.ndim == 3 and query_points.shape[0] != tokens.shape[0]:
                raise ValueError("batched query_points must match the input batch size")
            if query_points.dtype not in (torch.float32, torch.float64):
                raise TypeError("query_points must have float32 or float64 dtype")
            if (
                not torch.isfinite(query_points).all()
                or ((query_points < 0) | (query_points > 1)).any()
            ):
                raise ValueError("query_points must be finite and lie in [0, 1]^2")
            points = query_points.to(device=tokens.device, dtype=tokens.dtype)
        if points.ndim == 2:
            points = points.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        query_count = points.shape[1]
        if query_count == 0:
            return tokens[:, :0, 0] + points[..., 0] * 0
        chunk_size = query_count if query_chunk_size is None else query_chunk_size

        predictions = []
        for start in range(0, query_count, chunk_size):
            query = points[:, start : start + chunk_size]
            decoded = self.query_embedding(self._coordinate_features(query))
            for layer, source in zip(self.decoder, prepared):
                decoded = layer(decoded, tokens, weights, prepared_source=source)
            prediction = self.output(decoded).squeeze(-1)
            if self.hard_boundary:
                x, y = query.unbind(-1)
                prediction = prediction * (16 * x * (1 - x) * y * (1 - y))
            predictions.append(prediction)
        prediction = torch.cat(predictions, dim=1)
        if full_grid:
            return prediction.reshape(tokens.shape[0], self.grid_size, self.grid_size)
        return prediction

    def forward(self, a, f, query_points=None, *, query_chunk_size=1024):
        """Return [B,H,W], or [B,Q] for [Q,2]/[B,Q,2] unit-square queries.

        Inputs and model must share float32/64 dtype and device.  Source tokens
        and each decoder's K/V are computed once per call.  A positive chunk
        size bounds each query attention matrix; None disables query chunking.
        Training autograd still retains all chunks until backward is complete.
        """
        return self.decode_queries(
            self.encode_inputs(a, f), query_points, query_chunk_size=query_chunk_size
        )
