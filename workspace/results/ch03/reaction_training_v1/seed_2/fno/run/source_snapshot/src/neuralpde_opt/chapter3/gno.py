"""A teaching baseline for quadrature-weighted graph neural operators.

This is a local, factored-kernel GNO, not a reproduction of a general
matrix-kernel architecture. It approximates an integral by physical quadrature
weights, rather than by an unweighted average over the graph degree. A finite
stack of local layers has a finite graph receptive field; elliptic solution
operators can require additional global or multilevel communication.

Book/code notation:
Book 3.3.2: graph-node predictions represent y_theta. The current 17x17-node output is reconstructed by P1 interpolation; gradient error depends on that output space.
"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F


def radius_graph(points: torch.Tensor, radius: float, *, max_edges=2_000_000):
    """Build directed ``[source, target]`` edges, including self edges.

    A CPU spatial index avoids forming an N-by-N distance matrix. Count the
    edges before expanding neighbor lists and reject graphs above ``max_edges``.
    This function does not cache a graph. Coordinates determine topology
    through a detached copy; gradients can still pass through the edge kernel
    on the resulting fixed graph, but not through edge membership changes.
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points must have shape [N, 2] with N > 0")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not isinstance(max_edges, int) or max_edges < 1:
        raise ValueError("max_edges must be a positive integer")
    coordinates = points.detach().cpu().double().numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("points must be finite")
    tree = cKDTree(coordinates)
    counts = tree.query_ball_point(coordinates, radius, return_length=True)
    edge_count = int(np.sum(counts))
    if edge_count > max_edges:
        raise ValueError(
            f"radius graph has {edge_count} edges, above the teaching baseline "
            f"limit {max_edges}; reduce radius/grid size or provide a suitable graph"
        )
    edges = np.empty((2, edge_count), dtype=np.int64)
    cursor = 0
    # Limit temporary Python neighbor lists as well as kernel message tensors.
    for start in range(0, len(coordinates), 256):
        neighbors = tree.query_ball_point(
            coordinates[start : start + 256], radius, return_sorted=True
        )
        for offset, source_indices in enumerate(neighbors):
            stop = cursor + len(source_indices)
            edges[0, cursor:stop] = source_indices
            edges[1, cursor:stop] = start + offset
            cursor = stop
    return torch.from_numpy(edges).to(device=points.device)


def trapezoid_weights(nodes: torch.Tensor):
    """Positive 1D trapezoid weights for actual, possibly nonuniform, nodes."""
    if nodes.ndim != 1 or nodes.numel() < 2:
        raise ValueError("nodes must be a 1D array with at least two entries")
    gaps = nodes[1:] - nodes[:-1]
    if not bool(torch.isfinite(nodes).all()) or not bool((gaps > 0).all()):
        raise ValueError("nodes must be finite and strictly increasing")
    return torch.cat((gaps[:1] / 2, (gaps[:-1] + gaps[1:]) / 2, gaps[-1:] / 2))


def p1_interpolate_rectilinear(values, x_nodes, y_nodes, x_query, y_query):
    """Differentiable P1 interpolation on rectangles split along SW--NE.

    Values are ``[B,Nx,Ny]`` with indexing='ij'; query axes need not be uniform.
    The formula uses actual cell widths, including a short final stride cell.
    Differentiation through nodal values is supported; the chosen cell/triangle
    is discrete and does not have a derivative at changes of membership.
    """
    if values.ndim != 3 or values.shape[1:] != (x_nodes.numel(), y_nodes.numel()):
        raise ValueError("values must have shape [B,len(x_nodes),len(y_nodes)]")
    for axis, query in ((x_nodes, x_query), (y_nodes, y_query)):
        trapezoid_weights(axis)  # Validate strict ordering and positive cell widths.
        if query.ndim != 1 or not bool(torch.isfinite(query).all()):
            raise ValueError("query axes must be finite 1D arrays")
        if query.numel() and (bool((query < axis[0]).any()) or bool((query > axis[-1]).any())):
            raise ValueError("query axes must lie inside the node domain")
        if axis.dtype != values.dtype or query.dtype != values.dtype:
            raise ValueError("interpolation dtypes must match")
        if axis.device != values.device or query.device != values.device:
            raise ValueError("interpolation devices must match")
    ix = (torch.searchsorted(x_nodes, x_query.contiguous(), right=True) - 1).clamp(
        0, len(x_nodes) - 2
    )
    iy = (torch.searchsorted(y_nodes, y_query.contiguous(), right=True) - 1).clamp(
        0, len(y_nodes) - 2
    )
    tx = ((x_query - x_nodes[ix]) / (x_nodes[ix + 1] - x_nodes[ix]))[:, None]
    ty = ((y_query - y_nodes[iy]) / (y_nodes[iy + 1] - y_nodes[iy]))[None, :]
    v00 = values[:, ix[:, None], iy[None, :]]
    v10 = values[:, (ix + 1)[:, None], iy[None, :]]
    v01 = values[:, ix[:, None], (iy + 1)[None, :]]
    v11 = values[:, (ix + 1)[:, None], (iy + 1)[None, :]]
    below = (1 - tx) * v00 + (tx - ty) * v10 + ty * v11
    above = (1 - ty) * v00 + (ty - tx) * v01 + tx * v11
    return torch.where(tx >= ty, below, above)


class IntegralKernel2d(nn.Module):
    """Compute ``sum_j w_j diag(g_theta(e_ij)) W v_j`` at each target i.

    Edge features are ``(x_i, x_j, x_j-x_i, |x_j-x_i|)`` (seven scalars).
    The diagonal gate times a shared channel matrix is an explicit restriction
    of a general edge-dependent C-by-C matrix kernel, chosen for readability
    and to avoid materializing a channel matrix on every edge.
    """

    def __init__(self, width: int, edge_chunk_size=8192):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("width must be a positive integer")
        if not isinstance(edge_chunk_size, int) or edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be a positive integer")
        self.width = width
        self.edge_chunk_size = edge_chunk_size
        self.channel_map = nn.Linear(width, width, bias=False)
        self.kernel = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, values, points, weights, edge_index):
        transformed = self.channel_map(values)
        result = torch.zeros_like(transformed)
        for start in range(0, edge_index.shape[1], self.edge_chunk_size):
            source, target = edge_index[:, start : start + self.edge_chunk_size]
            displacement = points[source] - points[target]
            edge_features = torch.cat(
                (
                    points[target],
                    points[source],
                    displacement,
                    torch.linalg.vector_norm(displacement, dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            gate = self.kernel(edge_features)
            message = transformed[:, source] * gate.unsqueeze(0) * weights[source].reshape(1, -1, 1)
            result = result.index_add(1, target, message)
        return result


class GNO2d(nn.Module):
    """Joint ``(a,f) -> y`` baseline on a square grid or a shared point cloud.

    ``forward(a,f)`` accepts ``[B,H,W]`` on the closed unit-square grid and uses
    tensor trapezoid quadrature. ``forward_nodes`` accepts ``[B,N]``, common
    physical coordinates ``[N,2]``, and strictly positive quadrature weights
    ``[N]`` (e.g. mass-lumped finite-element weights). Inputs must match model
    dtype/device. A supplied edge list uses rows ``[source,target]`` with each
    intended directed edge once; it may describe another explicitly chosen
    topology. The most recent radius topology is cached only while coordinates,
    radius, edge budget, dtype and device match; no features/autograd graph are
    cached. ``node_stride>1`` samples input grid nodes (always including both
    endpoints), predicts there and uses SW--NE P1 interpolation for full-grid
    output. It does not change the shared reference labels or dataset.
    """

    supports_queries = False

    def __init__(
        self,
        width=8,
        layers=3,
        radius=0.15,
        hard_boundary=True,
        *,
        edge_chunk_size=8192,
        max_edges=2_000_000,
        node_stride=1,
    ):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("width must be a positive integer")
        if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not isinstance(max_edges, int) or max_edges < 1:
            raise ValueError("max_edges must be a positive integer")
        if not isinstance(node_stride, int) or isinstance(node_stride, bool) or node_stride < 1:
            raise ValueError("node_stride must be a positive integer")
        self.radius = float(radius)
        self.hard_boundary = bool(hard_boundary)
        self.max_edges = max_edges
        self.node_stride = node_stride
        self._graph_cache = None
        self.lift = nn.Linear(4, width)
        self.integral_layers = nn.ModuleList(
            [IntegralKernel2d(width, edge_chunk_size) for _ in range(layers)]
        )
        self.local_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, 1)
        )

    def _apply(self, fn, recurse=True):
        # Plain cached tensors are intentionally absent from state_dict and must
        # not retain an old CUDA allocation when model.to()/float()/double() runs.
        self._graph_cache = None
        return super()._apply(fn, recurse=recurse)

    def representation_metadata(self):
        """JSON-serializable finite representation and architectural limits."""
        return {
            "architecture": "GNO2d",
            "supports_queries": False,
            "input_representation": "joint a,f samples with physical ij coordinates",
            "node_stride": self.node_stride,
            "input_sampling": "grid indices 0,stride,... plus final endpoint on each axis; forward_nodes uses all supplied nodes",
            "endpoint_policy": "always include final endpoint; last cell can be shorter",
            "quadrature": "tensor trapezoid on actual sampled coordinates; explicit positive weights for point clouds",
            "output_reconstruction": "SW-NE P1 on actual rectilinear node cells",
            "output_grid": "full original H-by-W grid; shared labels are not downsampled",
            "width": self.lift.out_features,
            "layers": len(self.integral_layers),
            "kernel": "diag(g_theta(x_i,x_j,x_j-x_i,|x_j-x_i|)) @ W",
            "kernel_is_general_matrix": False,
            "radius": self.radius,
            "global_communication": self.radius >= math.sqrt(2),
            "global_communication_definition": "one radius neighborhood covers the unit square when radius >= sqrt(2); no additional global/multilevel module",
            "multilevel_module": False,
            "local_receptive_distance_upper_bound": self.radius * len(self.integral_layers),
            "hard_boundary": self.hard_boundary,
            "max_edges": self.max_edges,
            "topology_cache": "one detached coordinate snapshot and edge list; excluded from state_dict",
        }

    def _radius_edges(self, points):
        key = (self.radius, self.max_edges, points.dtype, points.device, tuple(points.shape))
        if (
            self._graph_cache is not None
            and self._graph_cache[0] == key
            and torch.equal(self._graph_cache[1], points.detach())
        ):
            return self._graph_cache[2]
        edges = radius_graph(points, self.radius, max_edges=self.max_edges)
        self._graph_cache = (key, points.detach().clone(), edges)
        return edges

    def forward_nodes(self, a, f, points, weights, edge_index=None):
        if not isinstance(a, torch.Tensor) or not isinstance(f, torch.Tensor):
            raise TypeError("a and f must be torch tensors")
        if a.ndim != 2 or a.shape[0] == 0 or f.shape != a.shape:
            raise ValueError("a and f must have matching shapes [B,N]")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(f).all()):
            raise ValueError("a and f must be finite")
        count = a.shape[1]
        if count == 0 or points.shape != (count, 2) or weights.shape != (count,):
            raise ValueError("points and weights must have shapes [N,2] and [N], N > 0")
        if a.dtype not in (torch.float32, torch.float64):
            raise TypeError("inputs must use float32 or float64")
        for value in (a, f, points, weights):
            if value.dtype != self.lift.weight.dtype or value.device != self.lift.weight.device:
                raise ValueError("all input dtypes/devices must match the model")
        if not bool(torch.isfinite(points).all()) or not bool(
            ((points >= 0) & (points <= 1)).all()
        ):
            raise ValueError("points must be finite physical coordinates in [0,1]^2")
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("quadrature weights must be finite and strictly positive")
        if edge_index is None:
            edge_index = self._radius_edges(points)
        elif (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
            or edge_index.device != a.device
        ):
            raise ValueError("edge_index must be a long [2,E] tensor on the input device")
        if edge_index.shape[1] > self.max_edges:
            raise ValueError("supplied graph exceeds max_edges")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= count):
            raise ValueError("edge_index contains an out-of-range node index")
        coordinates = points.expand(a.shape[0], -1, -1)
        values = self.lift(torch.cat((a.unsqueeze(-1), f.unsqueeze(-1), coordinates), dim=-1))
        for integral, local in zip(self.integral_layers, self.local_layers):
            values = F.gelu(local(values) + integral(values, points, weights, edge_index))
        result = self.project(values).squeeze(-1)
        if self.hard_boundary:
            x, y = points.unbind(-1)
            result = result * (16 * x * (1 - x) * y * (1 - y))
        return result

    def forward(self, a, f):
        if not isinstance(a, torch.Tensor) or not isinstance(f, torch.Tensor):
            raise TypeError("a and f must be torch tensors")
        if a.ndim != 3 or a.shape[0] == 0 or f.shape != a.shape:
            raise ValueError("a and f must have matching shapes [B,H,W]")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(f).all()):
            raise ValueError("a and f must be finite")
        height, width = a.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("both grid axes must contain at least two points")
        if a.dtype not in (torch.float32, torch.float64):
            raise TypeError("inputs must use float32 or float64")
        x_query = torch.linspace(0, 1, height, dtype=a.dtype, device=a.device)
        y_query = torch.linspace(0, 1, width, dtype=a.dtype, device=a.device)
        ix = list(range(0, height - 1, self.node_stride)) + [height - 1]
        iy = list(range(0, width - 1, self.node_stride)) + [width - 1]
        x, y = x_query[ix], y_query[iy]
        points = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)
        weights = (trapezoid_weights(x)[:, None] * trapezoid_weights(y)[None, :]).reshape(-1)
        sampled_a = a[:, ix, :][:, :, iy].reshape(a.shape[0], -1)
        sampled_f = f[:, ix, :][:, :, iy].reshape(f.shape[0], -1)
        result = self.forward_nodes(sampled_a, sampled_f, points, weights)
        if self.node_stride == 1:
            return result.reshape_as(a)
        return p1_interpolate_rectilinear(
            result.reshape(a.shape[0], len(ix), len(iy)), x, y, x_query, y_query
        )
