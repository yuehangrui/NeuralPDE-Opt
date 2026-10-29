"""Quadrature, geometry and differentiability checks for the teaching GNO."""

import json
import os
import unittest
from unittest.mock import patch

import torch

from neuralpde_opt.chapter3.gno import (GNO2d, IntegralKernel2d, radius_graph,
                                       p1_interpolate_rectilinear, trapezoid_weights)


def grid(n, m=None, dtype=torch.float64):
    m = m or n
    x, y = torch.linspace(0, 1, n, dtype=dtype), torch.linspace(0, 1, m, dtype=dtype)
    points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
    # Independent trapezoid construction via differences between cell centers.
    wx = torch.cat((x[1:2] / 2, (x[2:] - x[:-2]) / 2, (1 - x[-2:-1]) / 2))
    wy = torch.cat((y[1:2] / 2, (y[2:] - y[:-2]) / 2, (1 - y[-2:-1]) / 2))
    return points, (wx[:, None] * wy[None, :]).reshape(-1)


def constant_kernel(width=1, chunk=7):
    integral = IntegralKernel2d(width, edge_chunk_size=chunk).double()
    with torch.no_grad():
        integral.channel_map.weight.copy_(torch.eye(width, dtype=torch.float64))
        for parameter in integral.kernel.parameters():
            parameter.zero_()
        integral.kernel[-1].bias.fill_(1)
    return integral


class GraphOperatorTests(unittest.TestCase):
    def test_field_validation_and_grid_only_contract(self):
        model = GNO2d(width=4, layers=1, radius=.4, node_stride=2)
        self.assertFalse(model.supports_queries)
        self.assertFalse(model.representation_metadata()['supports_queries'])
        self.assertTrue(GNO2d(radius=1.5).representation_metadata()['global_communication'])
        for value in (torch.full((1,5,5), float('nan')),
                      torch.full((1,5,5), float('inf')), torch.empty((0,5,5))):
            with self.assertRaises(ValueError):
                model(value, value)
        points, weights = grid(3, dtype=torch.float32)
        with self.assertRaises(ValueError):
            model.forward_nodes(torch.full((1,9), float('nan')), torch.ones(1,9), points, weights)

    def test_integral_uses_source_quadrature_without_degree_normalization(self):
        points = torch.tensor([[0., 0.], [.2, .2], [.5, .5]], dtype=torch.float64)
        weights = torch.tensor([.1, .2, .4], dtype=torch.float64)
        values = torch.tensor([[[1.], [2.], [4.]]], dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2], [2, 2, 2]])
        result = constant_kernel()(values, points, weights, edges)
        torch.testing.assert_close(result[:, :2], torch.zeros(1, 2, 1, dtype=result.dtype))
        torch.testing.assert_close(result[:, 2], torch.tensor([[2.1]], dtype=result.dtype))
        doubled = constant_kernel()(values, points, 2 * weights, edges)
        torch.testing.assert_close(doubled, 2 * result)

    def test_constant_integral_is_consistent_under_grid_refinement(self):
        integral = constant_kernel(width=2, chunk=13)
        for n in (3, 5, 9):
            points, weights = grid(n)
            values = torch.ones(1, n * n, 2, dtype=torch.float64)
            result = integral(values, points, weights, radius_graph(points, 2.0))
            torch.testing.assert_close(result, values, rtol=1e-12, atol=1e-12)

    def test_radius_edges_and_edge_budget(self):
        points = torch.tensor([[0., 0.], [.25, 0.], [1., 1.]], dtype=torch.float64)
        edges = radius_graph(points, .3)
        self.assertEqual(set(map(tuple, edges.t().tolist())),
                         {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)})
        distances = (points[edges[0]] - points[edges[1]]).norm(dim=-1)
        self.assertTrue(bool((distances <= .3).all()))
        with self.assertRaises(ValueError):
            radius_graph(points, 2.0, max_edges=8)

    def test_node_permutation_equivariance(self):
        torch.manual_seed(117)
        model = GNO2d(width=3, layers=2, radius=.6, edge_chunk_size=11).double()
        points, weights = grid(4)
        a, f = torch.randn(2, 16, dtype=torch.float64), torch.randn(2, 16, dtype=torch.float64)
        permutation = torch.randperm(16)
        original = model.forward_nodes(a, f, points, weights)
        permuted = model.forward_nodes(a[:, permutation], f[:, permutation],
                                       points[permutation], weights[permutation])
        torch.testing.assert_close(permuted, original[:, permutation], rtol=1e-12, atol=1e-12)

    def test_regular_grid_matches_explicit_points_and_has_zero_boundary(self):
        torch.manual_seed(121)
        for dtype in (torch.float32, torch.float64):
            model = GNO2d(width=3, layers=2, radius=.4).to(dtype=dtype)
            for n, m in ((4, 5), (5, 5)):
                a = torch.randn(2, n, m, dtype=dtype)
                f = torch.randn_like(a)
                result = model(a, f)
                points, weights = grid(n, m, dtype=dtype)
                explicit = model.forward_nodes(a.reshape(2, -1), f.reshape(2, -1),
                                                points, weights)
                torch.testing.assert_close(result.reshape(2, -1), explicit)
                self.assertEqual(result.dtype, dtype)
                boundary = torch.cat((result[:, 0], result[:, -1],
                                      result[:, :, 0], result[:, :, -1]), dim=-1)
                self.assertEqual(torch.count_nonzero(boundary).item(), 0)
                self.assertGreater(torch.count_nonzero(result[:, 1:-1, 1:-1]).item(), 0)

    def test_input_and_parameter_gradients(self):
        torch.manual_seed(213)
        model = GNO2d(width=3, layers=2, radius=.6, edge_chunk_size=7).double()
        a = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
        model(a, f).square().mean().backward()
        for gradient in (a.grad, f.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(gradient.abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        points, weights = grid(2)
        values = torch.randn(1, 4, 1, dtype=torch.float64, requires_grad=True)
        integral = constant_kernel()
        edges = radius_graph(points, 2.0)
        self.assertTrue(torch.autograd.gradcheck(
            lambda v: integral(v, points, weights, edges), (values,), fast_mode=True))

    def test_default_radius_supports_public_65_grid_without_dense_graph(self):
        model = GNO2d(width=2, layers=1)
        a = torch.ones(1, 65, 65)
        with torch.no_grad():
            result = model(a, a)
        self.assertEqual(result.shape, a.shape)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(torch.count_nonzero(result[:, 0]).item(), 0)

    def test_optional_boundary_and_invalid_inputs(self):
        model = GNO2d(width=2, layers=1, hard_boundary=False).double()
        with torch.no_grad():
            model.project[-1].weight.zero_()
            model.project[-1].bias.fill_(1)
        a = torch.zeros(1, 3, 3, dtype=torch.float64)
        torch.testing.assert_close(model(a, a), torch.ones_like(a))
        points, weights = grid(3)
        with self.assertRaises(ValueError):
            model.forward_nodes(a.reshape(1, -1), a.reshape(1, -1), points, -weights)
        with self.assertRaises(ValueError):
            model.forward_nodes(a.reshape(1, -1), a.reshape(1, -1), points + 1, weights)
        with self.assertRaises(ValueError):
            model.forward_nodes(a.reshape(1, -1), a.reshape(1, -1), points, weights,
                                torch.tensor([[0], [9]]))
        with self.assertRaises(ValueError):
            model(a.float(), a.float())

    def test_nonuniform_trapezoid_and_p1_linear_precision(self):
        x = torch.tensor([0., .4, .8, 1.], dtype=torch.float64)
        y = torch.tensor([0., .3, .6, .9, 1.], dtype=torch.float64)
        torch.testing.assert_close(trapezoid_weights(x),
                                   torch.tensor([.2, .4, .3, .1], dtype=x.dtype))
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        nodal = (2 + 3 * xx - 4 * yy).unsqueeze(0).requires_grad_()
        qx, qy = torch.linspace(0, 1, 13, dtype=x.dtype), torch.linspace(0, 1, 12, dtype=x.dtype)
        result = p1_interpolate_rectilinear(nodal, x, y, qx, qy)
        exact = 2 + 3 * qx[:, None] - 4 * qy[None, :]
        torch.testing.assert_close(result[0], exact, rtol=1e-12, atol=1e-12)
        result.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(nodal.grad).all()))
        self.assertGreater(nodal.grad.abs().sum().item(), 0)

    def test_p1_uses_sw_ne_triangles_rather_than_bilinear_shape_functions(self):
        nodes = torch.tensor([0., 1.], dtype=torch.float64)
        nodal = torch.tensor([[[0., 0.], [0., 1.]]], dtype=nodes.dtype)
        query = torch.tensor([.25, .75], dtype=nodes.dtype)
        actual = p1_interpolate_rectilinear(nodal, nodes, nodes, query, query)
        torch.testing.assert_close(actual[0], torch.minimum(query[:, None], query[None, :]))

    def test_stride_preserves_short_final_cell_and_full_output_boundary(self):
        model = GNO2d(width=3, layers=1, radius=.6, node_stride=4).double()
        a = torch.randn(1, 10, 11, dtype=torch.float64, requires_grad=True)
        f = torch.randn_like(a, requires_grad=True)
        output = model(a, f)
        x_full = torch.linspace(0, 1, 10, dtype=a.dtype)
        y_full = torch.linspace(0, 1, 11, dtype=a.dtype)
        ix, iy = [0, 4, 8, 9], [0, 4, 8, 10]
        x, y = x_full[ix], y_full[iy]
        points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
        weights = (trapezoid_weights(x)[:, None] * trapezoid_weights(y)[None, :]).reshape(-1)
        coarse = model.forward_nodes(a[:, ix, :][:, :, iy].reshape(1, -1),
                                     f[:, ix, :][:, :, iy].reshape(1, -1), points, weights)
        expected = p1_interpolate_rectilinear(coarse.reshape(1, 4, 4), x, y, x_full, y_full)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        boundary = torch.cat((output[:, 0], output[:, -1], output[:, :, 0], output[:, :, -1]), -1)
        self.assertEqual(torch.count_nonzero(boundary).item(), 0)
        output.square().mean().backward()
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertGreater(f.grad.abs().sum().item(), 0)
        # Unsampled inputs deliberately have no influence in this representation.
        self.assertEqual(torch.count_nonzero(a.grad[:, 1, :]).item(), 0)

    def test_topology_cache_matches_coordinates_radius_dtype_and_is_not_serialized(self):
        model = GNO2d(width=2, layers=1, radius=.4).double()
        points, weights = grid(3)
        a = torch.ones(1, 9, dtype=torch.float64)
        with patch("neuralpde_opt.chapter3.gno.radius_graph", wraps=radius_graph) as build:
            model.forward_nodes(a, a, points, weights)
            model.forward_nodes(a, a, points.clone(), weights)
            self.assertEqual(build.call_count, 1)
            changed = points.clone()
            changed[4, 0] += .01
            model.forward_nodes(a, a, changed, weights)
            self.assertEqual(build.call_count, 2)
            model.radius = .5
            model.forward_nodes(a, a, changed, weights)
            self.assertEqual(build.call_count, 3)
            self.assertFalse(model._graph_cache[1].requires_grad)
            self.assertFalse(model._graph_cache[2].requires_grad)
            model.float()
            self.assertIsNone(model._graph_cache)
            model.forward_nodes(a.float(), a.float(), changed.float(), weights.float())
            self.assertEqual(build.call_count, 4)
        self.assertFalse(any("cache" in key for key in model.state_dict()))
        metadata = json.loads(json.dumps(model.representation_metadata()))
        self.assertEqual(metadata["node_stride"], 1)
        self.assertFalse(metadata["global_communication"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_deterministic_forward_backward_and_device_cache_reset(self):
        # Set before the first CUDA linear operation; the runner also exports
        # this official cuBLAS determinism setting before process startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            model = GNO2d(width=3, layers=2, radius=.6, node_stride=2).cuda()
            a = torch.randn(2, 9, 10, device="cuda", requires_grad=True)
            f = torch.randn_like(a, requires_grad=True)
            result = model(a, f)
            result.square().mean().backward()
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertTrue(bool(torch.isfinite(a.grad).all()))
            self.assertTrue(bool(torch.isfinite(f.grad).all()))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertEqual(model._graph_cache[1].device.type, "cuda")
            model.cpu()
            self.assertIsNone(model._graph_cache)
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
