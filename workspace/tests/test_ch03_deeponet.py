"""Representation, query, boundary and differentiation checks for DeepONet."""

import json
import unittest

import torch

from neuralpde_opt.chapter3.deeponet import DeepONet


class Chapter3DeepONetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    @staticmethod
    def controlled_model(hard_boundary=True):
        model = DeepONet(grid_size=5, sensor_stride=3, latent_dim=1,
                         branch_width=1, trunk_width=1, depth=1,
                         hard_boundary=hard_boundary).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # y(q) = tanh(a[0,0] + f[0,0]) * boundary_factor(q).
            model.branch[0].weight[0, 0] = 1
            model.branch[0].weight[0, model.num_sensors] = 1
            model.branch[-1].weight[0, 0] = 1
            model.trunk[-1].bias[0] = 1
        return model

    def test_sensor_layout_includes_endpoint_and_records_order(self):
        model = DeepONet(grid_size=8, sensor_stride=3)
        axis = torch.tensor([0, 3, 6, 7])
        expected = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
        torch.testing.assert_close(model.sensor_axis_indices, axis)
        torch.testing.assert_close(model.sensor_indices, expected)
        torch.testing.assert_close(model.sensor_points, expected.double() / 7)
        self.assertEqual(model.num_sensors, 16)
        sparse = DeepONet(grid_size=8, sensor_stride=100)
        self.assertEqual(sparse.sensor_axis_indices.tolist(), [0, 7])

    def test_shared_batched_queries_and_full_grid_are_consistent(self):
        model = DeepONet(grid_size=7, sensor_stride=2, latent_dim=5,
                         branch_width=8, trunk_width=9).double()
        a, f = torch.randn(2, 7, 7, dtype=torch.float64), torch.randn(2, 7, 7, dtype=torch.float64)
        full = model(a, f)
        flat = model(a, f, model.grid_points)
        self.assertEqual(full.shape, (2, 7, 7))
        torch.testing.assert_close(full.flatten(1), flat)
        shared = torch.tensor([[.2, .3], [.5, .7], [.8, .4]], dtype=torch.float64)
        batched = torch.stack((shared, shared.flip(0)))
        prediction = model(a, f, batched)
        torch.testing.assert_close(prediction[0], model(a, f, shared)[0])
        torch.testing.assert_close(prediction[1], model(a, f, shared.flip(0))[1])
        torch.testing.assert_close(model(a, f, shared[:0]), torch.empty(2, 0, dtype=torch.float64))

    def test_boundary_factor_enforces_edges_for_all_query_layouts(self):
        model = self.controlled_model()
        a = torch.ones(2, 5, 5, dtype=torch.float64)
        f = a.clone()
        edges = torch.tensor([[0., .3], [1., .2], [.8, 0.], [.3, 1.],
                              [0., 0.], [1., 1.]], dtype=torch.float64)
        self.assertEqual(torch.count_nonzero(model(a, f, edges)).item(), 0)
        self.assertEqual(torch.count_nonzero(model(a, f, edges.expand(2, -1, -1))).item(), 0)
        output = model(a, f)
        self.assertEqual(torch.count_nonzero(output[:, [0, -1], :]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[:, :, [0, -1]]).item(), 0)
        torch.testing.assert_close(output[:, 2, 2], torch.tanh(torch.tensor([2., 2.], dtype=torch.float64)))
        unbounded = self.controlled_model(hard_boundary=False)
        self.assertGreater(torch.abs(unbounded(a, f, edges)).min().item(), .9)

    def test_joint_nonlinear_branch_has_coefficient_source_interaction(self):
        model = self.controlled_model()
        zero = torch.zeros(1, 5, 5, dtype=torch.float64)
        one = torch.ones_like(zero)
        query = torch.tensor([[.5, .5]], dtype=torch.float64)
        interaction = model(one, one, query) - model(one, zero, query) - model(zero, one, query) + model(zero, zero, query)
        self.assertGreater(interaction.abs().item(), .5)

    def test_fixed_sensors_ignore_unobserved_values(self):
        model = self.controlled_model()
        a = torch.zeros(1, 5, 5, dtype=torch.float64)
        f = a.clone()
        query = torch.tensor([[.5, .5]], dtype=torch.float64)
        baseline = model(a, f, query)
        a[:, 1, 1] = 123
        f[:, 2, 2] = -456
        torch.testing.assert_close(model(a, f, query), baseline)
        a[:, 0, 0] = 1
        self.assertGreater((model(a, f, query) - baseline).abs().item(), .7)

    def test_float64_field_and_query_gradients(self):
        model = DeepONet(grid_size=5, sensor_stride=2, latent_dim=7,
                         branch_width=8, trunk_width=9).double()
        a = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        query = torch.tensor([[.2, .4], [.6, .7]], dtype=torch.float64, requires_grad=True)
        prediction = model(a, f, query)
        self.assertEqual(prediction.dtype, torch.float64)
        prediction.square().sum().backward()
        for tensor in (a, f, query):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        # Unobserved input locations cannot receive a branch gradient.
        self.assertEqual(a.grad[:, 1, 1].abs().sum().item(), 0)

    def test_chunked_predictions_and_all_gradients_match(self):
        model = DeepONet(grid_size=5, sensor_stride=2, latent_dim=5,
                         branch_width=7, trunk_width=6).double()
        a = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        queries = torch.rand(2, 7, 2, dtype=torch.float64, requires_grad=True)
        variables = (a, f, queries, *model.parameters())
        baseline = model(a, f, queries, query_chunk_size=None)
        baseline_grads = torch.autograd.grad(baseline.square().sum(), variables)
        chunked = model(a, f, queries, query_chunk_size=2)
        chunked_grads = torch.autograd.grad(chunked.square().sum(), variables)
        torch.testing.assert_close(chunked, baseline)
        for actual, expected in zip(chunked_grads, baseline_grads):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        order = torch.tensor([3, 0, 5, 1, 6, 2, 4])
        torch.testing.assert_close(model(a, f, queries[:, order], query_chunk_size=3),
                                   baseline[:, order])

    def test_explicit_encoding_is_reusable_and_never_cached_on_module(self):
        model = DeepONet(grid_size=5, latent_dim=3, branch_width=5, trunk_width=5).double()
        a, f = torch.randn(2, 5, 5, dtype=torch.float64), torch.randn(2, 5, 5, dtype=torch.float64)
        queries = torch.rand(2, 6, 2, dtype=torch.float64)
        calls = []
        state_keys, attributes = set(model.state_dict()), set(model.__dict__)
        hook = model.branch.register_forward_hook(lambda module, args, output: calls.append(1))
        try:
            context = model.encode_inputs(a, f)
            first = model.decode_queries(context, queries[:, :2], query_chunk_size=1)
            second = model.decode_queries(context, queries[:, 2:], query_chunk_size=2)
        finally:
            hook.remove()
        self.assertEqual(calls, [1])
        torch.testing.assert_close(torch.cat((first, second), -1), model(a, f, queries))
        self.assertEqual(set(model.state_dict()), state_keys)
        self.assertEqual(set(model.__dict__), attributes)
        (first.square().sum() + second.square().sum()).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertEqual(model.decode_queries(context, torch.empty(0, 2, dtype=torch.float64)).shape, (2, 0))
        self.assertEqual(model.decode_queries(context, torch.empty(2, 0, 2, dtype=torch.float64)).shape, (2, 0))
        with self.assertRaisesRegex(ValueError, "this model"):
            model.decode_queries({})
        other = DeepONet(grid_size=5, latent_dim=3, branch_width=5, trunk_width=5).double()
        with self.assertRaisesRegex(ValueError, "this model"):
            other.decode_queries(context, queries)

    def test_representation_metadata_and_legacy_state_keys(self):
        model = DeepONet(grid_size=8, sensor_stride=3, latent_dim=3,
                         branch_width=5, trunk_width=5, depth=1)
        metadata = json.loads(json.dumps(model.representation_metadata(), allow_nan=False))
        self.assertTrue(model.supports_queries)
        self.assertEqual(metadata["input_grid"]["shape"], [8, 8])
        self.assertEqual(metadata["sensors"]["axis_indices"], [0, 3, 6, 7])
        self.assertEqual(metadata["sensors"]["count"], 16)
        self.assertTrue(metadata["query"]["independent_queries"])
        self.assertIn("unobserved", metadata["finite_input_limit"])
        expected = {"bias", "sensor_axis_indices", "sensor_indices", "sensor_points", "grid_points"}
        for network in ("branch", "trunk"):
            for layer in (0, 2):
                expected.update({f"{network}.{layer}.weight", f"{network}.{layer}.bias"})
        self.assertEqual(set(model.state_dict()), expected)
        restored = DeepONet(8, 3, 3, 5, 5, 1)
        restored.load_state_dict(model.state_dict(), strict=True)
        a, f = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
        torch.testing.assert_close(restored(a, f), model(a, f))

    def test_finite_values_supported_dtypes_and_explicit_query_cast(self):
        model = DeepONet(grid_size=5, latent_dim=4, branch_width=5, trunk_width=5)
        a = torch.zeros(2, 5, 5)
        for value in (float("nan"), float("inf"), -float("inf")):
            invalid = a.clone()
            invalid[:, 1, 1] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                model(invalid, a)
            with self.assertRaisesRegex(ValueError, "finite"):
                model(a, invalid)
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaisesRegex(TypeError, "float32 or float64"):
                model(a.to(dtype), a.to(dtype))
            with self.assertRaisesRegex(TypeError, "float32 or float64"):
                model(a, a, torch.zeros(1, 2, dtype=dtype))
        with self.assertRaisesRegex(ValueError, "same dtype and device"):
            model(a.double(), a.double())
        query = torch.tensor([[.4, .6]], dtype=torch.float64, requires_grad=True)
        output = model(a, torch.ones_like(a), query)
        self.assertEqual(output.dtype, torch.float32)
        output.sum().backward()
        self.assertEqual(query.grad.dtype, torch.float64)
        self.assertTrue(torch.isfinite(query.grad).all())
        with self.assertRaises(ValueError):
            model(a, a, query_chunk_size=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_two_actual_optimization_steps_full_65_grid(self):
        model = DeepONet(grid_size=65, latent_dim=32, branch_width=32,
                         trunk_width=32, depth=2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        a = torch.ones(2, 65, 65, device="cuda") + .1 * torch.randn(2, 65, 65, device="cuda")
        f = torch.randn_like(a)
        axis = torch.linspace(0, 1, 65, device="cuda")
        target = (torch.sin(torch.pi * axis)[:, None] * torch.sin(torch.pi * axis)[None, :]).expand(2, -1, -1)
        initial = model.branch[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(a, f, query_chunk_size=1024)
            loss = (prediction - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            optimizer.step()
        self.assertGreater((model.branch[0].weight - initial).abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(prediction[:, [0, -1], :]).item(), 0)
        with self.assertRaisesRegex(ValueError, "same dtype and device"):
            model(a.cpu(), f.cpu())

    def test_invalid_fields_and_fixed_input_resolution(self):
        model = DeepONet(grid_size=5)
        a = torch.zeros(2, 5, 5)
        invalid = [(a[0], a[0]), (torch.zeros(2, 7, 7), torch.zeros(2, 7, 7)),
                   (a, a[:1]), (a, a.double()), (a[:0], a[:0])]
        for coefficient, source in invalid:
            with self.subTest(shape=coefficient.shape), self.assertRaises(ValueError):
                model(coefficient, source)
        with self.assertRaises(TypeError):
            model(a.long(), a.long())
        with self.assertRaises(TypeError):
            model([], a)

    def test_invalid_queries_and_hyperparameters(self):
        model = DeepONet(grid_size=5)
        a = torch.zeros(2, 5, 5)
        for points in (torch.zeros(2), torch.zeros(2, 3), torch.zeros(1, 3, 2),
                       torch.tensor([[1.1, .5]]), torch.tensor([[float("nan"), .5]])):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                model(a, a, points)
        with self.assertRaises(TypeError):
            model(a, a, torch.tensor([[0, 1]]))
        with self.assertRaises(TypeError):
            model(a, a, [[0., 1.]])
        for arguments in ({"grid_size": 1}, {"sensor_stride": 0}, {"latent_dim": -1},
                          {"branch_width": 0}, {"trunk_width": True}, {"depth": 1.5},
                          {"hard_boundary": "yes"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                DeepONet(**arguments)


if __name__ == "__main__":
    unittest.main()
