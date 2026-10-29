"""Quadrature, independent-query and differentiation checks for Transformer."""

import json
import unittest

import torch

from neuralpde_opt.chapter3.transformer import (TransformerOperator,
                                               WeightedMultiheadAttention,
                                               weighted_scaled_dot_product_attention)


class Chapter3TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    @staticmethod
    def small_model(**kwargs):
        config = dict(grid_size=5, sensor_stride=3, width=8, heads=2, layers=1)
        config.update(kwargs)
        return TransformerOperator(**config)

    def test_nonuniform_attention_weights_are_normalized_spatial_weights(self):
        query = torch.zeros(2, 3, dtype=torch.float64)
        key = torch.zeros(3, 3, dtype=torch.float64)
        value = torch.tensor([[1.], [3.], [8.]], dtype=torch.float64)
        weights = torch.tensor([.2, .3, .5], dtype=torch.float64)
        output = weighted_scaled_dot_product_attention(query, key, value, weights)
        torch.testing.assert_close(output, torch.full((2, 1), 5.1, dtype=torch.float64))
        scaled = weighted_scaled_dot_product_attention(query, key, value, weights * 13)
        torch.testing.assert_close(output, scaled)
        self.assertGreater((output - value.mean()).abs().min().item(), 1)

    def test_duplicate_token_split_weight_and_source_order_invariance(self):
        query = torch.randn(2, 3, 4, dtype=torch.float64)
        key = torch.randn(2, 4, 4, dtype=torch.float64)
        value = torch.randn(2, 4, 5, dtype=torch.float64)
        weights = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
        baseline = weighted_scaled_dot_product_attention(query, key, value, weights)
        duplicate = torch.tensor([0, 1, 1, 2, 3])
        split_weights = torch.tensor([.1, .1, .1, .3, .4], dtype=torch.float64)
        split = weighted_scaled_dot_product_attention(query, key[:, duplicate],
                                                      value[:, duplicate], split_weights)
        torch.testing.assert_close(baseline, split)
        order = torch.tensor([3, 1, 0, 2])
        shuffled = weighted_scaled_dot_product_attention(query, key[:, order],
                                                         value[:, order], weights[order])
        torch.testing.assert_close(baseline, shuffled)

    def test_multihead_per_batch_weights_match_individual_calls(self):
        module = WeightedMultiheadAttention(8, 2).double()
        query = torch.randn(2, 3, 8, dtype=torch.float64)
        source = torch.randn(2, 4, 8, dtype=torch.float64)
        weights = torch.tensor([[.1, .2, .3, .4], [.4, .1, .2, .3]], dtype=torch.float64)
        batched = module(query, source, weights)
        for batch in range(2):
            individual = module(query[batch:batch + 1], source[batch:batch + 1], weights[batch])
            torch.testing.assert_close(batched[batch:batch + 1], individual)

    def test_sensor_layout_and_tensor_trapezoid_weights(self):
        model = self.small_model(grid_size=8)
        self.assertEqual(model.sensor_axis_indices.tolist(), [0, 3, 6, 7])
        axis_weights = torch.tensor([1.5, 3., 2., .5], dtype=torch.float64) / 7
        torch.testing.assert_close(model.sensor_axis_weights, axis_weights)
        torch.testing.assert_close(model.sensor_weights,
                                   (axis_weights[:, None] * axis_weights[None, :]).flatten())
        self.assertTrue((model.sensor_weights > 0).all())
        self.assertAlmostEqual(model.sensor_weights.sum().item(), 1.)
        for coordinate in range(2):
            self.assertAlmostEqual((model.sensor_weights * model.sensor_points[:, coordinate]).sum().item(), .5)
        self.assertAlmostEqual((model.sensor_weights * model.sensor_points.prod(-1)).sum().item(), .25)
        default = TransformerOperator()
        self.assertEqual(default.num_sensors, 289)
        endpoints = self.small_model(sensor_stride=100)
        self.assertEqual(endpoints.sensor_axis_indices.tolist(), [0, 4])
        torch.testing.assert_close(endpoints.sensor_weights, torch.full((4,), .25, dtype=torch.float64))

    def test_full_shared_batched_chunked_and_reordered_queries(self):
        model = self.small_model().double()
        a = torch.randn(2, 5, 5, dtype=torch.float64)
        f = torch.randn_like(a)
        queries = torch.tensor([[.2, .3], [.4, .7], [.8, .4], [.6, .2]], dtype=torch.float64)
        together = model(a, f, queries, query_chunk_size=None)
        for chunk_size in (1, 2, 3, 99):
            torch.testing.assert_close(model(a, f, queries, query_chunk_size=chunk_size), together)
        order = torch.tensor([2, 0, 3, 1])
        reordered = model(a, f, queries[order], query_chunk_size=3)
        torch.testing.assert_close(reordered, together[:, order])
        batched_queries = torch.stack((queries, queries[order]))
        expected = torch.stack((together[0], together[1, order]))
        torch.testing.assert_close(model(a, f, batched_queries, query_chunk_size=1), expected)
        full = model(a, f, query_chunk_size=4)
        flat = model(a, f, model.grid_points, query_chunk_size=7)
        self.assertEqual(full.shape, (2, 5, 5))
        torch.testing.assert_close(full.flatten(1), flat)
        self.assertEqual(model(a, f, queries[:0]).shape, (2, 0))

    def test_default_grid_uses_sparse_source_and_bounded_query_blocks(self):
        model = TransformerOperator()
        source_shapes, query_shapes = [], []
        source_hook = model.encoder[0].attention.register_forward_pre_hook(
            lambda module, args: source_shapes.append((args[0].shape[1], args[1].shape[1])))
        query_hook = model.decoder[0].attention.register_forward_pre_hook(
            lambda module, args: query_shapes.append((args[0].shape[1], args[1].shape[1])))
        try:
            with torch.no_grad():
                output = model(torch.ones(1, 65, 65), torch.ones(1, 65, 65))
        finally:
            source_hook.remove()
            query_hook.remove()
        self.assertEqual(output.shape, (1, 65, 65))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(source_shapes, [(289, 289)])
        self.assertEqual(query_shapes, [(1024, 289)] * 4 + [(129, 289)])

    def test_exact_boundary_and_optional_unconstrained_output(self):
        model = self.small_model().double()
        a, f = torch.randn(2, 5, 5, dtype=torch.float64), torch.randn(2, 5, 5, dtype=torch.float64)
        edges = torch.tensor([[0., .4], [1., .7], [.4, 0.], [.2, 1.]], dtype=torch.float64)
        self.assertEqual(torch.count_nonzero(model(a, f, edges)).item(), 0)
        grid = model(a, f)
        self.assertEqual(torch.count_nonzero(grid[:, [0, -1], :]).item(), 0)
        self.assertEqual(torch.count_nonzero(grid[:, :, [0, -1]]).item(), 0)
        model.hard_boundary = False
        self.assertGreater(model(a, f, edges).abs().sum().item(), 1e-6)

    def test_field_query_and_parameter_gradients_float32_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = self.small_model().to(dtype=dtype)
                a = torch.randn(2, 5, 5, dtype=dtype, requires_grad=True)
                f = torch.randn(2, 5, 5, dtype=dtype, requires_grad=True)
                queries = torch.tensor([[.2, .3], [.5, .6], [.7, .4]], dtype=dtype, requires_grad=True)
                prediction = model(a, f, queries, query_chunk_size=1)
                self.assertEqual(prediction.dtype, dtype)
                prediction.square().sum().backward()
                for tensor in (a, f, queries):
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(a.grad[:, 1, 1].abs().sum().item(), 0)
                self.assertEqual(f.grad[:, 1, 1].abs().sum().item(), 0)

    def test_unobserved_field_values_do_not_change_prediction(self):
        model = self.small_model().double()
        a = torch.randn(1, 5, 5, dtype=torch.float64)
        f = torch.randn_like(a)
        queries = torch.tensor([[.3, .7], [.5, .5]], dtype=torch.float64)
        baseline = model(a, f, queries)
        a[:, 1, 1] = 1e5
        f[:, 2, 2] = -1e5
        torch.testing.assert_close(model(a, f, queries), baseline)
        a[:, 0, 0] += 1
        self.assertGreater((model(a, f, queries) - baseline).abs().sum().item(), 1e-8)

    def test_explicit_encoding_projects_each_source_once_and_is_not_cached(self):
        model = self.small_model(layers=2).double()
        a = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        queries = torch.rand(2, 7, 2, dtype=torch.float64, requires_grad=True)
        projections, normalizations = [], []
        state_keys = set(model.state_dict())
        attributes = {name: set(module.__dict__) for name, module in model.named_modules()}
        hooks = []
        for index, layer in enumerate(model.decoder):
            hooks.append(layer.attention.key_projection.register_forward_hook(
                lambda module, args, output, index=index: projections.append(index)))
            hooks.append(layer.norm_source.register_forward_hook(
                lambda module, args, output, index=index: normalizations.append(index)))
        try:
            context = model.encode_inputs(a, f)
            first = model.decode_queries(context, queries[:, :3], query_chunk_size=1)
            second = model.decode_queries(context, queries[:, 3:], query_chunk_size=2)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(projections, [0, 1])
        self.assertEqual(normalizations, [0, 1])
        torch.testing.assert_close(torch.cat((first, second), dim=-1), model(a, f, queries))
        self.assertEqual(set(model.state_dict()), state_keys)
        for name, module in model.named_modules():
            self.assertEqual(set(module.__dict__), attributes[name])
        (first.square().sum() + second.square().sum()).backward()
        for tensor in (a, f, queries):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertEqual(model.decode_queries(context, queries[:, :0]).shape, (2, 0))
        self.assertEqual(model.decode_queries(context, queries[0, :0]).shape, (2, 0))
        with self.assertRaisesRegex(ValueError, "this model"):
            model.decode_queries({})
        other = self.small_model(layers=2).double()
        with self.assertRaisesRegex(ValueError, "this model"):
            other.decode_queries(context, queries)

    def test_chunking_preserves_input_query_and_parameter_gradients(self):
        model = self.small_model(layers=2).double()
        a = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        query = torch.rand(2, 7, 2, dtype=torch.float64, requires_grad=True)
        variables = (a, f, query, *model.parameters())
        baseline = model(a, f, query, query_chunk_size=None)
        baseline_grads = torch.autograd.grad(baseline.square().sum(), variables)
        chunked = model(a, f, query, query_chunk_size=2)
        chunked_grads = torch.autograd.grad(chunked.square().sum(), variables)
        torch.testing.assert_close(chunked, baseline)
        for actual, expected in zip(chunked_grads, baseline_grads):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)

    def test_representation_metadata_and_unchanged_checkpoint_structure(self):
        model = self.small_model(grid_size=8)
        metadata = json.loads(json.dumps(model.representation_metadata(), allow_nan=False))
        self.assertTrue(model.supports_queries)
        self.assertEqual(metadata["input_grid"]["shape"], [8, 8])
        self.assertEqual(metadata["sensors"]["axis_indices"], [0, 3, 6, 7])
        self.assertEqual(metadata["sensors"]["count"], 16)
        self.assertFalse(metadata["attention"]["query_self_attention"])
        self.assertIn("unobserved", metadata["finite_input_limit"])
        self.assertIn("Not a GNOT", metadata["reproduction"])
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        for name in ("sensor_axis_indices", "sensor_indices", "sensor_points",
                     "sensor_axis_weights", "sensor_weights", "grid_points",
                     "decoder.0.attention.key_projection.weight",
                     "decoder.0.attention.value_projection.weight"):
            self.assertIn(name, state)
        restored = self.small_model(grid_size=8)
        restored.load_state_dict(state, strict=True)
        a, f = torch.randn(2, 8, 8), torch.randn(2, 8, 8)
        torch.testing.assert_close(restored(a, f), model(a, f))

    def test_finite_fields_supported_dtypes_and_query_cast(self):
        model = self.small_model()
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_two_actual_optimization_steps_full_65_grid(self):
        model = TransformerOperator(grid_size=65, sensor_stride=4, width=16,
                                    heads=2, layers=1).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        a = torch.ones(2, 65, 65, device="cuda") + .1 * torch.randn(2, 65, 65, device="cuda")
        f = torch.randn_like(a)
        axis = torch.linspace(0, 1, 65, device="cuda")
        target = (torch.sin(torch.pi * axis)[:, None] * torch.sin(torch.pi * axis)[None, :]).expand(2, -1, -1)
        initial = model.input_embedding[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(a, f, query_chunk_size=1024)
            loss = (prediction - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            optimizer.step()
        self.assertGreater((model.input_embedding[0].weight - initial).abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(prediction[:, [0, -1], :]).item(), 0)
        with self.assertRaisesRegex(ValueError, "same dtype and device"):
            model(a.cpu(), f.cpu())

    def test_invalid_quadrature_weights_and_attention_shapes(self):
        query, key, value = torch.randn(2, 3), torch.randn(4, 3), torch.randn(4, 2)
        for weights in (torch.ones(3), torch.tensor([1., 0., 1., 1.]),
                        torch.tensor([1., -1., 1., 1.]),
                        torch.tensor([1., float("inf"), 1., 1.]), torch.ones(2, 4)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                weighted_scaled_dot_product_attention(query, key, value, weights)
        with self.assertRaises(ValueError):
            weighted_scaled_dot_product_attention(query, key[:, :2], value, torch.ones(4))
        with self.assertRaises(TypeError):
            weighted_scaled_dot_product_attention(query, key, value, torch.ones(4, dtype=torch.long))
        invalid = query.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            weighted_scaled_dot_product_attention(invalid, key, value, torch.ones(4))

    def test_input_and_configuration_validation(self):
        model = self.small_model()
        a = torch.zeros(2, 5, 5)
        for coefficient, source in ((a[0], a[0]), (a, a[:1]), (a, a.double()),
                                    (torch.zeros(1, 6, 6), torch.zeros(1, 6, 6))):
            with self.assertRaises(ValueError):
                model(coefficient, source)
        with self.assertRaises(TypeError):
            model(a.long(), a.long())
        for queries in (torch.zeros(1, 2, 2), torch.zeros(3, 4),
                        torch.tensor([[.5, -.1]]), torch.tensor([[.5, float("nan")]])):
            with self.assertRaises(ValueError):
                model(a, a, queries)
        with self.assertRaises(TypeError):
            model(a, a, torch.tensor([[0, 1]]))
        with self.assertRaises(ValueError):
            model(a, a, query_chunk_size=0)
        for arguments in ({"grid_size": 1}, {"sensor_stride": 0}, {"heads": 3},
                          {"width": 0}, {"layers": 0}, {"hard_boundary": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                TransformerOperator(**arguments)


if __name__ == "__main__":
    unittest.main()
