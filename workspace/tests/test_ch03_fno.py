"""Behavioral checks for the small Chapter 3 Fourier operator baseline."""

import math
import json
import unittest

import torch

from neuralpde_opt.chapter3.fno import FNO2d, SpectralConv2d


class FourierOperatorTests(unittest.TestCase):
    def test_representation_contract_and_nonfinite_rejection(self):
        model = FNO2d(modes=2, width=4, layers=1)
        self.assertFalse(model.supports_queries)
        metadata = model.representation_metadata()
        json.dumps(metadata, allow_nan=False)
        a = torch.ones(1, 5, 5)
        for invalid in (a*float('nan'), a*float('inf'), a[:0]):
            with self.assertRaises(ValueError):
                model(invalid, invalid)
        with self.assertRaises(ValueError):
            FNO2d(hard_boundary='yes')

    def test_closed_grid_axes(self):
        a = torch.zeros(1, 5, 7, dtype=torch.float64)
        x, y = FNO2d._coordinates(a)
        torch.testing.assert_close(x[:, 0], torch.linspace(0, 1, 5, dtype=a.dtype))
        torch.testing.assert_close(y[0, :], torch.linspace(0, 1, 7, dtype=a.dtype))
        self.assertTrue(torch.equal(x[:, 0], x[:, -1]))
        self.assertTrue(torch.equal(y[0, :], y[-1, :]))

    def test_one_model_runs_on_odd_even_and_finer_grids(self):
        torch.manual_seed(271)
        for dtype in (torch.float32, torch.float64):
            model = FNO2d(modes=8, width=4, layers=2).to(dtype=dtype)
            for n in (32, 33, 65):
                a = torch.rand(1, n, n, dtype=dtype)
                result = model(a, torch.randn_like(a))
                self.assertEqual(result.shape, a.shape)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(bool(torch.isfinite(result).all()))
                boundary = torch.cat((result[:, 0], result[:, -1],
                                      result[:, :, 0], result[:, :, -1]), dim=-1)
                self.assertEqual(torch.count_nonzero(boundary).item(), 0)
                self.assertGreater(torch.count_nonzero(result[:, 1:-1, 1:-1]).item(), 0)

    def test_spectral_identity_when_all_frequencies_retained(self):
        # Oversized modes must clip without overlapping the two x blocks.
        for height, width in ((2, 2), (4, 6), (5, 7), (8, 9)):
            layer = SpectralConv2d(1, 1, modes=16).double()
            with torch.no_grad():
                for weight in (layer.weight_positive, layer.weight_negative):
                    weight.zero_()
                    weight[..., 0] = 1
            values = torch.randn(2, 1, height, width, dtype=torch.float64)
            torch.testing.assert_close(layer(values), values, rtol=1e-12, atol=1e-12)

    def test_signed_frequency_blocks_and_high_mode_rejection(self):
        height, width = 9, 11
        x = torch.arange(height, dtype=torch.float64)[:, None] / height
        y = torch.arange(width, dtype=torch.float64)[None, :] / width
        layer = SpectralConv2d(1, 1, modes=2).double()
        with torch.no_grad():
            layer.weight_positive.zero_()
            layer.weight_negative.zero_()
            layer.weight_positive[..., 0] = 2
            layer.weight_negative[..., 0] = 3
        positive = torch.cos(2 * math.pi * (x + y))[None, None]
        negative = torch.cos(2 * math.pi * (-x + y))[None, None]
        high = torch.cos(2 * math.pi * (4 * x + y))[None, None]
        torch.testing.assert_close(layer(positive), 2 * positive, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(layer(negative), 3 * negative, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(layer(high), torch.zeros_like(high), rtol=0, atol=1e-12)

    def test_real_parameter_pairs_follow_double_precision(self):
        layer = SpectralConv2d(1, 2, modes=2).double()
        values = torch.randn(1, 1, 5, 7, dtype=torch.float64, requires_grad=True)
        self.assertEqual(layer.weight_positive.dtype, torch.float64)
        self.assertEqual(layer(values).dtype, torch.float64)
        self.assertTrue(torch.autograd.gradcheck(layer, (values,), eps=1e-6,
                                                atol=1e-5, rtol=1e-3, fast_mode=True))

    def test_clipping_preserves_negative_frequency_parameter_identity(self):
        layer = SpectralConv2d(1, 1, modes=8).double()
        with torch.no_grad():
            layer.weight_positive.zero_()
            layer.weight_negative.zero_()
            layer.weight_negative[:, :, -1, 1, 0] = 3
        for height, width in ((4, 7), (9, 11), (32, 33)):
            x = torch.arange(height, dtype=torch.float64)[:, None] / height
            y = torch.arange(width, dtype=torch.float64)[None, :] / width
            wave = torch.cos(2 * math.pi * (-x + y))[None, None]
            torch.testing.assert_close(layer(wave), 3 * wave, rtol=1e-12, atol=1e-12)

    def test_full_model_backpropagates_to_both_inputs_and_spectral_blocks(self):
        torch.manual_seed(182)
        model = FNO2d(modes=3, width=4, layers=2).double()
        a = torch.rand(2, 7, 9, dtype=torch.float64, requires_grad=True)
        f = torch.randn(2, 7, 9, dtype=torch.float64, requires_grad=True)
        model(a, f).square().mean().backward()
        for gradient in (a.grad, f.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(gradient.abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_optional_boundary_and_validation(self):
        model = FNO2d(modes=2, width=2, layers=1, hard_boundary=False)
        with torch.no_grad():
            model.project[-1].weight.zero_()
            model.project[-1].bias.fill_(1)
        a = torch.zeros(1, 3, 4)
        torch.testing.assert_close(model(a, a), torch.ones_like(a))
        with self.assertRaises(ValueError):
            model(a, a[:, :, :-1])
        with self.assertRaises(ValueError):
            model(a[:, :1], a[:, :1])
        with self.assertRaises(TypeError):
            model(a.long(), a.long())
        with self.assertRaises(ValueError):
            model(a.double(), a.double())
        for kwargs in ({"modes": 0}, {"width": -1}, {"layers": 0}):
            with self.assertRaises(ValueError):
                FNO2d(**kwargs)


if __name__ == "__main__":
    unittest.main()
