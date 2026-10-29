import copy
import json
import unittest
from pathlib import Path
import numpy as np
from neuralpde_opt.chapter3.fields import FieldPair, draw_parameters
from neuralpde_opt.chapter3.reference import (Manufactured, analytic_errors,
                                             evaluate_p1, solve_fields)


class Chapter3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parents[1]/'configs/ch03_pilot.json').read_text())

    def test_append_stability_split_separation_and_arbitrary_rank(self):
        c = copy.deepcopy(self.config)
        c['fields']['a']['modes'] = 5
        c['fields']['f']['modes'] = 11
        first = draw_parameters(c, 0, 3)
        c['splits']['train'] += 1
        self.assertTrue(np.array_equal(first['a_coefficients'], draw_parameters(c, 0, 3)['a_coefficients']))
        self.assertEqual(first['f_coefficients'].shape, (11, 11))
        self.assertFalse(np.array_equal(first['f_coefficients'], draw_parameters(c, 1, 3)['f_coefficients']))
        fields = FieldPair(first, c)
        p = np.random.default_rng(81).random((43, 2))
        self.assertTrue(np.all((fields.coefficient(p) >= .5) & (fields.coefficient(p) <= 1.5)))

    def test_nested_inputs_and_axes(self):
        fields = FieldPair(draw_parameters(self.config, 1, 2), self.config)
        t = np.linspace(0, 1, 33)
        p = np.stack(np.meshgrid(t, t, indexing='ij'), -1)
        self.assertTrue(np.array_equal(fields.source(p)[::2, ::2], fields.source(p[::2, ::2])))
        # A single orthonormal mode fixes x/y orientation independently.
        params = {'a_coefficients': np.zeros((3, 3)), 'f_coefficients': np.zeros((3, 3))}
        params['f_coefficients'][1, 0] = 1
        source = FieldPair(params, self.config).source(p)
        np.testing.assert_allclose(source, np.sqrt(2)*np.cos(np.pi*p[..., 0]), atol=1e-14)

    def test_variable_coefficient_manufactured_convergence(self):
        field = Manufactured(True)
        errors = [analytic_errors(solve_fields(field, n, 5), field.exact, field.gradient)
                  for n in (8, 16, 32)]
        self.assertGreater(np.log2(errors[-2]['relative_l2']/errors[-1]['relative_l2']), 1.8)
        self.assertGreater(np.log2(errors[-2]['relative_h1_seminorm']/errors[-1]['relative_h1_seminorm']), .9)

    def test_source_linearity_and_coefficient_scaling(self):
        class Pair:
            def __init__(self, a, f):
                self.a, self.f = a, f
            def coefficient(self, p):
                return np.full(p.shape[:-1], self.a)
            def source(self, p):
                return self.f(p)
        f1 = lambda p: np.ones(p.shape[:-1])
        f2 = lambda p: p[..., 0] + 2*p[..., 1]
        y1 = solve_fields(Pair(1., f1), 12).values
        y2 = solve_fields(Pair(1., f2), 12).values
        mix = solve_fields(Pair(2., lambda p: 3*f1(p)-.7*f2(p)), 12)
        np.testing.assert_allclose(mix.values, (3*y1-.7*y2)/2, atol=1e-14)
        self.assertLess(mix.diagnostics['relative_algebraic_residual'], 1e-12)
        self.assertEqual(mix.diagnostics['boundary_max'], 0.)

    def test_p1_reconstruction_linear_precision(self):
        n = 7
        t = np.linspace(0, 1, n+1)
        x, y = np.meshgrid(t, t, indexing='ij')
        values = (2+3*x-4*y).ravel()
        points = np.vstack((np.random.default_rng(90).random((151, 2)),
                            [[0, 0], [0, 1], [1, 0], [1, 1]]))
        out, grad = evaluate_p1(values, n, points)
        np.testing.assert_allclose(out, 2+3*points[:, 0]-4*points[:, 1], atol=1e-14)
        np.testing.assert_allclose(grad, np.broadcast_to([3., -4.], grad.shape), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
