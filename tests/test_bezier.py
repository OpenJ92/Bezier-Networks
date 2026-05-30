import unittest

import numpy as np

from bezier_network.bezier.bezier import Bezier, bezierCurve


class BezierCoreTests(unittest.TestCase):
    def test_linear_curve_interpolates_endpoints_and_midpoint(self):
        curve = Bezier(np.asarray([[0.0, 10.0], [4.0, 8.0]]), collapse_axes=(1,))

        np.testing.assert_allclose(curve(0.0), np.asarray([0.0, 4.0]))
        np.testing.assert_allclose(curve(0.5), np.asarray([5.0, 6.0]))
        np.testing.assert_allclose(curve(1.0), np.asarray([10.0, 8.0]))

    def test_quadratic_curve_uses_bernstein_basis(self):
        curve = Bezier(np.asarray([[0.0, 2.0, 4.0]]), collapse_axes=(1,))

        np.testing.assert_allclose(curve(0.25), np.asarray([1.0]))

    def test_identity_constructor_returns_input_parameters(self):
        identity = Bezier.identity(3)

        np.testing.assert_allclose(identity([0.25, 0.5, 0.75]), np.asarray([0.25, 0.5, 0.75]))

    def test_requires_one_parameter_per_collapse_axis(self):
        curve = Bezier(np.asarray([[0.0, 1.0]]), collapse_axes=(1,))

        with self.assertRaises(ValueError):
            curve([0.0, 1.0])


class BezierCurveCompatibilityTests(unittest.TestCase):
    def test_legacy_curve_keeps_column_vector_shape(self):
        curve = bezierCurve(
            np.asarray([0, 4]),
            np.asarray([10, 8]),
            np.asarray([[0.0, 10.0], [4.0, 8.0]]),
        )

        np.testing.assert_allclose(curve.evaluate(0.5), np.asarray([[5.0], [6.0]]))

    def test_legacy_call_returns_ceil_integer_shape(self):
        curve = bezierCurve(
            np.asarray([0, 4]),
            np.asarray([5, 7]),
            np.asarray([[0.0, 2.5, 5.0], [4.0, 5.5, 7.0]]),
        )

        result = curve(0.5)

        self.assertEqual(result.dtype.kind, "i")
        np.testing.assert_array_equal(result, np.asarray([[3], [6]]))


if __name__ == "__main__":
    unittest.main()
