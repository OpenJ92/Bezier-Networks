from math import comb

import numpy as np


class Bezier:
    """Tensor-valued Bezier map over one or more control-point axes."""

    def __init__(self, control_points, collapse_axes):
        self.control_points = np.asarray(control_points, dtype=float)
        self.collapse_axes = tuple(collapse_axes)
        self._validate_collapse_axes()

        self._binoms = tuple(
            np.asarray(
                [comb(self.control_points.shape[axis] - 1, i)
                 for i in range(self.control_points.shape[axis])],
                dtype=float,
            )
            for axis in self.collapse_axes
        )

    def __call__(self, ts):
        return self.evaluate(ts)

    def _validate_collapse_axes(self):
        if not self.collapse_axes:
            raise ValueError("Bezier requires at least one collapse axis")

        ndim = self.control_points.ndim
        if len(set(self.collapse_axes)) != len(self.collapse_axes):
            raise ValueError("Bezier collapse axes must be unique")

        for axis in self.collapse_axes:
            if axis < 0 or axis >= ndim:
                raise ValueError(
                    f"Bezier collapse axis {axis} is out of bounds for {ndim} dimensions"
                )
            if self.control_points.shape[axis] < 2:
                raise ValueError("Bezier collapse axes must have at least two points")

    @staticmethod
    def _basis(n, t, binoms):
        i = np.arange(n + 1, dtype=float)
        return binoms * ((1.0 - t) ** (n - i)) * (t ** i)

    @staticmethod
    def _collapse_axis(points, axis, basis):
        shape = [1] * points.ndim
        shape[axis] = points.shape[axis]
        return np.sum(points * basis.reshape(shape), axis=axis)

    def evaluate(self, ts):
        ts = np.asarray(ts, dtype=float)
        if ts.ndim == 0:
            ts = ts.reshape(1)
        if len(ts) != len(self.collapse_axes):
            raise ValueError("Bezier expected one parameter per collapse axis")

        points = self.control_points
        axes = list(self.collapse_axes)

        for i, t in enumerate(ts):
            axis = axes[i]
            n = points.shape[axis] - 1
            basis = self._basis(n, t, self._binoms[i])
            points = self._collapse_axis(points, axis, basis)
            axes = [a if a < axis else a - 1 for a in axes]

        return points

    @classmethod
    def identity(cls, k):
        shape = (2,) * k
        grid = np.indices(shape, dtype=float)
        control_points = np.moveaxis(grid, 0, -1)
        collapse_axes = tuple(range(k))
        return cls(control_points, collapse_axes)

    @classmethod
    def ID(cls, k):
        return cls.identity(k)


class bezierCurve:
    """
    Backward-compatible architectural shape curve.

    The historical API stores control points with shape
    ``(tensor_dimension, number_of_control_points)`` and returns integer tensor
    shapes. The new core keeps continuous Bezier evaluation separate from this
    rounding policy.
    """

    def __init__(self, shape_in, shape_out, control_points):
        self.shape_in_ = np.asarray(shape_in)
        self.shape_out_ = np.asarray(shape_out)
        self.control_points = np.asarray(control_points, dtype=float)
        self.bezier = Bezier(self.control_points, collapse_axes=(1,))

    def __call__(self, t):
        return np.ceil(self.evaluate(t)).astype("int")

    def evaluate(self, t):
        return self.bezier(t).reshape(self.shape_in_.shape[0], 1)


class ShapeBezierCurve:
    """Bezier curve specialized for tensor-shape paths."""

    def __init__(self, shape_in, shape_out, control_points, quantizer=np.ceil):
        self.shape_in = np.asarray(shape_in, dtype=int)
        self.shape_out = np.asarray(shape_out, dtype=int)
        self.control_points = np.asarray(control_points, dtype=float)
        self.quantizer = quantizer
        self.curve = Bezier(self.control_points, collapse_axes=(1,))

        if self.control_points.ndim != 2:
            raise ValueError("shape control points must be a 2D array")
        if self.control_points.shape[0] != self.shape_in.shape[0]:
            raise ValueError("control point dimension must match shape dimension")

    @classmethod
    def linear(cls, shape_in, shape_out, quantizer=np.rint):
        shape_in = np.asarray(shape_in, dtype=int)
        shape_out = np.asarray(shape_out, dtype=int)
        control_points = np.stack([shape_in, shape_out], axis=1)
        return cls(shape_in, shape_out, control_points, quantizer=quantizer)

    def evaluate(self, t):
        return self.curve(t)

    def quantize(self, values):
        return self.quantizer(values).astype(int)

    def shape_at(self, t):
        return self.quantize(self.evaluate(t))

    def sample(self, num_samples):
        if num_samples < 2:
            raise ValueError("a shape path requires at least two samples")
        return np.asarray(
            [self.shape_at(t) for t in np.linspace(0, 1, num_samples)],
            dtype=int,
        )
