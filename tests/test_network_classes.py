import unittest

import numpy as np
import torch

from bezier_network.conv1d.conv1d_bezier_network import Conv1dBezierNetwork, conv1DbezierNetwork
from bezier_network.conv2d.conv2d_bezier_network import Conv2dBezierNetwork, conv2dbezierNetwork
from bezier_network.conv3d.conv3d_bezier_network import Conv3dBezierNetwork, conv3dbezierNetwork
from bezier_network.dense.dense_bezier_network import DenseBezierNetwork, densebezierNetwork


class NetworkClassTests(unittest.TestCase):
    def test_dense_network_samples_and_forwards(self):
        control_points = np.asarray([[4.0, 3.0]])
        network = DenseBezierNetwork(
            shape_in=np.asarray([4]),
            shape_out=np.asarray([3]),
            control_points=control_points,
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.sample_bezier(2), np.asarray([[4], [3]]))
        output = network(torch.ones(2, 4))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_dense_network_samples_curved_control_points(self):
        network = DenseBezierNetwork(
            shape_in=np.asarray([4]),
            shape_out=np.asarray([8]),
            control_points=np.asarray([[4.0, 5.2, 8.0]]),
            bezier_samples=3,
            layers=1,
        )

        np.testing.assert_array_equal(network.sample_bezier(3), np.asarray([[4], [6], [8]]))

    def test_conv1d_network_samples_and_forwards(self):
        control_points = np.asarray([[2.0, 3.0], [4.0, 6.0]])
        network = Conv1dBezierNetwork(
            shape_in=np.asarray([2, 4]),
            shape_out=np.asarray([3, 6]),
            control_points=control_points,
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.sample_bezier(2), np.asarray([[2, 4], [3, 6]]))
        output = network(torch.ones(2, 2, 4))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_conv1d_network_contracts_and_traces_shape_path(self):
        network = Conv1dBezierNetwork(
            shape_in=np.asarray([3, 6]),
            shape_out=np.asarray([2, 4]),
            control_points=np.asarray([[3.0, 2.0], [6.0, 4.0]]),
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.tensor_shape(), np.asarray([[3, 6], [2, 4]]))
        output = network(torch.ones(2, 3, 6))
        self.assertEqual(tuple(output.shape), (2, 2, 4))

    def test_conv1d_network_multi_sample_trace(self):
        network = Conv1dBezierNetwork(
            shape_in=np.asarray([2, 4]),
            shape_out=np.asarray([4, 8]),
            control_points=np.asarray([[2.0, 3.0, 4.0], [4.0, 5.2, 8.0]]),
            bezier_samples=3,
            layers=1,
        )

        trace = network.tensor_shape()

        self.assertEqual(tuple(trace[0]), (2, 4))
        self.assertEqual(tuple(trace[-1]), (4, 8))
        self.assertEqual(trace.shape[1], 2)

    def test_conv2d_network_samples_and_forwards(self):
        control_points = np.asarray([[2.0, 3.0], [4.0, 6.0], [4.0, 6.0]])
        network = Conv2dBezierNetwork(
            shape_in=np.asarray([2, 4, 4]),
            shape_out=np.asarray([3, 6, 6]),
            control_points=control_points,
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.sample_bezier(2), np.asarray([[2, 4, 4], [3, 6, 6]]))
        output = network(torch.ones(2, 2, 4, 4))
        self.assertEqual(tuple(output.shape), (2, 3, 6, 6))

    def test_conv2d_network_contracts_and_traces_shape_path(self):
        network = Conv2dBezierNetwork(
            shape_in=np.asarray([3, 6, 6]),
            shape_out=np.asarray([2, 4, 4]),
            control_points=np.asarray([[3.0, 2.0], [6.0, 4.0], [6.0, 4.0]]),
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.tensor_shape(), np.asarray([[3, 6, 6], [2, 4, 4]]))
        output = network(torch.ones(2, 3, 6, 6))
        self.assertEqual(tuple(output.shape), (2, 2, 4, 4))

    def test_conv2d_network_multi_sample_trace(self):
        network = Conv2dBezierNetwork(
            shape_in=np.asarray([2, 4, 4]),
            shape_out=np.asarray([4, 8, 8]),
            control_points=np.asarray([[2.0, 3.0, 4.0], [4.0, 5.2, 8.0], [4.0, 5.2, 8.0]]),
            bezier_samples=3,
            layers=1,
        )

        trace = network.tensor_shape()

        self.assertEqual(tuple(trace[0]), (2, 4, 4))
        self.assertEqual(tuple(trace[-1]), (4, 8, 8))
        self.assertEqual(trace.shape[1], 3)

    def test_conv3d_network_samples_and_forwards(self):
        control_points = np.asarray([[2.0, 3.0], [4.0, 6.0], [4.0, 6.0], [4.0, 6.0]])
        network = Conv3dBezierNetwork(
            shape_in=np.asarray([2, 4, 4, 4]),
            shape_out=np.asarray([3, 6, 6, 6]),
            control_points=control_points,
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.sample_bezier(2), np.asarray([[2, 4, 4, 4], [3, 6, 6, 6]]))
        output = network(torch.ones(2, 2, 4, 4, 4))
        self.assertEqual(tuple(output.shape), (2, 3, 6, 6, 6))

    def test_conv3d_network_contracts_and_traces_shape_path(self):
        network = Conv3dBezierNetwork(
            shape_in=np.asarray([3, 6, 6, 6]),
            shape_out=np.asarray([2, 4, 4, 4]),
            control_points=np.asarray([[3.0, 2.0], [6.0, 4.0], [6.0, 4.0], [6.0, 4.0]]),
            bezier_samples=2,
            layers=0,
        )
        network.eval()

        np.testing.assert_array_equal(network.tensor_shape(), np.asarray([[3, 6, 6, 6], [2, 4, 4, 4]]))
        output = network(torch.ones(2, 3, 6, 6, 6))
        self.assertEqual(tuple(output.shape), (2, 2, 4, 4, 4))

    def test_conv3d_network_multi_sample_trace(self):
        network = Conv3dBezierNetwork(
            shape_in=np.asarray([2, 4, 4, 4]),
            shape_out=np.asarray([4, 8, 8, 8]),
            control_points=np.asarray(
                [[2.0, 3.0, 4.0], [4.0, 5.2, 8.0], [4.0, 5.2, 8.0], [4.0, 5.2, 8.0]]
            ),
            bezier_samples=3,
            layers=1,
        )

        trace = network.tensor_shape()

        self.assertEqual(tuple(trace[0]), (2, 4, 4, 4))
        self.assertEqual(tuple(trace[-1]), (4, 8, 8, 8))
        self.assertEqual(trace.shape[1], 4)

    def test_legacy_aliases_still_point_to_modern_classes(self):
        self.assertIs(densebezierNetwork, DenseBezierNetwork)
        self.assertIs(conv1DbezierNetwork, Conv1dBezierNetwork)
        self.assertIs(conv2dbezierNetwork, Conv2dBezierNetwork)
        self.assertIs(conv3dbezierNetwork, Conv3dBezierNetwork)


if __name__ == "__main__":
    unittest.main()
