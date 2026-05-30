import unittest


class ImportTests(unittest.TestCase):
    def test_core_modules_import(self):
        import bezier_network
        import bezier_network.bezier
        import bezier_network.bezier.bezier
        import bezier_network.bezier.control_points
        import bezier_network.conv1d
        import bezier_network.dense.dense_bezier_network
        import bezier_network.conv1d.conv1d_bezier_network
        import bezier_network.conv2d
        import bezier_network.conv2d.conv2d_bezier_network
        import bezier_network.conv3d
        import bezier_network.conv3d.conv3d_bezier_network

        self.assertTrue(hasattr(bezier_network, "Bezier"))


if __name__ == "__main__":
    unittest.main()
