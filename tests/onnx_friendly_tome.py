import unittest

import torch

from tomesd.merge import bipartite_soft_matching_random2d


class TestOnnxFriendlyToMeOperations(unittest.TestCase):
    def test_component_correctness(self):
        c = 320
        w = h = 64
        r = 0.2
        sx = sy = 2
        x = torch.rand(2, w * h, c)
        m_orig, u_orig = bipartite_soft_matching_random2d(x, w, h, sx, sy, int(w * h * r), no_rand=True)
        m_onnx, u_onnx = bipartite_soft_matching_random2d(x, w, h, sx, sy, int(w * h * r), no_rand=True, onnx_friendly=True)
        torch.testing.assert_close(u_orig(m_orig(x)), u_onnx(m_onnx(x)))


if __name__ == '__main__':
    unittest.main()
