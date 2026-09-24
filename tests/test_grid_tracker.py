import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grid_tracker import GridTracker


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([(80 + 29*x, 70 + 28*y) for y in range(7) for x in range(9)], dtype=float)
        self.tracker = GridTracker(7, 9)

    def test_stationary_and_motion_with_shuffled_detections(self):
        self.tracker.update(self.points[::-1])
        flow = self.tracker.update(self.points)
        np.testing.assert_allclose(flow[2], flow[0])
        np.testing.assert_allclose(flow[3], flow[1])
        moved = self.points + [3, -2]
        flow = self.tracker.update(moved[np.random.default_rng(1).permutation(63)])
        np.testing.assert_allclose(flow[2] - flow[0], 3)
        np.testing.assert_allclose(flow[3] - flow[1], -2)
        self.assertTrue((flow[4] >= 0).all())

    def test_missing_extra_and_recovery(self):
        self.tracker.update(self.points)
        flow = self.tracker.update(np.vstack([self.points[1:], [0, 0]]))
        self.assertEqual((flow[4] >= 0).sum(), 62)
        self.assertEqual(flow[4][0, 0], -1)
        self.assertTrue((self.tracker.update(self.points)[4] >= 0).all())

    def test_ambiguous_and_large_jump_rejected(self):
        self.tracker.update(self.points)
        flow = self.tracker.update(np.vstack([self.points, self.points[0] + 1]))
        self.assertEqual(flow[4][0, 0], -1)
        flow = self.tracker.update(self.points + [12, 12])
        self.assertTrue((flow[4] == -1).all())

    def test_initialization_and_reset(self):
        self.assertIsNone(self.tracker.update(self.points[:-1]))
        bad = self.points.copy()
        bad[-1] = [0, 0]
        self.assertIsNone(self.tracker.update(bad))
        self.tracker.update(self.points)
        self.tracker.reset()
        flow = self.tracker.update(self.points + 3)
        np.testing.assert_allclose(flow[0], flow[2])


if __name__ == '__main__':
    unittest.main()
