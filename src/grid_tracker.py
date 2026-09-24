"""Conservative local tracking for a roughly horizontal marker grid."""
import numpy as np


class GridTracker:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.reset()

    def reset(self):
        self.reference = None
        self.previous = None
        self.gate = None

    def update(self, centers):
        points = np.asarray(centers, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Marker coordinates must be finite")
        count = self.rows * self.cols
        if self.reference is None:
            if len(points) != count:
                return None
            grid = points[np.argsort(points[:, 1])].reshape(self.rows, self.cols, 2)
            grid = np.stack([row[np.argsort(row[:, 0])] for row in grid])
            dx = np.median(np.diff(grid[:, :, 0], axis=1))
            dy = np.median(np.diff(grid[:, :, 1], axis=0))
            if min(dx, dy) <= 0:
                return None
            # Reject malformed grids and extra blobs replacing missing markers.
            ideal = np.empty_like(grid)
            ideal[:, :, 0] = np.median(grid[:, :, 0], axis=0)[None, :]
            ideal[:, :, 1] = np.median(grid[:, :, 1], axis=1)[:, None]
            if np.max(np.linalg.norm(grid - ideal, axis=2)) > 0.3 * min(dx, dy):
                return None
            if (np.any(np.abs(np.diff(grid[:, :, 0], axis=1) - dx) > 0.4 * dx)
                    or np.any(np.abs(np.diff(grid[:, :, 1], axis=0) - dy) > 0.4 * dy)):
                return None
            self.reference = grid.reshape(-1, 2).copy()
            self.previous = self.reference.copy()
            self.gate = 0.4 * min(dx, dy)

        current = self.previous.copy()
        occupied = np.full(count, -1, dtype=int)
        if len(points):
            distances = np.linalg.norm(self.previous[:, None, :] - points[None, :, :], axis=2)
            candidates = distances < self.gate
            # Accept only an unambiguous one-to-one association inside the gate.
            for i in range(count):
                js = np.flatnonzero(candidates[i])
                if len(js) == 1 and candidates[:, js[0]].sum() == 1:
                    j = js[0]
                    current[i] = points[j]
                    occupied[i] = j
        self.previous = current
        shape = (self.rows, self.cols)
        return tuple(a.reshape(shape) for a in (
            self.reference[:, 0], self.reference[:, 1],
            current[:, 0], current[:, 1], occupied,
        ))
