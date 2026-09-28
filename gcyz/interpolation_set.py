"""Point sets of the sample (Y and Z) with their function values."""
import numpy as np


class InterpolationSet:
    """
    A thin container for an interpolation set:
    - ``points``: array of points relative to the trust-region centre, shape (m, d)
    - ``values``: array of function values, shape (m,)

    Provides small utilities for adding/removing points, shifting, and
    selecting the furthest point from a reference.
    """

    def __init__(self, points, values):
        self.points = np.asarray(points)
        m, self.n = self.points.shape
        self.values = np.asarray(values)
        if len(self.values) != m:
            raise ValueError(
                f"InterpolationSet: points has {m} rows but values has {len(self.values)} entries."
            )

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        """(point, value) of entry ``idx``, as copies."""
        return (self.points[idx].copy(), self.values[idx].copy())

    def add_point(self, point, value=np.nan):
        """Append ``point`` (a flat or row array of length n) with its function value. The point is copied:
        the set never aliases the caller's array."""
        point = np.array(point, dtype=float)
        if point.shape not in ((self.n,), (1, self.n)):
            raise ValueError(f"InterpolationSet: a point of shape {point.shape} for a set in dimension {self.n}")
        point = point.reshape(1, -1)
        if self.points.size == 0:
            self.points = point
            self.values = np.array([value])
        else:
            self.points = np.vstack([self.points, point])
            self.values = np.append(self.values, value)

    def delete_point(self, indices):
        """Remove the entries at ``indices`` (an index or a list of indices)."""
        self.points = np.delete(self.points, indices, axis=0)
        self.values = np.delete(self.values, indices)

    def get_furthest(self, s=None):
        """(index, point) of the point furthest from ``s`` (the origin, i.e. the centre, by default);
        (None, None) if the set is empty. Ties go to the first such point."""
        if self.points.size == 0:
            return None, None
        if s is None:
            s = np.zeros(self.points.shape[1])
        s = np.asarray(s)
        diffs = np.linalg.norm(self.points - s, axis=1)
        idx = int(np.argmax(diffs))
        return idx, self.points[idx]

    def append_origin(self, value=np.nan):
        """Append the origin (the current centre) with its value: how the old centre joins a set."""
        self.add_point(np.zeros(self.n), value=value)

    def shift(self, s):
        """Shift all points by subtracting s (broadcast over rows)."""
        if self.points.size != 0:
            self.points = self.points - np.asarray(s)


