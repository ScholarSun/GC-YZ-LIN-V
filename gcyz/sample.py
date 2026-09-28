"""The sample: the managed set Y (kept well spread with Lagrange polynomials) and the free set Z."""
import numpy as np
from scipy import linalg as sla

from .interpolation_set import InterpolationSet
from .model import SampleFactors


def _orthogonal_step(P, n, delta):
    """A step of length delta orthogonal to every row of P (any direction if P has no rows)."""
    c = np.linalg.svd(P, full_matrices=True)[2][-1] if P.shape[0] else np.eye(n)[0]
    return (c / np.linalg.norm(c)) * delta


class Sample:
    """The sample around the centre ``center`` (whose value is ``fc``). Points are stored relative to the centre.

    Y: up to n points, kept well spread with linear Lagrange polynomials (the only basis implemented).
    Z: the other points, up to ``z_capacity``. When Z is full, a closer point replaces its furthest one.
    """

    def __init__(self, x0, fc, Y_pts, Y_vals, z_capacity, Z_pts, Z_vals):
        self.n = len(x0)
        self.center = x0
        self.fc = fc
        self.z_capacity = int(z_capacity)
        self.Y = InterpolationSet(Y_pts, Y_vals)
        self.Z = InterpolationSet(Z_pts, Z_vals)
        self._factors = SampleFactors()  # LU factorisation reused by the model fits and the admission test
        self.y_rank_deficient = False  # True when Y's points are linearly dependent (set by get_lagrange_coef)

    @property
    def mY(self):
        return len(self.Y)

    @property
    def mZ(self):
        return len(self.Z)

    # -- linear Lagrange polynomials of Y -----------------------------------------
    def lagrange_poisedness(self, L_coefs, delta):
        """Maximize |l_i| over the ball of radius delta for Y's linear Lagrange polynomials l_i. Returns
        (i, maximiser, poisedness value) for the largest one. If Y's points are linearly dependent, returns +inf,
        with the replacement point from _degenerate_repair as the maximizer. If Y has fewer than n points, returns
        +inf with i = None and a point orthogonal to Y, to be added to it."""
        if self.y_rank_deficient or not np.all(np.isfinite(L_coefs)):
            idx, step = self._degenerate_repair(delta)
            return idx, step, np.inf
        if self.mY < self.n:
            # a short Y cannot be poised in the full space
            return None, _orthogonal_step(self.Y.points, self.n, delta), np.inf
        norms = np.linalg.norm(L_coefs, axis=0)
        if not np.all(np.isfinite(norms)):
            # a coefficient above ~1e154 makes the norm overflow: Y is extremely badly poised, so repair it as if
            # its points were dependent
            idx, step = self._degenerate_repair(delta)
            return idx, step, np.inf
        idx = int(np.argmax(norms))
        direction = L_coefs[:, idx]
        lag_step = (direction / np.linalg.norm(direction)) * delta
        return idx, lag_step, float(direction @ lag_step)

    def _degenerate_repair(self, delta):
        """For a linearly dependent Y: returns (idx, step), where step (length delta) is orthogonal to the rest of Y
        and replaces Y[idx]."""
        P = self.Y.points
        finite = np.all(np.isfinite(P), axis=1)
        if not np.all(finite):
            idx = int(np.argmin(finite))  # the first non-finite point
            rest = P[finite]
        else:
            U = np.linalg.svd(P, full_matrices=False)[0]
            idx = int(np.argmax(np.abs(U[:, -1])))  # the largest weight in the smallest singular direction
            rest = np.delete(P, idx, axis=0)
        return idx, _orthogonal_step(rest, self.n, delta)

    def get_lagrange_coef(self):
        """Coefficients of Y's linear Lagrange polynomials, they exist only for a full Y. A short
        Y then counts as not poised (lagrange_poisedness adds a point), and ``y_rank_deficient`` is set for dependent or
        non-finite points (lagrange_poisedness replaces one)."""
        P = self.Y.points
        self.y_rank_deficient = False
        if not np.all(np.isfinite(P)):
            self.y_rank_deficient = True  # a non-finite point: it is replaced
            return np.zeros((P.shape[1], P.shape[0]))
        if P.shape[0] < self.n:
            return np.zeros((self.n, P.shape[0]))  # a short Y
        try:
            return np.linalg.inv(P)
        except np.linalg.LinAlgError:
            self.y_rank_deficient = True  # an exactly singular Y
            return np.zeros((self.n, self.n))

    def lagrange_step_col(self, L_coefs, i, delta):
        """The point in the ball of radius delta where |l_i| is largest"""
        dirv = L_coefs[:, i]
        nrm = np.linalg.norm(dirv)
        return (dirv / nrm) * delta if nrm > 0 else None

    # -- admission test of GC-YZ-V -----------------------------------------------
    def mfn_values_all(self, p):
        """Values at p of the MFN quadratic Lagrange polynomials of the whole sample (Y, Z and the centre). Computed on
        the points divided by the sample radius. Uses pre-built LU factorizations. None if Y is empty or anything is not finite."""
        if len(self.Y) == 0:
            return None
        if self.Z.points.size:
            rows = np.vstack([self.Y.points, self.Z.points, np.zeros((1, self.n))])
        else:
            rows = np.vstack([self.Y.points, np.zeros((1, self.n))])
        n = self.n
        with np.errstate(all='ignore'):
            s = float(np.max(np.linalg.norm(rows, axis=1)))
            if not np.isfinite(s) or s <= 0.0:
                return None
            Xs = rows / s
            q = np.asarray(p, float).ravel() / s
            if not (np.all(np.isfinite(Xs)) and np.all(np.isfinite(q))):
                return None
            phi = np.concatenate([[1.0], q, 0.5 * (Xs @ q) ** 2])
            v = sla.lu_solve(self._factors.lu(Xs), phi, check_finite=False)[1 + n:]
            if not np.all(np.isfinite(v)):
                return None
        return v

    # -- housekeeping ------------------------------------------------------------
    def compact(self):
        """Copy both point sets into fresh arrays. No value changes, but NumPy's row norms can differ in the last
        bit between memory layouts, which can change which point counts as the furthest. This makes runs depend
        on the values only. Do not remove: results would change."""
        for S in (self.Z, self.Y):
            S.points = np.delete(S.points, [], axis=0)
            S.values = np.delete(S.values, [])

    # -- adding points to Z ------------------------------------------------------
    def update_Z(self, kicked_list):
        """Add (point, value) pairs to Z. When Z is full, a new point replaces Z's furthest point if it is closer."""
        for kicked_point, kicked_val in kicked_list:
            xk = self.center + kicked_point
            if np.array_equal(xk, self.center) or any(np.array_equal(self.center + z, xk) for z in self.Z.points):
                continue  # skip the centre and points Z already holds (compared as the objective sees them)
            if self.mZ < self.z_capacity:
                self.Z.add_point(kicked_point, kicked_val)
            else:
                far_idx, far_point = self.Z.get_furthest()
                if far_idx is None:
                    continue  # z_capacity == 0: Z keeps nothing
                if np.linalg.norm(far_point) > np.linalg.norm(kicked_point):
                    self.Z.delete_point(far_idx)
                    self.Z.add_point(kicked_point, kicked_val)
