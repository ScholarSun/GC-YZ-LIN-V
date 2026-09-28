"""Quadratic models: the fresh minimum-Frobenius-norm (MFN) fit, the least-change chain, the hedge that serves the
better of the two, and the Hessian cap."""
import hashlib
import math
import warnings

import numpy as np
from scipy import linalg as sla

from ._hessian_cap import cap_refit_feasible, fit_least_squares_capped, have_cvxpy
from .trust_sub import trust_sub_exact, trust_sub_CG

_EPS = np.finfo(float).eps


def _kkt_matrix(Xs):
    """Matrix of the linear system of the minimum-Frobenius-norm fit on the rows Xs (points relative to the centre,
    scaled by the sample radius; the centre is the last row)."""
    m, n = Xs.shape
    A = np.empty((1 + n + m, 1 + n + m))
    A[:1 + n, :1 + n] = 0
    A[0, 1 + n:] = 1
    A[1 + n:, 0] = 1
    A[1:1 + n, 1 + n:] = Xs.T
    A[1 + n:, 1:1 + n] = Xs
    A[1 + n:, 1 + n:] = np.dot(Xs, Xs.T) ** 2 / 2
    return A


def _interpolates(X, df, g, H, fX):
    """True if the model (g, H) interpolates the whole sample to a relative tolerance. X: the sample points (Y, Z,
    centre) relative to the centre; df: f(x) - f(centre) at them; fX: f(x) at them (sets the tolerance floor)."""
    with np.errstate(over='ignore', invalid='ignore'):
        pred = (X @ g) + 0.5 * np.sum((X @ H) * X, axis=1)
        resid = np.abs(pred - df)
        tol = max(1e-6 * float(np.max(np.abs(df))), 10.0 * _EPS * float(np.max(np.abs(fX))))
        return bool(np.all(np.isfinite(resid))) and float(np.max(resid)) <= tol


def _lstsq(M, b):
    """Least-squares solve, with a backup driver if the default one fails."""
    try:
        return np.linalg.lstsq(M, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sla.lstsq(M, b, lapack_driver='gelsy', check_finite=False)[0]


def _linear_model(samp):
    """(g, H = 0): the linear fit through Y, exact for a full Y of independent points.
    Used when the exact fit fails (fit_MFN) and when the Hessian cap cannot refit (_cap_hessian); H = 0 
    meets any cap."""
    return _lstsq(samp.Y.points, samp.Y.values - samp.fc), np.zeros((samp.n, samp.n))


def _min_frobenius_fit(X, df, fX, H_prev, factors):
    """Fits a minimum-frobenius norm model. The points are scaled by the sample radius s (exact: g = g_s / s, H = H_s / s^2) and the fit is solved by LU;
    `factors` (SampleFactors) holds the factorisation for reuse. Raises LinAlgError on non-finite data or a
    non-finite solution (an exactly singular system)."""
    m, n = X.shape
    s = float(np.max(np.linalg.norm(X, axis=1)))
    if not np.isfinite(s):
        # row norms overflow for finite entries above ~1e154: use the largest entry instead; non-finite data
        # raises below
        s = float(np.max(np.abs(X))) * math.sqrt(n)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    Xs = X / s
    ss = s * s
    # if s * s overflows or underflows (s outside ~1e-154..1e154), divide by s twice instead; otherwise use s * s,
    # so ordinary runs keep their exact rounding
    two = not (0.0 < ss < np.inf)
    by_ss = (lambda a: (a / s) / s) if two else (lambda a: a / ss)
    rhs = df - 0.5 * np.sum((Xs @ ((H_prev * s) * s if two else H_prev * ss)) * Xs, axis=1)
    if not (np.all(np.isfinite(Xs)) and np.all(np.isfinite(rhs))):
        raise np.linalg.LinAlgError("non-finite interpolation data")
    b = np.zeros(1 + n + m)
    b[-m:] = rhs
    lam = sla.lu_solve(factors.lu(Xs), b, check_finite=False)
    g, H = lam[1:1 + n] / s, H_prev + by_ss(np.dot(Xs.T * lam[-m:], Xs))
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        raise np.linalg.LinAlgError("singular interpolation system")
    return g, H


class SampleFactors:
    """Cache of the LU factorisation of one sample's fit system, shared by every exact solve on it in an iteration"""

    def __init__(self):
        self._key = None
        self._lu = None

    def lu(self, Xs):
        """(lu, piv): LU factorisation of the fit system of Xs."""
        key = (Xs.shape, hashlib.blake2b(np.ascontiguousarray(Xs).tobytes(), digest_size=16).digest())
        if key != self._key:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sla.LinAlgWarning)
                self._key, self._lu = key, sla.lu_factor(_kkt_matrix(Xs), check_finite=False)
        return self._lu


class Model:
    """Quadratic model ``m(s) = fc + g.s + 0.5 s'Hs`` on the current sample.

    ``kind``:
      'mfn'   -- minimum-Frobenius-norm fit, from scratch each iteration.
      'chain' -- least-change fit: min ||H - H_anchor||_F, where the anchor is the last accepted chain fit
                 (started from an MFN fit).
      'hedge' -- both are fitted each iteration; the one that has recently predicted the function changes
                 better is used.
    """

    def __init__(self, n, options):
        self.n = n
        self.kind = str(options['model'])  # checked by options.validate_options
        self.g = None
        self.H = None
        self.delta = 0.0  # trust-region radius (set by the solver)

        # least-change chain
        self.scale_reseed = float(options['mcfn_scale_reseed'] or 0.0)
        self.H_anchor = None
        self.seed_delta = None
        self.mcfn_updates = 0     # accepted chain fits
        self.mcfn_fallbacks = 0   # rejected chain fits, replaced by the MFN fit
        self.mfn_noninterp = 0    # fresh MFN fits that do not interpolate (least-squares fits)
        self.mfn_linear_fallbacks = 0  # fresh MFN fits that failed and were replaced by the linear model
        self.scale_reseeds = 0

        # hedge
        self.hedge_beta = float(options['hedge_beta'])
        self.hedge_hyst = float(options['hedge_hysteresis'])
        self.hedge_warmup = int(options['hedge_warmup'])
        self.hedge_chain_max_err = options['hedge_chain_max_err']
        self._hedge_models = {}
        self.hedge_err = {'fresh': None, 'chain': None}
        self.hedge_scored = {'fresh': 0, 'chain': 0}
        self.hedge_active = 'fresh'
        self.hedge_picks = {'fresh': 0, 'chain': 0}
        self.hedge_switches = 0

        # Hessian cap (0 or None = off): see _cap_hessian
        self.hessian_cap = float(options['hessian_cap'] or 0.0)
        self.hessian_cap_refits = 0    # iterations on which the cap triggered a refit
        self.hessian_cap_linear = 0    # of those, the ones that used the linear model
        self._warned_cap_skip = False
        self._warned_cap_finite = False

    # ------------------------------------------------------------------ fit --
    def fit(self, samp):
        """Algorithm 5, lines 1-5: fit the model(s) and pick the one to use (lines 1-4), then apply the Hessian
        cap (line 5). Sets g, H."""
        if self.kind == 'mfn':
            g, H = self.fit_MFN(samp)
        elif self.kind == 'chain':
            g, H = self.fit_MCFN(samp)
        else:
            g, H = self.fit_hedged(samp)
        if self.hessian_cap > 0.0:
            g, H = self._cap_hessian(samp, g, H)
        self.g = g
        self.H = H

    def _cap_hessian(self, samp, g, H):
        """Hessian cap (Algorithm 5, line 5): if ||H||_2 > hessian_cap, replace the model by the least-squares fit
        with ||H||_2 <= hessian_cap (fit_least_squares_capped), or by the linear model when that refit is skipped
        or fails (the cases are listed in _hessian_cap.py)."""
        try:
            Ha = np.asarray(H, float)
            # ||H||_2 <= ||H||_F: the cheap Frobenius norm settles almost every call, the 2-norm (an SVD) is
            # needed only when the Frobenius norm is above the cap
            hf = float(np.linalg.norm(Ha))
            hn = hf if np.isfinite(hf) and hf <= self.hessian_cap else float(np.linalg.norm(Ha, 2))
        except (np.linalg.LinAlgError, ValueError):
            hn = np.inf
        if np.isfinite(hn) and hn <= self.hessian_cap:
            return g, H
        self.hessian_cap_refits += 1
        if not self._warned_cap_finite:
            vals = np.concatenate([samp.Y.values, samp.Z.values, [samp.fc]])
            if np.all(np.abs(vals) < 1e200):
                # the cap is absolute: an objective curved more than the cap triggers a refit every iteration
                warnings.warn(f"hessian_cap = {self.hessian_cap:g} reached by a model of finite data (||H|| = {hn:.3g}): "
                              "the objective's curvature exceeds the cap; rescale it or raise hessian_cap",
                              RuntimeWarning, stacklevel=3)
                self._warned_cap_finite = True
        if not have_cvxpy():
            feasible, reason = False, "cvxpy is not installed (pip install 'gcyz[rescue]')"
        else:
            feasible, reason = cap_refit_feasible(samp)
        if feasible:
            try:
                g, H, used_linear = fit_least_squares_capped(samp, self.hessian_cap)
            except MemoryError:
                g, H = _linear_model(samp)
                used_linear = True
        else:
            if not self._warned_cap_skip:
                warnings.warn(f"hessian_cap triggered but the capped least-squares refit was skipped ({reason}): "
                              "using the linear model through Y instead", RuntimeWarning, stacklevel=3)
                self._warned_cap_skip = True
            g, H = _linear_model(samp)
            used_linear = True
        self.hessian_cap_linear += int(used_linear)
        return g, H

    @staticmethod
    def _stack(samp):
        """(X, fX): the whole sample's points relative to the centre (Y, then Z, then the centre = 0), and their
        values."""
        X = np.vstack([samp.Y.points, samp.Z.points, np.zeros((1, samp.n))])
        fX = np.concatenate([samp.Y.values, samp.Z.values, [samp.fc]])
        return X, fX

    def fit_MFN(self, samp, count=True):
        """Minimum-Frobenius-norm fit of the whole sample (see _min_frobenius_fit). `count=False` leaves the
        counters alone."""
        assert samp.n == self.n, 'Dimensions of model and sample mismatch.'
        X, fX = self._stack(samp)
        try:
            g, H = _min_frobenius_fit(X, fX - samp.fc, fX, np.zeros((self.n, self.n)), samp._factors)
            if not _interpolates(X, fX - samp.fc, g, H, fX):
                # the fit misses the data (a nearly singular sample, or different values at nearly the same
                # point): counted, still used
                self.mfn_noninterp += count
            return g, H
        except np.linalg.LinAlgError:
            # Raised on non-finite data or an exactly singular sample. On finite data, use the linear model
            # through Y; on non-finite data, re-raise and the run ends with the best point found (GCYZ.optimize).
            if np.all(np.isfinite(X)) and np.all(np.isfinite(fX)) and np.isfinite(samp.fc):
                self.mfn_linear_fallbacks += count
                return _linear_model(samp)
            raise

    def fit_MCFN(self, samp):
        """Least-change fit (the paper's MCFN model): min ||H - H_anchor||_F subject to interpolation.

        A fit that fails or does not interpolate is rejected: the MFN fit is used in its place for this iteration
        (and scored in its place by the hedge), and the anchor stays at the last accepted fit."""
        count_mfn = self.kind == 'chain'  # under the hedge the MFN fit is already counted as the fresh model
        if self.scale_reseed > 1.0 and self.H_anchor is not None:
            # drop an anchor started at a radius more than `scale_reseed` times smaller than the current one
            # (seed_delta is the radius when the chain was started; accepted fits do not reset it)
            ds = self.seed_delta
            if ds > 0 and (self.delta / ds) > self.scale_reseed:
                self.H_anchor = None
                self.scale_reseeds += 1
        H0 = self.H_anchor
        if H0 is None:
            g, H = self.fit_MFN(samp, count=count_mfn)
            self.H_anchor = H
            self.seed_delta = float(self.delta)
            return g, H

        X, fX = self._stack(samp)
        df = fX - samp.fc
        try:
            g, H = _min_frobenius_fit(X, df, fX, H0, samp._factors)
        except np.linalg.LinAlgError:
            g = H = None
        if g is None or not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))) or \
                not _interpolates(X, df, g, H, fX):
            self.mcfn_fallbacks += 1
            return self.fit_MFN(samp, count=count_mfn)

        self.mcfn_updates += 1
        self.H_anchor = H
        return g, H

    def fit_hedged(self, samp):
        """Fit both models and use the one with the lower recent prediction error. The used model changes only
        when the other one's error is below hedge_hysteresis times its own."""
        g_f, H_f = self.fit_MFN(samp)
        g_c, H_c = self.fit_MCFN(samp)
        self._hedge_models = {'fresh': (g_f, H_f), 'chain': (g_c, H_c)}
        ready = (self.hedge_scored['fresh'] >= self.hedge_warmup and
                 self.hedge_scored['chain'] >= self.hedge_warmup and
                 self.hedge_err['fresh'] is not None and
                 self.hedge_err['chain'] is not None)
        prev_active = self.hedge_active
        if ready:
            cur = self.hedge_active
            other = 'chain' if cur == 'fresh' else 'fresh'
            if self.hedge_err[other] < self.hedge_hyst * self.hedge_err[cur]:
                self.hedge_active = other
        if self.hedge_chain_max_err is not None and self.hedge_active == 'chain' and \
                self.hedge_err['chain'] > self.hedge_chain_max_err:
            # GC-YZ-V: use the chain only while its error is below hedge_chain_max_err
            self.hedge_active = 'fresh'
        if self.hedge_active != prev_active:
            self.hedge_switches += 1  # counts changes of the used model
        self.hedge_picks[self.hedge_active] += 1
        return self._hedge_models[self.hedge_active]

    def hedge_update(self, step, actual_df):
        """Score both hedge models on an evaluated step: error = |predicted - actual| / max(|predicted|, |actual|)"""
        if self.kind != 'hedge':
            return
        s = np.asarray(step, dtype=float).ravel()
        for k, (g, H) in self._hedge_models.items():
            pred = float(np.dot(np.asarray(g).ravel(), s) + 0.5 * float(s @ (H @ s)))
            denom = max(abs(actual_df), abs(pred), 1e-300)
            e = abs(pred - actual_df) / denom
            if not np.isfinite(e):
                e = 2.0
            prev = self.hedge_err[k]
            self.hedge_err[k] = e if prev is None else \
                self.hedge_beta * prev + (1.0 - self.hedge_beta) * e
            self.hedge_scored[k] += 1

    # ------------------------------------------------------------- TR step --
    def minimize(self):
        """Trust-region step: returns (step, predicted decrease). The exact solver can overshoot the radius slightly
        (about 1e-7 relative): a step up to 1e-6 outside is scaled back onto the boundary. A step further outside
        is replaced by the truncated CG step (never seen in practice)."""
        v, val = trust_sub_exact(self.H, self.g, self.delta)
        nv = np.linalg.norm(v)
        if nv > self.delta * (1.0 + 1e-8):
            if nv <= self.delta * (1.0 + 1e-6):
                v = v * (self.delta / nv)
                val = float(np.dot(self.g, v) + 0.5 * np.dot(v, np.dot(self.H, v)))
            else:
                v, val = trust_sub_CG(self.H, self.g, self.delta)
                val = val[0]
                nv = np.linalg.norm(v)
                if not (np.all(np.isfinite(v)) and np.isfinite(val) and nv <= self.delta * (1.0 + 1e-6)):
                    # the CG step failed too (an overflowing model): stop cleanly rather than take a NaN step
                    raise np.linalg.LinAlgError("trust-region step not finite or outside the ball")
        return v, -val
