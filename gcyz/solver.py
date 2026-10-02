"""GC-YZ solver: the GCYZ class and minimize(). The two variants (LIN, V) are described in the README; their
options are in options.py."""
import contextlib
import math
import numbers
import time
import warnings

import numpy as np

from .sample import Sample, _orthogonal_step
from .model import Model
from .oracle import Oracle, BudgetExhausted, ObjectiveUnbounded, _real_value, _standin, is_failure, _BIG as _BIG_VALUE
from .options import DEFAULT_OPTIONS, GC_YZ_LIN, GC_YZ_V, validate_options, _resolve

__all__ = ["GCYZ", "minimize", "Result", "GC_YZ_LIN", "GC_YZ_V", "DEFAULT_OPTIONS"]


class DesignCollapsed(ValueError):
    """Raised when x0 is so large compared with the initial radius tr_delta that x0 + tr_delta rounds back to x0,
    so the starting points would all coincide. Fix: rescale the problem or use a larger tr_delta."""


def _design_z(n, sample_max):
    """How many of the n points x0 - Delta q_i of the initial design go to Z (Y takes the n points x0 + Delta q_i):
    as many as Z holds, sample_max - 1 - n."""
    return min(n, max(0, sample_max - 1 - n))


def _design_size(options, n):
    """Number of evaluations of the initial design points (f(x0) not included)."""
    sample_max = max(int(_resolve(options.get('sample_max', DEFAULT_OPTIONS['sample_max']), n)), n + 1)
    return n + _design_z(n, sample_max)


def _checked_fx0(fx0):
    """A given f(x0) converted as objective values are (NaN, +-inf -> +-1e200); None if not given."""
    if fx0 is None:
        return None
    try:
        return _standin(_real_value(fx0))
    except TypeError:
        raise TypeError(f"fx0 must be a real number, got {type(fx0).__name__}") from None


def _stop_reason(stop):
    """info['stop_reason'] for a run ended by the budget or by a -inf value."""
    if isinstance(stop, BudgetExhausted):
        return 'max function evaluations reached'
    return 'objective unbounded below (returned -inf)'


class Result:
    """Return value of minimize()."""

    def __init__(self, x, fun, nfev, fval_history, info, message):
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.fval_history = fval_history
        self.info = info
        self.message = message

    def __repr__(self):
        return f"Result(fun={self.fun!r}, nfev={self.nfev}, message={self.message!r})"


# Rounding tolerances for three "equal" tests: rho = rhoend, Delta = rho, and a point on the trust-region boundary.
_RHOEND_RTOL = 1e-10
_FLOOR_RTOL = 1e-12
_FAR_RTOL = 1e-8

# Model counters copied into info at the end of a run: info key -> how to read it from the Model.
_MODEL_COUNTERS = {
    'mcfn_update_counter': lambda m: m.mcfn_updates,
    'mcfn_fallback_counter': lambda m: m.mcfn_fallbacks,
    'mfn_noninterp_counter': lambda m: m.mfn_noninterp,
    'mfn_linear_fallback_counter': lambda m: m.mfn_linear_fallbacks,
    'scale_reseed_counter': lambda m: m.scale_reseeds,
    'hedge_pick_fresh': lambda m: m.hedge_picks['fresh'],
    'hedge_pick_chain': lambda m: m.hedge_picks['chain'],
    'hedge_switch_counter': lambda m: m.hedge_switches,
    'hessian_cap_counter': lambda m: m.hessian_cap_refits,
    'hessian_cap_linear_counter': lambda m: m.hessian_cap_linear,
}


class GCYZ:
    """Solver state. Use :meth:`optimize` (or the module-level :func:`minimize`)."""

    def __init__(self, x0, oracle, fx0=None, options=None, rng=None):
        x0 = np.asarray(x0, dtype=float).ravel()
        self.n = int(x0.size)
        self.oracle = oracle
        opts = dict(DEFAULT_OPTIONS)
        opts.update(options or {})
        validate_options(opts, n=self.n)
        opts['sample_max'] = max(int(_resolve(opts['sample_max'], self.n)), self.n + 1)  # room for Y and the centre
        self.options = opts

        self.info = self._new_info()
        self._floor_failed = None     # set when the resolution floor cannot be verified: the run then stops
        self._far_gate_tried = False  # GC-YZ-V: the far-point guard already spent an evaluation this time

        self.samp = self._initial_sample(x0, fx0, rng)
        self.model = Model(self.n, opts)
        self.rho = opts['tr_delta']  # resolution floor

    @staticmethod
    def _new_info():
        """The counters of a run before its first iteration (_finish adds the final state)."""
        return {
            'start_time': time.time(),
            'iteration': 0,
            'success': 0,
            'nfeval': 0,
            'predicted_decrease': np.inf,
            'delta_history': [],
            'small_grad_bad_geom_counter': 0,
            'small_grad_good_geom_counter': 0,
            'short_step_counter': 0,
            'short_step_far_counter': 0,
            'gc_far_point_counter': 0,
            'gc_sc_counter': 0,
            'gc_gc_counter': 0,
            'nwa_admit_y_counter': 0,
            'nwa_admit_z_counter': 0,
            'nwa_discard_counter': 0,
            'rho_far_gate_counter': 0,
            'rho_lambda_gate_counter': 0,
            'rho_converged': False,
            'floor_geometry_counter': 0,
        }

    def _initial_sample(self, x0, fx0, rng):
        """f(x0) (unless given) and the 2n design points x0 +/- tr_delta q_i, where the q_i are the coordinate axes
        or a random rotation of them (init_frame). Failed evaluations are replaced or dropped."""
        o = self.options
        oracle = self.oracle
        if o['init_frame'] == 'identity':
            Q = np.eye(self.n)
        elif rng is None:
            Q, _ = np.linalg.qr(np.random.randn(self.n, self.n))
        else:
            Q, _ = np.linalg.qr(rng.standard_normal((self.n, self.n)))
        all_pts = np.vstack([o['tr_delta'] * Q.T, o['tr_delta'] * -Q.T])
        if any(np.array_equal(x0 + p, x0) for p in all_pts):
            raise DesignCollapsed(f"a design point x0 +/- tr_delta q_i equals x0 in floating point (tr_delta = "
                                  f"{o['tr_delta']:g}, max|x0| = {float(np.max(np.abs(x0))):g}): rescale the problem "
                                  f"or set tr_delta to the scale of x0")
        z_cap = max(0, o['sample_max'] - 1 - self.n)
        n_z = _design_z(self.n, o['sample_max'])
        Y_pts = all_pts[:self.n]
        Z_pts = all_pts[self.n:self.n + n_z]
        fc = float(oracle(x0)) if fx0 is None else _checked_fx0(fx0)
        if fx0 is not None and fc <= -_BIG_VALUE:
            raise ObjectiveUnbounded("fx0 is -inf: the objective is unbounded below at x0")
        Y_vals = np.array([float(oracle(x0 + y)) for y in Y_pts])
        Z_vals = np.array([float(oracle(x0 + z)) for z in Z_pts])
        if any(is_failure(v) for v in Y_vals) or any(is_failure(v) for v in Z_vals):
            Y_pts, Y_vals, Z_pts, Z_vals = self._replace_failed_design(x0, Y_pts, Y_vals, Z_pts, Z_vals)
        return Sample(x0, fc, Y_pts, Y_vals, z_cap, Z_pts=Z_pts, Z_vals=Z_vals)

    # ------------------------------------------------------------ helpers --
    def _log(self, level, msg):
        """Print ``msg`` when the verbosity is at least ``level``."""
        if self.options['verbosity'] >= level:
            print(msg)

    def _replace_failed_design(self, x0, Y_pts, Y_vals, Z_pts, Z_vals):
        """Keep failed evaluations (NaN, +inf) out of the initial sample. A failed Y design point x0 + Delta q_i is
        replaced by the opposite design point x0 - Delta q_i (moved over from Z, or evaluated if Z does not hold it),
        so Y keeps a point along q_i; if that one failed too, or is already in Y, the point is dropped. Failed Z
        points are dropped."""
        Y_pts, Y_vals, Z_pts, Z_vals = Y_pts.copy(), Y_vals.copy(), Z_pts.copy(), Z_vals.copy()
        keep_y = np.ones(len(Y_pts), dtype=bool)
        keep_z = np.array([not is_failure(v) for v in Z_vals], dtype=bool)
        for i in range(len(Y_pts)):
            if not is_failure(Y_vals[i]):
                continue
            mirror = -Y_pts[i]  # the opposite design point
            in_z = [j for j in range(len(Z_pts)) if np.array_equal(Z_pts[j], mirror)]
            if in_z:
                value = Z_vals[in_z[0]]
                keep_z[in_z[0]] = False  # it moves from Z to Y (or it failed as well)
            elif any(np.array_equal(y, mirror) for y in Y_pts):
                value = np.nan  # already in Y: using it would duplicate a point
            else:
                value = float(self.oracle(x0 + mirror))
            if is_failure(value):
                keep_y[i] = False
            else:
                Y_pts[i], Y_vals[i] = mirror, value
        return Y_pts[keep_y], Y_vals[keep_y], Z_pts[keep_z], Z_vals[keep_z]

    def _repair_geometry(self, idx, lag_step):
        """Replace Y[idx] by a new point at lag_step (one evaluation); with idx None (a short Y) the point is added
        instead. If that evaluation fails, the mirror point -lag_step, where the Lagrange polynomial has the same
        size, is tried instead. Returns the list of removed (point, value) pairs, or None (Y unchanged) if no finite
        value was found."""
        for s in (lag_step, -lag_step):
            x_s = self.samp.center + s
            if np.array_equal(x_s, self.samp.center) or \
                    any(np.array_equal(self.samp.center + y, x_s) for y in self.samp.Y.points):
                continue  # same point as the centre or a Y point in floating point
            value = self.oracle(x_s)
            if not is_failure(value):
                kicked = []
                if idx is not None:
                    kicked.append(self.samp.Y[idx])
                    self.samp.Y.delete_point(idx)
                self.samp.Y.add_point(s, value=value)
                dup_z = [j for j in range(self.samp.mZ) if np.array_equal(self.samp.center + self.samp.Z.points[j], x_s)]
                if dup_z:  # the new point was already in Z: it now lives in Y only
                    self.samp.Z.delete_point(dup_z)
                return kicked
            if self.oracle.get_evaluation_count() >= self.options['stop_nfeval']:
                break
        return None

    def _repair(self, idx, lag_step):
        """_repair_geometry, then offer the removed point (if any) to Z. True if Y changed."""
        kicked = self._repair_geometry(idx, lag_step)
        if kicked is None:
            return False
        self.samp.update_Z(kicked)
        return True

    def _stale_far_point(self, L, mult):
        """(idx, step) for the furthest Y point if it lies beyond mult * Delta, with the step that would replace
        it; None if there is none, mult is 0 (off), or no usable step exists. mult below 1 counts as 1."""
        if mult <= 0.0 or self.samp.mY == 0:
            return None
        mult = max(mult, 1.0)
        far_idx, far_point = self.samp.Y.get_furthest()
        if np.linalg.norm(far_point) <= mult * self.model.delta * (1.0 + _FAR_RTOL):
            return None
        far_step = self.samp.lagrange_step_col(L, far_idx, self.model.delta)
        if far_step is None or not np.all(np.isfinite(far_step)) or np.linalg.norm(far_step) == 0:
            return None
        return far_idx, far_step

    def _rho_far_gate(self):
        """GC-YZ-V guard: do not lower rho while a Y point lies beyond rho_advance_far_gate * Delta; replace it
        instead (one evaluation). True if the advance was blocked."""
        m = float(self.options['rho_advance_far_gate'])
        self._far_gate_tried = False
        if m <= 0.0 or self.samp.mY == 0:
            return False
        far_idx, far_point = self.samp.Y.get_furthest()
        if np.linalg.norm(far_point) <= m * self.model.delta:
            return False
        lag_step = self.samp.lagrange_step_col(self.samp.get_lagrange_coef(), far_idx, self.model.delta)
        if lag_step is None or not np.all(np.isfinite(lag_step)) or np.linalg.norm(lag_step) == 0:
            return False
        self._far_gate_tried = True
        if not self._repair(far_idx, lag_step):
            self._floor_failed = 'no usable geometry point at the resolution floor'
            return True
        self.info['rho_far_gate_counter'] += 1
        self._log(1, "GC: rho advance blocked -- stale managed point dragged in")
        return True

    def _rho_lambda_gate(self):
        """GC-YZ-V guard: do not lower rho while Y is badly poised (above rho_advance_lambda_gate); repair it
        instead (one evaluation). True if the advance was blocked."""
        thr = float(self.options['rho_advance_lambda_gate'])
        if thr <= 0.0 or self.samp.mY == 0:
            return False
        L = self.samp.get_lagrange_coef()
        idx, lag_step, poised = self.samp.lagrange_poisedness(L, self.model.delta)
        if poised <= thr:
            return False
        if not np.all(np.isfinite(lag_step)) or np.linalg.norm(lag_step) == 0:
            return False
        if not self._repair(idx, lag_step):
            self._floor_failed = 'no usable geometry point at the resolution floor'
            return True
        self.info['rho_lambda_gate_counter'] += 1
        self._log(1, "GC: rho advance blocked -- badly poised set repaired")
        return True

    def _try_advance_rho(self):
        """Delta is at the floor: lower rho, or stop at rhoend (both variants). First GC-YZ-V's two guards may block
        it by fixing Y instead (they are off in GC-YZ-LIN), and Y is checked to be full and well poised."""
        if self._rho_far_gate():
            return
        if not self._far_gate_tried and self._rho_lambda_gate():  # skip it if the far gate just tried the same repair
            return
        o = self.options
        if not self._verify_floor_geometry():
            self._floor_failed = 'no usable geometry point at the resolution floor'
            return
        if self.rho <= o['rhoend'] * (1.0 + _RHOEND_RTOL):
            self.info['rho_converged'] = True
        else:
            new_rho = max(o['rho_shrink'] * self.rho, o['rhoend'])
            self.model.delta = max(0.5 * self.model.delta, new_rho)  # as NEWUOA: halve Delta, not below the new rho
            self.rho = new_rho

    def _verify_floor_geometry(self):
        """Sanity check before lowering rho: Y must be full and well poised. The algorithm normally guarantees this;
        it only acts in edge cases (failed evaluations, a short Y). False if Y cannot be fixed: the run then stops."""
        samp, o = self.samp, self.options
        while samp.mY < samp.n:
            # a new point along a direction orthogonal to the current Y points
            step = _orthogonal_step(samp.Y.points if samp.mY else np.zeros((0, samp.n)), samp.n, self.model.delta)
            if not (np.all(np.isfinite(step)) and np.linalg.norm(step) > 0):
                return False
            added = False
            for cand in (step, -step):
                x_c = samp.center + cand
                if np.array_equal(x_c, samp.center) or any(np.array_equal(samp.center + y, x_c) for y in samp.Y.points) \
                        or any(np.array_equal(samp.center + z, x_c) for z in samp.Z.points):
                    continue
                value = self.oracle(x_c)
                if not is_failure(value):
                    samp.Y.add_point(cand, value=value)
                    added = True
                    break
            if not added:
                return False
            self.info['floor_geometry_counter'] += 1
            self._log(1, "GC: managed set refilled at the floor")
        for _ in range(samp.mY + 1):
            L = samp.get_lagrange_coef()
            idx, lag_step, poised = samp.lagrange_poisedness(L, self.model.delta)
            if poised <= o['big_lambda']:
                return True
            if lag_step is None or not np.all(np.isfinite(lag_step)) or np.linalg.norm(lag_step) == 0:
                return False
            if not self._repair(idx, lag_step):
                return False
            self.info['floor_geometry_counter'] += 1
            self.info['gc_gc_counter'] += 1
            self._log(1, "GC: managed set repaired at the floor")
        return False

    def _at_floor(self):
        """True when Delta has come down to rho (up to rounding)."""
        return self.model.delta <= self.rho * (1.0 + _FLOOR_RTOL)

    def _shrink(self):
        """Shrink Delta by tr_shrink, but not below rho."""
        self.model.delta = max(self.options['tr_shrink'] * self.model.delta, self.rho)

    def _shrink_or_advance(self):
        """Shrink Delta, or lower rho when Delta is already at rho."""
        if self._at_floor():
            self._try_advance_rho()
        else:
            self._shrink()

    # ----------------------------------------------------------- main loop --
    @classmethod
    def optimize(cls, x0, oracle, options=None, *, fx0=None, rng=None):
        """Run the method. Returns (x, f, info) for the final trust-region centre, which need not be the best
        point evaluated (minimize() returns that one). oracle is called on points x and returns a float; use
        gcyz.Oracle to count evaluations and record the history."""
        try:
            opt = cls(x0, oracle, fx0, options, rng=rng)
        except (BudgetExhausted, ObjectiveUnbounded) as stop:
            if oracle.in_fun:
                raise  # raised inside the objective itself (a nested gcyz run): not this run's stop
            # the run ended during the initial design: return x0 and the counters of a run that never iterated
            f0 = _checked_fx0(fx0)
            tr0 = float((options or {}).get('tr_delta', DEFAULT_OPTIONS['tr_delta']))
            info = cls._new_info()
            info.update(nfeval=oracle.get_evaluation_count(), stop_reason=_stop_reason(stop), end_time=time.time(),
                        tr_final=tr0, rho_final=tr0)
            info.update({key: 0 for key in _MODEL_COUNTERS})
            return np.asarray(x0, dtype=float).ravel(), (np.inf if f0 is None else f0), info
        opt.model.delta = opt.options['tr_delta']

        while True:
            try:
                if opt._iteration():
                    break
            except (BudgetExhausted, ObjectiveUnbounded) as stop:
                if oracle.in_fun:
                    raise  # raised inside the objective itself (a nested gcyz run): not this run's stop
                opt.info['stop_reason'] = _stop_reason(stop)
                opt._finish(None)
                break
            except Exception as exc:
                try:
                    exc.gcyz_info = dict(opt.info)  # the counters so far, for the result of a run cut short
                except (AttributeError, TypeError):
                    pass
                raise

        return opt.samp.center, opt.samp.fc, opt.info

    def _iteration(self):
        """One iteration of Algorithm 5. Returns True when the run must stop (the reason is in info)."""
        o = self.options
        oracle = self.oracle
        self.info['iteration'] += 1
        self.info['nfeval'] = oracle.get_evaluation_count()

        # ---- fit the model (Algorithm 5, lines 1-5) ----
        try:
            self.model.fit(self.samp)
        except (np.linalg.LinAlgError, ValueError) as exc:
            self._finish(f"fit: {exc}")
            return True

        # ---- geometry of Y (line 6) ----
        L = self.samp.get_lagrange_coef()
        idx, lag_step, poised_val = self.samp.lagrange_poisedness(L, self.model.delta)
        gnorm = np.linalg.norm(self.model.g)
        small_grad = gnorm < o['tr_toexpand'] * self.model.delta
        # with a small gradient, first replace a Y point outside the ball (poisedness in the ball assumes Y is
        # inside it), otherwise repair a badly poised Y
        stale = self._stale_far_point(L, 1.0) if small_grad else None
        if stale is not None:
            idx, lag_step = stale

        if small_grad and (stale is not None or poised_val > o['big_lambda']):
            # small gradient, bad geometry: repair Y, no trial point (line 7)
            if not self._repair(idx, lag_step):
                self._shrink_or_advance()  # no usable repair point: change the radius instead
            self.info['small_grad_bad_geom_counter'] += 1
            self._log(1, "small grad - gc")

        elif small_grad and poised_val <= o['big_lambda']:
            # small gradient, good geometry: shrink Delta, or lower rho at the floor
            self._shrink_or_advance()
            self.info['small_grad_good_geom_counter'] += 1
            self._log(1, "small grad - shrink")

        else:
            # ---- trial step (line 8) ----
            try:
                step, self.info['predicted_decrease'] = self.model.minimize()
            except (np.linalg.LinAlgError, ValueError) as exc:
                self._finish(f"minimize: {exc}")
                return True
            if self.info['predicted_decrease'] <= 0:
                # no predicted decrease: the ratio test below then treats the step as unsuccessful
                self.info['predicted_decrease'] = np.inf

            if np.linalg.norm(step) < o['small_step_gate'] * self.rho:
                # step too short to evaluate (line 9): first bring back a Y point beyond gate_far_mult * Delta,
                # otherwise repair a badly poised Y; if the geometry is fine, shrink Delta or lower rho
                self.info['short_step_counter'] += 1
                stale = self._stale_far_point(L, o['gate_far_mult'])
                if stale is not None:
                    idx, lag_step = stale
                if stale is not None or poised_val > o['big_lambda']:
                    if self._repair(idx, lag_step):
                        self.info['gc_gc_counter'] += 1
                        if stale is not None:
                            self.info['short_step_far_counter'] += 1
                    else:
                        self._shrink_or_advance()  # no usable repair point: change the radius instead
                else:
                    self._shrink_or_advance()
            else:
                self._evaluate_step(step)

        self.samp.compact()
        self.info['delta_history'].append((oracle.get_evaluation_count(), self.model.delta))

        if self._floor_failed:
            self._finish(self._floor_failed)
            return True
        if self._stop():
            self._finish(None)
            return True
        return False

    def _evaluate_step(self, step):
        """Lines 10-13: evaluate the trial point, score both models on it, then accept or reject the step."""
        samp = self.samp
        # The trial point may equal the centre or a sample point (e.g. a step onto a design point). It is then not
        # added to the sample again (a duplicate makes the fit singular), and its known value is reused -- except
        # from Z, which may hold a made-up value for a failed point, so that point is evaluated again.
        x_new = samp.center + step
        zero = np.array_equal(x_new, samp.center)
        dup_y = next((i for i in range(samp.mY) if np.array_equal(samp.center + samp.Y.points[i], x_new)), None)
        dup = zero or dup_y is not None or any(np.array_equal(samp.center + z, x_new) for z in samp.Z.points)
        if zero:
            f_new = samp.fc
        elif dup_y is not None:
            f_new = float(samp.Y.values[dup_y])
        else:
            f_new = self.oracle(x_new)
        # a failed evaluation (NaN, +inf) is an unsuccessful step and never enters the sample
        failed = is_failure(f_new)
        ratio = (samp.fc - f_new) / self.info['predicted_decrease']

        # line 11: score both hedge models on the actual change
        if not (failed or is_failure(samp.fc)):
            self.model.hedge_update(step, f_new - samp.fc)

        if ratio >= self.options['tr_toaccept'] and not failed:
            self._accept_step(step, f_new)
        else:
            self._reject_step(step, f_new, failed, dup)

    def _accept_step(self, step, f_new):
        """Line 12, successful step: move the centre to the trial point, add the old centre to the sample
        (removing the point furthest from the new centre if the set is full), and expand Delta."""
        o = self.options
        samp = self.samp
        self.info['success'] += 1
        y_max_idx, y_furthest = samp.Y.get_furthest(step)
        z_max_idx, z_furthest = samp.Z.get_furthest(step)
        if is_failure(samp.fc):
            pass  # f(x0) failed: the old centre does not join the sample
        elif samp.mY < samp.n:
            samp.Y.append_origin(value=samp.fc)
        elif z_max_idx is not None and \
                np.linalg.norm(y_furthest - step) > np.linalg.norm(z_furthest - step):
            # the furthest Y point is further than the furthest Z point: it makes room in Y
            evicted = samp.Y[y_max_idx]
            samp.Y.delete_point(y_max_idx)
            samp.Y.append_origin(value=samp.fc)
            if samp.mZ < samp.z_capacity:
                samp.Z.add_point(evicted[0], value=evicted[1])  # Z has room: keep the removed point
        elif samp.z_capacity == 0:
            samp.Y.delete_point(y_max_idx)
            samp.Y.append_origin(value=samp.fc)
        elif samp.mZ >= samp.z_capacity:
            samp.Z.delete_point(z_max_idx)
            samp.Z.append_origin(value=samp.fc)
        else:
            samp.Z.append_origin(value=samp.fc)
        # if the new centre was already a sample point (the step landed on it), remove that point: the centre
        # is stored separately, and keeping it in Y or Z as well would duplicate it
        x_new = samp.center + step
        for S in (samp.Y, samp.Z):
            at_centre = [i for i in range(len(S)) if np.array_equal(samp.center + S.points[i], x_new)]
            if at_centre:
                S.delete_point(at_centre)
        samp.Y.shift(step)
        samp.Z.shift(step)
        samp.center = x_new
        samp.fc = f_new
        self.model.delta *= o['tr_expand']
        self.model.delta = max(self.model.delta, self.rho)
        self._log(1, "accepted")

    def _reject_step(self, step, f_new, failed, dup=False):
        """Line 13, unsuccessful step: update the sample with the trial point (GC-YZ-V's admission test, or the
        steps of Algorithm 2 for GC-YZ-LIN), then shrink Delta, or lower rho if the sample did not improve."""
        update = self._reject_update_v if self.options['newuoa_admission'] else self._reject_update_lin
        improved, kicked_list = update(step, f_new, failed, dup)
        if improved:
            self._shrink()  # the sample improved: shrink only
        else:
            self._shrink_or_advance()  # nothing improved: shrink, or lower rho at the floor
        self.samp.update_Z(kicked_list)

    def _failed_point(self, step):
        """A failed trial point as it is offered to Z: (step, f(centre) + predicted decrease), so the model stops
        stepping towards it. None if that value is not available."""
        pred = self.info['predicted_decrease']
        if np.isfinite(pred) and not is_failure(self.samp.fc):
            return step, self.samp.fc + pred
        return None

    def _reject_update_v(self, step, f_new, failed, dup):
        """GC-YZ-V (as NEWUOA): the trial point replaces the sample point with the largest distance-weighted
        Lagrange value if that value is above 1 + nwa_margin, and is discarded otherwise. A duplicate is ignored,
        a failed point is offered to Z, and a Y that is not full just takes the point. Returns (improved, the
        points to offer to Z)."""
        o, samp = self.options, self.samp
        if dup:
            return False, []
        if failed:
            barrier = self._failed_point(step)
            return False, [barrier] if barrier is not None else []
        if samp.mY < samp.n:
            samp.Y.add_point(step, value=f_new)
            self._log(1, "GC: Added point to Y")
            return True, []
        lv = samp.mfn_values_all(step)
        score = np.zeros(0)
        if lv is not None:
            norms = np.linalg.norm(samp.Y.points, axis=1)
            if samp.mZ:
                norms = np.concatenate([norms, np.linalg.norm(samp.Z.points, axis=1)])
            rref = max(0.1 * float(self.model.delta), float(self.rho))
            w = np.maximum(1.0, norms ** 2 / max(rref, 1e-300) ** 2) ** 3
            score = w * np.asarray(lv[:norms.size], float) ** 2
            score[~np.isfinite(score)] = -1.0
        if not (score.size and score.max() > 1.0 + float(o['nwa_margin'])):
            self.info['nwa_discard_counter'] += 1
            self._log(1, "GC: NEWUOA admission -- trial discarded")
            # nwa_no_discard: offer the trial point to Z instead of throwing the evaluation away
            return False, [(step, f_new)] if o['nwa_no_discard'] else []
        t = int(np.argmax(score))
        kicked_list = []
        if t < samp.mY:
            kicked_list.append(samp.Y[t])
            samp.Y.delete_point(t)
            samp.Y.add_point(step, value=f_new)
            self.info['nwa_admit_y_counter'] += 1
        else:
            samp.Z.delete_point(t - samp.mY)
            samp.Z.add_point(step, value=f_new)
            self.info['nwa_admit_z_counter'] += 1
        self._log(1, "GC: NEWUOA admission swap")
        return True, kicked_list

    def _reject_update_lin(self, step, f_new, failed, dup):
        """GC-YZ-LIN, Algorithm 2, after an unsuccessful step. The trial point: a duplicate adds nothing, and a
        failed one is offered to Z with a stand-in value; otherwise a short Y takes it, or (i) it replaces a Y point
        outside the ball (which ends the update), or (ii) it replaces the Y point whose Lagrange polynomial is large
        there, or else it is offered to Z. Then (iii) a still badly poised Y is repaired (one evaluation). Returns
        (improved, the points to offer to Z)."""
        o, samp = self.options, self.samp
        kicked_list = []
        improved = False
        L = None  # Lagrange coefficients of Y, while Y is unchanged
        if dup:
            pass  # the trial point is already in the sample
        elif failed:
            barrier = self._failed_point(step)
            if barrier is not None:
                kicked_list.append(barrier)
        elif samp.mY < samp.n:
            samp.Y.add_point(step, value=f_new)
            self._log(1, "GC: Added point to Y")
            return True, kicked_list
        else:
            L = samp.get_lagrange_coef()
            far_idx, far_point = samp.Y.get_furthest()
            if np.linalg.norm(far_point) > self.model.delta * (1.0 + _FAR_RTOL):
                # (i) replace the furthest Y point by the trial point; if its Lagrange polynomial is zero there
                # use the polynomial's maximiser over the ball instead (one evaluation)
                l_at_s = float(np.abs(step @ L)[far_idx]) if np.all(np.isfinite(L)) else 1.0
                if l_at_s > 0.0:
                    kicked_list.append(samp.Y[far_idx])
                    samp.Y.delete_point(far_idx)
                    samp.Y.add_point(step, value=f_new)
                else:
                    far_step = samp.lagrange_step_col(L, far_idx, self.model.delta)
                    kicked = self._repair_geometry(far_idx, far_step) if far_step is not None and \
                        np.all(np.isfinite(far_step)) and np.linalg.norm(far_step) > 0 else None
                    if kicked is not None:
                        kicked_list += kicked
                    kicked_list.append((step, f_new))  # the trial point goes to Z
                    self.info['gc_gc_counter'] += kicked is not None
                self.info['gc_far_point_counter'] += 1
                self._log(1, "GC: Replace far point")
                return True, kicked_list
            if o['self_correcting']:
                # (ii) self-correcting swap
                lagrange_at_s = np.abs(step @ L)
                idx = int(np.argmax(lagrange_at_s))
                if lagrange_at_s[idx] > o['sc_lambda']:
                    kicked_list.append(samp.Y[idx])
                    samp.Y.delete_point(idx)
                    samp.Y.add_point(step, value=f_new)
                    improved = True
                    self.info['gc_sc_counter'] += 1
                    self._log(1, "GC: Replace by Lagrange poly (self correcting)")
                    L = None  # Y changed
            if not improved:
                kicked_list.append((step, f_new))  # the trial point is offered to Z
        # (iii) repair Y if it is still badly poised
        if L is None:
            L = samp.get_lagrange_coef()  # a duplicate or failed point, or Y changed in (ii)
        idx, lag_step, poised_val = samp.lagrange_poisedness(L, self.model.delta)
        if poised_val > o['big_lambda']:
            kicked = self._repair_geometry(idx, lag_step)
            if kicked is not None:
                kicked_list += kicked
                improved = True
                self.info['gc_gc_counter'] += 1
                self._log(1, "GC: Replace bad point by Lagrange poly")
        return improved, kicked_list

    # -------------------------------------------------------------- stop --
    def _stop(self):
        """Apply the stopping rules; record the reason in info and return True when the run must stop."""
        o = self.options
        info = self.info
        info['nfeval'] = self.oracle.get_evaluation_count()  # include this iteration's evaluations
        reason = None
        if info['iteration'] >= o['stop_iter']:
            reason = 'max iterations reached'
        elif info['nfeval'] >= o['stop_nfeval']:
            reason = 'max function evaluations reached'
        elif info.get('rho_converged', False):
            reason = 'rho reached rhoend'
        if reason is None:
            return False
        info['stop_reason'] = reason
        self._log(1, f"Exiting: {reason}.")
        return True

    def _finish(self, degenerate):
        """Record the final state in info; degenerate (not None) is the reason for a degenerate stop."""
        info = self.info
        if degenerate is not None:
            info['degenerate_stop'] = degenerate
            info['stop_reason'] = f"degenerate model ({degenerate})"
            self._log(1, f"Exiting: {degenerate}.")
        info['nfeval'] = self.oracle.get_evaluation_count()
        info['end_time'] = time.time()
        info['tr_final'] = self.model.delta
        info['rho_final'] = self.rho
        info.update({key: read(self.model) for key, read in _MODEL_COUNTERS.items()})


def _check_arguments(x0, budget, variant, options, fx0, rng):
    """Check minimize's arguments; return (x0 as a flat float array, the full options, fx0 as the solver sees it)."""
    if np.iscomplexobj(np.asarray(x0)):
        raise ValueError("x0 must be real")
    x0 = np.asarray(x0, dtype=float).ravel()
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be finite")
    if not isinstance(variant, str) or variant.upper() not in ('V', 'LIN'):
        raise ValueError(f"variant must be 'LIN' or 'V', got {variant!r}")
    if options is not None and not isinstance(options, dict):
        raise ValueError(f"options must be a dict, got {type(options).__name__}")
    opts = dict(GC_YZ_V if variant.upper() == 'V' else GC_YZ_LIN)
    if options:
        opts.update(options)
    if not (isinstance(budget, numbers.Real) and math.isfinite(budget) and float(budget).is_integer()):
        raise ValueError(f"budget must be a whole number, got {budget!r}")
    opts['stop_nfeval'] = int(budget)
    validate_options(opts, n=x0.size)
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator (numpy.random.default_rng(seed)) or None, got "
                        f"{type(rng).__name__}; it is used only with init_frame='random'")
    fx0 = _checked_fx0(fx0)
    need = _design_size(opts, x0.size) + (fx0 is None)
    if budget < need:
        raise ValueError(f"budget must cover the initial sample, {need} evaluations (f(x0) and the design "
                         f"points x0 +/- tr_delta q_i), got {budget}")
    return x0, opts, fx0


def minimize(fun, x0, budget, variant='LIN', options=None, *, fx0=None, rng=None, keep_x=False):
    """Minimise fun, starting from x0, with at most budget evaluations of fun. variant is 'LIN' or 'V'.
    Returns a Result; its x and fun are the best point found. See the README for the arguments and options."""
    # check before the try block below, which would turn a ValueError into an early stop
    x0, opts, fx0 = _check_arguments(x0, budget, variant, options, fx0, rng)
    oracle = Oracle(fun, keep_x=keep_x, max_nfev=int(budget))
    try:
        with np.errstate(under='ignore'):  # underflow to zero is harmless here
            x, f, info = GCYZ.optimize(x0, oracle, opts, fx0=fx0, rng=rng)
    except BaseException as exc:
        counters = getattr(exc, 'gcyz_info', None)  # the run's counters so far, attached by GCYZ.optimize
        if oracle.in_fun:
            # the objective raised: re-raise it, with the best point so far attached as exc.gcyz_result
            oracle.clear_error()
            with contextlib.suppress(AttributeError, TypeError):  # some exception types take no attributes
                exc.gcyz_result = _best_so_far(x0, fx0, oracle, keep_x, counters,
                                               f'stopped early: the objective raised {type(exc).__name__}: {exc}')
            raise
        if isinstance(exc, DesignCollapsed) or \
                not isinstance(exc, (np.linalg.LinAlgError, ValueError, FloatingPointError)):
            raise  # an input error, an interrupt, or a bug
        # the solver's linear algebra broke down: warn, and return the best point so far
        reason = f'stopped early: {exc}'
        warnings.warn(f"gcyz {reason}; returning the best point evaluated", RuntimeWarning, stacklevel=2)
        return _best_so_far(x0, fx0, oracle, keep_x, counters, reason)
    if oracle.best_x is not None and oracle.best_f < f:  # return the best point, not the final centre
        x, f = oracle.best_x, oracle.best_f
    message = info.get('stop_reason', '')
    if is_failure(f):
        message += " (no evaluation returned a finite value)"
    if keep_x:
        info['x_history'] = oracle.x_history
    return Result(np.asarray(x), float(f), oracle.nfev, list(oracle.f_history), info, message)


def _best_so_far(x0, fx0, oracle, keep_x, counters, reason):
    """Result of a run cut short: the best point so far (x0 if none is better) and the counters so far."""
    if oracle.best_x is None:
        x, f = x0, (fx0 if fx0 is not None else np.inf)
    elif fx0 is not None and fx0 <= oracle.best_f:
        x, f = x0, fx0
    else:
        x, f = oracle.best_x, oracle.best_f
    info = dict(counters or {}, stop_reason=reason)
    if keep_x:
        info['x_history'] = oracle.x_history
    return Result(np.asarray(x), float(f), oracle.nfev, list(oracle.f_history), info, reason)
