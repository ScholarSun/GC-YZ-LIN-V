"""The bounded-Hessian least-squares refit behind the ``hessian_cap`` option (``Model._cap_hessian``). It needs
``cvxpy`` (optional dependency, ``pip install 'gcyz[rescue]'``).

The cap is checked on every model about to be served (``Model.fit``). When ||H||_2 <= hessian_cap the model is
served as it is and nothing here runs. Otherwise the served model is replaced, by one of two models that both
satisfy the bound:

* the **bounded least-squares refit** (``fit_least_squares_capped``): exact interpolation on Y, least squares on
  Z, ||H||_2 <= K, a semidefinite program solved by cvxpy;
* the **linear interpolant through Y** with H = 0 (``model._linear_model``, or the same model in scaled arithmetic
  inside the refit).

The linear model is served instead, in this order of checks:

1. no cvxpy (``Model._cap_hessian``, warns once);
2. ``cap_refit_feasible`` refuses: n > 60 (large memory issues), a non-finite sample, or cond(Y) > 1e12
   (the solver does not converge); warns once with the reason;
3. before the solve: the spread of the values overflows (the +1e200 stand-in of a failed f(x0)), or the scaled cap
   K zs^2 / f_scale < 1e-7 (every admissible H is numerically zero; e.g. values >= 1e100 under the default cap);
4. after the solve: the solver raised or returned nothing, or its solution is not finite, exceeds the cap by more
   than 0.1 %, or misses a Y value by more than 1e-6 (scaled units); or a ``MemoryError``.
"""
import numpy as np


# Errors that are not solver failures and must propagate (Ctrl-C, interpreter exit).
_NOT_SOLVER_ERRORS = (KeyboardInterrupt, SystemExit, GeneratorExit)

# Limits above which the linear model is used without attempting the refit: its memory grows fast with n
# (~3 GB at n = 100, > 32 GB at n = 200), and the solver does not converge on ill-conditioned samples.
_CAP_REFIT_MAX_N = 60
_CAP_REFIT_MAX_COND = 1e12

# Refit tolerances. Below _CAP_MIN_SCALED the (scaled) cap is finer than the solver's precision, so the
# solver is skipped and the linear model (H = 0) is used. A returned H may exceed the cap by _CAP_EXCESS_RTOL
# and must match the Y values to _Y_INTERP_RTOL.
_CAP_MIN_SCALED = 1e-7
_CAP_EXCESS_RTOL = 1e-3
_Y_INTERP_RTOL = 1e-6


def have_cvxpy():
    """True when cvxpy can be imported."""
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return False
    return True


def _solvers(cp):
    """Solvers to try, in order: CLARABEL (comes with cvxpy), then MOSEK if installed."""
    return [cp.CLARABEL] + ([cp.MOSEK] if 'MOSEK' in cp.installed_solvers() else [])


def _scales(samp):
    """(f_scale, zs, spread_overflow): the standard deviation of the sample f values, the largest per-axis standard
    deviation of the sample points (each 1.0 when zero or not finite), and whether f_scale was not finite."""
    all_values = np.concatenate([samp.Z.values, samp.Y.values, [samp.fc]])
    with np.errstate(over='ignore', invalid='ignore'):
        f_scale = np.std(all_values)
    spread_overflow = not np.isfinite(f_scale)
    if f_scale == 0 or not np.isfinite(f_scale):
        f_scale = 1.0
    z_scale = np.std(np.vstack([samp.Z.points, samp.Y.points]), axis=0)
    z_scale[z_scale == 0] = 1.0
    return f_scale, float(np.max(z_scale)), spread_overflow


def cap_refit_feasible(samp):
    """(ok, reason): whether to attempt the refit (case 2 of the module docstring). Refuses n > _CAP_REFIT_MAX_N,
    a non-finite sample, and cond(Y) > _CAP_REFIT_MAX_COND."""
    if samp.n > _CAP_REFIT_MAX_N:
        return False, f"n = {samp.n} > {_CAP_REFIT_MAX_N}"
    Y = samp.Y.points
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(samp.Y.values)) and np.isfinite(samp.fc)
            and np.all(np.isfinite(samp.Z.points)) and np.all(np.isfinite(samp.Z.values))):
        return False, "non-finite sample"
    if len(Y):
        sv = np.linalg.svd(Y, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        if not np.isfinite(cond) or cond > _CAP_REFIT_MAX_COND:
            return False, f"cond(Y) = {cond:.1e} > {_CAP_REFIT_MAX_COND:g}"
    return True, ""


def fit_least_squares_capped(samp, K):
    """Least-squares model with a bounded Hessian (option ``hessian_cap``)::

        min   sum_{z in Z} (m(z) - f(z))^2
        s.t.  m(y) = f(y) for every y in Y,    ||H||_2 <= K,

    where m(x) = f(centre) + g.x + 0.5 x'Hx, points are relative to the centre, and
    with Z empty the objective is ||H||_F. Solved on the scaled data of _scales, where
    the cap becomes ||H_s||_2 <= K zs^2 / f_scale.

    Falls back to the linear interpolant through Y with H = 0 in cases outlined by module docstring.

    Returns (g, H, used_linear).
    """
    import cvxpy as cp
    n = samp.n
    f_scale, zs, spread_overflow = _scales(samp)
    K_s = float(K) * zs * zs / f_scale

    fc_s = samp.fc / f_scale
    Z_s, Zv_s = samp.Z.points / zs, samp.Z.values / f_scale
    Y_s, Yv_s = samp.Y.points / zs, samp.Y.values / f_scale
    # linear interpolant through Y in scaled units (model._linear_model computes the same model unscaled, for the
    # paths that skip the refit; they agree to rounding)
    g_lin = np.linalg.lstsq(Y_s, Yv_s - fc_s, rcond=None)[0]

    def linear_fallback():
        return g_lin * (f_scale / zs), np.zeros((n, n)), True

    if spread_overflow:
        # Checks for blow-ups (fvals with 1e200, indicative of NaN's or overflows) and calls linear model
        return linear_fallback()
    if not np.isfinite(K_s) or K_s < _CAP_MIN_SCALED:
        # every H the solver could tell from zero already violates the cap
        return linear_fallback()

    # Divide the objective by its value at the linear model (a feasible point), so it is at most 1 at the optimum.
    # Same minimiser; without it a tight cap can make the objective so large that the solver wrongly reports
    # infeasibility.
    obj_scale = float(np.sum((Z_s @ g_lin - (Zv_s - fc_s)) ** 2)) if len(Z_s) else 1.0
    obj_scale = obj_scale if np.isfinite(obj_scale) and obj_scale > 0 else 1.0

    g_s = cp.Variable(n)
    H_s = cp.Variable((n, n), symmetric=True)

    def change(P):
        """g.p + 0.5 p'Hp for every row p of P (affine in g and H)."""
        return P @ g_s + 0.5 * cp.sum(cp.multiply(P @ H_s, P), axis=1)

    # ||H||_2 <= K_s as two n-by-n constraints (largest and smallest eigenvalue) instead of cvxpy's
    # norm(H, 2) <= K_s, which builds one 2n-by-2n constraint: same feasible set, 3-5x faster, less memory.
    constraints = [cp.lambda_max(H_s) <= K_s, cp.lambda_min(H_s) >= -K_s]
    if len(Y_s):
        constraints.append(change(Y_s) == Yv_s - fc_s)
    if len(Z_s):
        objective = cp.Minimize(cp.sum_squares(change(Z_s) - (Zv_s - fc_s)) / obj_scale)
    else:
        objective = cp.Minimize(cp.norm(H_s, 'fro'))
    problem = cp.Problem(objective, constraints)

    # Move to the next solver only if this one raises or returns nothing; the first solution returned is checked
    # and, if it fails, replaced by the linear model.
    for solver in _solvers(cp):
        try:
            problem.solve(solver=solver, verbose=False)
        except BaseException as exc:
            if isinstance(exc, _NOT_SOLVER_ERRORS):
                raise
            continue
        if g_s.value is None:
            continue
        g = np.asarray(g_s.value, dtype=float) * (f_scale / zs)
        H = np.asarray(H_s.value, dtype=float) * (f_scale / (zs * zs))
        ok = np.all(np.isfinite(g)) and np.all(np.isfinite(H)) and \
            float(np.linalg.norm(H, 2)) <= float(K) * (1 + _CAP_EXCESS_RTOL)
        if ok and len(Y_s):
            # the solution must also reproduce the Y values to solver accuracy (scaled units)
            target = Yv_s - fc_s
            Hs_val = np.asarray(H_s.value, dtype=float)
            resid = Y_s @ np.asarray(g_s.value, dtype=float) + 0.5 * np.sum((Y_s @ Hs_val) * Y_s, axis=1) - target
            ok = float(np.max(np.abs(resid))) <= _Y_INTERP_RTOL * max(1.0, float(np.max(np.abs(target))))
        return (g, H, False) if ok else linear_fallback()
    return linear_fallback()
