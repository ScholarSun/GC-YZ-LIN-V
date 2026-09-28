"""
Solvers for the trust-region subproblem: minimise the model g.s + 0.5 s'Hs over the ball ||s|| <= delta.
trust_sub_exact solves it exactly (the default); trust_sub_CG is a cheaper approximate backup.

Author: Anahita Hassanzadeh (Anahita.Hassanzadeh@gmail.com), later modified by Liyuan Cao (liyuancao7@gmail.com),
and modified further for this package.
"""

from scipy import linalg as LA
import numpy as np

def _boundary_alpha(s, p, delta):
    """How far to move from s along p to reach the trust-region boundary: alpha >= 0 with ||s + alpha p|| = delta.
    If p is so large or small that p.p overflows or underflows, p is rescaled first. Returns 0 for a non-finite p."""
    a = np.dot(p, p)
    if np.isfinite(a) and a >= np.finfo(float).tiny:
        b = np.dot(s, p) * 2
        c = np.dot(s, s) - np.float64(delta) ** 2
        return (-b + (b**2 - 4*a*c)**0.5) / 2 / a
    pm = float(np.max(np.abs(p)))
    if not np.isfinite(pm) or pm == 0.0:
        return 0.0
    q = p / pm
    a = np.dot(q, q)
    b = np.dot(s, q) * 2
    c = np.dot(s, s) - np.float64(delta) ** 2
    return (-b + max(b**2 - 4*a*c, 0.0)**0.5) / 2 / a / pm


def trust_sub_CG(H, g, delta):
    """
    Approximate solution by truncated conjugate gradients. Stops when it hits the boundary,
    meets negative curvature, or the residual drops below 1e-5.
    Input: g (n,) model gradient, H (n, n) model Hessian, delta radius.
    Output: (s, val) with s the step and val = [g.s + 0.5 s'Hs] (a 1-element array).

    Written by Liyuan Cao @Lehigh University in October 2017.
    """
    n = g.shape[0]

    s = np.zeros(n)
    if not np.any(g):
        # zero gradient: no direction to start from, so return the zero step
        return (s, np.zeros(1))
    r = np.dot(H, s) + g
    p = -r

    for k in range(n):
        temp = np.dot( np.dot(p.reshape(1,n), H), p)
        if temp <= 0:
            alpha = _boundary_alpha(s, p, delta)
            s += alpha * p
            break
        else:
            alpha = np.dot(r, r) / temp

        if np.linalg.norm(s + alpha * p) > delta:
            alpha = _boundary_alpha(s, p, delta)
            s += alpha * p
            break
        else:
            s += alpha * p
            r0 = r.copy()   # keep the old residual for beta (r changes in place next)
            r += alpha * np.dot(H, p)

        if np.linalg.norm(r) < 1e-5:
            break
        else:
            p = -r + np.dot(r, r) / np.dot(r0, r0) * p

    val = np.dot(g, s) + np.dot( np.dot(s.reshape(1,n), H), s) / 2

    return (s, val)

def _secular_eqn(lambda_0, eigval, alpha, delta):
    """
    The secular equation 1/delta - 1/||s(lambda)|| at each of the points lambda_0. Its root gives the lambda
    that puts the step on the boundary.
    """
    m = lambda_0.size
    n = len(eigval)
    unn = np.ones((n, 1))
    unm = np.ones((m, 1))
    M = np.dot(eigval, unm.T) + np.dot(unn, lambda_0.T)
    MC = M.copy()
    MM = np.dot(alpha, unm.T)
    # infinities here are expected (a pole, or overflow on a huge model): they give ||s|| = inf
    with np.errstate(over='ignore', divide='ignore'):
        M[M != 0.0] = MM[M != 0.0] / M[M != 0.0]
        M[MC == 0.0] = np.inf * np.ones(MC[MC == 0.0].size)
        M = M*M
        value = np.sqrt(unm / np.dot(M.T, unn))

    if len(value[np.where(value == np.inf)]):
        inf_arg = np.where(value == np.inf)
        value[inf_arg] = np.zeros((len(value[inf_arg]), 1))

    value = (1.0/delta) * unm - value

    return value


def rfzero(x, itbnd, eigval, alpha, delta, tol):
    """
    Finds a root of the secular equation to the RIGHT of x: first doubles a step until the sign changes,
    then narrows the bracket (bisection or interpolation, as in MATLAB's fzero). Returns (root, other
    bracket end, number of evaluations).
    """
    itfun = 0
    # Step size for the bracket search, in the problem's own units: the root lies below x + ||alpha|| / delta,
    # so steps of that size reach it quickly at any scale of f and x. (||alpha|| is estimated from max|alpha_i|
    # if the norm overflows.)
    x0 = abs(float(np.asarray(x, dtype=float).reshape(-1)[0]))
    na = float(np.linalg.norm(alpha))
    if not np.isfinite(na):
        na = float(np.max(np.abs(alpha))) * np.sqrt(alpha.size)
    unit = max(x0, na / delta, float(np.max(np.abs(eigval))), 0.0)
    if unit == 0.0:
        unit = 1.0
    elif not np.isfinite(unit):
        unit = np.finfo(float).max / 8.0
    # never start the doubling from a rounding-level |x| / 2, or it runs out of iterations
    dx = max(abs(x) / 2, 0.5 * unit)

    a = x
    c = a
    fa = _secular_eqn(a, eigval, alpha, delta)
    fa = float(fa.item())
    itfun = itfun + 1

    b = x + unit   # first bracket point; then x + 2dx, x + 4dx, ...
    fb = _secular_eqn(b, eigval, alpha, delta)
    fb = float(fb.item())
    itfun = itfun + 1

    # double the step until the sign changes
    while ((fa > 0) == (fb > 0)):

        dx = 2*dx

        b = x + dx
        fb = _secular_eqn(b, eigval, alpha, delta)
        fb = float(fb.item())
        itfun = itfun + 1

        if (itfun > itbnd):
            break

    fc = fb

    # narrow the bracket
    while (fb != 0):
        # keep b = best point so far, a = previous b, c = on the other side of the root
        if (fb > 0) == (fc > 0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # stop?
        if itfun > itbnd:
            break

        m = 0.5 * (c-b)
        # bracket width tolerance, relative to the root itself
        rel_tol = 4.0 * np.finfo(float).eps * max(abs(b), 1e-300)

        if (abs(m) <= rel_tol) or (abs(fb) < tol):
            break

        # bisection or interpolation
        if (abs(e) < rel_tol) or (abs(fa) <= abs(fb)):
            # bisection
            d = e = m
        else:
            # interpolation
            s = float(fb)/fa
            if a == c:
                # linear interpolation
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # inverse quadratic interpolation
                q = float(fa)/fc
                r = float(fb)/fc
                p = s * (2.0 * m * q * (q-r) - (b-a) * (r-1.0))
                q = (q-1.0) * (r-1.0) * (s-1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            # use the interpolated point only if it stays well inside the bracket
            if (2.0*p < 3.0*m*q - abs(rel_tol*q)) and (p < abs(0.5*e*q)):
                e = d
                d = float(p.item())/q
            else:
                d = m
                e = m

        # next point
        a = b
        fa = fb
        if (abs(d) > rel_tol):
            b = b + d
        else:
            if b > c:
                b = b - rel_tol
            else:
                b = b + rel_tol

        fb = _secular_eqn(b, eigval, alpha, delta)
        fb = float(fb.item())
        itfun = itfun + 1

    return (b, c, itfun)

def trust_sub_exact(H, g, delta, eig=None):
    """
    Exact solution of
        min g's + 0.5 s'Hs  subject to  ||s|| <= delta
    for any symmetric H (including zero, indefinite and the "hard case"), using the eigendecomposition of H.
    Returns (s, g's + 0.5 s'Hs).
    eig: optional precomputed (D, V) = eigh(0.5 (H + H')), eigenvalues ascending; skips the eigendecomposition.
    """

    g = g.reshape(g.shape[0],1)
    delta = float(delta)
    if not delta > 0:
        # empty ball (the solver never asks for one): zero step
        return np.zeros(g.shape[0]), 0.0
    if delta < 1e-100 or delta > 1e100:
        # extreme radius: solve at radius 1 with g / delta and scale back, so delta^2 does not underflow
        # or overflow
        u, v = trust_sub_exact(H, g / delta, 1.0, eig=eig)
        d = np.float64(delta)
        return np.asarray(u, dtype=float).reshape(-1) * d, float(v * d * d)
    # accept a root when ||s|| is within about 1e-7 (relative) of delta
    tol_seqeq = 1e-7
    # the root search itself stops much tighter (about 1e-11 relative in ||s||)
    tol = 1e-4 * tol_seqeq / delta
    # key: which case we are in. 0 = not decided yet, 1 = Newton step inside the ball, 2 = step on the boundary,
    # 3-5 = hard case (no usable root was found)
    key = 0
    itbnd = 200      # max secular-equation evaluations (each is cheap)
    n = len(g)
    coeff = np.zeros((n, 1))

    H = np.atleast_2d(np.squeeze(np.asarray(H)))

    # eigendecomposition of H
    if eig is None:
        D, V = LA.eigh(0.5 * (H.T + H))
    else:
        D, V = eig
    eigval = D[np.newaxis].T
    jmin = np.argmin(eigval)
    mineig = np.amin(eigval)

    alpha = np.dot(-V.T, g)
    sig = (np.sign(alpha[jmin]) + (alpha[jmin] == 0).sum())[0]

    # H positive definite: try the Newton step
    if mineig > 0:
        coeff = alpha * (1/eigval)
        s = np.dot(V, coeff)
        nrms = LA.norm(s)
        if nrms <= delta:
            # Newton step inside the ball: done
            key = 1
        else:
            laminit = np.array([[0]])
    else:
        laminit = -mineig

    # otherwise: find the boundary step
    if key == 0:
        if _secular_eqn(laminit, eigval, alpha, delta) > 0:
            b, _, _ = rfzero(laminit, itbnd, eigval, alpha, delta, tol)

            if abs(_secular_eqn(b, eigval, alpha, delta)) <= tol_seqeq / delta:
                lambda_0 = b
                key = 2
                lam = lambda_0 * np.ones((n, 1))

                coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

                if (nrms > (1.0 + 1e-6) * delta or nrms < (1.0 - 1e-6) * delta):
                    # the root does not put the step on the boundary: hard case
                    key = 5
                    lambda_0 = -mineig
            else:
                key = 3
                lambda_0 = -mineig
        else:
            key = 4
            lambda_0 = -mineig

        lam = lambda_0 * np.ones((n, 1))

        if key > 2:
            # drop the gradient components along eigenvalues equal to -lambda (up to rounding relative to ||H||)
            hscale = max(float(np.max(np.abs(eigval))), np.finfo(float).tiny)
            arg = abs(eigval + lam) < 10 * np.finfo(float).eps * hscale
            alpha[arg] = 0.0

        coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

        if key > 2 and nrms < delta:
            # hard case: extend the step to the boundary along the eigenvector of the smallest eigenvalue
            beta = np.sqrt(np.float64(delta) ** 2 - nrms**2)
            s = s + np.dot(beta, np.dot(sig, V[:, jmin])).reshape(n, 1)

        if key > 2 and nrms > delta:
            b, _, _ = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
            lambda_0 = b
            lam = lambda_0 * np.ones((n, 1))

            coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

    # predicted change of the model
    val = np.dot(g.T, s) + 0.5 * np.dot(s.T, np.dot(H,s))

    return (s[:,0], val[0][0])

def compute_step(alpha, eigval, coeff, V, lam):
    """The step s(lambda) = -(H + lambda I)^-1 g in the eigenbasis. Returns (coeff, s, ||s||, eigval + lambda)."""
    w = eigval + lam
    arg1 = np.logical_and(w == 0, alpha == 0)
    arg2 = np.logical_and(w == 0, alpha != 0)
    coeff[w != 0] = alpha[w != 0] / w[w != 0]
    coeff[arg1] = 0
    coeff[arg2] = np.inf
    coeff[np.isnan(coeff)] = 0
    s = np.dot(V, coeff)
    nrms = LA.norm(s)
    return(coeff, s, nrms, w)
