"""Counting wrapper around a black-box objective."""
import math
import numbers

import numpy as np

_BIG = 1e200


def _standin(fx):
    """A real value as the solver sees it: NaN -> +1e200, +inf -> +1e200, -inf -> -1e200."""
    if math.isnan(fx):
        return _BIG
    if math.isinf(fx):
        return math.copysign(_BIG, fx)
    return fx


def is_failure(fx):
    """True for a failed evaluation as the solver sees it: NaN or +inf, recorded as +1e200 (any value that
    large counts). The solver never interpolates such a value."""
    return not fx < _BIG


def _real_value(fx):
    """The objective's value as a Python float. A real number (Python or NumPy scalar) or a one-element real
    array is accepted; anything else (None, a string, a complex number, an array with several elements)
    raises TypeError."""
    if isinstance(fx, (bool, np.bool_)):
        raise TypeError("the objective returned a boolean, not a number")
    if isinstance(fx, numbers.Real):
        return float(fx)
    try:
        a = np.asarray(fx)
    except Exception:
        a = None
    if a is not None and a.size == 1 and a.dtype.kind in 'biuf':
        return float(a.reshape(()))
    shape = f" of shape {a.shape}" if a is not None and a.ndim else ""
    raise TypeError(f"the objective must return a real number, got {type(fx).__name__}{shape}")


class BudgetExhausted(Exception):
    """Raised by :class:`Oracle` when a call would exceed ``max_nfev``: the solver ends the run cleanly."""


class ObjectiveUnbounded(Exception):
    """Raised by :class:`Oracle` after recording a value of -inf: the objective is unbounded below at that point,
    nothing can improve on it, and a model through the value -1e200 would be meaningless; the solver ends the run
    there and returns that point."""


class Oracle:
    """Wraps ``fun`` so that every call is counted and its value recorded, and enforces the evaluation
    budget: a call that would exceed ``max_nfev`` raises :class:`BudgetExhausted` instead of calling ``fun``.

    ``fun`` is called on a copy of the point, so it cannot change the caller's or the solver's arrays. It
    must return a real number (a Python or NumPy scalar, or a one-element real array); anything else raises
    TypeError, which is handled like an error raised by ``fun`` itself. Values are recorded as Python floats.
    Non-finite values are sanitised so that the solver never sees them, and are recorded that way:
    NaN -> +1e200, +inf -> +1e200, -inf -> -1e200. The solver treats +1e200 (see :func:`is_failure`) as a
    failed evaluation and never interpolates it. Floating-point warnings raised inside ``fun`` are
    suppressed.
    """

    def __init__(self, fun, keep_x=False, max_nfev=None):
        if not callable(fun):
            raise TypeError("fun must be callable")
        self.fun = fun
        self.keep_x = keep_x
        self.max_nfev = None if max_nfev is None else int(max_nfev)
        self.nfev = 0
        self.f_history = []
        self.x_history = []
        self.best_f = np.inf
        self.best_x = None
        self._in_fun = False  # True while fun runs: an exception escaping then came from fun
        self.n_raised = 0  # calls in which fun raised: they count against the budget but leave no value

    def __call__(self, x):
        if self.max_nfev is not None and self.nfev + self.n_raised >= self.max_nfev:
            raise BudgetExhausted(f"the evaluation budget of {self.max_nfev} is exhausted")
        x = np.array(x, dtype=float).ravel()  # the point evaluated, recorded as given
        self._in_fun = True
        try:
            with np.errstate(all="ignore"):
                # fun gets its own copy; a return value that is not a real number raises TypeError here,
                # while _in_fun is set, so it is reported like an error raised by fun
                fx = _real_value(self.fun(x.copy()))
        except BaseException:
            self.n_raised += 1  # _in_fun stays set: the caller reads it to attribute the error to fun
            raise
        self._in_fun = False
        fx = _standin(fx)
        self.nfev += 1
        self.f_history.append(fx)
        if self.keep_x:
            self.x_history.append(x.copy())
        if fx < self.best_f and not is_failure(fx):  # a failed evaluation is never the best point
            self.best_f, self.best_x = fx, x.copy()
        if fx <= -_BIG:
            raise ObjectiveUnbounded("the objective returned -inf (or a value at or below -1e200): unbounded below "
                                     "at the point evaluated")
        return fx

    def get_evaluation_count(self):
        """Number of evaluations so far (``nfev``)."""
        return self.nfev

    @property
    def in_fun(self):
        """True while ``fun`` runs, and after it raised: an exception escaping at that moment came from ``fun``
        itself (``clear_error`` resets it when the oracle is reused after such an error)."""
        return self._in_fun

    def clear_error(self):
        """Reset the in-``fun`` flag left set by an error raised by ``fun``."""
        self._in_fun = False
