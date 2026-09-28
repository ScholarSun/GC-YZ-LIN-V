"""Options: defaults, the two variants, and validation. Every option is described in the README (Options table)."""
import ast
import math
import numbers

import numpy as np

DEFAULT_OPTIONS = {
    # --- trust region (Delta) ---
    'tr_delta': 0.5,
    'tr_toaccept': 0.01,
    'tr_toexpand': 5e-9,
    'tr_expand': 1.3,
    'tr_shrink': 0.8,
    # --- resolution ladder (rho) ---
    'rho_shrink': 0.1,
    'rhoend': 1e-12,
    'small_step_gate': 0.5,
    'gate_far_mult': 5.0,
    # --- sample and geometry ---
    'sample_max': '2*n+1',
    'big_lambda': 1000.0,
    'sc_lambda': 2.0,
    'self_correcting': True,
    'init_frame': 'identity',
    # --- model ---
    'model': 'hedge',
    'hedge_beta': 0.8,
    'hedge_hysteresis': 0.8,
    'hedge_warmup': 1,
    'hessian_cap': 1e100,
    'hedge_chain_max_err': None,
    'mcfn_scale_reseed': 0.0,
    # --- GC-YZ-V admission and ladder guards ---
    'newuoa_admission': False,
    'nwa_margin': 0.0,
    'nwa_no_discard': False,
    'rho_advance_far_gate': 0.0,
    'rho_advance_lambda_gate': 0.0,
    # --- stopping ---
    'stop_nfeval': 2000,
    'stop_iter': 100000,
    'verbosity': 0,
}

# The two variants: the options each one changes from DEFAULT_OPTIONS.
GC_YZ_LIN = {}
GC_YZ_V = {
    'newuoa_admission': True,
    'nwa_margin': 0.25,
    'nwa_no_discard': True,
    'mcfn_scale_reseed': 2.0,
    'hedge_chain_max_err': 1.0,
    'rho_advance_far_gate': 10.0,
    'rho_advance_lambda_gate': 1000.0,
    'gate_far_mult': 0.0,
}

# ---- allowed values ----
# Text options: the allowed values.
_CHOICES = {
    'model': ('mfn', 'chain', 'hedge'),
    'init_frame': ('identity', 'random'),
}
# Real-valued options: (low, high, low included, high included). These are the paper's ranges, plus 0 as an "off"
# value for some options.
_RANGES = {
    'tr_delta': (0.0, math.inf, False, False),
    'tr_toaccept': (0.0, 1.0, False, False),
    'tr_toexpand': (0.0, math.inf, True, False),
    'tr_expand': (1.0, math.inf, True, False),
    'tr_shrink': (0.0, 1.0, False, False),
    'rho_shrink': (0.0, 1.0, False, False),
    'rhoend': (0.0, math.inf, False, False),
    'small_step_gate': (0.0, 1.0, True, True),
    'gate_far_mult': (0.0, math.inf, True, False),
    'big_lambda': (1.0, math.inf, False, True),
    'sc_lambda': (1.0, math.inf, True, True),
    'hedge_beta': (0.0, 1.0, True, True),
    'hedge_hysteresis': (0.0, math.inf, False, False),
    'hessian_cap': (0.0, math.inf, True, False),
    'hedge_chain_max_err': (0.0, math.inf, False, True),
    'mcfn_scale_reseed': (0.0, math.inf, True, False),
    'nwa_margin': (-1.0, math.inf, False, False),
    'rho_advance_far_gate': (0.0, math.inf, True, False),
    'rho_advance_lambda_gate': (0.0, math.inf, True, True),
}
# Options for which None also means off.
_NONE_MEANS_OFF = ('hessian_cap', 'hedge_chain_max_err')
# Options that are 0 (off) or else at least / greater than a minimum: (minimum, minimum allowed).
_ZERO_OR_ABOVE = {
    'gate_far_mult': (1.0, True),
    'mcfn_scale_reseed': (1.0, False),
}
_COUNTS = ('hedge_warmup', 'stop_nfeval', 'stop_iter', 'verbosity')  # whole numbers >= 0
_FLAGS = ('self_correcting', 'newuoa_admission', 'nwa_no_discard')


def validate_options(options, n=None):
    """Raise ValueError for an unknown option name or an invalid value. With the dimension n, also check
    sample_max (which may be a formula in n)."""
    for key in options:
        if key not in DEFAULT_OPTIONS:
            raise ValueError(f"{key!r} is not a valid option name.")
    get = lambda key: options.get(key, DEFAULT_OPTIONS[key])
    for key, allowed in _CHOICES.items():
        if get(key) not in allowed:
            raise ValueError(f"{key} must be {', '.join(map(repr, allowed[:-1]))} or {allowed[-1]!r}, "
                             f"got {get(key)!r}")
    for key, (lo, hi, lo_in, hi_in) in _RANGES.items():
        v = get(key)
        if v is None and key in _NONE_MEANS_OFF:
            continue
        x = float(v) if isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) else math.nan
        if not ((x > lo or (lo_in and x == lo)) and (x < hi or (hi_in and x == hi))):
            interval = f"{'[' if lo_in else '('}{lo:g}, {hi:g}{']' if hi_in else ')'}"
            raise ValueError(f"{key} must be a real number in {interval}, got {v!r}")
    for key in _COUNTS:
        v = get(key)
        if isinstance(v, (bool, np.bool_)) or \
                not (isinstance(v, numbers.Real) and math.isfinite(v) and float(v).is_integer() and v >= 0):
            raise ValueError(f"{key} must be a whole number >= 0, got {v!r}")
    for key, (lo, lo_in) in _ZERO_OR_ABOVE.items():
        x = float(get(key))
        if 0.0 < x < lo or (x == lo and not lo_in):
            raise ValueError(f"{key} must be 0 (off) or {'at least' if lo_in else 'greater than'} {lo:g}, "
                             f"got {get(key)!r}")
    for key in _FLAGS:
        if not isinstance(get(key), (bool, np.bool_)):
            raise ValueError(f"{key} must be True or False, got {get(key)!r}")
    if not float(get('rho_shrink')) < float(get('tr_shrink')):
        raise ValueError(f"rho_shrink must be smaller than tr_shrink, got {get('rho_shrink')!r} >= "
                         f"{get('tr_shrink')!r}")
    if float(get('rhoend')) > float(get('tr_delta')):
        raise ValueError(f"rhoend must not exceed tr_delta, got {get('rhoend')!r} > {get('tr_delta')!r}")
    if n is not None:
        if n < 1:
            raise ValueError("x0 must have at least one element")
        v = get('sample_max')
        try:
            sample_max = _resolve(v, n)
            whole, why = float(sample_max).is_integer(), ""
        except Exception as exc:  # a malformed formula or a non-number
            whole, why = False, f": {exc}"
        if not whole:
            raise ValueError(f"sample_max must be a whole number or a formula in n giving one, got {v!r}{why}")
        sample_max = max(sample_max, n + 1)
        if sample_max > (n + 1) * (n + 2) // 2:
            raise ValueError(f"sample_max must not exceed (n + 1)(n + 2) / 2 = {(n + 1) * (n + 2) // 2}, the number of "
                             f"points a quadratic interpolates, got {sample_max}")


_FORMULA_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                  ast.Mod, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)


def _resolve(v, n):
    """Value of sample_max: an integer, or a formula in n such as '2*n+1'. A formula may only use n, integers
    and + - * / // % ( ) (checked before it is evaluated, so it can never run code) and must give a whole number."""
    if not isinstance(v, str):
        return v
    tree = ast.parse(v.strip(), mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _FORMULA_NODES) or (isinstance(node, ast.Name) and node.id != 'n') or \
                (isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, int))):
            raise ValueError(f"size formula {v!r} may only use n, integers and + - * / // % ( )")
    val = eval(compile(tree, '<size formula>', 'eval'), {"__builtins__": {}}, {"n": n})
    if not float(val).is_integer():
        raise ValueError(f"size formula {v!r} gives {val!r} at n = {n}, not a whole number")
    return int(val)
