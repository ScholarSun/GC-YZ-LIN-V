"""GC-YZ: geometry-correcting derivative-free trust-region optimisation.

>>> import numpy as np, gcyz
>>> res = gcyz.minimize(lambda x: float(np.sum((x - 1) ** 2)), np.zeros(5), budget=300)  # GC-YZ-LIN
>>> res.fun < 1e-10
True
"""
from .solver import GCYZ, minimize, Result, GC_YZ_LIN, GC_YZ_V, DEFAULT_OPTIONS
from .oracle import Oracle, BudgetExhausted

__version__ = "1.1.0"  # the single source of the version: pyproject.toml reads it
__all__ = ["GCYZ", "minimize", "Result", "GC_YZ_LIN", "GC_YZ_V", "DEFAULT_OPTIONS", "Oracle", "BudgetExhausted",
           "__version__"]
