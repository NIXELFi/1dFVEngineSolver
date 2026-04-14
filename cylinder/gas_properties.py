"""Gas properties — burned/unburned γ(T) and mixture R(xb), all SI, all @njit.

Source: 1d/engine_simulator/gas_dynamics/gas_properties.py  (read-only V1 file)
Copy date: 2026-04-13

Changes vs V1:
- Ported from module-level numpy-array-vectorised functions to @njit scalar
  free functions callable from V2 @njit kernels (valve BC ghost-cell fill).
  V1's use of np.clip / np.asarray is replaced with scalar min/max.
- Dropped all Benson non-dimensionalization helpers (V2 is SI throughout).
- Dropped friction and heat-transfer correlations from this file; they live
  in solver/sources.py (also ported from V1 with their own header).

V1 coefficients (preserved verbatim):
    R_AIR    = 287.0   J/(kg·K)
    R_BURNED = 295.0   J/(kg·K)  (approximate stoich gasoline-air)
    γ_unburned(T) = 1.38 − 1.2e-4 · clamp(T, 300..900) offset 300
    γ_burned(T)   = 1.30 − 8.0e-5 · clamp(T, 300..3000) offset 300
    γ_mix(T, xb)  = (1-xb)·γ_unburned + xb·γ_burned
    R_mix(xb)     = (1-xb)·R_AIR + xb·R_BURNED

These polynomials are JIT-friendly as written; no table lookups.
"""

from __future__ import annotations

import numpy as np
from numba import njit


R_AIR = 287.0       # J/(kg·K), dry air
R_BURNED = 295.0    # J/(kg·K), burned stoichiometric gasoline-air approx


@njit(cache=True, fastmath=False)
def gamma_unburned(T: float) -> float:
    """γ(T) for unburned air-fuel mixture, valid 300..900 K."""
    T_clamped = T
    if T_clamped < 300.0:
        T_clamped = 300.0
    elif T_clamped > 900.0:
        T_clamped = 900.0
    return 1.38 - 1.2e-4 * (T_clamped - 300.0)


@njit(cache=True, fastmath=False)
def gamma_burned(T: float) -> float:
    """γ(T) for burned gas, valid 300..3000 K."""
    T_clamped = T
    if T_clamped < 300.0:
        T_clamped = 300.0
    elif T_clamped > 3000.0:
        T_clamped = 3000.0
    return 1.30 - 8.0e-5 * (T_clamped - 300.0)


@njit(cache=True, fastmath=False)
def gamma_mixture(T: float, x_b: float) -> float:
    """Mass-fraction weighted γ during combustion. x_b in [0, 1]."""
    if x_b <= 0.0:
        return gamma_unburned(T)
    if x_b >= 1.0:
        return gamma_burned(T)
    return (1.0 - x_b) * gamma_unburned(T) + x_b * gamma_burned(T)


@njit(cache=True, fastmath=False)
def R_mixture(x_b: float) -> float:
    """Mass-fraction weighted specific gas constant."""
    if x_b <= 0.0:
        return R_AIR
    if x_b >= 1.0:
        return R_BURNED
    return (1.0 - x_b) * R_AIR + x_b * R_BURNED


@njit(cache=True, fastmath=False)
def speed_of_sound(gamma: float, R: float, T: float) -> float:
    """a = sqrt(γ·R·T). Clamps T at 1 K to avoid sqrt of negatives on broken state."""
    if T < 1.0:
        T = 1.0
    return np.sqrt(gamma * R * T)
