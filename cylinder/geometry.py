"""Engine geometry — bore, stroke, con-rod, V(θ), dV/dθ, surface area.

Source: 1d/engine_simulator/engine/geometry.py  (read-only V1 file)
Copy date: 2026-04-13

Changes vs V1:
- Ported to @njit free functions (no class); cylinder integrator is
  Python-level but calls these per-step.
- Head + piston surface area exposed as a free function.

Formulas (standard slider-crank):
    r = stroke/2,  l = con_rod_length,  V_c = V_d / (CR - 1)
    θ_r = radians(θ − 180°)  (piston at TDC when θ = 0 or 720° CAD)
    s = r·cos(θ_r) + √(l² − r²·sin²(θ_r))   (piston position from crank)
    x_piston = r + l − s                     (measured from TDC)
    V(θ) = V_c + A_bore · x_piston
    dV/dθ = A_bore · dx/dθ
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def cylinder_volume(
    theta_deg: float, bore: float, stroke: float, con_rod: float, CR: float,
) -> float:
    """Instantaneous cylinder volume (m³) at crank angle θ_deg (degrees CAD)."""
    A_bore = 0.25 * np.pi * bore * bore
    V_d = A_bore * stroke
    V_c = V_d / (CR - 1.0)
    r = 0.5 * stroke
    l = con_rod
    # Convention: θ=0 and θ=720 at TDC firing; piston at BDC at θ=180 and 540.
    theta_r = np.radians(theta_deg)
    sin_sq = np.sin(theta_r) ** 2
    s = r * np.cos(theta_r) + np.sqrt(max(l * l - r * r * sin_sq, 0.0))
    x_from_tdc = r + l - s
    return V_c + A_bore * x_from_tdc


@njit(cache=True, fastmath=False)
def cylinder_dVdtheta(
    theta_deg: float, bore: float, stroke: float, con_rod: float,
) -> float:
    """dV/dθ in m³ per degree (divide by omega in rad/s to get dV/dt)."""
    A_bore = 0.25 * np.pi * bore * bore
    r = 0.5 * stroke
    l = con_rod
    theta_r = np.radians(theta_deg)
    sin_theta = np.sin(theta_r)
    cos_theta = np.cos(theta_r)
    inner = max(l * l - r * r * sin_theta * sin_theta, 1e-20)
    sqrt_inner = np.sqrt(inner)
    # ds/dθ_r (m per rad)
    ds_dtheta_r = -r * sin_theta - (r * r * sin_theta * cos_theta) / sqrt_inner
    # dV/dx_piston = A_bore, dx/dθ_r = -ds/dθ_r, convert rad → deg
    return A_bore * (-ds_dtheta_r) * (np.pi / 180.0)


@njit(cache=True, fastmath=False)
def cylinder_surface_area(
    theta_deg: float, bore: float, stroke: float, con_rod: float, CR: float,
) -> float:
    """Approximate total internal surface area at θ: head + piston + liner."""
    A_bore = 0.25 * np.pi * bore * bore
    V_d = A_bore * stroke
    V_c = V_d / (CR - 1.0)
    V = cylinder_volume(theta_deg, bore, stroke, con_rod, CR)
    # Liner surface = circumference × exposed stroke length
    liner_len = max((V - V_c) / A_bore, 0.0)
    A_liner = np.pi * bore * liner_len
    # Flat head + flat piston
    return 2.0 * A_bore + A_liner
