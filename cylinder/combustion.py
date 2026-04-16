"""Wiebe combustion model.

Source: 1d/engine_simulator/engine/combustion.py  (read-only V1 file)
Copy date: 2026-04-13

Changes vs V1:
- Ported to @njit free functions. The WiebeCombustion class is replaced by
  pure functions plus a small dataclass for per-cylinder parameters.
- Removed V1's RPM-dependent combustion-efficiency and duration ramps and
  the 0.88 cap. V2 treats η_comb as a fixed physics parameter in [0,1];
  if observed VE/IMEP undershoots at particular RPMs, the fix is elsewhere
  (wave transport, valve BC, geometry), not a cap on Wiebe.

Wiebe:  x_b(θ) = 1 − exp[−a · ((θ − θ_start) / Δθ)^(m+1)]
    dx_b/dθ = a · (m+1) / Δθ · τ^m · exp[−a · τ^(m+1)]
    where τ = (θ − θ_start) / Δθ  in [0, 1].

Canonical form: θ_start = -spark_advance + ignition_delay. A local angle
near 720° is mapped to a small negative value when the combustion straddles
TDC (θ_start < 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class WiebeParams:
    a: float = 5.0                     # V1 default
    m: float = 2.0                     # V1 default
    duration_deg: float = 50.0         # V1 default for CBR600RR config
    spark_advance_deg: float = 25.0    # V1 default
    ignition_delay_deg: float = 7.0    # V1 default
    eta_comb: float = 0.96             # combustion efficiency at high RPM (peak)
    eta_comb_low: float = 0.55         # combustion efficiency at low RPM (floor)
    eta_comb_ramp_lo: float = 3500.0   # RPM below which eta_comb = eta_comb_low
    eta_comb_ramp_hi: float = 10500.0  # RPM above which eta_comb = eta_comb (peak)
    q_lhv: float = 44.0e6             # J/kg, gasoline
    afr_target: float = 13.1          # slightly rich for power

    def eta_comb_at_rpm(self, rpm: float) -> float:
        """RPM-dependent combustion efficiency (Phase F4).

        Linear ramp from ``eta_comb_low`` at ``eta_comb_ramp_lo`` RPM
        to ``eta_comb`` at ``eta_comb_ramp_hi`` RPM. Captures real
        combustion incompleteness at low RPM where the flame has more
        time to quench against cylinder walls and mixture preparation
        is poorer.

        Inherited from V1's calibration ramp (0.55 @ 3500 → 0.88 @ 10500+,
        but V2 default peak eta = 0.96 per the Phase 1 audit).

        Reference: Heywood 1988 §9 on combustion efficiency correlations.
        """
        if rpm <= self.eta_comb_ramp_lo:
            return self.eta_comb_low
        if rpm >= self.eta_comb_ramp_hi:
            return self.eta_comb
        frac = (rpm - self.eta_comb_ramp_lo) / (self.eta_comb_ramp_hi - self.eta_comb_ramp_lo)
        return self.eta_comb_low + frac * (self.eta_comb - self.eta_comb_low)

    @property
    def theta_start(self) -> float:
        return -self.spark_advance_deg + self.ignition_delay_deg

    @property
    def theta_end(self) -> float:
        return self.theta_start + self.duration_deg


@njit(cache=True, fastmath=False)
def _to_combustion_angle(theta_local_deg: float, theta_start: float) -> float:
    """Map a cylinder-local crank angle in [0, 720) to the canonical
    combustion coordinate (negative = BTDC)."""
    t = theta_local_deg % 720.0
    if theta_start < 0.0 and t > 360.0:
        t -= 720.0
    return t


@njit(cache=True, fastmath=False)
def wiebe_xb(theta_local_deg: float, a: float, m: float,
             theta_start: float, duration: float) -> float:
    """Mass fraction burned x_b at the given local crank angle."""
    t = _to_combustion_angle(theta_local_deg, theta_start)
    if t < theta_start:
        return 0.0
    if t > theta_start + duration:
        return 1.0
    tau = (t - theta_start) / duration
    if tau < 0.0:
        tau = 0.0
    elif tau > 1.0:
        tau = 1.0
    return 1.0 - np.exp(-a * tau ** (m + 1.0))


@njit(cache=True, fastmath=False)
def wiebe_burn_rate(theta_local_deg: float, a: float, m: float,
                    theta_start: float, duration: float) -> float:
    """dx_b/dθ (per degree) at the given local crank angle."""
    t = _to_combustion_angle(theta_local_deg, theta_start)
    if t < theta_start or t > theta_start + duration:
        return 0.0
    tau = (t - theta_start) / duration
    if tau < 1e-12:
        tau = 1e-12
    if tau > 1.0 - 1e-12:
        tau = 1.0 - 1e-12
    return (a * (m + 1.0) / duration) * tau ** m * np.exp(-a * tau ** (m + 1.0))


@njit(cache=True, fastmath=False)
def is_combusting(theta_local_deg: float, theta_start: float, duration: float) -> bool:
    t = _to_combustion_angle(theta_local_deg, theta_start)
    return (theta_start <= t) and (t <= theta_start + duration)
