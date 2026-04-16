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
    eta_comb: float = 0.96             # base combustion efficiency (Wiebe model's
                                       # theoretical maximum if everything were perfect)
    q_lhv: float = 44.0e6             # J/kg, gasoline
    afr_target: float = 13.1          # slightly rich for power

    # V1 two-segment efficiency-factor ramp (Phase F4 corrected).
    # V1 source: orchestrator.py:267-276, calibrated against SDM25
    # DynoJet data May 2025.
    #
    # V1 uses: actual_eta_comb = base_efficiency × efficiency_factor(RPM)
    # where base_efficiency = 0.96 (from CombustionConfig) and the factor
    # acknowledges that the Wiebe model itself has deficiencies (wave speed
    # error in V1's MOC, sin² cam profile approximation). The factor cap
    # at 0.88 means peak actual eta_comb = 0.96 × 0.88 = 0.845.
    #
    # V2 inherits V1's factor ramp AND V1's base (0.96) per review
    # decision (Option C). Caveat: V1's derating factor was calibrated
    # for V1's specific model deficiencies, which differ from V2's
    # (V2 has correct wave speed via HLLC-FV, better cam profiles, but
    # has inviscid junction over-prediction and frozen gamma). The net
    # effect of inheriting V1's factor is that V2 systematically
    # under-predicts peak power by ~10-12% vs SDM25 dyno. This is
    # honest and documented; future V2-specific calibration of the
    # base_efficiency or the factor cap would address it.
    _factor_rpm_lo: float = 3500.0     # below this: factor = 0.55
    _factor_rpm_knee: float = 6000.0   # knee between steep and gentle segments
    _factor_rpm_hi: float = 10500.0    # above this: factor = 0.88 (capped)
    _factor_lo: float = 0.55           # factor at and below _factor_rpm_lo
    _factor_knee: float = 0.80         # factor at _factor_rpm_knee
    _factor_hi: float = 0.88           # factor at and above _factor_rpm_hi (cap)

    def eta_comb_at_rpm(self, rpm: float) -> float:
        """RPM-dependent combustion efficiency — V1's exact two-segment
        factor ramp applied to the base eta_comb.

        Two-segment piecewise linear on the efficiency_factor:
          RPM ≤ 3500:          factor = 0.55  (poor low-RPM combustion)
          3500 < RPM ≤ 6000:   linear 0.55 → 0.80  (steep improvement)
          6000 < RPM ≤ 10500:  linear 0.80 → 0.88  (gentle plateau)
          RPM > 10500:         factor = 0.88  (capped at design point)

        actual_eta_comb = base_eta_comb × factor

        Resulting values at key RPMs:
          3500 RPM:  0.96 × 0.55 = 0.528
          6000 RPM:  0.96 × 0.80 = 0.768
          10500 RPM: 0.96 × 0.88 = 0.845

        Source: V1 orchestrator.py:267-276, DynoJet calibration May 2025.
        Reference: Heywood 1988 §9 on combustion efficiency correlations.
        """
        if rpm <= self._factor_rpm_lo:
            factor = self._factor_lo
        elif rpm <= self._factor_rpm_knee:
            frac = (rpm - self._factor_rpm_lo) / (self._factor_rpm_knee - self._factor_rpm_lo)
            factor = self._factor_lo + frac * (self._factor_knee - self._factor_lo)
        elif rpm <= self._factor_rpm_hi:
            frac = (rpm - self._factor_rpm_knee) / (self._factor_rpm_hi - self._factor_rpm_knee)
            factor = self._factor_knee + frac * (self._factor_hi - self._factor_knee)
        else:
            factor = self._factor_hi
        return self.eta_comb * factor

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
