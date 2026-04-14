"""Sanity tests for ported cylinder submodels — Wiebe, Woschni, valve, geometry.

These are identities and basic property checks, not full validation. Full
validation is the end-to-end SDM26 single-RPM run (Phase 3 step).
"""

from __future__ import annotations

import numpy as np

from cylinder.combustion import (
    WiebeParams, wiebe_xb, wiebe_burn_rate, is_combusting,
)
from cylinder.geometry import (
    cylinder_volume, cylinder_dVdtheta, cylinder_surface_area,
)
from cylinder.heat_transfer import (
    WoschniParams, woschni_h, mean_piston_speed, motored_pressure,
)
from cylinder.valve import (
    valve_lift, valve_Cd, valve_reference_area, valve_effective_area,
    INTAKE_LD_TABLE, INTAKE_CD_TABLE, EXHAUST_LD_TABLE, EXHAUST_CD_TABLE,
)


# ---- Wiebe ----------------------------------------------------------------

def test_wiebe_xb_bounds():
    p = WiebeParams()
    # Before start: 0. After end: 1.
    assert wiebe_xb(-30.0, p.a, p.m, p.theta_start, p.duration_deg) == 0.0
    assert wiebe_xb(+90.0, p.a, p.m, p.theta_start, p.duration_deg) == 1.0


def test_wiebe_xb_at_full_duration_is_large():
    p = WiebeParams()  # a=5 → x_b at τ=1 = 1 − exp(-5) ≈ 0.9933
    xb = wiebe_xb(p.theta_start + p.duration_deg - 1e-9, p.a, p.m,
                  p.theta_start, p.duration_deg)
    assert 0.98 < xb < 1.0


def test_wiebe_integral_matches_xb():
    """Integrated burn rate over [θ_start, θ_end] must equal 1 − exp(-a)."""
    p = WiebeParams()
    thetas = np.linspace(p.theta_start, p.theta_start + p.duration_deg, 2001)
    dtheta = thetas[1] - thetas[0]
    rates = np.array([
        wiebe_burn_rate(t, p.a, p.m, p.theta_start, p.duration_deg)
        for t in thetas
    ])
    integrated = float(np.sum(rates) * dtheta)
    expected = 1.0 - np.exp(-p.a)
    assert abs(integrated - expected) < 5e-3


def test_is_combusting_straddles_tdc():
    p = WiebeParams()  # θ_start = -18 (i.e. 702 CAD-wrap), θ_end = +32
    # 710° → -10° BTDC: inside window
    assert is_combusting(710.0, p.theta_start, p.duration_deg)
    # 10° ATDC: inside window
    assert is_combusting(10.0, p.theta_start, p.duration_deg)
    # 50° ATDC: past end (32°)
    assert not is_combusting(50.0, p.theta_start, p.duration_deg)
    # 680° (-40° BTDC): before start
    assert not is_combusting(680.0, p.theta_start, p.duration_deg)


# ---- Geometry --------------------------------------------------------------

def test_cylinder_volume_at_tdc_and_bdc():
    # CBR600RR parameters
    bore = 0.067
    stroke = 0.0425
    con_rod = 0.0963
    CR = 12.2
    A_bore = 0.25 * np.pi * bore ** 2
    V_d = A_bore * stroke
    V_c = V_d / (CR - 1.0)
    V_tdc = cylinder_volume(0.0, bore, stroke, con_rod, CR)
    V_bdc = cylinder_volume(180.0, bore, stroke, con_rod, CR)
    assert abs(V_tdc - V_c) / V_c < 1e-10
    assert abs(V_bdc - (V_c + V_d)) / (V_c + V_d) < 1e-10


def test_dVdtheta_sign():
    bore = 0.067; stroke = 0.0425; con_rod = 0.0963
    # Post-TDC (θ = 90°): piston moving down, dV/dθ > 0
    dV = cylinder_dVdtheta(90.0, bore, stroke, con_rod)
    assert dV > 0
    # Post-BDC (θ = 270°): piston moving up, dV/dθ < 0
    dV = cylinder_dVdtheta(270.0, bore, stroke, con_rod)
    assert dV < 0


# ---- Woschni ---------------------------------------------------------------

def test_mean_piston_speed_formula():
    assert abs(mean_piston_speed(0.0425, 10500) - (2 * 0.0425 * 10500 / 60)) < 1e-12


def test_motored_pressure_isentropic():
    # Expansion: V doubles → p drops by factor 2^1.35 ≈ 2.55
    p_ref = 15.0e5
    V_ref = 50e-6
    p_mot = motored_pressure(2 * V_ref, p_ref, V_ref, 1.35)
    assert abs(p_mot - p_ref / 2 ** 1.35) / p_ref < 1e-9


def test_woschni_h_positive():
    params = WoschniParams(bore=0.067, stroke=0.0425, T_wall=450.0)
    h = woschni_h(
        p=30e5, T=2000.0, rpm=10500, V=5e-5, V_d=1.5e-4,
        phase=2, p_ref=15e5, T_ref=600.0, V_ref=50e-6,
        bore=params.bore, stroke=params.stroke,
        C1_gx=params.C1_gas_exchange, C1_co=params.C1_compression,
        C1_cb=params.C1_combustion, C2_cb=params.C2_combustion,
    )
    # A few kW/m²/K is typical for Woschni during combustion
    assert 100.0 < h < 50000.0, f"h = {h:.1f} W/m²/K"


# ---- Valve -----------------------------------------------------------------

def test_valve_lift_peak_at_midpoint():
    # For intake on a CBR600RR: open 350, close 585, max 8.56 mm
    open_a, close_a, max_l = 350.0, 585.0, 0.00856
    peak = (open_a + close_a) / 2
    L = valve_lift(peak, open_a, close_a, max_l)
    assert abs(L - max_l) / max_l < 1e-6


def test_valve_Cd_table_monotone():
    # V1 tables should be monotone non-decreasing
    for k in range(len(INTAKE_CD_TABLE) - 1):
        assert INTAKE_CD_TABLE[k] <= INTAKE_CD_TABLE[k + 1]
    for k in range(len(EXHAUST_CD_TABLE) - 1):
        assert EXHAUST_CD_TABLE[k] <= EXHAUST_CD_TABLE[k + 1]


def test_valve_effective_area_zero_when_closed():
    # Outside the event window
    A = valve_effective_area(
        0.0, 350.0, 585.0, 0.00856, 0.0275, np.radians(45.0), 2,
        INTAKE_LD_TABLE, INTAKE_CD_TABLE,
    )
    assert A == 0.0


def test_valve_effective_area_positive_mid_event():
    # Mid-event, should be roughly Cd_peak · port_area · 2
    A = valve_effective_area(
        467.5, 350.0, 585.0, 0.00856, 0.0275, np.radians(45.0), 2,
        INTAKE_LD_TABLE, INTAKE_CD_TABLE,
    )
    port = 0.25 * np.pi * 0.0275 ** 2
    assert 0.1 * port < A < 5 * port
