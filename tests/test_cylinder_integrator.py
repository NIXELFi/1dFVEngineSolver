"""Sanity test for the ported CylinderModel: motored compression.

Initialize cylinder with atmospheric charge at BDC intake, hold valves
closed, sweep 180° → 360° (compression to TDC firing), verify peak
pressure is in the isentropic-compression range p_BDC · CR^γ.

No combustion (m_fuel = 0 → dQ_comb = 0) so the pressure rise is purely
polytropic (with wall heat losses reducing it modestly).
"""

from __future__ import annotations

import numpy as np

from cylinder.combustion import WiebeParams
from cylinder.cylinder import CylinderGeom, CylinderModel
from cylinder.heat_transfer import WoschniParams
from cylinder.valve import (
    INTAKE_LD_TABLE, INTAKE_CD_TABLE, EXHAUST_LD_TABLE, EXHAUST_CD_TABLE,
    ValveParams,
)


def _make_cylinder():
    geom = CylinderGeom(bore=0.067, stroke=0.0425, con_rod=0.0963, CR=12.2)
    wiebe = WiebeParams()
    woschni = WoschniParams(bore=geom.bore, stroke=geom.stroke, T_wall=450.0)
    intake = ValveParams(
        diameter=0.0275, max_lift=0.00856,
        open_angle_deg=350.0, close_angle_deg=585.0,
        seat_angle_deg=45.0, n_valves=2,
        ld_table=INTAKE_LD_TABLE, cd_table=INTAKE_CD_TABLE,
    )
    exhaust = ValveParams(
        diameter=0.023, max_lift=0.00735,
        open_angle_deg=140.0, close_angle_deg=365.0,
        seat_angle_deg=45.0, n_valves=2,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )
    return CylinderModel(geom, wiebe, woschni, intake, exhaust, phase_offset_deg=0.0)


def test_motored_compression_pre_combustion():
    """Sweep from θ = 600° (post-IVC) to θ = 690° (pre-combustion-start at
    702°). Pure compression, no fuel added. Peak p and T must land in
    a physically reasonable isentropic range.

    Expected (γ≈1.35, V_ratio ≈ 5.5): p ≈ 1.0 MPa, T ≈ 550 K, reduced by
    Woschni heat loss to the 450 K wall.
    """
    cyl = _make_cylinder()
    # Initialise AT 600° so IVC bookkeeping (at 585) doesn't fire inside the
    # loop. Zero fuel: no combustion regardless.
    cyl.initialize(p=101325.0, T=300.0, theta_global_deg=600.0)
    cyl.state.m_fuel = 0.0
    # Mass was set at θ=600° volume, not BDC volume. Record it for later
    # conservation check (closed cycle keeps m constant).
    m0 = cyl.state.m
    rpm = 10500.0
    omega = 2 * np.pi * rpm / 60.0

    theta = 600.0
    dtheta = 0.25
    n_steps = int(90.0 / dtheta)
    for _ in range(n_steps):
        dt = np.radians(dtheta) / omega
        cyl.state.mdot_intake = 0.0
        cyl.state.mdot_exhaust = 0.0
        cyl.advance(theta, dtheta, rpm, dt)
        theta += dtheta

    assert 0.6e6 < cyl.state.p < 1.5e6, f"p at 690° = {cyl.state.p:.2e} Pa"
    assert 400.0 < cyl.state.T < 700.0, f"T at 690° = {cyl.state.T:.1f} K"
    # Closed cycle preserves mass
    assert abs(cyl.state.m - m0) / m0 < 1e-10, (
        f"mass drift: {cyl.state.m:.4e} vs {m0:.4e}"
    )
    # Wiebe has not fired yet (θ_start = -18 BTDC = 702°, end at 32° ATDC)
    assert cyl.state.x_b == 0.0
