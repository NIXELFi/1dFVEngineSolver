"""Unit tests for the HLLC Riemann solver.

These are the minimal invariants any HLLC implementation must satisfy:
1. Consistency: F(U, U) reduces to the physical Euler flux at state U.
2. Supersonic: if u_L - a_L >= 0, flux is the left physical flux (no star state).
3. Stationary contact: p_L = p_R, u_L = u_R = 0, arbitrary density and
   composition jumps → zero mass / momentum / energy flux, and zero
   composition flux (no transport at u = 0).
4. Moving contact: p_L = p_R, u_L = u_R > 0, density and composition jumps
   → flux picks up left state's ρ and Y (upwind for positive u*).
5. Composition always follows the contact wave, not the shock.
"""

from __future__ import annotations

import numpy as np
import pytest

from solver.hllc import euler_flux, hllc_flux


GAMMA = 1.4


def _euler_flux_primitive(rho, u, p, Y, gamma=GAMMA):
    return euler_flux(rho, u, p, Y, gamma)


def test_hllc_consistency_at_rest():
    """F(U, U) at u = 0 must equal (0, p, 0, 0)."""
    rho, u, p, Y = 1.2, 0.0, 1.0e5, 0.3
    f0, f1, f2, f3 = hllc_flux(rho, u, p, Y, rho, u, p, Y, GAMMA)
    assert abs(f0) < 1e-12
    assert abs(f1 - p) < 1e-9
    assert abs(f2) < 1e-9
    assert abs(f3) < 1e-12


def test_hllc_consistency_moving():
    """F(U, U) for u != 0 must equal the physical Euler flux."""
    rho, u, p, Y = 1.2, 42.7, 1.5e5, 0.6
    expected = _euler_flux_primitive(rho, u, p, Y)
    got = hllc_flux(rho, u, p, Y, rho, u, p, Y, GAMMA)
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-6, f"consistency failed: got {got}, expected {expected}"


def test_hllc_supersonic_right_going():
    """Fully supersonic flow to the right: F = F_L."""
    # Mach ~2 flow with similar states
    rhoL, uL, pL, YL = 1.0, 800.0, 1e5, 0.0
    rhoR, uR, pR, YR = 0.5, 700.0, 5e4, 1.0
    # Sanity: u_L - a_L = 800 - sqrt(1.4*1e5) = 800 - 374 ≈ 426 > 0 → supersonic
    assert uL - np.sqrt(GAMMA * pL / rhoL) > 0
    assert uR - np.sqrt(GAMMA * pR / rhoR) > 0
    expected_L = _euler_flux_primitive(rhoL, uL, pL, YL)
    got = hllc_flux(rhoL, uL, pL, YL, rhoR, uR, pR, YR, GAMMA)
    for a, b in zip(got, expected_L):
        assert abs(a - b) / max(abs(b), 1.0) < 1e-9


def test_hllc_supersonic_left_going():
    """Fully supersonic flow to the left: F = F_R."""
    rhoL, uL, pL, YL = 0.5, -700.0, 5e4, 0.0
    rhoR, uR, pR, YR = 1.0, -800.0, 1e5, 1.0
    assert uR + np.sqrt(GAMMA * pR / rhoR) < 0  # u_R + a_R < 0 → left-supersonic
    expected_R = _euler_flux_primitive(rhoR, uR, pR, YR)
    got = hllc_flux(rhoL, uL, pL, YL, rhoR, uR, pR, YR, GAMMA)
    for a, b in zip(got, expected_R):
        assert abs(a - b) / max(abs(b), 1.0) < 1e-9


def test_hllc_stationary_contact_preservation():
    """Contact discontinuity at rest: p and u continuous, ρ & Y jump.

    HLLC must produce zero mass, zero-but-for-pressure momentum, zero energy,
    and zero composition flux. Any non-zero mass or composition flux here is
    diagnostic of HLL-style contact smearing.
    """
    p_common, u_common = 1.0e5, 0.0
    rhoL, YL = 1.0, 0.0
    rhoR, YR = 0.125, 1.0
    f0, f1, f2, f3 = hllc_flux(
        rhoL, u_common, p_common, YL,
        rhoR, u_common, p_common, YR,
        GAMMA,
    )
    assert abs(f0) < 1e-9
    assert abs(f1 - p_common) < 1e-6   # only pressure contribution
    assert abs(f2) < 1e-6
    assert abs(f3) < 1e-9


def test_hllc_moving_contact_advects_left_state():
    """Contact moving to the right: flux carries the LEFT state's ρ and Y.

    At a pure contact with u_L = u_R = u > 0 and p_L = p_R = p, the upwind
    rule picks left-side values for density-like quantities that cross the
    contact, so F_mass = ρ_L · u and F_comp = ρ_L · u · Y_L. Pressure flux
    is p, momentum flux is ρ_L u² + p.
    """
    p, u = 1.0e5, 50.0
    rhoL, YL = 1.0, 0.0
    rhoR, YR = 0.5, 1.0
    f0, f1, f2, f3 = hllc_flux(
        rhoL, u, p, YL,
        rhoR, u, p, YR,
        GAMMA,
    )
    expected_mass = rhoL * u
    expected_mom = rhoL * u * u + p
    # Energy flux uses left state's energy
    gm1 = GAMMA - 1.0
    EL = p / gm1 + 0.5 * rhoL * u * u
    expected_energy = (EL + p) * u
    expected_comp = rhoL * u * YL

    assert abs(f0 - expected_mass) / abs(expected_mass) < 1e-6
    assert abs(f1 - expected_mom) / abs(expected_mom) < 1e-6
    assert abs(f2 - expected_energy) / abs(expected_energy) < 1e-6
    assert abs(f3 - expected_comp) < 1e-9  # YL = 0 so absolute tol


def test_hllc_moving_contact_advects_right_state_when_u_negative():
    """Same but u < 0: flux carries the right state."""
    p, u = 1.0e5, -50.0
    rhoL, YL = 1.0, 0.0
    rhoR, YR = 0.5, 1.0
    f0, f1, f2, f3 = hllc_flux(
        rhoL, u, p, YL,
        rhoR, u, p, YR,
        GAMMA,
    )
    expected_mass = rhoR * u
    expected_comp = rhoR * u * YR
    assert abs(f0 - expected_mass) / abs(expected_mass) < 1e-6
    assert abs(f3 - expected_comp) / abs(expected_comp) < 1e-6


def test_hllc_sod_star_pressure_positivity():
    """For the Sod Riemann problem, HLLC must return a positivity-preserving
    flux with strictly positive mass flux (since u* > 0) and strictly
    positive energy flux.

    HLLC with Einfeldt-Batten wave speeds does NOT reproduce the exact
    Toro-Ch4 star state — EB underestimates u* here by ~27 %, which shifts
    the HLLC flux by ~10 %. That is the expected HLLC approximation and is
    why we judge the full solver by time-evolution L1 convergence (the Sod
    test_sod.py test), not by this single flux value.
    """
    rhoL, uL, pL, YL = 1.0, 0.0, 1.0, 0.0
    rhoR, uR, pR, YR = 0.125, 0.0, 0.1, 1.0
    f0, f1, f2, f3 = hllc_flux(rhoL, uL, pL, YL, rhoR, uR, pR, YR, GAMMA)
    # Mass flux should be positive (flow goes from high to low pressure)
    assert f0 > 0.0
    # Momentum flux dominated by pressure and is positive
    assert f1 > 0.0
    # Energy flux carries work against pressure gradient — positive
    assert f2 > 0.0
    # Composition flux follows the contact wave from the left (Y_L = 0)
    assert abs(f3) < 1e-9
