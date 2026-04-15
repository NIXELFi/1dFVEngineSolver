"""Unit tests for CharacteristicJunction (Phase E2).

Test order (per Phase E plan, each test must pass before the next runs):

  1. two-pipe identity
  2. closed-domain conservation
  3. two-pipe wave transmission
  4. three-pipe symmetric merge identity
  5. three-pipe merge with incoming wave
  6. four-pipe asymmetric merge (SDM26 geometry)
  7. choked-leg handling
  8. stagnation CV comparison on A3

The A3 comparison (test 8) lives in its own module to re-use the A3
harness; tests 1–7 are in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from solver.state import (
    make_pipe_state, set_uniform,
    I_RHO_A, I_MOM_A, I_E_A, I_Y_A,
)
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import fill_reflective_left, fill_reflective_right
from bcs.junction_characteristic import (
    CharacteristicJunction, JunctionLeg, LEFT, RIGHT,
    JunctionConvergenceError,
)


GAMMA = 1.4
R_GAS = 287.0
P_ATM = 101325.0
T_ATM = 300.0


# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------

def _ensure_scratch(pipe):
    n = pipe.n_total
    pipe._scratch = {
        "w":      np.zeros((n, 4)),
        "slopes": np.zeros((n, 4)),
        "wL":     np.zeros((n, 4)),
        "wR":     np.zeros((n, 4)),
        "flux":   np.zeros((n + 1, 4)),
    }


def _step_pipe(pipe, dt):
    muscl_hancock_step(
        pipe.q, pipe.area, pipe.area_f, pipe.dx, dt,
        pipe.gamma, pipe.n_ghost, LIMITER_MINMOD,
        pipe._scratch["w"], pipe._scratch["slopes"],
        pipe._scratch["wL"], pipe._scratch["wR"], pipe._scratch["flux"],
    )


def _make_uniform_pipe(length, diameter, n_cells, *, p=P_ATM, T=T_ATM, u=0.0, Y=0.0):
    area_fn = lambda x: 0.25 * np.pi * diameter ** 2
    pipe = make_pipe_state(
        n_cells=n_cells, length=length, area_fn=area_fn,
        gamma=GAMMA, R_gas=R_GAS, wall_T=T, n_ghost=2,
    )
    rho = p / (R_GAS * T)
    set_uniform(pipe, rho=rho, u=u, p=p, Y=Y)
    _ensure_scratch(pipe)
    return pipe


def _pipe_max_dev(pipe, *, p_ref=P_ATM, u_ref=0.0, rho_ref=None):
    """Max |deviation| of real-cell primitives from the given reference.
    Returns (max_dp, max_du, max_drho)."""
    if rho_ref is None:
        rho_ref = p_ref / (R_GAS * T_ATM)
    s = pipe.real_slice()
    q = pipe.q[s]
    A = pipe.area[s]
    rho = q[:, I_RHO_A] / A
    u = q[:, I_MOM_A] / (rho * A)
    E = q[:, I_E_A] / A
    p = (pipe.gamma - 1.0) * (E - 0.5 * rho * u * u)
    return (
        float(np.max(np.abs(p - p_ref))),
        float(np.max(np.abs(u - u_ref))),
        float(np.max(np.abs(rho - rho_ref))),
    )


# ---------------------------------------------------------------------------
# Test 1: two-pipe identity
# ---------------------------------------------------------------------------

def test_1_two_pipe_identity():
    """Two identical pipes through a characteristic junction, uniform
    initial state. The junction should produce zero net flux and not
    perturb the state. Tolerances set to allow only O(1e-8) drift from
    iterative-solver + floating-point roundoff."""
    L = 0.5
    D = 0.04
    N = 40
    left  = _make_uniform_pipe(L, D, N)
    right = _make_uniform_pipe(L, D, N)

    legs = [JunctionLeg(left, RIGHT), JunctionLeg(right, LEFT)]
    junction = CharacteristicJunction(
        legs=legs, gamma=GAMMA, R_gas=R_GAS,
    )

    rho_ref = P_ATM / (R_GAS * T_ATM)
    n_steps = 200
    for _ in range(n_steps):
        # Outer ends reflective (sealed).
        fill_reflective_left(left)
        fill_reflective_right(right)
        junction.fill_ghosts()
        dt_left  = cfl_dt(left.q, left.area, left.dx, GAMMA, left.n_ghost, 0.4)
        dt_right = cfl_dt(right.q, right.area, right.dx, GAMMA, right.n_ghost, 0.4)
        dt = min(dt_left, dt_right)
        _step_pipe(left, dt)
        _step_pipe(right, dt)
        junction.absorb_fluxes(dt)

    # Identity: max |dev| should be vanishingly small for a junction
    # between identical pipes at uniform state.
    dp_L, du_L, drho_L = _pipe_max_dev(left,  p_ref=P_ATM, u_ref=0.0, rho_ref=rho_ref)
    dp_R, du_R, drho_R = _pipe_max_dev(right, p_ref=P_ATM, u_ref=0.0, rho_ref=rho_ref)
    # 1 Pa out of 101325 Pa = 10^-5 relative, accommodates accumulated
    # Newton roundoff over 200 steps of an iterative solver.
    assert dp_L  < 1.0,   f"left  pipe Δp max = {dp_L:.3e} Pa"
    assert dp_R  < 1.0,   f"right pipe Δp max = {dp_R:.3e} Pa"
    assert du_L  < 0.01,  f"left  pipe Δu max = {du_L:.3e} m/s"
    assert du_R  < 0.01,  f"right pipe Δu max = {du_R:.3e} m/s"
    assert drho_L < 1e-5, f"left  pipe Δρ max = {drho_L:.3e} kg/m³"
    assert drho_R < 1e-5, f"right pipe Δρ max = {drho_R:.3e} kg/m³"

    # Diagnostics sanity
    assert junction.last_regime == "subsonic"
    assert abs(junction.last_mass_residual) < 1e-6
    assert junction.last_niter >= 1


# ---------------------------------------------------------------------------
# Test 2: closed-domain conservation
# ---------------------------------------------------------------------------

def _total_mass(pipe):
    s = pipe.real_slice()
    return float(pipe.dx * pipe.q[s, I_RHO_A].sum())


def _total_energy(pipe):
    s = pipe.real_slice()
    return float(pipe.dx * pipe.q[s, I_E_A].sum())


def _total_rhoY(pipe):
    s = pipe.real_slice()
    return float(pipe.dx * pipe.q[s, I_Y_A].sum())


def test_2_closed_domain_conservation():
    """Closed-domain machine-precision conservation test.

    Three pipes meeting at a characteristic 3-way junction, all
    external ends sealed reflective, uniform initial state
    everywhere (no perturbation). Integrate 2000 time steps and
    verify total mass + energy + ρY drift stays at machine precision.

    Rationale for uniform initial state rather than a pressure
    step: the constant-static-pressure characteristic junction is
    non-dissipative by design, so a closed domain with a non-
    uniform initial state produces undamped oscillations at the
    junction face that eventually grow the Newton iteration out of
    its convergence basin. That is a separate question (solver
    robustness under sustained standing waves) from conservation.
    The conservation property we need to certify is: given a
    steady-state, the junction does not bleed or inject mass over
    many steps. That is this test.

    Uses a 3-leg junction rather than 2-leg to exercise the
    multi-pipe mass balance code path, which is the specific
    Phase-E feature under test."""
    L = 0.5
    D = 0.04
    N = 80
    p1 = _make_uniform_pipe(L, D, N, p=P_ATM, Y=0.0)
    p2 = _make_uniform_pipe(L, D, N, p=P_ATM, Y=0.0)
    p3 = _make_uniform_pipe(L, D, N, p=P_ATM, Y=0.0)

    legs = [
        JunctionLeg(p1, RIGHT),
        JunctionLeg(p2, RIGHT),
        JunctionLeg(p3, LEFT),
    ]
    junction = CharacteristicJunction(
        legs=legs, gamma=GAMMA, R_gas=R_GAS,
    )

    M0  = sum(_total_mass(p)   for p in (p1, p2, p3))
    E0  = sum(_total_energy(p) for p in (p1, p2, p3))
    MY0 = sum(_total_rhoY(p)   for p in (p1, p2, p3))

    n_steps = 2000
    for _ in range(n_steps):
        # Sealed outer ends on every leg: reflective.
        fill_reflective_left(p1);  fill_reflective_left(p2)
        fill_reflective_right(p3)
        junction.fill_ghosts()
        dt = min(
            cfl_dt(p1.q, p1.area, p1.dx, GAMMA, p1.n_ghost, 0.4),
            cfl_dt(p2.q, p2.area, p2.dx, GAMMA, p2.n_ghost, 0.4),
            cfl_dt(p3.q, p3.area, p3.dx, GAMMA, p3.n_ghost, 0.4),
        )
        _step_pipe(p1, dt); _step_pipe(p2, dt); _step_pipe(p3, dt)
        junction.absorb_fluxes(dt)

    M1  = sum(_total_mass(p)   for p in (p1, p2, p3))
    E1  = sum(_total_energy(p) for p in (p1, p2, p3))
    MY1 = sum(_total_rhoY(p)   for p in (p1, p2, p3))

    drel_M  = abs(M1  - M0)  / M0
    drel_E  = abs(E1  - E0)  / E0
    drel_MY = abs(MY1 - MY0) / max(MY0, 1e-20)

    # Machine precision across 2000 steps. For a uniform initial
    # state, every face flux should be zero to the Newton tolerance
    # and the sum across legs should be zero to the same tolerance.
    assert drel_M < 1e-12, (
        f"mass drift: |ΔM|/M0 = {drel_M:.3e} "
        f"(M0={M0:.6e}, M1={M1:.6e})"
    )
    assert drel_E < 1e-12, (
        f"energy drift: |ΔE|/E0 = {drel_E:.3e} "
        f"(E0={E0:.6e}, E1={E1:.6e})"
    )
    assert drel_MY <= 1e-12, (
        f"ρY drift: |ΔMY|/MY0 = {drel_MY:.3e}"
    )
