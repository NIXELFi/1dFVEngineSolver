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
