"""Test 6: quasi-1D converging-diverging nozzle — area source validation.

Two tests:

1. Tapered sealed pipe conservation. Extends the closed-domain conservation
   test to a non-uniform area profile. Verifies that the p·dA/dx momentum
   source does not spoil mass or energy conservation (ρY too).

2. Subsonic steady nozzle. Mild (factor-of-2) area contraction with fixed
   subsonic inflow and back pressure. After the transient, mass flux ρuA
   must be uniform along the pipe to within 1 %, and the flow field must
   stop changing (converged steady state).

The nozzle profile is a smooth cosine:
    A(x) = 0.5·(A_max+A_min) + 0.5·(A_max−A_min)·cos(π·x/L)·(-1 if throat in
           middle else +1)

Simpler: A(x) = A_max − (A_max−A_min)·(x/L)   — linear contraction.
"""

from __future__ import annotations

import numpy as np

from solver.state import make_pipe_state, set_uniform
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import fill_reflective_left, fill_reflective_right
from bcs.subsonic import fill_subsonic_inflow_left, fill_subsonic_outflow_right


GAMMA = 1.4


def _area_converging(x):
    """Linear contraction from A=1.0 at x=0 to A=0.5 at x=1.0."""
    return 1.0 - 0.5 * x


def _run_sealed_tapered(n_cells=100, n_steps=500):
    state = make_pipe_state(
        n_cells=n_cells, length=1.0,
        area_fn=lambda x: 1.5 - 0.5 * np.cos(2 * np.pi * x),  # two humps
        gamma=GAMMA, n_ghost=2,
    )
    set_uniform(state, rho=1.2, u=10.0, p=1.0e5, Y=0.3)

    def totals():
        s = state.real_slice()
        return (
            float(state.dx * state.q[s, 0].sum()),
            float(state.dx * state.q[s, 2].sum()),
            float(state.dx * state.q[s, 3].sum()),
        )
    m0, e0, y0 = totals()

    n = state.n_total
    w = np.zeros((n, 4)); slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4)); wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    for _ in range(n_steps):
        fill_reflective_left(state)
        fill_reflective_right(state)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, 0.85, state.n_ghost)
        if dt <= 0.0:
            raise RuntimeError("positivity failure")
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            GAMMA, state.n_ghost, LIMITER_MINMOD,
            w, slopes, wL, wR, flux,
        )
    m1, e1, y1 = totals()
    return m0, m1, e0, e1, y0, y1


def test_tapered_sealed_conservation():
    """Quasi-1D area source preserves mass/energy/composition exactly on a
    sealed non-uniform domain."""
    m0, m1, e0, e1, y0, y1 = _run_sealed_tapered(n_cells=100, n_steps=500)
    assert abs(m1 - m0) / m0 < 1e-12, f"mass drift {abs(m1-m0)/m0:.2e}"
    assert abs(e1 - e0) / e0 < 1e-12, f"energy drift {abs(e1-e0)/e0:.2e}"
    assert abs(y1 - y0) / y0 < 1e-12, f"composition drift {abs(y1-y0)/y0:.2e}"


def _run_subsonic_nozzle(n_cells=200, n_steps=4000):
    """Linear contraction nozzle A(x) = 1.0 − 0.5x. Fixed subsonic inflow,
    fixed back-pressure outflow. Run until steady."""
    L = 1.0
    state = make_pipe_state(
        n_cells=n_cells, length=L, area_fn=_area_converging,
        gamma=GAMMA, n_ghost=2,
    )
    # Initial guess: uniform flow at inlet conditions
    rho_in, u_in, p_in, Y_in = 1.2, 30.0, 1.0e5, 0.0
    set_uniform(state, rho=rho_in, u=u_in, p=p_in, Y=Y_in)
    # Back pressure: slightly below inlet static, driving accelerating flow
    p_back = 0.93 * p_in

    n = state.n_total
    w = np.zeros((n, 4)); slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4)); wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    for _ in range(n_steps):
        fill_subsonic_inflow_left(state, rho_in, u_in, p_in, Y_in)
        fill_subsonic_outflow_right(state, p_back)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, 0.85, state.n_ghost)
        if dt <= 0.0:
            raise RuntimeError("positivity failure")
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            GAMMA, state.n_ghost, LIMITER_MINMOD,
            w, slopes, wL, wR, flux,
        )
    return state


def test_subsonic_nozzle_mass_flux_uniform():
    """Once steady, ρuA must be uniform along the pipe to within 1 %."""
    state = _run_subsonic_nozzle(n_cells=200, n_steps=6000)
    s = state.real_slice()
    mdot = state.q[s, 1]  # ρuA is stored directly
    mdot_mean = float(np.mean(mdot))
    rel_spread = float((mdot.max() - mdot.min()) / abs(mdot_mean))
    assert rel_spread < 0.02, f"ρuA spread {rel_spread:.3%} across nozzle"


def test_subsonic_nozzle_reached_steady_state():
    """After the long run, another 500 steps should change the solution by
    at most 1 e-4 in relative norm (steady state reached)."""
    state = _run_subsonic_nozzle(n_cells=200, n_steps=6000)
    q_before = state.q.copy()

    n = state.n_total
    w = np.zeros((n, 4)); slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4)); wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))
    p_back = 0.93e5
    for _ in range(500):
        fill_subsonic_inflow_left(state, 1.2, 30.0, 1.0e5, 0.0)
        fill_subsonic_outflow_right(state, p_back)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, 0.85, state.n_ghost)
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            GAMMA, state.n_ghost, LIMITER_MINMOD,
            w, slopes, wL, wR, flux,
        )
    s = state.real_slice()
    rel = float(np.max(np.abs(state.q[s] - q_before[s])) /
                (np.max(np.abs(q_before[s])) + 1e-30))
    assert rel < 1e-3, f"not steady: max rel change = {rel:.2e}"
