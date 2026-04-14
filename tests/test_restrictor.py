"""Test 8: choked FSAE restrictor feeding a downstream pipe.

Setup: 20 mm throat (A_t = π·(0.01)² = 3.1416e-4 m²), Cd = 0.967,
upstream stagnation p_0 = 101325 Pa, T_0 = 300 K. Downstream pipe is
40 mm diameter, 0.5 m long. Subsonic outflow BC with back pressure set
well below the critical ratio (so the throat stays choked).

Acceptance: steady-state ṁ (measured via ρuA at any pipe cell) matches
the theoretical choked-nozzle ṁ to within 0.1 %.

Theoretical choked ṁ for γ=1.4:
    ṁ = Cd · A_t · p_0 · √(γ/(R·T_0)) · (2/(γ+1))^((γ+1)/(2(γ-1)))
    ≈ 0.967 · 3.1416e-4 · 101325 · √(1.4/(287·300)) · 0.5787
    ≈ 0.0718 kg/s
"""

from __future__ import annotations

import numpy as np

from solver.state import make_pipe_state, set_uniform
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.restrictor import fill_choked_restrictor_left, restrictor_mdot
from bcs.subsonic import fill_subsonic_outflow_right


GAMMA = 1.4
R_GAS = 287.0
P0 = 101325.0
T0 = 300.0
CD = 0.967
A_THROAT = np.pi * (0.02 / 2) ** 2


def _theoretical_choked_mdot():
    choke_factor = (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
    return CD * A_THROAT * P0 * np.sqrt(GAMMA / (R_GAS * T0)) * choke_factor


def test_theoretical_mdot_value():
    """Sanity check that the analytical formula matches documented ~72 g/s."""
    mdot = _theoretical_choked_mdot()
    assert 0.06 < mdot < 0.08, f"theoretical ṁ = {mdot:.4f} kg/s (expected ~0.072)"


def test_restrictor_mdot_is_choked_for_strong_back_pressure():
    """restrictor_mdot should return the choked value when p_down is below
    the critical ratio."""
    crit = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))
    p_critical = P0 * crit  # about 53526 Pa
    mdot_theory = _theoretical_choked_mdot()
    # Any p_down ≤ p_critical gives the same (choked) mass flow
    for p_down in [0.0, 0.3 * P0, 0.5 * P0, p_critical]:
        m = restrictor_mdot(p_down, P0, T0, A_THROAT, CD, GAMMA, R_GAS)
        assert abs(m - mdot_theory) / mdot_theory < 1e-12, (
            f"p_down={p_down}: ṁ={m:.6f}, expected {mdot_theory:.6f}"
        )


def test_restrictor_mdot_is_subsonic_for_weak_back_pressure():
    """Above the critical ratio, ṁ decreases continuously to 0 at p_down=p_0."""
    mdot_theory = _theoretical_choked_mdot()
    m_subs = restrictor_mdot(0.9 * P0, P0, T0, A_THROAT, CD, GAMMA, R_GAS)
    assert 0.0 < m_subs < mdot_theory
    m_zero = restrictor_mdot(P0, P0, T0, A_THROAT, CD, GAMMA, R_GAS)
    assert m_zero == 0.0


def test_choked_restrictor_delivers_theoretical_mdot_into_pipe():
    """Drive a 40-mm downstream pipe with the choked restrictor BC and a
    low back-pressure outflow. After enough steps, ρuA anywhere in the pipe
    equals the theoretical choked ṁ to within 0.1 %.
    """
    D_pipe = 0.040
    A_pipe = np.pi * (D_pipe / 2) ** 2
    L = 0.3  # modest length
    state = make_pipe_state(
        n_cells=100, length=L,
        area_fn=lambda x: A_pipe,
        gamma=GAMMA, R_gas=R_GAS, wall_T=300.0, n_ghost=2,
    )
    # Initial guess: static atmosphere
    set_uniform(state, rho=P0 / (R_GAS * T0), u=0.0, p=P0, Y=0.0)

    # Back pressure well below the critical ratio to ensure the throat
    # stays choked. We pick 0.3·P0.
    p_back = 0.3 * P0

    n = state.n_total
    w = np.zeros((n, 4)); slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4)); wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    mdot_hist = []
    for step in range(15000):
        fill_choked_restrictor_left(state, P0, T0, A_THROAT, CD)
        fill_subsonic_outflow_right(state, p_back)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, 0.85, state.n_ghost)
        if dt <= 0.0:
            raise RuntimeError(f"positivity failure at step {step}")
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            GAMMA, state.n_ghost, LIMITER_MINMOD,
            w, slopes, wL, wR, flux,
        )
        if step % 500 == 0:
            s = state.real_slice()
            mdot_mid = float(state.q[s, 1][50])
            mdot_hist.append(mdot_mid)

    mdot_theory = _theoretical_choked_mdot()
    s = state.real_slice()
    mdot_profile = state.q[s, 1]
    mdot_mean = float(mdot_profile.mean())
    rel = abs(mdot_mean - mdot_theory) / mdot_theory
    spread = float((mdot_profile.max() - mdot_profile.min()) / mdot_theory)

    assert spread < 0.01, f"ṁ profile spread {spread:.3%} (non-uniform flow)"
    assert rel < 0.001, (
        f"ṁ = {mdot_mean:.6f} kg/s, theory = {mdot_theory:.6f} kg/s, rel = {rel:.3%}"
    )
