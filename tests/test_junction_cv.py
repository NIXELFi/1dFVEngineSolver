"""Standalone conservation test for the 0D junction CV.

Setup: two pipes of equal length + area connected by a junction CV in
the middle. Outer ends are sealed (reflective). Initial state has a
pressure imbalance between the two pipes to drive flow through the CV.

Acceptance: total mass, total energy, and total ρY across the two pipes
and the junction CV remain constant to machine precision (< 1e-12
relative) over N steps.
"""

from __future__ import annotations

import numpy as np

from solver.state import make_pipe_state, set_uniform, I_RHO_A, I_E_A, I_Y_A
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import fill_reflective_left, fill_reflective_right
from bcs.junction_cv import JunctionCV, JunctionCVLeg, LEFT, RIGHT


GAMMA = 1.4
R_GAS = 287.0


def _ensure_scratch(pipe):
    n = pipe.n_total
    pipe._scratch = {
        "w":      np.zeros((n, 4)),
        "slopes": np.zeros((n, 4)),
        "wL":     np.zeros((n, 4)),
        "wR":     np.zeros((n, 4)),
        "flux":   np.zeros((n + 1, 4)),
    }


def _step_pipe(pipe, dt, gamma=GAMMA):
    muscl_hancock_step(
        pipe.q, pipe.area, pipe.area_f, pipe.dx, dt,
        gamma, pipe.n_ghost, LIMITER_MINMOD,
        pipe._scratch["w"], pipe._scratch["slopes"],
        pipe._scratch["wL"], pipe._scratch["wR"], pipe._scratch["flux"],
    )


def _run_two_pipe_junction(
    n_cells: int = 80, length: float = 0.5, n_steps: int = 1000,
    rhoL: float = 1.5, pL: float = 2.0e5, YL: float = 0.0,
    rhoR: float = 1.0, pR: float = 1.0e5, YR: float = 1.0,
):
    area = lambda x: 1e-3  # 10 cm² pipe
    left_pipe = make_pipe_state(n_cells, length, area_fn=area,
                                 gamma=GAMMA, R_gas=R_GAS, wall_T=300.0, n_ghost=2)
    right_pipe = make_pipe_state(n_cells, length, area_fn=area,
                                  gamma=GAMMA, R_gas=R_GAS, wall_T=300.0, n_ghost=2)
    set_uniform(left_pipe,  rho=rhoL, u=0.0, p=pL, Y=YL)
    set_uniform(right_pipe, rho=rhoR, u=0.0, p=pR, Y=YR)
    _ensure_scratch(left_pipe)
    _ensure_scratch(right_pipe)

    # Junction CV sits between left_pipe RIGHT and right_pipe LEFT.
    p_init = 0.5 * (pL + pR)
    T_init = 0.5 * (pL / (rhoL * R_GAS) + pR / (rhoR * R_GAS))
    junction = JunctionCV.from_legs(
        [JunctionCVLeg(left_pipe, RIGHT), JunctionCVLeg(right_pipe, LEFT)],
        p_init=p_init, T_init=T_init, Y_init=0.5,
        gamma=GAMMA, R_gas=R_GAS,
    )

    def totals():
        sL = left_pipe.real_slice()
        sR = right_pipe.real_slice()
        M = (float(left_pipe.dx * left_pipe.q[sL, I_RHO_A].sum()) +
             float(right_pipe.dx * right_pipe.q[sR, I_RHO_A].sum()) +
             junction.M)
        E = (float(left_pipe.dx * left_pipe.q[sL, I_E_A].sum()) +
             float(right_pipe.dx * right_pipe.q[sR, I_E_A].sum()) +
             junction.E)
        MY = (float(left_pipe.dx * left_pipe.q[sL, I_Y_A].sum()) +
              float(right_pipe.dx * right_pipe.q[sR, I_Y_A].sum()) +
              junction.M_Y)
        return M, E, MY

    M0, E0, MY0 = totals()

    for _ in range(n_steps):
        # Sealed outer ends
        fill_reflective_left(left_pipe)
        fill_reflective_right(right_pipe)
        # Junction fills inner ends
        junction.fill_ghosts()
        # dt
        dt = min(
            cfl_dt(left_pipe.q, left_pipe.area, left_pipe.dx, GAMMA, 0.85, left_pipe.n_ghost),
            cfl_dt(right_pipe.q, right_pipe.area, right_pipe.dx, GAMMA, 0.85, right_pipe.n_ghost),
        )
        if dt <= 0:
            raise RuntimeError("positivity failure")
        _step_pipe(left_pipe, dt)
        _step_pipe(right_pipe, dt)
        # Junction absorbs face fluxes
        junction.absorb_fluxes(dt)

    M1, E1, MY1 = totals()
    return M0, M1, E0, E1, MY0, MY1, junction


def test_junction_cv_mass_conserved():
    M0, M1, _, _, _, _, _ = _run_two_pipe_junction(n_steps=500)
    rel = abs(M1 - M0) / M0
    assert rel < 1e-12, f"relative mass drift = {rel:.2e}"


def test_junction_cv_energy_conserved():
    _, _, E0, E1, _, _, _ = _run_two_pipe_junction(n_steps=500)
    rel = abs(E1 - E0) / E0
    assert rel < 1e-12, f"relative energy drift = {rel:.2e}"


def test_junction_cv_composition_conserved():
    _, _, _, _, MY0, MY1, _ = _run_two_pipe_junction(n_steps=500)
    rel = abs(MY1 - MY0) / MY0
    assert rel < 1e-12, f"relative ρY drift = {rel:.2e}"


def test_junction_cv_long_run():
    """Even with 5000 steps, no drift accumulates."""
    M0, M1, E0, E1, MY0, MY1, _ = _run_two_pipe_junction(n_steps=5000)
    assert abs(M1 - M0) / M0 < 1e-12
    assert abs(E1 - E0) / E0 < 1e-12
    assert abs(MY1 - MY0) / MY0 < 1e-12


def test_three_pipe_junction_conserves():
    """Extends to 3 legs — the 4-2-1-exhaust-relevant case."""
    area = lambda x: 1e-3
    pipes = []
    for _ in range(3):
        p = make_pipe_state(50, 0.3, area_fn=area, gamma=GAMMA, R_gas=R_GAS,
                             wall_T=300.0, n_ghost=2)
        _ensure_scratch(p)
        pipes.append(p)
    set_uniform(pipes[0], rho=1.5, u=0.0, p=2.0e5, Y=0.0)
    set_uniform(pipes[1], rho=1.5, u=0.0, p=1.5e5, Y=0.5)
    set_uniform(pipes[2], rho=1.0, u=0.0, p=1.0e5, Y=1.0)

    junction = JunctionCV.from_legs(
        [JunctionCVLeg(pipes[0], RIGHT),
         JunctionCVLeg(pipes[1], RIGHT),
         JunctionCVLeg(pipes[2], LEFT)],
        p_init=1.5e5, T_init=300.0, Y_init=0.5,
        gamma=GAMMA, R_gas=R_GAS,
    )

    def totals():
        M = E = MY = 0.0
        for p in pipes:
            s = p.real_slice()
            M += float(p.dx * p.q[s, I_RHO_A].sum())
            E += float(p.dx * p.q[s, I_E_A].sum())
            MY += float(p.dx * p.q[s, I_Y_A].sum())
        return M + junction.M, E + junction.E, MY + junction.M_Y

    M0, E0, MY0 = totals()
    for _ in range(2000):
        fill_reflective_left(pipes[0])
        fill_reflective_left(pipes[1])
        fill_reflective_right(pipes[2])
        junction.fill_ghosts()
        dt = min(cfl_dt(p.q, p.area, p.dx, GAMMA, 0.85, p.n_ghost) for p in pipes)
        if dt <= 0:
            raise RuntimeError("positivity failure")
        for p in pipes:
            _step_pipe(p, dt)
        junction.absorb_fluxes(dt)

    M1, E1, MY1 = totals()
    assert abs(M1 - M0) / M0 < 1e-12, f"3-pipe mass drift = {abs(M1-M0)/M0:.2e}"
    assert abs(E1 - E0) / E0 < 1e-12, f"3-pipe energy drift = {abs(E1-E0)/E0:.2e}"
    assert abs(MY1 - MY0) / MY0 < 1e-12, f"3-pipe ρY drift = {abs(MY1-MY0)/MY0:.2e}"
