"""Shared runner for 1D shock-tube validation tests.

Provides a single `run_riemann` function that advances a MUSCL-Hancock + HLLC
simulation from given left/right Riemann initial data to a final time, under
either transmissive or reflective BCs. Returns the primitive solution on the
real cells at t_end.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from solver.state import make_pipe_state, set_left_right, primitives_array
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import (
    fill_transmissive_left, fill_transmissive_right,
    fill_reflective_left, fill_reflective_right,
)


def run_riemann(
    n_cells: int,
    length: float,
    x0: float,
    left: tuple,             # (ρ, u, p, Y)
    right: tuple,            # (ρ, u, p, Y)
    t_end: float,
    gamma: float = 1.4,
    cfl: float = 0.85,
    limiter: int = LIMITER_MINMOD,
    bc: str = "transmissive",  # or "reflective"
    area_fn: Callable | None = None,  # default unit area
    progress: bool = False,
):
    if area_fn is None:
        area_fn = lambda x: 1.0
    state = make_pipe_state(
        n_cells=n_cells, length=length,
        area_fn=area_fn, gamma=gamma,
        n_ghost=2,
    )
    set_left_right(
        state, x0=x0,
        rhoL=left[0], uL=left[1], pL=left[2], YL=left[3],
        rhoR=right[0], uR=right[1], pR=right[2], YR=right[3],
    )

    n = state.n_total
    w = np.zeros((n, 4))
    slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4))
    wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    if bc == "transmissive":
        fill_L = fill_transmissive_left
        fill_R = fill_transmissive_right
    elif bc == "reflective":
        fill_L = fill_reflective_left
        fill_R = fill_reflective_right
    else:
        raise ValueError(f"bc must be 'transmissive' or 'reflective', got {bc!r}")

    t = 0.0
    step = 0
    while t < t_end:
        fill_L(state)
        fill_R(state)
        dt = cfl_dt(state.q, state.area, state.dx, gamma, cfl, state.n_ghost)
        if dt <= 0.0:
            raise RuntimeError(f"positivity failed at t={t}, step={step}")
        if t + dt > t_end:
            dt = t_end - t
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            gamma, state.n_ghost, limiter,
            w, slopes, wL, wR, flux,
        )
        t += dt
        step += 1

    w_final = primitives_array(state)
    s = state.real_slice()
    x_centres = (np.arange(n_cells) + 0.5) * state.dx
    return x_centres, w_final[s], state, step
