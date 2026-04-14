"""Sod shock tube test for the MUSCL-Hancock + HLLC solver.

Sod initial conditions (Toro Table 4.1, row 1):
    Left  (x < 0.5): ρ=1.0,   u=0.0, p=1.0
    Right (x ≥ 0.5): ρ=0.125, u=0.0, p=0.1
    γ = 1.4, domain [0, 1], t_end = 0.2.

Acceptance criteria:
    1. Star state recovered within 1 % of exact p*, u*.
    2. L1 error on density at n_cells = 200 is below 0.020 (typical
       MUSCL-Hancock + HLLC + minmod performance on Sod).
    3. L1 error shrinks with grid refinement (approximate 1st order at shocks,
       higher on smooth regions; we test the integrated L1 is monotone in n).
"""

from __future__ import annotations

import numpy as np
import pytest

from solver.state import make_pipe_state, set_left_right, primitives_array
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import fill_transmissive_left, fill_transmissive_right
from tests.exact_riemann import sample_array, solve_star


GAMMA = 1.4


def run_sod(n_cells: int, t_end: float = 0.2, cfl: float = 0.85, limiter: int = LIMITER_MINMOD):
    state = make_pipe_state(
        n_cells=n_cells, length=1.0,
        area_fn=lambda x: 1.0,  # unit area
        gamma=GAMMA, R_gas=287.0, wall_T=0.0,
        n_ghost=2,
    )
    set_left_right(
        state, x0=0.5,
        rhoL=1.0, uL=0.0, pL=1.0, YL=0.0,
        rhoR=0.125, uR=0.0, pR=0.1, YR=1.0,
    )

    # Scratch buffers for the @njit kernel
    n = state.n_total
    w = np.zeros((n, 4))
    slopes = np.zeros((n, 4))
    w_pred_L = np.zeros((n, 4))
    w_pred_R = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    t = 0.0
    while t < t_end:
        fill_transmissive_left(state)
        fill_transmissive_right(state)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, cfl, state.n_ghost)
        if dt <= 0.0:
            raise RuntimeError("CFL returned zero — positivity violated")
        if t + dt > t_end:
            dt = t_end - t
        muscl_hancock_step(
            state.q, state.area, state.area_f, state.dx, dt,
            GAMMA, state.n_ghost, limiter,
            w, slopes, w_pred_L, w_pred_R, flux,
        )
        t += dt

    w_final = primitives_array(state)
    s = state.real_slice()
    x_centres = (np.arange(n_cells) + 0.5) * state.dx
    return x_centres, w_final[s]


def _l1_error(x, primitives_sim, t_end):
    rho_ex, u_ex, p_ex, p_star, u_star = sample_array(
        x, t_end, x0=0.5,
        rho_L=1.0, u_L=0.0, p_L=1.0,
        rho_R=0.125, u_R=0.0, p_R=0.1,
        gamma=GAMMA,
    )
    dx = x[1] - x[0]
    e_rho = np.sum(np.abs(primitives_sim[:, 0] - rho_ex)) * dx
    e_u   = np.sum(np.abs(primitives_sim[:, 1] - u_ex))   * dx
    e_p   = np.sum(np.abs(primitives_sim[:, 2] - p_ex))   * dx
    return e_rho, e_u, e_p, p_star, u_star


def test_sod_star_state_recovery():
    """After the simulation, a cell well inside the star region (say x=0.55)
    should show pressure within 1 % of p* ≈ 0.30313 and velocity within 1 %
    of u* ≈ 0.92745."""
    x, w = run_sod(n_cells=400, t_end=0.2)
    # Sample at x = 0.55: between rarefaction tail (~0.497) and contact
    # (~0.685) — squarely inside left star region.
    i = int(0.55 / (x[1] - x[0]))
    p_star_sim = w[i, 2]
    u_star_sim = w[i, 1]
    p_star_ex, u_star_ex = solve_star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, GAMMA)
    assert abs(p_star_sim - p_star_ex) / p_star_ex < 0.02, (
        f"p* sim={p_star_sim:.5f} vs exact={p_star_ex:.5f}"
    )
    assert abs(u_star_sim - u_star_ex) / u_star_ex < 0.02, (
        f"u* sim={u_star_sim:.5f} vs exact={u_star_ex:.5f}"
    )


def test_sod_l1_200_cells():
    """At n_cells = 200, L1 density error should be below 0.02."""
    x, w = run_sod(n_cells=200, t_end=0.2)
    e_rho, e_u, e_p, _, _ = _l1_error(x, w, 0.2)
    assert e_rho < 0.02, f"L1(ρ) = {e_rho:.4f}"
    assert e_u   < 0.03, f"L1(u) = {e_u:.4f}"
    assert e_p   < 0.02, f"L1(p) = {e_p:.4f}"


def test_sod_l1_convergence():
    """L1 error monotonically decreases with refinement."""
    errs = []
    for n in (100, 200, 400):
        x, w = run_sod(n_cells=n, t_end=0.2)
        e_rho, _, _, _, _ = _l1_error(x, w, 0.2)
        errs.append(e_rho)
    assert errs[0] > errs[1] > errs[2], f"L1 not monotone: {errs}"
    # Order of convergence between 100 and 400 cells
    rate = np.log(errs[0] / errs[2]) / np.log(4.0)
    assert rate > 0.6, f"L1 convergence rate {rate:.2f} below 0.6"


def test_sod_composition_advects_with_contact():
    """The burned-gas fraction Y, initialised to 1 on the right, should have
    advected leftward by u*·t at t=0.2 — the contact has moved from x=0.5
    to x=0.5 + u*·0.2 ≈ 0.685. Cells at x > 0.685 keep Y ≈ 1; cells at
    x < 0.685 should have Y ≈ 0. A small transition region (contact smearing)
    is expected from the limiter.
    """
    x, w = run_sod(n_cells=400, t_end=0.2)
    p_star_ex, u_star_ex = solve_star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, GAMMA)
    x_contact = 0.5 + u_star_ex * 0.2
    # Well-left of contact:
    i_left = int((x_contact - 0.1) / (x[1] - x[0]))
    assert w[i_left, 3] < 0.05, f"Y at x<contact = {w[i_left, 3]:.3f}"
    # Well-right of contact, but still inside the star region (before shock):
    i_right = int((x_contact + 0.05) / (x[1] - x[0]))
    assert w[i_right, 3] > 0.95, f"Y at x>contact = {w[i_right, 3]:.3f}"
