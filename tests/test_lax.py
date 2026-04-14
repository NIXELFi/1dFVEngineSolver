"""Lax shock tube: Toro Table 4.1 row 2.

Initial data:
    L: ρ=0.445, u=0.698, p=3.528
    R: ρ=0.5,   u=0.0,   p=0.571
    γ = 1.4, domain [0, 1], x0=0.5, t_end = 0.15.

Stronger shock than Sod; stresses the Riemann solver but HLLC handles it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.harness import run_riemann
from tests.exact_riemann import sample_array, solve_star


GAMMA = 1.4
LEFT = (0.445, 0.698, 3.528, 0.0)
RIGHT = (0.5, 0.0, 0.571, 1.0)
T_END = 0.15


def test_lax_star_state():
    x, w, _, _ = run_riemann(
        n_cells=400, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    p_star_ex, u_star_ex = solve_star(*LEFT[:3], *RIGHT[:3], gamma=GAMMA)
    # Sample well inside the left star region
    x_contact = 0.5 + u_star_ex * T_END
    i = int((x_contact - 0.05) / (x[1] - x[0]))
    assert abs(w[i, 2] - p_star_ex) / p_star_ex < 0.02, f"p* sim={w[i,2]:.4f} exact={p_star_ex:.4f}"
    assert abs(w[i, 1] - u_star_ex) / u_star_ex < 0.05, f"u* sim={w[i,1]:.4f} exact={u_star_ex:.4f}"


def test_lax_l1_at_200_cells():
    x, w, _, _ = run_riemann(
        n_cells=200, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    rho_ex, u_ex, p_ex, _, _ = sample_array(
        x, T_END, 0.5, *LEFT[:3], *RIGHT[:3], gamma=GAMMA,
    )
    dx = x[1] - x[0]
    e_rho = np.sum(np.abs(w[:, 0] - rho_ex)) * dx
    # Lax's stronger waves make slightly larger L1 than Sod — a bound of 0.05
    # on normalized-density L1 is standard for MUSCL-Hancock+HLLC+minmod.
    assert e_rho < 0.05, f"L1(ρ) Lax = {e_rho:.4f}"
