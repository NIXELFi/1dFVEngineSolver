"""Test 4: stationary contact discontinuity.

A pure contact has p and u continuous but ρ and composition Y jump. The
contact should remain at its initial position (u = 0) with no pressure
or velocity oscillation and only limiter-controlled numerical smearing
in ρ and Y.

Initial data: L = (1.0, 0, 1e5, 0), R = (0.125, 0, 1e5, 1), γ = 1.4,
domain [0, 1], x0=0.5, t_end = 0.01 s (long enough that spurious
oscillations would be visible).
"""

from __future__ import annotations

import numpy as np

from tests.harness import run_riemann


GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0e5, 0.0)
RIGHT = (0.125, 0.0, 1.0e5, 1.0)
T_END = 0.01


def test_stationary_contact_pressure_constant():
    """Pressure is uniform and equal to the initial value everywhere."""
    x, w, _, _ = run_riemann(
        n_cells=200, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    p_max, p_min = w[:, 2].max(), w[:, 2].min()
    assert (p_max - p_min) / 1e5 < 1e-6, f"pressure range {p_max - p_min:.2e}"


def test_stationary_contact_velocity_zero():
    """Velocity remains zero (machine precision)."""
    x, w, _, _ = run_riemann(
        n_cells=200, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    assert np.max(np.abs(w[:, 1])) < 1e-6, f"max |u| = {np.max(np.abs(w[:,1])):.2e}"


def test_stationary_contact_density_jump_preserved():
    """Far from the interface, ρ holds its initial values."""
    x, w, _, _ = run_riemann(
        n_cells=200, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    # Left side: x < 0.4 should still be ~1.0
    i_far_left = np.argmin(np.abs(x - 0.3))
    assert abs(w[i_far_left, 0] - 1.0) < 1e-3
    # Right side: x > 0.6 should still be ~0.125
    i_far_right = np.argmin(np.abs(x - 0.7))
    assert abs(w[i_far_right, 0] - 0.125) < 1e-3


def test_stationary_contact_composition_jump_preserved():
    """Same for the composition Y."""
    x, w, _, _ = run_riemann(
        n_cells=200, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    i_far_left = np.argmin(np.abs(x - 0.3))
    assert w[i_far_left, 3] < 1e-3
    i_far_right = np.argmin(np.abs(x - 0.7))
    assert w[i_far_right, 3] > 1.0 - 1e-3
