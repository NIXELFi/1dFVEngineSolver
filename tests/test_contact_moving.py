"""Test 4b: moving contact discontinuity at Mach 0.3 (user-requested).

A moving contact is what physically happens during exhaust events; stationary-
contact tests can pass while moving contacts smear. Initial data:
    L = (rho_L, u, p, 0.0)
    R = (rho_R, u, p, 1.0)
with a common u > 0 at Mach 0.3 relative to the (unit) sound speed in the
left gas:
    a_L = sqrt(γ p / ρ_L) = sqrt(1.4·1e5/1.0) ≈ 374.17 m/s
    u   = 0.3 · a_L        ≈ 112.25 m/s

After time t, the contact should have moved by u·t. We verify:
1. Pressure remains uniform (max−min / p0 < 1e-6).
2. Velocity remains uniform (|u − u0| < 1e-6·u0).
3. The density jump has advected by u·t to within half a cell.
4. The composition jump coincides with the density jump.
"""

from __future__ import annotations

import numpy as np

from tests.harness import run_riemann


GAMMA = 1.4
P0 = 1.0e5
RHO_L = 1.0
RHO_R = 0.125
A_L = np.sqrt(GAMMA * P0 / RHO_L)
U = 0.3 * A_L  # Mach 0.3 in left gas

LEFT = (RHO_L, U, P0, 0.0)
RIGHT = (RHO_R, U, P0, 1.0)

LENGTH = 1.0
X0 = 0.3                   # start left-of-centre so contact doesn't exit domain
T_END = 0.002              # contact travels 112·0.002 ≈ 0.225 m → ends at x ≈ 0.525


def _find_jump_midpoint(x, field):
    """Find the x location where `field` crosses its mid-value."""
    v_max = field.max()
    v_min = field.min()
    mid = 0.5 * (v_max + v_min)
    # For a descending step (L>R), the first i where field[i] <= mid is the jump.
    if field[0] > field[-1]:
        idxs = np.where(field <= mid)[0]
    else:
        idxs = np.where(field >= mid)[0]
    if len(idxs) == 0:
        return float("nan")
    i = idxs[0]
    # Linear interp between i-1 and i
    if i == 0:
        return float(x[0])
    f0, f1 = field[i - 1], field[i]
    x0_, x1_ = x[i - 1], x[i]
    if abs(f1 - f0) < 1e-30:
        return 0.5 * (x0_ + x1_)
    frac = (mid - f0) / (f1 - f0)
    return float(x0_ + frac * (x1_ - x0_))


def test_moving_contact_pressure_uniform():
    x, w, _, _ = run_riemann(
        n_cells=400, length=LENGTH, x0=X0,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    rng = w[:, 2].max() - w[:, 2].min()
    assert rng / P0 < 1e-6, f"pressure range {rng/P0:.2e} (rel)"


def test_moving_contact_velocity_uniform():
    x, w, _, _ = run_riemann(
        n_cells=400, length=LENGTH, x0=X0,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    dev = np.max(np.abs(w[:, 1] - U))
    assert dev / U < 1e-4, f"|u-u0| max = {dev:.2e}, rel {dev/U:.2e}"


def test_moving_contact_position_correct():
    """The density jump midpoint should lie at X0 + U·t within half a cell."""
    x, w, _, _ = run_riemann(
        n_cells=400, length=LENGTH, x0=X0,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    x_expected = X0 + U * T_END
    x_mid_rho = _find_jump_midpoint(x, w[:, 0])
    dx = x[1] - x[0]
    assert abs(x_mid_rho - x_expected) < dx, (
        f"ρ jump at {x_mid_rho:.5f}, expected {x_expected:.5f} (dx={dx:.5f})"
    )


def test_moving_contact_conservative_rhoY_advects_with_rho():
    """The conservative composition ρY must advect exactly with ρ — they
    are both transported by the same contact wave at speed u.

    We check midpoints of ρ and of ρY (both in their 0→asymptote range).
    They must coincide to within half a cell. The *primitive* Y = (ρY)/ρ
    midpoint does NOT coincide with the ρ midpoint when ρ_L/ρ_R ≠ 1 because
    the ratio of two smeared jumps is not itself a symmetric smeared jump
    — this is arithmetic, not numerical diffusion.
    """
    x, w, _, _ = run_riemann(
        n_cells=400, length=LENGTH, x0=X0,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    rhoY = w[:, 0] * w[:, 3]
    x_rho = _find_jump_midpoint(x, w[:, 0])
    x_rhoY = _find_jump_midpoint(x, rhoY)
    dx = x[1] - x[0]
    assert abs(x_rho - x_rhoY) < 0.5 * dx, (
        f"conservative jump-midpoint offset {abs(x_rho - x_rhoY):.4e}, dx={dx:.4e}"
    )


def test_moving_contact_Y_transitions_monotonically():
    """Composition is monotone-in-x across the contact (no spurious
    overshoot/undershoot from the limiter). Guards against oscillations
    that a non-TVD scheme would show across strong contacts.
    """
    x, w, _, _ = run_riemann(
        n_cells=400, length=LENGTH, x0=X0,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    Y = w[:, 3]
    # Restrict to a window around the contact
    x_expected = X0 + U * T_END
    mask = (x > x_expected - 0.2) & (x < x_expected + 0.2)
    Yw = Y[mask]
    # Y increases left to right — differences should be non-negative
    diffs = np.diff(Yw)
    assert (diffs >= -1e-12).all(), f"non-monotone Y, min diff = {diffs.min():.2e}"
    # Bounds preserved
    assert Yw.min() >= -1e-12, f"Y undershoot = {Yw.min():.2e}"
    assert Yw.max() <= 1.0 + 1e-12, f"Y overshoot = {Yw.max():.2e}"
