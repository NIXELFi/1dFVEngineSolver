"""Toro test 2: the 123 problem (two strong symmetric rarefactions).

Initial data (Toro Table 4.1 row 3):
    L: ρ=1.0, u=-2.0, p=0.4
    R: ρ=1.0, u=+2.0, p=0.4
    γ = 1.4, domain [0, 1], x0=0.5, t_end = 0.15.

The flow creates near-vacuum in the middle. Many solvers fail (generate
negative pressure or density). HLLC with Einfeldt-Batten wave speeds
handles it.
"""

from __future__ import annotations

import numpy as np

from tests.harness import run_riemann
from tests.exact_riemann import sample_array, solve_star


GAMMA = 1.4
LEFT = (1.0, -2.0, 0.4, 0.0)
RIGHT = (1.0, 2.0, 0.4, 1.0)
T_END = 0.15


def test_123_positivity_preserved():
    """No cell ever shows ρ ≤ 0 or p ≤ 0 through the entire evolution."""
    x, w, state, steps = run_riemann(
        n_cells=400, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    assert (w[:, 0] > 0).all(), "density positivity violated"
    assert (w[:, 2] > 0).all(), "pressure positivity violated"


def test_123_star_region_near_vacuum():
    """Density in the central region (near x=0.5) should collapse to near
    zero. Exact star-state density for this problem is ~0.003 (near vacuum)."""
    x, w, _, _ = run_riemann(
        n_cells=400, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    i_mid = len(x) // 2
    assert w[i_mid, 0] < 0.03, f"central ρ = {w[i_mid,0]:.4f}, expected near-vacuum"


def test_123_symmetric():
    """The solution is symmetric about x=0.5. Check ρ(x) ≈ ρ(1-x) and
    u(x) ≈ -u(1-x) in a cell-wise L-infinity sense."""
    x, w, _, _ = run_riemann(
        n_cells=400, length=1.0, x0=0.5,
        left=LEFT, right=RIGHT, t_end=T_END, gamma=GAMMA,
    )
    n = len(x)
    # Flip and compare
    rho_flip = w[::-1, 0]
    u_flip = -w[::-1, 1]
    p_flip = w[::-1, 2]
    rho_dev = np.max(np.abs(w[:, 0] - rho_flip))
    u_dev = np.max(np.abs(w[:, 1] - u_flip))
    p_dev = np.max(np.abs(w[:, 2] - p_flip))
    assert rho_dev < 1e-8, f"symmetry ρ dev = {rho_dev:.2e}"
    assert u_dev   < 1e-8, f"symmetry u dev = {u_dev:.2e}"
    assert p_dev   < 1e-8, f"symmetry p dev = {p_dev:.2e}"
