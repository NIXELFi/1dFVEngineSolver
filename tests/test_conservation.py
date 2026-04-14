"""Test 5: closed-domain mass and energy conservation.

Sealed (reflective) ends, non-trivial initial state, 1000+ steps. Total
mass and total energy must be preserved to machine precision. This is the
V1-failure test: V1 MOC leaks mass because its BCs set flux that the
interior doesn't enforce; V2's FV + HLLC scheme conserves by construction
because face fluxes are the only thing updating cell averages, and each
interior face's flux appears with opposite signs in the two neighboring
cells' updates. Reflective ghost-cell fills at the ends give zero
boundary flux.

We also check that the composition scalar ρY is conserved.
"""

from __future__ import annotations

import numpy as np

from solver.state import make_pipe_state, set_uniform, total_mass, total_energy
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from bcs.simple import fill_reflective_left, fill_reflective_right


GAMMA = 1.4


def _run_sealed(n_cells: int, n_steps: int, seed: int = 0):
    """Initialise with sine-perturbed density/energy and a uniform baseline,
    then run sealed for n_steps. Return (mass0, mass_final, e0, e_final, rhoY0, rhoY_final)."""
    rng = np.random.default_rng(seed)
    state = make_pipe_state(
        n_cells=n_cells, length=1.0,
        area_fn=lambda x: 1.0,  # constant area
        gamma=GAMMA, n_ghost=2,
    )
    # Nontrivial state: pressure ~1 bar with a sine bump, velocity zero,
    # density with sine variation, composition linearly ramping 0→1.
    ng = state.n_ghost
    nc = state.n_cells
    for i in range(state.n_total):
        i_real = i - ng
        x = (i_real + 0.5) * state.dx
        rho = 1.0 + 0.5 * np.sin(2 * np.pi * x)
        p = 1.0e5 + 3e4 * np.cos(4 * np.pi * x)
        u = 0.0
        Y = max(0.0, min(1.0, x))  # clamp to [0,1]
        gm1 = GAMMA - 1.0
        E = p / gm1 + 0.5 * rho * u * u
        A = state.area[i]
        state.q[i, 0] = rho * A
        state.q[i, 1] = rho * u * A
        state.q[i, 2] = E * A
        state.q[i, 3] = rho * Y * A

    def totals():
        s = state.real_slice()
        return (
            float(state.dx * state.q[s, 0].sum()),
            float(state.dx * state.q[s, 2].sum()),
            float(state.dx * state.q[s, 3].sum()),
        )

    m0, e0, rhoY0 = totals()

    n = state.n_total
    w = np.zeros((n, 4))
    slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4))
    wR = np.zeros((n, 4))
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

    m1, e1, rhoY1 = totals()
    return m0, m1, e0, e1, rhoY0, rhoY1


def test_mass_conserved_to_machine_precision():
    m0, m1, _, _, _, _ = _run_sealed(n_cells=100, n_steps=1000)
    rel = abs(m1 - m0) / m0
    assert rel < 1e-12, f"relative mass drift after 1000 steps = {rel:.2e}"


def test_energy_conserved_to_machine_precision():
    _, _, e0, e1, _, _ = _run_sealed(n_cells=100, n_steps=1000)
    rel = abs(e1 - e0) / e0
    assert rel < 1e-12, f"relative energy drift after 1000 steps = {rel:.2e}"


def test_composition_conserved_to_machine_precision():
    _, _, _, _, rhoY0, rhoY1 = _run_sealed(n_cells=100, n_steps=1000)
    rel = abs(rhoY1 - rhoY0) / rhoY0
    assert rel < 1e-12, f"relative ρY drift after 1000 steps = {rel:.2e}"


def test_conservation_scales_with_steps():
    """Drift should stay at roundoff regardless of step count."""
    m0a, m1a, e0a, e1a, _, _ = _run_sealed(n_cells=80, n_steps=500)
    m0b, m1b, e0b, e1b, _, _ = _run_sealed(n_cells=80, n_steps=2000)
    assert abs(m1a - m0a) / m0a < 1e-12
    assert abs(m1b - m0b) / m0b < 1e-12
    assert abs(e1a - e0a) / e0a < 1e-12
    assert abs(e1b - e0b) / e0b < 1e-12
