"""Subsonic inflow and back-pressure outflow BCs for steady nozzle tests.

These are simple over-prescribed BCs suitable for steady quasi-1D nozzle
validation. They are NOT characteristic-correct and must not be used in
the engine model. The engine uses purpose-specific ghost-cell BCs.

Subsonic inflow: we impose all four primitives (ρ, u, p, Y) in the ghost
cells. For a true 1D subsonic inflow only 3 characteristics enter, so this
is mildly overdetermined. For a prescribed steady state it is adequate:
the simulation discards the outgoing characteristic and settles to the
prescribed state.

Subsonic outflow: we impose p in the ghost cells and extrapolate ρ, u, Y
from the adjacent interior cell. Only 1 characteristic enters, so this is
correct for subsonic outflow.
"""

from __future__ import annotations

import numpy as np

from solver.state import PipeState, I_RHO_A, I_MOM_A, I_E_A, I_Y_A


def fill_subsonic_inflow_left(state: PipeState, rho: float, u: float, p: float, Y: float) -> None:
    ng = state.n_ghost
    gamma = state.gamma
    gm1 = gamma - 1.0
    E = p / gm1 + 0.5 * rho * u * u
    for i in range(ng):
        A = state.area[i]
        state.q[i, I_RHO_A] = rho * A
        state.q[i, I_MOM_A] = rho * u * A
        state.q[i, I_E_A]   = E * A
        state.q[i, I_Y_A]   = rho * Y * A


def fill_subsonic_outflow_right(state: PipeState, p_back: float) -> None:
    """Set right ghosts to the interior primitives (ρ, u, Y) with p replaced."""
    ng = state.n_ghost
    nc = state.n_cells
    gamma = state.gamma
    gm1 = gamma - 1.0
    src = ng + nc - 1
    A_src = state.area[src]
    rho = state.q[src, I_RHO_A] / A_src
    u = state.q[src, I_MOM_A] / (rho * A_src)
    Y = state.q[src, I_Y_A] / (rho * A_src)
    p = p_back
    E = p / gm1 + 0.5 * rho * u * u
    for i in range(ng + nc, state.n_total):
        A = state.area[i]
        state.q[i, I_RHO_A] = rho * A
        state.q[i, I_MOM_A] = rho * u * A
        state.q[i, I_E_A]   = E * A
        state.q[i, I_Y_A]   = rho * Y * A
