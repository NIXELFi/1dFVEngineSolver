"""Choked / subsonic restrictor BC — ghost-cell filler.

Models an FSAE-style isentropic nozzle with discharge coefficient as the
upstream boundary. Given stagnation conditions (p_0, T_0), throat area
A_t and Cd, and the downstream pipe static state, compute the ṁ that
the restrictor delivers and fill the ghost cells such that the HLLC flux
at the boundary face carries that ṁ.

Two regimes:
  - Choked (p_down/p_0 ≤ critical): ṁ is independent of p_down, set by
    choked-nozzle formula.
  - Subsonic (critical < p_down/p_0 < 1): ṁ depends on p_down via the
    compressible-orifice equation.

After ṁ is known, ghost-cell primitives are derived from:
  - p_ghost: copy p from interior first real cell (zero-gradient in p,
    which is correct for subsonic inflow — the pressure information comes
    from downstream).
  - T_ghost: from stagnation conservation along the isentrope:
      T_ghost = T_0 · (p_ghost / p_0)^((γ-1)/γ)
  - ρ_ghost = p_ghost / (R · T_ghost)
  - u_ghost = ṁ / (ρ_ghost · A_pipe)   (mass conservation into the pipe)
  - Y_ghost = 0 (fresh atmospheric air)

The ghost cells are filled with these primitives; the subsequent MUSCL
reconstruction and HLLC face flux naturally deliver the chosen ṁ into
the first real cell.
"""

from __future__ import annotations

import numpy as np

from solver.state import PipeState, I_RHO_A, I_MOM_A, I_E_A, I_Y_A


def _critical_pressure_ratio(gamma: float) -> float:
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def _choke_factor(gamma: float) -> float:
    return (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def restrictor_mdot(
    p_down: float, p_0: float, T_0: float,
    A_t: float, Cd: float, gamma: float, R_gas: float,
) -> float:
    """Mass flow through an isentropic nozzle with discharge coefficient.

    Choked when p_down/p_0 ≤ (2/(γ+1))^(γ/(γ-1)); otherwise subsonic."""
    if p_0 <= 0.0 or A_t <= 0.0 or Cd <= 0.0:
        return 0.0
    pr = max(p_down / p_0, 0.0)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    if pr >= 1.0:
        return 0.0
    if pr <= _critical_pressure_ratio(gamma):
        # Choked
        return Cd * A_t * p_0 * np.sqrt(gamma / (R_gas * T_0)) * _choke_factor(gamma)
    # Subsonic
    t1 = pr ** (2.0 / gamma)
    t2 = pr ** (gp1 / gamma)
    flow_fn = np.sqrt(max(2.0 * gamma / gm1 * (t1 - t2), 0.0))
    return Cd * A_t * p_0 / np.sqrt(R_gas * T_0) * flow_fn


def fill_choked_restrictor_left(
    state: PipeState,
    p_0: float, T_0: float,
    A_t: float, Cd: float,
) -> float:
    """Fill the left ghost cells to deliver the restrictor's ṁ into the pipe.

    Returns the ṁ that the BC is imposing (kg/s) so the caller can log it.
    """
    ng = state.n_ghost
    gamma = state.gamma
    gm1 = gamma - 1.0
    R_gas = state.R_gas

    src = ng  # first real cell
    A_src = state.area[src]
    rho_src = state.q[src, I_RHO_A] / A_src
    u_src = state.q[src, I_MOM_A] / (rho_src * A_src)
    E_src = state.q[src, I_E_A] / A_src
    p_src = gm1 * (E_src - 0.5 * rho_src * u_src * u_src)
    p_src = max(p_src, 1e-3 * p_0)

    mdot = restrictor_mdot(p_src, p_0, T_0, A_t, Cd, gamma, R_gas)

    # Ghost primitives
    p_ghost = p_src  # zero-gradient in p for subsonic inflow
    T_ghost = T_0 * (p_ghost / p_0) ** (gm1 / gamma)
    rho_ghost = p_ghost / (R_gas * max(T_ghost, 1.0))

    for i in range(ng):
        A_g = state.area[i]
        u_ghost = mdot / (rho_ghost * A_g)
        E_ghost = p_ghost / gm1 + 0.5 * rho_ghost * u_ghost * u_ghost
        state.q[i, I_RHO_A] = rho_ghost * A_g
        state.q[i, I_MOM_A] = rho_ghost * u_ghost * A_g
        state.q[i, I_E_A]   = E_ghost * A_g
        state.q[i, I_Y_A]   = 0.0

    return float(mdot)
