"""Entropy-aware valve ghost-cell BC — the V1 fix.

V1's MOC valve BC carried pipe-side entropy (AA) across the valve,
underpredicting exhaust pipe temperature by ~75 % and wave speed by ~2×.
V2's V2 valve BC tracks composition Y = burned mass fraction as a
conservative scalar and sets the ghost cell's (ρ, T, Y) to the reservoir
side (cylinder for exhaust outflow, pipe for intake outflow), so the
HLLC contact wave carries the correct entropy into the pipe.

Mechanism:
  1. Compute effective valve flow area A_eff(θ) from lift × Cd tables.
  2. If A_eff ≈ 0: valve closed, use reflective ghost (wall).
  3. Determine flow direction from (p_cyl, p_pipe_interior).
  4. Compute ṁ via compressible-orifice equation using UPSTREAM gas
     properties (p_up, T_up, R_up, γ_up, composition Y_up).
  5. Set ghost primitives:
        p_ghost = p_pipe_interior  (zero-gradient for subsonic coupling)
        T_ghost = T_up · (p_ghost / p_up)^((γ-1)/γ)  (isentropic expansion)
        ρ_ghost = p_ghost / (R_up · T_ghost)
        u_ghost = ṁ / (ρ_ghost · A_pipe)  with sign from flow direction
        Y_ghost = Y_up           (composition of upstream gas)

HLLC at the ghost/interior face then carries u, ρ, p, Y through the
contact wave correctly; the computed flux is what actually enters/leaves
the pipe, and the cylinder reads the same flux back (see
caller in models/sdm26.py for the flux-return plumbing).

SI units throughout. No Benson non-dim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from solver.state import PipeState, I_RHO_A, I_MOM_A, I_E_A, I_Y_A
from cylinder.gas_properties import gamma_mixture, R_mixture, R_AIR
from cylinder.valve import ValveParams, valve_effective_area


@dataclass
class ValveBC:
    """Configures a valve boundary at one end of a pipe."""
    pipe_end: str              # "left" (exhaust) or "right" (intake)
    valve_type: str            # "intake" or "exhaust"
    valve: ValveParams


def _mass_flow_orifice(
    p_up: float, T_up: float, p_down: float,
    A_eff: float, gamma: float, R_gas: float,
) -> float:
    """Compressible-orifice ṁ from upstream stagnation state to downstream p.

    A_eff here should ALREADY include n_valves · Cd (that's what
    cylinder.valve.valve_effective_area returns), so this function does not
    re-multiply by Cd.
    """
    if p_up <= 0.0 or A_eff <= 0.0:
        return 0.0
    pr = p_down / p_up
    if pr >= 1.0:
        return 0.0
    if pr < 0.0:
        pr = 0.0
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if pr <= pr_crit:
        choke = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        return A_eff * p_up * np.sqrt(gamma / (R_gas * max(T_up, 100.0))) * choke
    t1 = pr ** (2.0 / gamma)
    t2 = pr ** ((gamma + 1.0) / gamma)
    flow_fn = np.sqrt(max(2.0 * gamma / (gamma - 1.0) * (t1 - t2), 0.0))
    return A_eff * p_up / np.sqrt(R_gas * max(T_up, 100.0)) * flow_fn


def _fill_reflective_at_end(state: PipeState, pipe_end: str) -> None:
    """Wall-like ghost fill for a closed valve."""
    from bcs.simple import fill_reflective_left, fill_reflective_right
    if pipe_end == "left":
        fill_reflective_left(state)
    else:
        fill_reflective_right(state)


def fill_valve_ghost(
    state: PipeState,
    pipe_end: str,
    valve_type: str,
    vp: ValveParams,
    theta_local_deg: float,
    p_cyl: float, T_cyl: float, xb_cyl: float,
) -> float:
    """Fill valve-side ghost cells. Returns the BC's claimed ṁ (kg/s).

    Sign convention for the returned ṁ:
        Exhaust valve (pipe_end='left'): positive ṁ = flow cyl → pipe.
        Intake valve  (pipe_end='right'): positive ṁ = flow pipe → cyl.

    The cylinder's mass balance will be updated by the caller using the
    ACTUAL HLLC flux at the boundary face (not this "claimed" ṁ), so this
    return value is advisory.
    """
    ng = state.n_ghost
    gamma_pipe = state.gamma  # hardware γ used by HLLC; pipe is frozen γ=1.4

    # Effective flow area
    A_eff = valve_effective_area(
        theta_local_deg, vp.open_angle_deg, vp.close_angle_deg, vp.max_lift,
        vp.diameter, np.radians(vp.seat_angle_deg), vp.n_valves,
        vp.ld_table, vp.cd_table,
    )
    if A_eff < 1e-12:
        _fill_reflective_at_end(state, pipe_end)
        return 0.0

    # Index into pipe for the first real cell on the valve side
    if pipe_end == "left":
        i_real = ng
    else:
        i_real = ng + state.n_cells - 1
    A_pipe = state.area[i_real]
    gm1 = gamma_pipe - 1.0
    rho_pipe = state.q[i_real, I_RHO_A] / A_pipe
    u_pipe = state.q[i_real, I_MOM_A] / (rho_pipe * A_pipe)
    E_pipe = state.q[i_real, I_E_A] / A_pipe
    p_pipe = gm1 * (E_pipe - 0.5 * rho_pipe * u_pipe * u_pipe)
    p_pipe = max(p_pipe, 1.0)
    Y_pipe = state.q[i_real, I_Y_A] / (rho_pipe * A_pipe)

    # Decide flow direction
    # Forward (cyl → pipe at exhaust, pipe → cyl at intake) happens when
    # the UPSTREAM pressure exceeds the DOWNSTREAM pressure.
    if valve_type == "exhaust":
        # Cylinder is upstream for forward; pipe is upstream for reverse
        forward = p_cyl >= p_pipe
    else:
        # Intake: pipe is upstream for forward (pipe→cyl); cyl upstream for reverse
        forward = p_pipe >= p_cyl

    # Upstream reservoir state
    if forward and valve_type == "exhaust":
        # Cylinder → pipe. Use cylinder γ, R, composition.
        p_up = p_cyl
        T_up = T_cyl
        gamma_up = gamma_mixture(T_cyl, xb_cyl)
        R_up = R_mixture(xb_cyl)
        Y_up = xb_cyl
        p_down = p_pipe
    elif (not forward) and valve_type == "exhaust":
        # Pipe → cylinder. Pipe is upstream.
        p_up = p_pipe
        T_up = max(p_pipe / (max(rho_pipe, 1e-6) * R_AIR), 100.0)
        gamma_up = gamma_pipe
        R_up = R_AIR  # could use R_mixture(Y_pipe) but R_pipe frozen
        Y_up = Y_pipe
        p_down = p_cyl
    elif forward and valve_type == "intake":
        # Pipe → cylinder. Pipe is upstream.
        p_up = p_pipe
        T_up = max(p_pipe / (max(rho_pipe, 1e-6) * R_AIR), 100.0)
        gamma_up = gamma_pipe
        R_up = R_AIR
        Y_up = Y_pipe
        p_down = p_cyl
    else:
        # Intake backflow: cylinder → pipe
        p_up = p_cyl
        T_up = T_cyl
        gamma_up = gamma_mixture(T_cyl, xb_cyl)
        R_up = R_mixture(xb_cyl)
        Y_up = xb_cyl
        p_down = p_pipe

    # ṁ through the valve orifice
    mdot = _mass_flow_orifice(p_up, T_up, p_down, A_eff, gamma_up, R_up)

    # Ghost primitive state (reservoir-biased isentropic expansion to pipe p)
    # p_ghost takes pipe pressure (subsonic coupling; for choked, the ghost
    # pressure doesn't actually matter because upstream is unaware).
    p_ghost = p_pipe
    ratio = max(p_ghost / max(p_up, 1.0), 1e-6)
    T_ghost = max(T_up * ratio ** (gm1 / gamma_up), 100.0)
    rho_ghost = p_ghost / (R_up * T_ghost)
    # Ghost velocity (sign) — flow-direction-aware:
    # - Exhaust pipe (LEFT end): forward flow has u_ghost > 0 (into pipe).
    # - Intake pipe (RIGHT end): forward flow has u_ghost > 0 (out of pipe,
    #   toward cylinder). But since ghost is OUTSIDE the pipe to the right,
    #   flow from pipe INTO the ghost means u_ghost > 0 also. Same sign!
    if forward:
        u_mag = mdot / max(rho_ghost * A_pipe, 1e-20)
    else:
        u_mag = -mdot / max(rho_ghost * A_pipe, 1e-20)

    # Apply to ghost cells
    if pipe_end == "left":
        ghost_indices = range(0, ng)
    else:
        ghost_indices = range(ng + state.n_cells, state.n_total)

    gm1_pipe = gamma_pipe - 1.0  # the FV update uses frozen-γ for energy close
    for i in ghost_indices:
        A_g = state.area[i]
        # Use pipe's frozen γ for E calc (the pipe HLLC uses this γ)
        E_ghost = p_ghost / gm1_pipe + 0.5 * rho_ghost * u_mag * u_mag
        state.q[i, I_RHO_A] = rho_ghost * A_g
        state.q[i, I_MOM_A] = rho_ghost * u_mag * A_g
        state.q[i, I_E_A]   = E_ghost * A_g
        state.q[i, I_Y_A]   = rho_ghost * Y_up * A_g

    # Signed return value per the docstring convention
    if valve_type == "exhaust":
        return mdot if forward else -mdot  # + = cyl→pipe
    else:
        return mdot if forward else -mdot  # + = pipe→cyl
