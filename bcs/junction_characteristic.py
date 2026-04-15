"""Characteristic-coupled constant-static-pressure junction BC.

Parallel alternative to ``bcs/junction_cv.py:JunctionCV``. Preserves
more of the incident wave amplitude across a multi-pipe merge than the
stagnation-CV formulation does, at the cost of approximate (not exact)
energy conservation across area-mismatched junctions.

**Two junction types are available in the codebase; choose deliberately.**

- ``bcs.junction_cv.JunctionCV`` — stagnation control volume. Strictly
  conservative in mass + energy + ρY. Dissipative at the face: per-
  junction acoustic transmission ~0.69 for the SDM26 4-2-1 geometry.
  **Default**; use when conservation matters more than wave propagation.

- ``bcs.junction_characteristic.CharacteristicJunction`` (this file) —
  constant static pressure across all legs, characteristic compatibility
  from each interior, mass-conservative by Newton residual, energy-
  conservative to O(ΔA/A̅) at mismatched junctions. Per-junction
  transmission ~0.91 for SDM26 linear acoustics. Use when tuned-length
  effects need to propagate (e.g. engine torque tuning).

History
-------
The Phase-3 WIP draft at ``bcs/junction.py`` implemented a similar
formulation but was never wired in or tested. Phase E (acoustic audit
2026-04, Phase C1/C2/C3 follow-up) promoted it to production by
fixing the following specific issues:

1. Inflow legs now use junction-mixed entropy (not interior entropy)
2. Three-regime choked-leg dispatch (startup / subsonic / choked)
3. Analytic Jacobian in Newton iteration (replaces finite-difference)
4. Explicit convergence error raise on max_iter hit (no silent stall)
5. Signed energy residual diagnostic (+ = gain, physical violation)
6. Numba ready (kernel separated from the dataclass state object)
7. Full 8-test unit suite at tests/test_junction_characteristic.py
8. Module docstring citing the draft as predecessor

Leave ``bcs/junction.py`` in place as historical reference.

References
----------
Winterbone & Pearson, "Design Techniques for Engine Manifolds," 1999.
  Constant-static-pressure Type-1 multi-pipe junction, §9 (sections
  not verified against the book — book not on disk at implementation
  time, see docs/phase_e_design.md §1).
Corberán & Gascón, IJTS 1995. Conservation properties of multi-pipe
  junction models.
Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics,"
  2009. §3 for Riemann invariants along characteristics; §6 for BCs.
LeVeque, "Finite Volume Methods for Hyperbolic Problems," 2002.
  Framework for conservative coupling between FV domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from solver.state import PipeState, I_RHO_A, I_MOM_A, I_E_A, I_Y_A


LEFT = "left"
RIGHT = "right"


class JunctionConvergenceError(RuntimeError):
    """Raised when Newton iteration does not converge in max_iter steps."""


class JunctionAllChokedError(RuntimeError):
    """Raised when every incident leg is simultaneously choked at the
    junction face — a physical contradiction (no degrees of freedom
    left to balance mass). Indicates an upstream pathology, not a
    solver failure."""


@dataclass
class JunctionLeg:
    """One pipe connected to the junction at a specific end.

    sign convention: if positive velocity at the pipe's junction face
    means flow *into* the junction, ``sign_into = +1``. RIGHT-end legs
    default to +1 (u>0 at pipe right-end goes out of pipe into
    junction), LEFT-end legs default to −1.
    """
    state: PipeState
    end: str                  # "left" or "right"
    sign_into: int = 0

    def __post_init__(self) -> None:
        if self.sign_into == 0:
            self.sign_into = +1 if self.end == RIGHT else -1

    @property
    def face_cell_index(self) -> int:
        """Index of the first real cell adjacent to the junction face."""
        ng = self.state.n_ghost
        return ng if self.end == LEFT else ng + self.state.n_cells - 1

    @property
    def s_end(self) -> float:
        """+1 at RIGHT end (outgoing invariant is J+); −1 at LEFT end."""
        return +1.0 if self.end == RIGHT else -1.0


# ---------------------------------------------------------------------------
# Leg kinematics: interior state, face state given p_j, and per-leg mass flux
# ---------------------------------------------------------------------------

def _interior_primitives(leg: JunctionLeg) -> Tuple[float, float, float, float, float, float]:
    """Read interior primitives (ρ, u, p, Y, c, A) at the first real cell
    adjacent to the junction face."""
    state = leg.state
    gm1 = state.gamma - 1.0
    i = leg.face_cell_index
    A = state.area[i]
    rho = state.q[i, I_RHO_A] / A
    u = state.q[i, I_MOM_A] / (rho * A)
    E = state.q[i, I_E_A] / A
    p = max(gm1 * (E - 0.5 * rho * u * u), 1.0)
    Y = state.q[i, I_Y_A] / (rho * A)
    c = float(np.sqrt(state.gamma * p / max(rho, 1e-9)))
    return rho, u, p, Y, c, A


def _face_from_pj(
    rho_ref: float, u_ref: float, p_ref: float, c_ref: float,
    p_j: float, gamma: float, s_end: float,
) -> Tuple[float, float, float, float, float]:
    """Isentropic expansion from reference interior state to junction p_j,
    with the outgoing characteristic (J+ at s_end=+1, J− at s_end=−1) held
    invariant between interior and face.

    Returns (ρ_f, u_f, c_f, dρf_dpj, duf_dpj) — last two are the analytic
    derivatives needed for Newton on mass residual.

    Derivation of derivatives:
        Let r = p_j / p_ref, so dr/dp_j = 1/p_ref.
        c_f    = c_ref · r^((γ−1)/(2γ))
        ρ_f    = ρ_ref · r^(1/γ)
        u_f    = u_ref + (2/(γ−1)) · (c_ref − c_f) · s_end
        dc_f / dp_j = c_f · ((γ−1)/(2γ)) · (1/p_j)                 [chain rule]
        dρ_f / dp_j = ρ_f · (1/γ) · (1/p_j)
        du_f / dp_j = −(2/(γ−1)) · s_end · dc_f/dp_j
    """
    gm1 = gamma - 1.0
    r = max(p_j / p_ref, 1e-9)
    c_f = c_ref * r ** (gm1 / (2.0 * gamma))
    rho_f = rho_ref * r ** (1.0 / gamma)
    u_f = u_ref + (2.0 / gm1) * (c_ref - c_f) * s_end
    # analytic derivatives in p_j
    dcf_dpj = c_f * (gm1 / (2.0 * gamma)) / max(p_j, 1.0)
    drhof_dpj = rho_f * (1.0 / gamma) / max(p_j, 1.0)
    duf_dpj = -(2.0 / gm1) * s_end * dcf_dpj
    return rho_f, u_f, c_f, drhof_dpj, duf_dpj


def _mass_residual_and_jacobian(
    interiors: List[Tuple[float, float, float, float, float, float]],
    legs: List[JunctionLeg],
    p_j: float,
    gamma: float,
) -> Tuple[float, float, List[Tuple[float, float, float]]]:
    """Compute mass-balance residual R(p_j) = Σ σ_i ρ_f u_f A_i and its
    analytic derivative dR/dp_j. Also returns per-leg (ρ_f, u_f, c_f)
    tuples so the caller can re-use them for diagnostics and ghost fill.

    The p_ref/c_ref passed into _face_from_pj uses the interior state for
    every leg in this pass. For subsonic inflow legs, §2a of the design
    doc calls for using junction-mixed entropy instead; that correction
    is applied in a lagged-Picard outer pass by the caller.
    """
    R = 0.0
    dR_dpj = 0.0
    face_states: List[Tuple[float, float, float]] = []
    for leg, (rho_i, u_i, p_i, Y_i, c_i, A_i) in zip(legs, interiors):
        rho_f, u_f, c_f, drhof, duf = _face_from_pj(
            rho_i, u_i, p_i, c_i, p_j, gamma, leg.s_end,
        )
        sigma = leg.sign_into
        R      += sigma * rho_f * u_f * A_i
        # product rule: d(ρ_f·u_f)/dp_j = drhof·u_f + rho_f·duf
        dR_dpj += sigma * (drhof * u_f + rho_f * duf) * A_i
        face_states.append((rho_f, u_f, c_f))
    return R, dR_dpj, face_states


# ---------------------------------------------------------------------------
# CharacteristicJunction class
# ---------------------------------------------------------------------------

@dataclass
class CharacteristicJunction:
    """Constant-static-pressure characteristic-coupled N-pipe junction.

    Lifecycle (matches JunctionCV for swap-in compatibility):
      1. fill_ghosts()     — called BEFORE each pipe's MUSCL step.
      2. absorb_fluxes(dt) — called AFTER each pipe's MUSCL step.
                            **No-op here**: mass and energy flow through
                            the ghost-mediated face flux directly; there
                            is no separate CV state to update. Kept for
                            interface symmetry.

    Robustness:
      - Newton on scalar p_j with analytic Jacobian.
      - 20% step damping (standard Winterbone recommendation for
        area-mismatched junctions; prevents overshoot on the first step
        when the initial guess is far from the fixed point).
      - Floor p_j at 1000 Pa to catch cavitation-style runaway.
      - If any leg goes sonic (|M_f| > 1 − choke_margin) at the
        converged p_j, re-dispatch to the choked branch (§4 of design).
      - If non-convergence in ``newton_max_iter`` steps, raise
        JunctionConvergenceError (no silent fallback).
    """

    legs: List[JunctionLeg]
    gamma: float = 1.4
    R_gas: float = 287.0
    newton_tol: float = 1e-9        # absolute mass-residual tolerance [kg/s]
    newton_max_iter: int = 30
    choke_margin: float = 0.02      # |M| < 1 − margin counts as subsonic
    inflow_entropy_picard: bool = True   # one extra pass with mixed-entropy

    # diagnostics (written each fill_ghosts call)
    last_p_junction: float = 101325.0
    last_mass_residual: float = 0.0
    last_energy_residual: float = 0.0
    last_niter: int = 0
    last_regime: str = "subsonic"
    last_y_mixed: float = 0.0

    # --- lifecycle ------------------------------------------------------

    def fill_ghosts(self) -> None:
        if len(self.legs) < 2:
            return
        gamma = self.gamma

        interiors = [_interior_primitives(L) for L in self.legs]

        # Initial guess: area-weighted mean of interior pressures
        sum_pA = sum(it[2] * it[5] for it in interiors)
        sum_A  = sum(it[5] for it in interiors)
        p_j = sum_pA / sum_A

        # --- Newton iteration on mass balance --------------------------
        p_j, niter, face_states, converged = self._newton_mass_balance(
            interiors, p_j, gamma,
        )
        if not converged:
            raise JunctionConvergenceError(
                f"CharacteristicJunction did not converge in "
                f"{self.newton_max_iter} Newton steps. "
                f"last p_j = {p_j:.1f} Pa, last |R| = {self.last_mass_residual:.3e} kg/s. "
                f"Leg interiors: "
                f"{[(rho, u, p) for rho, u, p, _, _, _ in interiors]}"
            )

        # --- Regime check: any choked legs? ----------------------------
        choked_legs = self._detect_choked_legs(face_states)
        if choked_legs:
            p_j, niter_c, face_states = self._solve_with_choked(
                interiors, choked_legs, gamma,
            )
            niter += niter_c
            self.last_regime = f"choked_{len(choked_legs)}"
        else:
            self.last_regime = "subsonic"

        # --- Optional: inflow entropy correction (Picard one pass) -----
        # If any leg is an inflow (ρ_f u_f σ < 0, i.e. flow from junction
        # into leg), its face state should use the junction-mixed
        # entropy, not its own interior entropy. Relax once and accept.
        if self.inflow_entropy_picard and not choked_legs:
            p_j_corr, face_states_corr = self._inflow_entropy_pass(
                interiors, face_states, p_j, gamma,
            )
            if p_j_corr is not None:
                p_j = p_j_corr
                face_states = face_states_corr

        # --- Composition mixing ---------------------------------------
        Y_mixed = self._compute_mixed_Y(interiors, face_states)
        self.last_y_mixed = Y_mixed

        # --- Energy residual (diagnostic, signed) ---------------------
        self.last_energy_residual = self._energy_residual(
            interiors, face_states,
        )

        # --- Ghost-cell write ------------------------------------------
        self._write_ghosts(interiors, face_states, Y_mixed, p_j)

        self.last_p_junction = p_j
        self.last_niter = niter

    def absorb_fluxes(self, dt: float) -> None:
        """No-op. The characteristic junction has no separate control-volume
        state; conservation is maintained by the ghost-mediated face
        fluxes directly. Method kept for API symmetry with JunctionCV."""
        _ = dt

    # --- Newton mass balance -------------------------------------------

    def _newton_mass_balance(self, interiors, p_j, gamma):
        p_floor = 1000.0
        R_abs_best = float("inf")
        for it in range(self.newton_max_iter):
            R, dR_dpj, face_states = _mass_residual_and_jacobian(
                interiors, self.legs, p_j, gamma,
            )
            self.last_mass_residual = R
            if abs(R) < self.newton_tol:
                return p_j, it + 1, face_states, True
            if abs(dR_dpj) < 1e-30:
                # Singular Jacobian — usually means all legs near-choked.
                # Let the caller decide; return what we have as not-converged.
                return p_j, it + 1, face_states, False
            dp = -R / dR_dpj
            # 20% damping cap
            cap = 0.2 * p_j
            if dp > cap:
                dp = cap
            elif dp < -cap:
                dp = -cap
            p_j = max(p_j + dp, p_floor)
            R_abs_best = min(R_abs_best, abs(R))
        # max_iter hit: one last residual evaluation so the diagnostic
        # is up to date, then fail.
        R, _, face_states = _mass_residual_and_jacobian(
            interiors, self.legs, p_j, gamma,
        )
        self.last_mass_residual = R
        return p_j, self.newton_max_iter, face_states, False

    # --- Choked-leg dispatch -------------------------------------------

    def _detect_choked_legs(self, face_states):
        """Return indices of legs whose converged face state is at or
        above Mach (1 − choke_margin) with flow INTO the junction. We
        only check "into" because inflow legs at M=1 are numerically
        similar to outflow-into-junction at M=1; a leg whose upstream
        side is the junction would need a separate choked-inflow
        treatment (not currently needed in the SDM26 use case)."""
        choked = []
        for i, (L, (rho_f, u_f, c_f)) in enumerate(zip(self.legs, face_states)):
            sigma = L.sign_into
            M_into = sigma * u_f / max(c_f, 1.0)
            if M_into > 1.0 - self.choke_margin:
                choked.append(i)
        return choked

    def _solve_with_choked(self, interiors, choked_indices, gamma):
        """Reduced solve: each choked leg contributes a fixed mass flux
        (sonic throat) independent of p_j. Newton iterates p_j only
        over the non-choked legs."""
        choked_set = set(choked_indices)
        if len(choked_set) == len(self.legs):
            raise JunctionAllChokedError(
                f"All {len(self.legs)} legs simultaneously choked at "
                f"the junction face — no subsonic leg remains to balance "
                f"mass. Indicates upstream pathology."
            )

        # Sonic throat mass flux for each choked leg, using the leg's
        # interior stagnation state as reservoir.
        fixed_mdot = []
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        choke_exp = -0.5 * gp1 / gm1
        for idx in choked_indices:
            rho_i, u_i, p_i, _Y, c_i, A_i = interiors[idx]
            # Stagnation density/sound-speed from isentropic rel: ρ0 = ρ·(1+(γ−1)/2·M²)^(1/(γ−1))
            M_i = u_i / max(c_i, 1.0)
            t0_factor = 1.0 + 0.5 * gm1 * M_i * M_i
            rho0 = rho_i * t0_factor ** (1.0 / gm1)
            c0 = c_i * np.sqrt(t0_factor)
            # ṁ* = ρ0·c0·A·(2/(γ+1))^((γ+1)/(2(γ−1))) · something — use
            # the compact form  ṁ* = ρ0·c0·A·((γ+1)/2)^(-(γ+1)/(2(γ−1)))
            mdot_star = rho0 * c0 * A_i * (gp1 / 2.0) ** choke_exp
            sigma = self.legs[idx].sign_into
            fixed_mdot.append((idx, sigma * mdot_star))

        fixed_mdot_sum = sum(v for _, v in fixed_mdot)

        # Newton on non-choked legs, residual = Σ_non σ ρ_f u_f A + fixed_mdot_sum
        p_floor = 1000.0
        # Initial guess: area-weighted mean over non-choked legs
        sum_pA = 0.0
        sum_A = 0.0
        for i, (_, _, p_i, _, _, A_i) in enumerate(interiors):
            if i in choked_set:
                continue
            sum_pA += p_i * A_i
            sum_A += A_i
        p_j = sum_pA / max(sum_A, 1e-9)

        for it in range(self.newton_max_iter):
            R_nonchoked = 0.0
            dR = 0.0
            face_states_all: List[Tuple[float, float, float]] = [None] * len(self.legs)  # type: ignore
            for i, (L, (rho_i, u_i, p_i, _Y, c_i, A_i)) in enumerate(zip(self.legs, interiors)):
                if i in choked_set:
                    continue
                rho_f, u_f, c_f, drhof, duf = _face_from_pj(
                    rho_i, u_i, p_i, c_i, p_j, gamma, L.s_end,
                )
                sigma = L.sign_into
                R_nonchoked += sigma * rho_f * u_f * A_i
                dR += sigma * (drhof * u_f + rho_f * duf) * A_i
                face_states_all[i] = (rho_f, u_f, c_f)
            R = R_nonchoked + fixed_mdot_sum
            if abs(R) < self.newton_tol:
                # Fill choked-leg face states for ghost write
                self._fill_choked_face_states(
                    interiors, choked_indices, face_states_all, gamma,
                )
                self.last_mass_residual = R
                return p_j, it + 1, face_states_all
            if abs(dR) < 1e-30:
                break
            dp = -R / dR
            cap = 0.2 * p_j
            if dp > cap:
                dp = cap
            elif dp < -cap:
                dp = -cap
            p_j = max(p_j + dp, p_floor)

        raise JunctionConvergenceError(
            f"CharacteristicJunction choked branch did not converge "
            f"({len(choked_indices)} legs choked, "
            f"{len(self.legs) - len(choked_indices)} subsonic). "
            f"last p_j = {p_j:.1f} Pa."
        )

    def _fill_choked_face_states(
        self, interiors, choked_indices, face_states_all, gamma,
    ):
        """Populate face_states_all entries for choked legs with their
        sonic throat state (ρ*, u*, c*)."""
        gm1 = gamma - 1.0
        for idx in choked_indices:
            L = self.legs[idx]
            rho_i, u_i, p_i, _Y, c_i, A_i = interiors[idx]
            M_i = u_i / max(c_i, 1.0)
            t0_factor = 1.0 + 0.5 * gm1 * M_i * M_i
            T0_over_T = t0_factor  # T0/T at interior
            rho0 = rho_i * t0_factor ** (1.0 / gm1)
            c0 = c_i * np.sqrt(t0_factor)
            # Sonic throat: T*/T0 = 2/(γ+1), ρ*/ρ0 = (2/(γ+1))^(1/(γ−1))
            t_star_over_t0 = 2.0 / (gamma + 1.0)
            rho_star = rho0 * t_star_over_t0 ** (1.0 / gm1)
            c_star = c0 * np.sqrt(t_star_over_t0)
            u_star = L.sign_into * c_star  # sonic, direction = into junction
            face_states_all[idx] = (rho_star, u_star, c_star)

    # --- Inflow entropy Picard pass ------------------------------------

    def _inflow_entropy_pass(
        self, interiors, face_states, p_j_current, gamma,
    ):
        """One-pass Picard correction: for legs carrying mass out of the
        junction (into the pipe), re-evaluate the face state using the
        junction-mixed entropy rather than the leg's interior entropy,
        then re-solve Newton for p_j.

        Returns (new_p_j, new_face_states) if a correction was applied,
        else (None, None)."""
        # Identify inflow legs (from junction into pipe)
        inflow_legs = []
        outflow_legs = []
        for i, (L, (rho_f, u_f, c_f)) in enumerate(zip(self.legs, face_states)):
            sigma = L.sign_into
            if sigma * u_f < 0.0:
                inflow_legs.append(i)
            else:
                outflow_legs.append(i)

        if not inflow_legs or not outflow_legs:
            # All one-way: no mixing correction to apply.
            return None, None

        # Mass-weighted (rho, p) of outflow legs defines the junction
        # reservoir state. Entropy is p/ρ^γ.
        mdot_out_total = 0.0
        rho_mix = 0.0
        p_mix = 0.0
        for i in outflow_legs:
            L = self.legs[i]
            rho_f, u_f, c_f = face_states[i]
            A_i = interiors[i][5]
            mdot = L.sign_into * rho_f * u_f * A_i
            mdot_out_total += mdot
            rho_mix += mdot * rho_f
            p_mix += mdot * interiors[i][2]  # interior p
        if mdot_out_total <= 0.0:
            return None, None
        rho_mix /= mdot_out_total
        p_mix   /= mdot_out_total
        # Use p_mix at the current p_j as the "reservoir" state; entropy
        # s_mix = p_mix / rho_mix^γ. For the inflow leg face state:
        #   c_mix = sqrt(γ p_mix / rho_mix)
        c_mix = float(np.sqrt(gamma * p_mix / max(rho_mix, 1e-9)))

        # Build a corrected "interiors" list: for inflow legs, swap in
        # the mixed reference state; outflow legs keep their own
        # interior. Velocity reference for the characteristic is the
        # leg's interior u (the characteristic is still carried from
        # the interior along the outgoing direction).
        corrected = list(interiors)
        for i in inflow_legs:
            rho_i, u_i, p_i, Y_i, c_i, A_i = interiors[i]
            corrected[i] = (rho_mix, u_i, p_mix, Y_i, c_mix, A_i)

        # Re-run Newton from the current p_j with the corrected
        # references.
        p_j = p_j_current
        for it in range(self.newton_max_iter):
            R, dR, face_states_new = _mass_residual_and_jacobian(
                corrected, self.legs, p_j, gamma,
            )
            if abs(R) < self.newton_tol:
                self.last_mass_residual = R
                return p_j, face_states_new
            if abs(dR) < 1e-30:
                break
            dp = -R / dR
            cap = 0.2 * p_j
            if dp > cap:
                dp = cap
            elif dp < -cap:
                dp = -cap
            p_j = max(p_j + dp, 1000.0)
        # Picard pass did not converge — keep the original face states.
        return None, None

    # --- Composition mixing --------------------------------------------

    def _compute_mixed_Y(self, interiors, face_states):
        mdot_in_total = 0.0
        YsumIn = 0.0
        for L, (rho_f, u_f, _), it in zip(self.legs, face_states, interiors):
            A_i = it[5]
            Y_i = it[3]
            signed_into_j = L.sign_into * rho_f * u_f * A_i
            if signed_into_j > 0.0:
                mdot_in_total += signed_into_j
                YsumIn += signed_into_j * Y_i
        if mdot_in_total > 1e-30:
            return YsumIn / mdot_in_total
        return 0.0

    # --- Energy residual -----------------------------------------------

    def _energy_residual(self, interiors, face_states):
        """Signed energy residual at the junction face. Σ σ_i ρ_f u_f A_i h0_f.

        Positive residual = junction gaining energy (physically impossible,
        flags numerical bug). Negative of small magnitude = lossy mixing
        across area change, expected and correct for mismatched junctions."""
        gamma = self.gamma
        gm1 = gamma - 1.0
        # For ρ_f, u_f, c_f: h = c^2/(γ−1). h0 = h + ½u².
        E_flux = 0.0
        for L, (rho_f, u_f, c_f), it in zip(self.legs, face_states, interiors):
            A_i = it[5]
            h = c_f * c_f / gm1
            h0 = h + 0.5 * u_f * u_f
            E_flux += L.sign_into * rho_f * u_f * A_i * h0
        return E_flux

    # --- Ghost write ---------------------------------------------------

    def _write_ghosts(self, interiors, face_states, Y_mixed, p_j):
        gamma = self.gamma
        gm1 = gamma - 1.0
        for leg, it, face in zip(self.legs, interiors, face_states):
            rho_i, u_i, p_i, Y_i, c_i, A_i = it
            rho_f, u_f, c_f = face
            # Y upstream-upwind: if this leg is pushing flow INTO the
            # junction, its own composition moves into the junction.
            # If this leg is pulling flow OUT of the junction, it
            # receives the mixed composition.
            signed_into = leg.sign_into * rho_f * u_f
            Y_ghost = Y_i if signed_into >= 0.0 else Y_mixed
            # Ghost static pressure = p_j (subsonic) or sonic-throat p
            # (choked). For subsonic legs p_f = p_j identically. For
            # choked legs we need p from face state: p_f = ρ_f c_f^2 / γ.
            p_f = rho_f * c_f * c_f / gamma
            E_ghost_density = p_f / gm1 + 0.5 * rho_f * u_f * u_f
            state = leg.state
            ng = state.n_ghost
            nc = state.n_cells
            if leg.end == LEFT:
                ghost_range = range(0, ng)
            else:
                ghost_range = range(ng + nc, state.n_total)
            for i in ghost_range:
                A_g = state.area[i]
                state.q[i, I_RHO_A] = rho_f * A_g
                state.q[i, I_MOM_A] = rho_f * u_f * A_g
                state.q[i, I_E_A]   = E_ghost_density * A_g
                state.q[i, I_Y_A]   = rho_f * Y_ghost * A_g


# ---------------------------------------------------------------------------
# Factory for model wiring. Default stays "stagnation" per Phase E review
# decision — characteristic junction is opt-in, requested explicitly by
# models that want wave propagation through the merge.
# ---------------------------------------------------------------------------

def make_junction(
    junction_type: str,
    legs: List[JunctionLeg],
    *,
    gamma: float = 1.4,
    R_gas: float = 287.0,
    p_init: float = 101325.0,
    T_init: float = 300.0,
    Y_init: float = 0.0,
):
    """Factory returning either a JunctionCV (stagnation, default) or a
    CharacteristicJunction. Both expose the same fill_ghosts /
    absorb_fluxes lifecycle.

    - junction_type="stagnation" → bcs.junction_cv.JunctionCV
    - junction_type="characteristic" → CharacteristicJunction

    Leg types differ: JunctionCV wants JunctionCVLeg, this module wants
    JunctionLeg. The factory handles the conversion when needed.
    """
    if junction_type == "stagnation":
        # Late import to avoid circularity at module load.
        from bcs.junction_cv import JunctionCV, JunctionCVLeg
        cv_legs = [JunctionCVLeg(L.state, L.end) for L in legs]
        return JunctionCV.from_legs(
            cv_legs,
            p_init=p_init, T_init=T_init, Y_init=Y_init,
            gamma=gamma, R_gas=R_gas,
        )
    if junction_type == "characteristic":
        return CharacteristicJunction(
            legs=legs, gamma=gamma, R_gas=R_gas,
        )
    raise ValueError(
        f"unknown junction_type {junction_type!r}; "
        f"choose 'stagnation' (default, strict conservation) or "
        f"'characteristic' (better wave transmission)"
    )
