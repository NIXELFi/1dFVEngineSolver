"""Phase E1 sanity check — run the dormant ``bcs/junction.py`` draft
through the A3 manifold test and report ``R_round_trip``.

Purpose
-------
Before implementing the new ``CharacteristicJunction`` we want to know
whether the existing (untested, Phase-3 WIP) characteristic-coupled
junction draft is closer to correct or fundamentally broken. This
script wires ``apply_junction`` (constant-static-pressure Newton
solver, no absorb-fluxes step because it is not a CV) into the full
A3 4-2-1 manifold harness and measures the round-trip reflection
coefficient in both linear and nominal amplitudes.

Outcomes
  - |R| > 0.3 → draft is close to correct; E2 is polish work
  - 0.1 < |R| ≤ 0.3 → partial; E2 is genuine work informed by the draft
  - |R| ≤ 0.1 → draft is not a useful baseline; E2 is effectively from scratch
  - crashes/NaN → draft is broken beyond useful; E2 from scratch

Not part of the test suite. Run standalone:

    python -m diagnostics.phase_e_draft_sanity
"""

from __future__ import annotations

import numpy as np

# The draft under test
from bcs.junction import JunctionLeg, apply_junction, LEFT, RIGHT
from bcs.simple import fill_transmissive_right
from bcs.valve import fill_valve_ghost_characteristic as fill_valve_ghost
from cylinder.valve import EXHAUST_CD_TABLE, EXHAUST_LD_TABLE

from tests.acoustic._helpers import (
    DIAG_DIR, GAMMA, P_ATM, R_AIR, RHO_ATM, T_ATM,
    ensure_scratch, make_always_open_valve, run_acoustic,
    set_uniform_atmosphere,
    windowed_signed_extremum, windowed_signed_impulse,
    save_timeseries_png,
)
from solver.state import make_pipe_state


# ---- SDM26 default geometry, matched to test_a3_sdm26_manifold.py ----
PRIMARY_L     = 0.308
PRIMARY_D     = 0.032
PRIMARY_NC    = 30
SECONDARY_L   = 0.392
SECONDARY_D   = 0.038
SECONDARY_NC  = 20
COLLECTOR_L   = 0.100
COLLECTOR_D   = 0.050
COLLECTOR_NC  = 20

EXH_VALVE_D = 0.023
EXH_VALVE_MAX_LIFT = 0.00735
EXH_VALVE_SEAT_DEG = 45.0
EXH_N_VALVES = 2

T_END_S = 15e-3
T_PULSE_S = 1.0e-3


def _build_pipe(length, diameter, n_cells, *, wall_T=1000.0):
    area_fn = lambda x: 0.25 * np.pi * diameter ** 2
    pipe = make_pipe_state(
        n_cells=n_cells, length=length, area_fn=area_fn,
        gamma=GAMMA, R_gas=R_AIR, wall_T=wall_T, n_ghost=2,
    )
    set_uniform_atmosphere(pipe)
    return pipe


def _build_manifold_draft():
    P = [_build_pipe(PRIMARY_L, PRIMARY_D, PRIMARY_NC, wall_T=1000.0) for _ in range(4)]
    S = [_build_pipe(SECONDARY_L, SECONDARY_D, SECONDARY_NC, wall_T=800.0) for _ in range(2)]
    C = _build_pipe(COLLECTOR_L, COLLECTOR_D, COLLECTOR_NC, wall_T=700.0)
    pipes = {
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "S0": S[0], "S1": S[1], "C": C,
    }
    for p in pipes.values():
        ensure_scratch(p)

    # Draft-junction leg groups (same topology as JunctionCV wiring).
    j_4_2_a = [JunctionLeg(P[0], RIGHT), JunctionLeg(P[3], RIGHT), JunctionLeg(S[0], LEFT)]
    j_4_2_b = [JunctionLeg(P[1], RIGHT), JunctionLeg(P[2], RIGHT), JunctionLeg(S[1], LEFT)]
    j_2_1   = [JunctionLeg(S[0], RIGHT), JunctionLeg(S[1], RIGHT), JunctionLeg(C,  LEFT)]
    return pipes, [j_4_2_a, j_4_2_b, j_2_1]


def run_a3_draft(*, p_cyl_peak_bar: float, label: str):
    pipes, junctions = _build_manifold_draft()
    P0, S0, C = pipes["P0"], pipes["S0"], pipes["C"]

    vp, theta_fixed = make_always_open_valve(
        diameter=EXH_VALVE_D, max_lift=EXH_VALVE_MAX_LIFT,
        seat_angle_deg=EXH_VALVE_SEAT_DEG, n_valves=EXH_N_VALVES,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )

    p_pulse = p_cyl_peak_bar * 1e5

    def bc_apply(t: float) -> None:
        for i, name in enumerate(["P0", "P1", "P2", "P3"]):
            primary = pipes[name]
            p_cyl = p_pulse if (i == 0 and t < T_PULSE_S) else P_ATM
            fill_valve_ghost(
                primary, pipe_end="left", valve_type="exhaust", vp=vp,
                theta_local_deg=theta_fixed,
                p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
            )
        # DRAFT junction fill: constant-static-p Newton, no CV absorb step.
        for legs in junctions:
            apply_junction(legs)
        fill_transmissive_right(C)

    # No post_step_hook: the draft has no CV state to update.
    probes_spec = {
        "P0": {
            "P0 valve  (x≈0)":     P0.dx * 1.5,
            "P0 mid    (x=L/2)":   PRIMARY_L * 0.5,
            "P0 jct    (x≈L)":     PRIMARY_L - P0.dx * 1.5,
        },
        "S0": {
            "S0 left   (x≈0)":     S0.dx * 1.5,
            "S0 mid    (x=L/2)":   SECONDARY_L * 0.5,
        },
        "C": {
            "C open    (x≈L)":     COLLECTOR_L - C.dx * 1.5,
        },
    }

    run = run_acoustic(
        pipes=pipes, bc_apply=bc_apply, post_step_hook=None,
        t_end=T_END_S, probes_spec=probes_spec,
        waterfall_rows=500, cfl=0.5,
    )

    probe = run.probes["P0"]["P0 valve  (x≈0)"]
    t_arr = np.array(probe.t)
    p_arr = np.array(probe.p)
    c0 = float(np.sqrt(GAMMA * P_ATM / RHO_ATM))

    one_way_m = PRIMARY_L + SECONDARY_L + COLLECTOR_L
    round_trip_s = 2.0 * one_way_m / c0
    pulse_w = T_PULSE_S

    t1_start, t1_end = 0.0, pulse_w + 2.0 * P0.dx / c0
    A1_imp = windowed_signed_impulse(t_arr, p_arr, t1_start, t1_end)
    A1_pk, t1_pk = windowed_signed_extremum(t_arr, p_arr, t1_start, t1_end)

    junction_slop_s = 1.0e-3
    t2_start = round_trip_s - 0.5 * junction_slop_s
    t2_end   = round_trip_s + pulse_w + junction_slop_s
    A2_imp = windowed_signed_impulse(t_arr, p_arr, t2_start, t2_end)
    A2_pk, t2_pk = windowed_signed_extremum(t_arr, p_arr, t2_start, t2_end)

    R_rt_imp  = A2_imp / A1_imp if abs(A1_imp) > 1e-20 else float("nan")
    R_rt_peak = A2_pk  / A1_pk  if abs(A1_pk)  > 1e-9  else float("nan")

    # Save P0 probe plot for visual sanity check of the draft's wave shape.
    out_png = DIAG_DIR / f"phase_e_draft_{label}_P0_probes.png"
    save_timeseries_png(
        run, "P0", out_path=out_png,
        title=(f"Phase-E draft / {label} — P0 probes.  "
               f"R_round_trip = {R_rt_imp:+.3f}"),
    )

    return {
        "label": label,
        "p_cyl_peak_bar": p_cyl_peak_bar,
        "round_trip_s": round_trip_s,
        "A1_peak": A1_pk, "A1_imp": A1_imp, "t1_peak": t1_pk,
        "A2_peak": A2_pk, "A2_imp": A2_imp, "t2_peak": t2_pk,
        "R_round_trip_peak": R_rt_peak,
        "R_round_trip":      R_rt_imp,
        "plot": str(out_png),
    }


def main() -> int:
    print("Phase E draft-junction sanity check")
    print("=" * 60)
    print("bcs/junction.py:apply_junction wired into A3 manifold test.")
    print("Comparing against the C3 JunctionCV baseline:")
    print("  nominal (5 bar):  R_round_trip = +0.011")
    print("  linear  (+5 kPa): R_round_trip = +0.228")
    print()

    try:
        R_lin = run_a3_draft(p_cyl_peak_bar=1.05, label="linear_5kPa")
        print(f"[linear  +5 kPa]   R_round_trip = {R_lin['R_round_trip']:+.4f}  "
              f"(peak {R_lin['R_round_trip_peak']:+.4f})")
        print(f"                   A1={R_lin['A1_imp']:+.3e}  A2={R_lin['A2_imp']:+.3e}")
        print(f"                   plot: {R_lin['plot']}")
    except Exception as e:
        print(f"[linear  +5 kPa]   CRASH: {type(e).__name__}: {e}")
        R_lin = None

    print()

    try:
        R_nom = run_a3_draft(p_cyl_peak_bar=5.0, label="nominal_5bar")
        print(f"[nominal 5 bar ]   R_round_trip = {R_nom['R_round_trip']:+.4f}  "
              f"(peak {R_nom['R_round_trip_peak']:+.4f})")
        print(f"                   A1={R_nom['A1_imp']:+.3e}  A2={R_nom['A2_imp']:+.3e}")
        print(f"                   plot: {R_nom['plot']}")
    except Exception as e:
        print(f"[nominal 5 bar ]   CRASH: {type(e).__name__}: {e}")
        R_nom = None

    print()
    print("=" * 60)
    if R_lin is None and R_nom is None:
        print("Verdict: draft crashed on both amplitudes. Not a useful baseline.")
        return 2
    if R_lin is not None and not np.isfinite(R_lin["R_round_trip"]):
        print("Verdict: draft returned NaN in linear regime. Not a useful baseline.")
        return 2
    if R_lin is not None and abs(R_lin["R_round_trip"]) > 0.3:
        print("Verdict: draft is close to correct; E2 is polish work.")
        return 0
    if R_lin is not None and abs(R_lin["R_round_trip"]) > 0.1:
        print("Verdict: draft is partial; E2 is genuine work informed by the draft.")
        return 0
    print("Verdict: draft is not a useful baseline; E2 is effectively from scratch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
