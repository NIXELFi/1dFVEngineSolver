"""Test 8 for Phase E2 — stagnation CV vs characteristic junction
side-by-side on the A3 4-2-1 manifold harness.

The A3 harness is already set up in tests/acoustic/test_a3_sdm26_manifold.py
for the stagnation-CV baseline. This test runs the SAME harness with
the CharacteristicJunction and reports R_round_trip for comparison.

Per the plan, this is a diagnostic, not a pass/fail test: the
individual test must at least *run successfully*; we don't fail it
on a specific |R| threshold because the Phase E4 sweep uses A3 as
part of the acceptance evaluation.

The integration test is long (it runs the full 15 ms manifold
simulation) so it is isolated in its own module to keep the fast
unit-test loop under 1 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from bcs.simple import fill_transmissive_right
from bcs.valve import fill_valve_ghost_characteristic as fill_valve_ghost
from bcs.junction_characteristic import (
    CharacteristicJunction, JunctionLeg, LEFT, RIGHT,
)
from cylinder.valve import EXHAUST_CD_TABLE, EXHAUST_LD_TABLE

from tests.acoustic._helpers import (
    DIAG_DIR, GAMMA, P_ATM, R_AIR, RHO_ATM, T_ATM,
    ensure_scratch, make_always_open_valve, run_acoustic,
    set_uniform_atmosphere,
    windowed_signed_extremum, windowed_signed_impulse,
)
from solver.state import make_pipe_state


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


def _build_manifold_characteristic():
    P = [_build_pipe(PRIMARY_L, PRIMARY_D, PRIMARY_NC, wall_T=1000.0) for _ in range(4)]
    S = [_build_pipe(SECONDARY_L, SECONDARY_D, SECONDARY_NC, wall_T=800.0) for _ in range(2)]
    C = _build_pipe(COLLECTOR_L, COLLECTOR_D, COLLECTOR_NC, wall_T=700.0)
    pipes = {
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "S0": S[0], "S1": S[1], "C":  C,
    }
    for p in pipes.values():
        ensure_scratch(p)

    j_4_2_a = CharacteristicJunction(
        legs=[JunctionLeg(P[0], RIGHT), JunctionLeg(P[3], RIGHT),
              JunctionLeg(S[0], LEFT)],
        gamma=GAMMA, R_gas=R_AIR,
    )
    j_4_2_b = CharacteristicJunction(
        legs=[JunctionLeg(P[1], RIGHT), JunctionLeg(P[2], RIGHT),
              JunctionLeg(S[1], LEFT)],
        gamma=GAMMA, R_gas=R_AIR,
    )
    j_2_1 = CharacteristicJunction(
        legs=[JunctionLeg(S[0], RIGHT), JunctionLeg(S[1], RIGHT),
              JunctionLeg(C,    LEFT)],
        gamma=GAMMA, R_gas=R_AIR,
    )
    return pipes, [j_4_2_a, j_4_2_b, j_2_1]


def _run_a3_characteristic(*, p_cyl_peak_bar: float, label: str):
    pipes, junctions = _build_manifold_characteristic()
    P0 = pipes["P0"]; S0 = pipes["S0"]; C = pipes["C"]

    vp, theta_fixed = make_always_open_valve(
        diameter=EXH_VALVE_D, max_lift=EXH_VALVE_MAX_LIFT,
        seat_angle_deg=EXH_VALVE_SEAT_DEG, n_valves=EXH_N_VALVES,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )

    p_pulse = p_cyl_peak_bar * 1e5

    def bc_apply(t: float, dt: float) -> None:
        for i, name in enumerate(["P0", "P1", "P2", "P3"]):
            primary = pipes[name]
            p_cyl = p_pulse if (i == 0 and t < T_PULSE_S) else P_ATM
            fill_valve_ghost(
                primary, pipe_end="left", valve_type="exhaust", vp=vp,
                theta_local_deg=theta_fixed,
                p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
            )
        for j in junctions:
            j.fill_ghosts(dt)
        fill_transmissive_right(C)

    def post_step_hook(_t: float, dt: float) -> None:
        for j in junctions:
            j.absorb_fluxes(dt)   # no-op for characteristic junction

    probes_spec = {
        "P0": {"P0 valve  (x≈0)": P0.dx * 1.5},
    }

    run = run_acoustic(
        pipes=pipes, bc_apply=bc_apply, post_step_hook=post_step_hook,
        t_end=T_END_S, probes_spec=probes_spec,
        waterfall_rows=500, cfl=0.5,
    )

    probe = run.probes["P0"]["P0 valve  (x≈0)"]
    t_arr = np.array(probe.t); p_arr = np.array(probe.p)
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

    return {
        "label": label,
        "A1_imp": A1_imp, "A2_imp": A2_imp,
        "R_round_trip": R_rt_imp,
        "R_round_trip_peak": R_rt_peak,
    }


def test_8_a3_characteristic_junction_comparison():
    """Run A3 manifold test with CharacteristicJunction in place of
    JunctionCV. Report R_round_trip for both amplitudes.

    Not pass/fail on a threshold — per plan: "this individual test
    must at least *run successfully*; the Phase E4 sweep uses A3 as
    part of the acceptance evaluation."

    Baseline (JunctionCV, from C3):
      linear +5 kPa : R_round_trip = +0.228
      nominal 5 bar : R_round_trip = +0.011

    Draft (bcs/junction.py, from Phase E1 sanity check):
      linear +5 kPa : R_round_trip = +0.698
      nominal 5 bar : R_round_trip = -0.042
    """
    R_lin = _run_a3_characteristic(p_cyl_peak_bar=1.05, label="linear_5kPa")
    R_nom = _run_a3_characteristic(p_cyl_peak_bar=5.0,  label="nominal_5bar")

    # Just verify the run completed and produced finite numbers.
    assert np.isfinite(R_lin["R_round_trip"]), (
        f"linear regime R_round_trip = {R_lin['R_round_trip']} "
        f"(A1={R_lin['A1_imp']}, A2={R_lin['A2_imp']})"
    )
    assert np.isfinite(R_nom["R_round_trip"]), (
        f"nominal regime R_round_trip = {R_nom['R_round_trip']} "
        f"(A1={R_nom['A1_imp']}, A2={R_nom['A2_imp']})"
    )
    assert abs(R_lin["A1_imp"]) > 1e-3, (
        f"launch pulse A1 = {R_lin['A1_imp']:.3e} Pa·s — too weak, test broken"
    )

    # Write a side-by-side summary for human inspection.
    summary_path = DIAG_DIR / "phase_e2_a3_characteristic_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        f.write("Phase E2 — A3 manifold comparison: CharacteristicJunction\n")
        f.write("=" * 60 + "\n\n")
        f.write("Baseline numbers for comparison:\n")
        f.write("  JunctionCV (C3):        linear=+0.228, nominal=+0.011\n")
        f.write("  Draft bcs/junction.py:  linear=+0.698, nominal=-0.042\n\n")
        f.write("CharacteristicJunction (Phase E2):\n")
        f.write(f"  linear +5 kPa : R_round_trip = {R_lin['R_round_trip']:+.4f}  "
                f"(peak {R_lin['R_round_trip_peak']:+.4f})\n")
        f.write(f"  nominal 5 bar : R_round_trip = {R_nom['R_round_trip']:+.4f}  "
                f"(peak {R_nom['R_round_trip_peak']:+.4f})\n")
    # Also print to stdout so pytest -s shows it.
    print(
        f"\nA3 / CharacteristicJunction:\n"
        f"  linear +5 kPa : R_rt = {R_lin['R_round_trip']:+.4f}  (baseline +0.228)\n"
        f"  nominal 5 bar : R_rt = {R_nom['R_round_trip']:+.4f}  (baseline +0.011)\n"
    )
