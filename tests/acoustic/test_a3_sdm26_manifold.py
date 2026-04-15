"""Test A3 — full SDM26 4-2-1 exhaust manifold round-trip reflection.

Geometry (SDM26 defaults; see models/sdm26.py)
----------------------------------------------
Four primaries, each 308 mm × 32 mm ID:                  P0, P1, P2, P3
Two 4→2 junctions:  (P0, P3) → S0 ;   (P1, P2) → S1
Two secondaries, each 392 mm × 38 mm ID:                 S0, S1
One 2→1 junction:   (S0, S1) → C
One collector,      100 mm × 50 mm ID:                   C
Right end of C: open-end BC to atmosphere (transmissive zero-gradient).

All four valve ends are connected to frozen cylinders held at P_ATM, and
all four exhaust valves are pinned at maximum lift the entire run
(``make_always_open_valve``).

Perturbation
------------
Cylinder 1's reservoir pressure is raised from 1 bar to 5 bar for 1.0 ms,
then returned to atmospheric. Cylinders 2, 3, 4 stay at atmospheric the
whole time. This launches a single blowdown wave down primary 1 (P0).

Probes
------
At six points along the wave's expected path:
  - valve end of P0 (the launch end)           — "P0 valve"
  - midpoint of P0                             — "P0 mid"
  - junction-side end of P0 (4→2 entry)        — "P0 jct"
  - left end of S0 (just past the 4→2 jct)     — "S0 left"
  - midpoint of S0                             — "S0 mid"
  - near collector open end                    — "C open"

Round-trip reflection coefficient
---------------------------------
The diagnostic the user explicitly asked for: at the valve end of P0,
take the ratio of the second arrival (the wave that has bounced once
off the open collector end and traversed back through both junctions)
to the first arrival (the launched pulse).

    A_1 = launch pulse passing the P0-valve probe outbound
    A_2 = pulse returning to P0-valve after one full round trip
    R_round_trip = A_2 / A_1   (impulse-based)

One-way path P0-valve → C-open is approximately:
    L_P0 + L_S0 + L_C ≈ 0.308 + 0.392 + 0.100 = 0.800 m
At c ≈ 347 m/s this gives a round-trip time ≈ 4.6 ms. The 15 ms run
duration provides ≈ 3 round trips before noise dominates.

Pass criteria (per user)
------------------------
|R_round_trip| ≥ 0.3   healthy (engine can predict tuned-length effects)
|R_round_trip| < 0.1   acoustically dead (engine cannot tune)
0.1–0.3 suspicious

Sign interpretation for the round trip is complicated (multiple junction
reflections + collector-end open BC contribute), so the user explicitly
relaxed the test to magnitude only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bcs.junction_cv import JunctionCV, JunctionCVLeg, LEFT, RIGHT
from bcs.simple import fill_transmissive_right
# Phase C1 fix (2026-04-14): characteristic + orifice valve BC
from bcs.valve import fill_valve_ghost_characteristic as fill_valve_ghost
from cylinder.valve import (
    EXHAUST_CD_TABLE,
    EXHAUST_LD_TABLE,
)

from tests.acoustic._helpers import (
    DIAG_DIR,
    GAMMA,
    P_ATM,
    R_AIR,
    RHO_ATM,
    T_ATM,
    cell_index_at,
    ensure_scratch,
    make_always_open_valve,
    pipe_pressure,
    run_acoustic,
    save_timeseries_png,
    save_waterfall_png,
    set_uniform_atmosphere,
    windowed_signed_extremum,
    windowed_signed_impulse,
)

from solver.state import make_pipe_state


# ---- SDM26 default geometry (must match models/sdm26.py SDM26Config defaults) ----
PRIMARY_L     = 0.308
PRIMARY_D     = 0.032
PRIMARY_NC    = 30
SECONDARY_L   = 0.392
SECONDARY_D   = 0.038
SECONDARY_NC  = 20
COLLECTOR_L   = 0.100
COLLECTOR_D   = 0.050
COLLECTOR_NC  = 20

# Exhaust valve geometry (SDM26 defaults)
EXH_VALVE_D = 0.023
EXH_VALVE_MAX_LIFT = 0.00735
EXH_VALVE_SEAT_DEG = 45.0
EXH_N_VALVES = 2

# Run config
T_END_S = 15e-3
T_PULSE_S = 1.0e-3   # user spec: 1.0 ms blowdown


def _build_pipe(length, diameter, n_cells, *, wall_T=1000.0):
    area_fn = lambda x: 0.25 * np.pi * diameter ** 2
    pipe = make_pipe_state(
        n_cells=n_cells, length=length, area_fn=area_fn,
        gamma=GAMMA, R_gas=R_AIR, wall_T=wall_T, n_ghost=2,
    )
    set_uniform_atmosphere(pipe)
    return pipe


def _build_manifold():
    """Build the 4-2-1 manifold geometry. Returns (pipes_dict, junctions)."""
    P = [
        _build_pipe(PRIMARY_L, PRIMARY_D, PRIMARY_NC, wall_T=1000.0)
        for _ in range(4)
    ]
    S = [
        _build_pipe(SECONDARY_L, SECONDARY_D, SECONDARY_NC, wall_T=800.0)
        for _ in range(2)
    ]
    C = _build_pipe(COLLECTOR_L, COLLECTOR_D, COLLECTOR_NC, wall_T=700.0)

    pipes = {
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "S0": S[0], "S1": S[1],
        "C":  C,
    }
    for p in pipes.values():
        ensure_scratch(p)

    # Junction CVs (mirroring models/sdm26.py SDM26Engine wiring)
    j_4_2_a = JunctionCV.from_legs(
        [JunctionCVLeg(P[0], RIGHT), JunctionCVLeg(P[3], RIGHT),
         JunctionCVLeg(S[0], LEFT)],
        p_init=P_ATM, T_init=T_ATM, Y_init=0.0,
    )
    j_4_2_b = JunctionCV.from_legs(
        [JunctionCVLeg(P[1], RIGHT), JunctionCVLeg(P[2], RIGHT),
         JunctionCVLeg(S[1], LEFT)],
        p_init=P_ATM, T_init=T_ATM, Y_init=0.0,
    )
    j_2_1   = JunctionCV.from_legs(
        [JunctionCVLeg(S[0], RIGHT), JunctionCVLeg(S[1], RIGHT),
         JunctionCVLeg(C, LEFT)],
        p_init=P_ATM, T_init=T_ATM, Y_init=0.0,
    )
    junctions = [j_4_2_a, j_4_2_b, j_2_1]
    return pipes, junctions


def run_a3(*, p_cyl_peak_bar: float, label: str):
    pipes, junctions = _build_manifold()
    P0 = pipes["P0"]
    S0 = pipes["S0"]
    C  = pipes["C"]

    vp, theta_fixed = make_always_open_valve(
        diameter=EXH_VALVE_D, max_lift=EXH_VALVE_MAX_LIFT,
        seat_angle_deg=EXH_VALVE_SEAT_DEG, n_valves=EXH_N_VALVES,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )

    p_pulse = p_cyl_peak_bar * 1e5

    def bc_apply(t: float, dt: float) -> None:
        # Cylinder 1: pulse. Cylinders 2-4: held at atmospheric.
        for i, name in enumerate(["P0", "P1", "P2", "P3"]):
            primary = pipes[name]
            if i == 0:
                p_cyl = p_pulse if t < T_PULSE_S else P_ATM
            else:
                p_cyl = P_ATM
            fill_valve_ghost(
                primary, pipe_end="left", valve_type="exhaust", vp=vp,
                theta_local_deg=theta_fixed,
                p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
            )
        # Junction CVs fill the inner ends of all incident pipes.
        for j in junctions:
            j.fill_ghosts()
        # Collector open end → transmissive (zero-gradient outflow)
        fill_transmissive_right(C)

    def post_step_hook(_t: float, dt: float) -> None:
        for j in junctions:
            j.absorb_fluxes(dt)

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
        pipes=pipes,
        bc_apply=bc_apply,
        post_step_hook=post_step_hook,
        t_end=T_END_S,
        probes_spec=probes_spec,
        waterfall_rows=500,
        cfl=0.5,
    )

    # ---- round-trip reflection coefficient at the P0 valve probe ----
    probe = run.probes["P0"]["P0 valve  (x≈0)"]
    t_arr = np.array(probe.t)
    p_arr = np.array(probe.p)
    c0 = float(np.sqrt(GAMMA * P_ATM / RHO_ATM))

    # One-way path length (P0 valve → collector open end)
    one_way_m = PRIMARY_L + SECONDARY_L + COLLECTOR_L
    round_trip_s = 2.0 * one_way_m / c0
    pulse_w = T_PULSE_S

    # A1: outbound pulse at the valve probe. The probe sits ≈ 1.5·dx from
    # the valve, so the launch passes it within a window
    # [0, T_PULSE + 2·dx/c] — wide enough for any small launch transient.
    t1_start = 0.0
    t1_end = pulse_w + 2.0 * P0.dx / c0
    A1_imp = windowed_signed_impulse(t_arr, p_arr, t1_start, t1_end)
    A1_pk, t1_pk = windowed_signed_extremum(t_arr, p_arr, t1_start, t1_end)

    # A2: round-trip return at the valve probe. Window centered on round-
    # trip time, width = pulse + slop for junction-induced dispersion.
    junction_slop_s = 1.0e-3   # generous slop for 0D-CV stagnation effects
    t2_start = round_trip_s - 0.5 * junction_slop_s
    t2_end   = round_trip_s + pulse_w + junction_slop_s
    A2_imp = windowed_signed_impulse(t_arr, p_arr, t2_start, t2_end)
    A2_pk, t2_pk = windowed_signed_extremum(t_arr, p_arr, t2_start, t2_end)

    R_rt_imp  = A2_imp / A1_imp if abs(A1_imp) > 1e-20 else float("nan")
    R_rt_peak = A2_pk  / A1_pk  if abs(A1_pk)  > 1e-9  else float("nan")

    R = {
        "label": label,
        "p_cyl_peak_bar": p_cyl_peak_bar,
        "c0_m_s": c0,
        "one_way_m": one_way_m,
        "round_trip_s": round_trip_s,
        "A1_peak": A1_pk, "A1_imp": A1_imp, "t1_peak": t1_pk,
        "A2_peak": A2_pk, "A2_imp": A2_imp, "t2_peak": t2_pk,
        "R_round_trip_peak": R_rt_peak,
        "R_round_trip":      R_rt_imp,
        "t1_window": (t1_start, t1_end),
        "t2_window": (t2_start, t2_end),
    }

    # ---- plots: probe time series for P0, plus a per-pipe waterfall ----
    save_timeseries_png(
        run, "P0",
        out_path=DIAG_DIR / f"a3_{label}_P0_probes.png",
        title=(
            f"A3/{label} — primary 0 (launch pipe) probes.  "
            f"R_round_trip = {R['R_round_trip']:+.3f}"
        ),
    )
    save_timeseries_png(
        run, "S0",
        out_path=DIAG_DIR / f"a3_{label}_S0_probes.png",
        title=f"A3/{label} — secondary 0 probes",
    )
    save_timeseries_png(
        run, "C",
        out_path=DIAG_DIR / f"a3_{label}_C_probes.png",
        title=f"A3/{label} — collector probes",
    )
    vmax_kPa = max(50.0, (p_cyl_peak_bar - 1.0) * 100.0)
    for pipe_name, length_m in [
        ("P0", PRIMARY_L), ("P1", PRIMARY_L), ("P2", PRIMARY_L), ("P3", PRIMARY_L),
        ("S0", SECONDARY_L), ("S1", SECONDARY_L), ("C", COLLECTOR_L),
    ]:
        save_waterfall_png(
            run, pipe_name, length_m,
            out_path=DIAG_DIR / f"a3_{label}_{pipe_name}_waterfall.png",
            title=f"A3/{label} waterfall — {pipe_name}",
            vmax_kPa=vmax_kPa,
        )
    return run, R


def _write_a3_summary(R_nominal: dict, R_linear: dict) -> None:
    path = DIAG_DIR / "a3_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Test A3 — full SDM26 4-2-1 manifold round-trip reflection\n")
        f.write("===========================================================\n\n")
        f.write("Geometry: 4×P (308 mm × 32 mm) → 4-2 junction → "
                "2×S (392 mm × 38 mm) → 2-1 junction → C (100 mm × 50 mm)\n")
        f.write(f"One-way path:    {R_nominal['one_way_m']*1000:.0f} mm\n")
        f.write(f"Round-trip time: {R_nominal['round_trip_s']*1000:.3f} ms "
                f"(at c = {R_nominal['c0_m_s']:.0f} m/s)\n\n")
        for title, R in [("nominal (5 bar pulse)", R_nominal),
                         ("linear  (+5 kPa)",     R_linear)]:
            f.write(f"--- {title} ---\n")
            f.write(f"  cyl 1 peak overpressure  = "
                    f"{(R['p_cyl_peak_bar']-1.0)*100.0:+.2f} kPa "
                    f"({R['p_cyl_peak_bar']:.3f} bar)\n")
            f.write(f"  A1 window=[{R['t1_window'][0]*1e3:6.3f}, "
                    f"{R['t1_window'][1]*1e3:6.3f}] ms\n")
            f.write(f"  A1_peak = {R['A1_peak']:+10.1f} Pa  "
                    f"(at t={R['t1_peak']*1e3:6.3f} ms)  "
                    f"impulse = {R['A1_imp']:+.4e} Pa·s\n")
            f.write(f"  A2 window=[{R['t2_window'][0]*1e3:6.3f}, "
                    f"{R['t2_window'][1]*1e3:6.3f}] ms\n")
            f.write(f"  A2_peak = {R['A2_peak']:+10.1f} Pa  "
                    f"(at t={R['t2_peak']*1e3:6.3f} ms)  "
                    f"impulse = {R['A2_imp']:+.4e} Pa·s\n")
            f.write(f"  R_round_trip_peak  = A2_peak/A1_peak = {R['R_round_trip_peak']:+.4f}\n")
            f.write(f"  R_round_trip       = A2_imp /A1_imp  = {R['R_round_trip']:+.4f}  <-- primary\n\n")


@pytest.fixture(scope="module")
def a3_runs():
    _, R_nom = run_a3(p_cyl_peak_bar=5.0,  label="nominal_5bar")
    _, R_lin = run_a3(p_cyl_peak_bar=1.05, label="linear_5kPa")
    _write_a3_summary(R_nom, R_lin)
    return R_nom, R_lin


def test_a3_nominal_round_trip(a3_runs):
    R_nom, _ = a3_runs
    assert np.isfinite(R_nom["R_round_trip"])
    assert abs(R_nom["A1_imp"]) > 1e-3, (
        f"A1 impulse {R_nom['A1_imp']:.3e} Pa·s implies the launch pulse never "
        f"developed at the P0 valve probe — test setup broken."
    )


def test_a3_linear_regime_round_trip(a3_runs):
    """Linear-regime diagnostic. Pass: |R_round_trip| ≥ 0.1
    (the user's failure floor for the manifold round trip)."""
    _, R_lin = a3_runs
    Rrt = R_lin["R_round_trip"]
    assert np.isfinite(Rrt)
    assert abs(Rrt) >= 0.1, (
        f"|R_round_trip| = {abs(Rrt):.3f} < 0.1 in the linear regime — the "
        f"manifold is acoustically dead. Tuned-length effects cannot be "
        f"predicted in the engine sweep with this BC layer."
    )


if __name__ == "__main__":
    _, R_nom = run_a3(p_cyl_peak_bar=5.0,  label="nominal_5bar")
    _, R_lin = run_a3(p_cyl_peak_bar=1.05, label="linear_5kPa")
    _write_a3_summary(R_nom, R_lin)
    print("\n=== nominal (5 bar) ===")
    print(f"  R_round_trip = {R_nom['R_round_trip']:+.4f}")
    print("\n=== linear  (+5 kPa) ===")
    print(f"  R_round_trip = {R_lin['R_round_trip']:+.4f}")
