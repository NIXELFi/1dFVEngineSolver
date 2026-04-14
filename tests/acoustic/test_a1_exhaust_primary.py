"""Test A1 — exhaust primary single-pipe reflection coefficient.

Geometry
--------
Straight pipe, 308 mm long, 32 mm ID (SDM26 exhaust primary).
Left end:  exhaust valve BC, valve held at max lift the entire run
           (open_angle=0, close_angle=720 so sin² lift peaks at θ=360).
           Connected to a "frozen" cylinder reservoir whose state is
           manipulated by hand to launch the acoustic perturbation.
Right end: reflective wall (fill_reflective_right) — a known perfect
           reflector with R_wall ≈ +1.

Perturbation
------------
User-specified: at t=0, instantaneously set cylinder p from 1 bar to
3 bar, hold for 0.5 ms, return to atmospheric. T_cyl held at 300 K.

Linear-regime companion test: same shape, 1.02 bar (+2 kPa) overpressure
— measures the BC reflection in the linear acoustic regime where
nonlinear shock-amplification effects vanish and R is unambiguous.

Instrumentation
---------------
Probes at x = 0, L/2, L. Run duration 20 ms (≈ 11 round trips).
x-t pressure waterfall plotted alongside probe time series.

Reflection coefficient extraction
---------------------------------
At the x = L/2 probe, each boundary reflection produces a new passage of
the pulse past the probe. With alternating end types (wall on the right,
valve-under-test on the left), the n-th arrival at the probe carries
amplitude:

    A_1 = (launch)·T_launch                     [outbound, no bounce]
    A_2 = R_wall  · A_1                         [right-end bounce]
    A_3 = R_valve · A_2                         [left-end bounce]
    A_4 = R_wall  · A_3                         [right-end bounce #2]
    …

R_wall = A_2 / A_1 is a sanity check (should be ≈ +1).
R_valve = A_3 / A_2 is the diagnosis.

We measure A_n by taking the signed extremum of (p − P_atm) within the
expected window for arrival n. This is robust to multi-component
wavetrains (compression leading + rarefaction trailing from the square
pulse) as long as the windows don't overlap, which holds for the first
3 arrivals at any interior probe.

Pass criteria (revised 2026-04-14)
----------------------------------
|R_valve| > 0.3 healthy, |R_valve| < 0.1 failure mode (absorbing BC),
0.1 ≤ |R| ≤ 0.3 suspicious. Sign is informative but not a gate:
  R > 0 → wall-like (compression reflects as compression)
  R < 0 → pressure-release-like (compression inverts to rarefaction)
  R ≈ 0 → the BC is absorbing wave energy (the failure we're hunting)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bcs.simple import fill_reflective_right
from bcs.valve import fill_valve_ghost
from cylinder.valve import (
    EXHAUST_CD_TABLE,
    EXHAUST_LD_TABLE,
)

from tests.acoustic._helpers import (
    DIAG_DIR,
    GAMMA,
    P_ATM,
    R_AIR,
    T_ATM,
    arrivals_at_probe,
    make_always_open_valve,
    reflection_from_windowed,
    run_acoustic,
    save_timeseries_png,
    save_waterfall_png,
    set_uniform_atmosphere,
)

from solver.state import make_pipe_state


# ---- geometry ----
LENGTH_M = 0.308
DIAMETER_M = 0.032
N_CELLS = 200
T_END_S = 20e-3
T_PULSE_S = 0.5e-3

# Exhaust valve geometry (SDM26 defaults)
EXH_VALVE_D = 0.023
EXH_VALVE_MAX_LIFT = 0.00735
EXH_VALVE_SEAT_DEG = 45.0
EXH_N_VALVES = 2


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _build_pipe():
    area_fn = lambda x: 0.25 * np.pi * DIAMETER_M ** 2
    pipe = make_pipe_state(
        n_cells=N_CELLS, length=LENGTH_M, area_fn=area_fn,
        gamma=GAMMA, R_gas=R_AIR, wall_T=1000.0, n_ghost=2,
    )
    set_uniform_atmosphere(pipe)
    return pipe


def run_a1(*, p_cyl_peak_bar: float, label: str):
    """Run A1 with a configurable cylinder overpressure.

    Returns (run, measurement_dict). Writes the waterfall + timeseries
    PNGs under DIAG_DIR tagged with ``label``.
    """
    pipe = _build_pipe()

    vp, theta_fixed = make_always_open_valve(
        diameter=EXH_VALVE_D, max_lift=EXH_VALVE_MAX_LIFT,
        seat_angle_deg=EXH_VALVE_SEAT_DEG, n_valves=EXH_N_VALVES,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )

    p_pulse = p_cyl_peak_bar * 1e5

    def bc_apply(t: float) -> None:
        p_cyl = p_pulse if t < T_PULSE_S else P_ATM
        fill_valve_ghost(
            pipe, pipe_end="left", valve_type="exhaust", vp=vp,
            theta_local_deg=theta_fixed,
            p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
        )
        fill_reflective_right(pipe)

    probe_x = {
        "x=0 (near valve)":  pipe.dx * 1.5,
        "x=L/2 (mid pipe)":  LENGTH_M * 0.5,
        "x=L (near wall)":   LENGTH_M - pipe.dx * 1.5,
    }

    run = run_acoustic(
        pipes={"primary": pipe},
        bc_apply=bc_apply,
        t_end=T_END_S,
        probes_spec={"primary": probe_x},
        waterfall_rows=500,
        cfl=0.5,
    )

    # Extract R using the mid-pipe probe.
    probe = run.probes["primary"]["x=L/2 (mid pipe)"]
    t_arr = np.array(probe.t)
    p_arr = np.array(probe.p)
    c0 = float(np.sqrt(GAMMA * P_ATM / (P_ATM / (R_AIR * T_ATM))))
    arrivals = arrivals_at_probe(
        t_arr, p_arr,
        length_m=LENGTH_M, c0_m_s=c0, pulse_width_s=T_PULSE_S,
        probe_x_m=LENGTH_M * 0.5, max_arrivals=3,
    )
    R = reflection_from_windowed(arrivals)
    R["c0_m_s"] = c0
    R["arrivals"] = arrivals
    R["label"] = label
    R["p_cyl_peak_bar"] = p_cyl_peak_bar

    # Plots (written every call — per-label filenames).
    save_timeseries_png(
        run, "primary",
        out_path=DIAG_DIR / f"a1_{label}_probes.png",
        title=(
            f"A1/{label} — exhaust primary, "
            f"p_cyl pulse = {p_cyl_peak_bar:.2f} bar for {T_PULSE_S*1e3:.1f} ms.  "
            f"R_wall={R['R_wall']:+.3f}  R_valve={R['R_valve']:+.3f}"
        ),
    )
    # Waterfall vmax scaled to pulse overpressure so small-amplitude runs don't vanish
    vmax_kPa = max(50.0, (p_cyl_peak_bar - 1.0) * 100.0)
    save_waterfall_png(
        run, "primary", LENGTH_M,
        out_path=DIAG_DIR / f"a1_{label}_waterfall.png",
        title=(
            f"A1/{label} waterfall — exhaust primary (308 mm × 32 mm ID).  "
            f"Valve open, reservoir p-pulse {p_cyl_peak_bar:.2f} bar "
            f"for {T_PULSE_S*1e3:.1f} ms."
        ),
        vmax_kPa=vmax_kPa,
    )

    return run, R


def _write_a1_summary(nominal: dict, linear: dict) -> None:
    path = DIAG_DIR / "a1_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Test A1 — exhaust primary reflection (windowed extrema)\n")
        f.write("=========================================================\n\n")
        f.write(f"Pipe length      : {LENGTH_M*1000:.1f} mm\n")
        f.write(f"Pipe diameter    : {DIAMETER_M*1000:.1f} mm ID\n")
        f.write(f"n_cells          : {N_CELLS}\n")
        f.write(f"Run duration     : {T_END_S*1000:.1f} ms\n")
        f.write(f"Pulse duration   : {T_PULSE_S*1000:.2f} ms\n")
        f.write(f"Sound speed      : {nominal['c0_m_s']:.1f} m/s\n\n")
        for label, R in [("nominal (3 bar)", nominal), ("linear (+2 kPa)", linear)]:
            f.write(f"--- {label} ---\n")
            f.write(f"  p_cyl peak            = {R['p_cyl_peak_bar']:.3f} bar\n")
            for a in R["arrivals"]:
                f.write(
                    f"  arrival n={a.n}  window=[{a.t_window[0]*1e3:6.3f}, "
                    f"{a.t_window[1]*1e3:6.3f}] ms  "
                    f"peak at t={a.t_peak*1e3:6.3f} ms  "
                    f"amp={a.amp:+10.1f} Pa  "
                    f"impulse={a.impulse:+.4e} Pa·s\n"
                )
            f.write(f"  A1_peak  = {R['A1_peak']:+.1f} Pa    "
                    f"A1_imp  = {R['A1_imp']:+.4e} Pa·s\n")
            f.write(f"  A2_peak  = {R['A2_peak']:+.1f} Pa    "
                    f"A2_imp  = {R['A2_imp']:+.4e} Pa·s\n")
            f.write(f"  A3_peak  = {R['A3_peak']:+.1f} Pa    "
                    f"A3_imp  = {R['A3_imp']:+.4e} Pa·s\n")
            f.write(f"  R_wall_peak  = A2_peak/A1_peak  = {R['R_wall_peak']:+.4f}\n")
            f.write(f"  R_wall       = A2_imp /A1_imp   = {R['R_wall']:+.4f}  <-- primary\n")
            f.write(f"  R_valve_peak = A3_peak/A2_peak  = {R['R_valve_peak']:+.4f}\n")
            f.write(f"  R_valve      = A3_imp /A2_imp   = {R['R_valve']:+.4f}  <-- primary\n\n")


# -----------------------------------------------------------------------------
# pytest entry points
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def a1_runs():
    _, R_nominal = run_a1(p_cyl_peak_bar=3.0, label="nominal_3bar")
    _, R_linear  = run_a1(p_cyl_peak_bar=1.02, label="linear_1p02bar")
    _write_a1_summary(R_nominal, R_linear)
    return R_nominal, R_linear


def test_a1_nominal_3bar_reflection(a1_runs):
    """User-specified 3 bar perturbation. Mildly nonlinear; used as the
    realistic case. The linear-regime test below is the clean diagnostic."""
    R_nominal, _ = a1_runs
    # No hard fail here — 3 bar is nonlinear enough that R_wall can be
    # slightly above unity due to shock amplification. We just require
    # the numbers are physically meaningful (finite, not zero).
    assert np.isfinite(R_nominal["R_wall"]), "R_wall is NaN"
    assert np.isfinite(R_nominal["R_valve"]), "R_valve is NaN"
    assert abs(R_nominal["A1_peak"]) > 500.0, (
        f"A1_peak amplitude {R_nominal['A1_peak']:.1f} Pa implies the pulse "
        f"never reached the probe — the test setup is broken."
    )


def test_a1_linear_regime_reflection(a1_runs):
    """+2 kPa linear-acoustic perturbation. This is the clean diagnostic:
    in the linear regime, R_wall must be ≈ +1 (sanity), and R_valve is
    the unambiguous BC reflection coefficient.

    Pass criteria: |R_valve| ≥ 0.1 (anything lower confirms the BC is
    absorbing the wave — the suspected failure mode).
    """
    _, R_linear = a1_runs
    Rw = R_linear["R_wall"]
    Rv = R_linear["R_valve"]

    assert np.isfinite(Rw) and np.isfinite(Rv)
    # Sanity: wall reflection should be close to +1 in the linear regime.
    assert 0.7 < Rw < 1.3, (
        f"linear-regime R_wall = {Rw:+.3f} is far from +1. Either MUSCL "
        f"numerical dissipation is excessive or the probe indexing is wrong. "
        f"The diagnostic cannot isolate R_valve when R_wall is broken."
    )
    # Diagnostic fail threshold per the user's revised criterion.
    assert abs(Rv) >= 0.1, (
        f"|R_valve| = {abs(Rv):.3f} < 0.1 in the linear regime — the valve "
        f"BC is absorbing the wave. This is the failure mode the diagnostic "
        f"was designed to catch. Expected healthy behavior |R| > 0.3 "
        f"(magnitude), suspicious 0.1–0.3, absorbing < 0.1."
    )


if __name__ == "__main__":
    _, R_nom = run_a1(p_cyl_peak_bar=3.0, label="nominal_3bar")
    _, R_lin = run_a1(p_cyl_peak_bar=1.02, label="linear_1p02bar")
    _write_a1_summary(R_nom, R_lin)
    print("\n=== nominal (3 bar) ===")
    print(f"  R_wall  = {R_nom['R_wall']:+.4f}")
    print(f"  R_valve = {R_nom['R_valve']:+.4f}")
    print("\n=== linear (+2 kPa) ===")
    print(f"  R_wall  = {R_lin['R_wall']:+.4f}")
    print(f"  R_valve = {R_lin['R_valve']:+.4f}")
