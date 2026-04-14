"""Test A2 — intake runner single-pipe reflection coefficient.

Geometry
--------
Straight pipe, 245 mm long, 38 mm ID (SDM26 intake runner).
Right end: intake valve BC, valve held at max lift throughout. Connected
           to a frozen cylinder reservoir.
Left end:  stagnation-inflow BC (``bcs.subsonic.fill_subsonic_inflow_left``)
           at 101325 Pa, 300 K, u=0, Y=0 — represents the plenum at rest.
           Physically: constant-pressure boundary, which in linear
           acoustics has reflection coefficient R_plenum = −1 (compression
           waves invert to rarefactions on reflection).

Perturbation
------------
User-specified: drop cyl p from 1 bar to 0.7 bar instantaneously, hold
for 0.5 ms, return to atmospheric. This launches a rarefaction wave
from the right end propagating leftward.

Linear-regime companion test: same shape, cyl drops by 2 kPa (0.98 bar)
— measures R in the linear acoustic regime for an unambiguous diagnosis.

Wave bookkeeping at the x = L/2 probe
-------------------------------------
Because the wave is launched from the RIGHT end in this test, the
effective launch-side probe distance is (L − x_probe). For a probe at
L/2 this is still L/2, so arrival timing is identical to A1, but the
sign convention differs:

    A_1 (outbound rarefaction, going left)              = ΔA  (< 0)
    A_2 (plenum-reflected: expect +, if R_plenum ≈ −1)  = R_plenum · A_1
    A_3 (valve-reflected)                               = R_valve  · A_2

So R_plenum = A_2 / A_1 and R_valve = A_3 / A_2, using the same
impulse-window extraction as A1.

Pass criteria (per user, revised 2026-04-14)
-------------------------------------------
|R_valve|  ≥ 0.1   (strict fail below this — BC is absorbing)
|R_plenum| ≥ 0.1   (and expected sign ≈ −1 for constant-p BC)

0.1 ≤ |R| ≤ 0.3 is "suspicious", |R| > 0.3 is "healthy". R close to 0
means the BC is eating the wave — the failure mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bcs.simple import fill_reflective_left
# Phase C2 (2026-04-14): use the characteristic-correct plenum BC, replacing
# the over-determined fill_subsonic_inflow_left that the audit identified as
# absorbing (R_plenum ≈ −0.07).
from bcs.subsonic import fill_subsonic_inflow_left_characteristic as fill_subsonic_inflow_left
# Phase C1 (2026-04-14): characteristic + orifice valve BC.
from bcs.valve import fill_valve_ghost_characteristic as fill_valve_ghost
from cylinder.valve import (
    INTAKE_CD_TABLE,
    INTAKE_LD_TABLE,
)

from tests.acoustic._helpers import (
    DIAG_DIR,
    GAMMA,
    P_ATM,
    R_AIR,
    RHO_ATM,
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
LENGTH_M = 0.245
DIAMETER_M = 0.038
N_CELLS = 200
T_END_S = 15e-3
T_PULSE_S = 0.5e-3

# Intake valve geometry (SDM26 defaults)
INT_VALVE_D = 0.0275
INT_VALVE_MAX_LIFT = 0.00856
INT_VALVE_SEAT_DEG = 45.0
INT_N_VALVES = 2


def _build_pipe():
    area_fn = lambda x: 0.25 * np.pi * DIAMETER_M ** 2
    pipe = make_pipe_state(
        n_cells=N_CELLS, length=LENGTH_M, area_fn=area_fn,
        gamma=GAMMA, R_gas=R_AIR, wall_T=320.0, n_ghost=2,
    )
    set_uniform_atmosphere(pipe)
    return pipe


def run_a2(
    *, p_cyl_trough_bar: float, label: str,
    far_end_bc: str = "plenum",
):
    """Run A2 with a configurable cylinder under-pressure.

    ``far_end_bc``:
      "plenum" — user-spec stagnation-inflow BC on the left end
                 (bcs.subsonic.fill_subsonic_inflow_left). Diagnoses the
                 plenum BC by measuring R_plenum = A2/A1.
      "wall"   — reflective wall on the left end. R_far ≈ +1 by
                 construction, which calibrates A2 so that R_valve =
                 A3/A2 can be extracted cleanly. Use this when the
                 "plenum" run shows R_plenum ≈ 0 (plenum-BC is the
                 absorber), because then A2 ≈ 0 and A3/A2 is noise/noise.
    """
    pipe = _build_pipe()

    vp, theta_fixed = make_always_open_valve(
        diameter=INT_VALVE_D, max_lift=INT_VALVE_MAX_LIFT,
        seat_angle_deg=INT_VALVE_SEAT_DEG, n_valves=INT_N_VALVES,
        ld_table=INTAKE_LD_TABLE, cd_table=INTAKE_CD_TABLE,
    )

    p_trough = p_cyl_trough_bar * 1e5

    if far_end_bc == "plenum":
        def _fill_left(_):
            fill_subsonic_inflow_left(pipe, rho=RHO_ATM, u=0.0, p=P_ATM, Y=0.0)
    elif far_end_bc == "wall":
        def _fill_left(_):
            fill_reflective_left(pipe)
    else:
        raise ValueError(f"far_end_bc must be 'plenum' or 'wall', got {far_end_bc!r}")

    def bc_apply(t: float) -> None:
        _fill_left(t)
        p_cyl = p_trough if t < T_PULSE_S else P_ATM
        fill_valve_ghost(
            pipe, pipe_end="right", valve_type="intake", vp=vp,
            theta_local_deg=theta_fixed,
            p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
        )

    probe_x = {
        "x=0 (near plenum)": pipe.dx * 1.5,
        "x=L/2 (mid pipe)":  LENGTH_M * 0.5,
        "x=L (near valve)":  LENGTH_M - pipe.dx * 1.5,
    }

    run = run_acoustic(
        pipes={"runner": pipe},
        bc_apply=bc_apply,
        t_end=T_END_S,
        probes_spec={"runner": probe_x},
        waterfall_rows=500,
        cfl=0.5,
    )

    # Wave launched from the RIGHT end. Effective "launch-to-probe" distance
    # for a probe at x_p is (L − x_p). For the mid-pipe probe this equals L/2.
    probe = run.probes["runner"]["x=L/2 (mid pipe)"]
    t_arr = np.array(probe.t)
    p_arr = np.array(probe.p)
    c0 = float(np.sqrt(GAMMA * P_ATM / RHO_ATM))
    effective_probe_x = LENGTH_M - LENGTH_M * 0.5
    arrivals = arrivals_at_probe(
        t_arr, p_arr,
        length_m=LENGTH_M, c0_m_s=c0, pulse_width_s=T_PULSE_S,
        probe_x_m=effective_probe_x, max_arrivals=3,
    )
    R = reflection_from_windowed(arrivals)

    # Relabel for the intake-runner context.
    # A2/A1 = "far-end" reflection (plenum BC or wall depending on config).
    # A3/A2 = "valve-end" reflection (only reliable when A2 is large enough,
    # which for the plenum variant requires R_plenum to NOT be ~0).
    R["R_far"]       = R["R_wall"]
    R["R_far_peak"]  = R["R_wall_peak"]
    if far_end_bc == "plenum":
        R["R_plenum"]      = R["R_wall"]
        R["R_plenum_peak"] = R["R_wall_peak"]
    # R_valve = A3/A2 unchanged from base dict

    R["c0_m_s"] = c0
    R["arrivals"] = arrivals
    R["label"] = label
    R["p_cyl_trough_bar"] = p_cyl_trough_bar
    R["far_end_bc"] = far_end_bc

    far_name = "R_plenum" if far_end_bc == "plenum" else "R_wall"
    save_timeseries_png(
        run, "runner",
        out_path=DIAG_DIR / f"a2_{label}_probes.png",
        title=(
            f"A2/{label} — intake runner ({far_end_bc} far end), "
            f"p_cyl trough = {p_cyl_trough_bar:.2f} bar for {T_PULSE_S*1e3:.1f} ms.  "
            f"{far_name}={R['R_far']:+.3f}  R_valve={R['R_valve']:+.3f}"
        ),
    )
    vmax_kPa = max(50.0, abs(p_cyl_trough_bar - 1.0) * 100.0)
    save_waterfall_png(
        run, "runner", LENGTH_M,
        out_path=DIAG_DIR / f"a2_{label}_waterfall.png",
        title=(
            f"A2/{label} waterfall — intake runner (245 mm × 38 mm ID).  "
            f"Valve open, reservoir p-trough {p_cyl_trough_bar:.2f} bar "
            f"for {T_PULSE_S*1e3:.1f} ms."
        ),
        vmax_kPa=vmax_kPa,
    )

    return run, R


def _write_a2_summary(runs: list) -> None:
    path = DIAG_DIR / "a2_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Test A2 — intake runner reflection\n")
        f.write("=====================================\n\n")
        f.write(f"Pipe length      : {LENGTH_M*1000:.1f} mm\n")
        f.write(f"Pipe diameter    : {DIAMETER_M*1000:.1f} mm ID\n")
        f.write(f"n_cells          : {N_CELLS}\n")
        f.write(f"Run duration     : {T_END_S*1000:.1f} ms\n")
        f.write(f"Pulse duration   : {T_PULSE_S*1000:.2f} ms\n\n")
        for title, R in runs:
            far = R["far_end_bc"]
            far_label = "R_plenum" if far == "plenum" else "R_wall"
            f.write(f"--- {title} (far end = {far}) ---\n")
            f.write(f"  p_cyl trough           = {R['p_cyl_trough_bar']:.3f} bar\n")
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
            f.write(f"  {far_label}_peak  = A2_peak/A1_peak = {R['R_far_peak']:+.4f}\n")
            f.write(f"  {far_label}       = A2_imp /A1_imp  = {R['R_far']:+.4f}  <-- primary\n")
            f.write(f"  R_valve_peak   = A3_peak/A2_peak    = {R['R_valve_peak']:+.4f}\n")
            f.write(f"  R_valve        = A3_imp /A2_imp     = {R['R_valve']:+.4f}  <-- primary\n\n")


@pytest.fixture(scope="module")
def a2_runs():
    # Primary user-spec: stagnation-inflow BC on the left (plenum).
    _, R_plen_nom = run_a2(
        p_cyl_trough_bar=0.7,  label="plenum_nominal_0p7bar",
        far_end_bc="plenum",
    )
    _, R_plen_lin = run_a2(
        p_cyl_trough_bar=0.98, label="plenum_linear_minus2kPa",
        far_end_bc="plenum",
    )
    # Supplementary: reflective wall on the left. Calibrates A2 so R_valve
    # can be extracted cleanly even if the plenum BC is absorbing.
    _, R_wall_nom = run_a2(
        p_cyl_trough_bar=0.7,  label="wall_nominal_0p7bar",
        far_end_bc="wall",
    )
    _, R_wall_lin = run_a2(
        p_cyl_trough_bar=0.98, label="wall_linear_minus2kPa",
        far_end_bc="wall",
    )
    _write_a2_summary([
        ("nominal, plenum far end",      R_plen_nom),
        ("linear,  plenum far end",      R_plen_lin),
        ("nominal, wall   far end (ref)", R_wall_nom),
        ("linear,  wall   far end (ref)", R_wall_lin),
    ])
    return R_plen_nom, R_plen_lin, R_wall_nom, R_wall_lin


def test_a2_plenum_bc_diagnosis(a2_runs):
    """Diagnose the plenum (stagnation-inflow) BC.

    In the linear regime, a constant-pressure reservoir has reflection
    coefficient R = −1. Anything with |R| < 0.1 means the BC is absorbing
    the wave rather than reflecting it.
    """
    _, R_plen_lin, _, _ = a2_runs
    Rp = R_plen_lin["R_far"]
    assert np.isfinite(Rp)
    assert abs(Rp) >= 0.1, (
        f"|R_plenum| = {abs(Rp):.3f} < 0.1 — the plenum (stagnation-inflow) "
        f"BC is absorbing the wave. Expected R_plenum ≈ −1 for a "
        f"constant-pressure reservoir."
    )


def test_a2_intake_valve_bc_diagnosis(a2_runs):
    """Diagnose the intake valve BC.

    Uses the WALL-far-end variant: A2 is dominated by the R_wall≈+1
    bounce, which cleanly feeds into the A3 valve reflection. Under the
    plenum-far-end config, A2 ≈ 0 makes A3/A2 unreliable noise.
    """
    _, _, _, R_wall_lin = a2_runs
    Rw = R_wall_lin["R_far"]      # should be close to +1 for a rigid wall
    Rv = R_wall_lin["R_valve"]
    assert np.isfinite(Rw) and np.isfinite(Rv)
    assert 0.6 < Rw < 1.4, (
        f"wall-far-end R_wall = {Rw:+.3f} is far from +1 in the linear "
        f"regime. Cannot isolate valve BC."
    )
    assert abs(Rv) >= 0.1, (
        f"|R_valve(intake)| = {abs(Rv):.3f} < 0.1 — the intake valve BC is "
        f"absorbing the wave. This is the failure mode the diagnostic was "
        f"designed to catch."
    )


if __name__ == "__main__":
    _, R_plen_nom = run_a2(p_cyl_trough_bar=0.7,  label="plenum_nominal_0p7bar",
                            far_end_bc="plenum")
    _, R_plen_lin = run_a2(p_cyl_trough_bar=0.98, label="plenum_linear_minus2kPa",
                            far_end_bc="plenum")
    _, R_wall_nom = run_a2(p_cyl_trough_bar=0.7,  label="wall_nominal_0p7bar",
                            far_end_bc="wall")
    _, R_wall_lin = run_a2(p_cyl_trough_bar=0.98, label="wall_linear_minus2kPa",
                            far_end_bc="wall")
    _write_a2_summary([
        ("nominal, plenum far end",       R_plen_nom),
        ("linear,  plenum far end",       R_plen_lin),
        ("nominal, wall   far end (ref)", R_wall_nom),
        ("linear,  wall   far end (ref)", R_wall_lin),
    ])
    print("\n=== plenum far end, nominal (0.7 bar) ===")
    print(f"  R_plenum = {R_plen_nom['R_far']:+.4f}")
    print(f"  R_valve  = {R_plen_nom['R_valve']:+.4f}   (unreliable if R_plenum≈0)")
    print("\n=== plenum far end, linear (−2 kPa) ===")
    print(f"  R_plenum = {R_plen_lin['R_far']:+.4f}")
    print(f"  R_valve  = {R_plen_lin['R_valve']:+.4f}   (unreliable if R_plenum≈0)")
    print("\n=== wall far end (ref), nominal (0.7 bar) ===")
    print(f"  R_wall   = {R_wall_nom['R_far']:+.4f}")
    print(f"  R_valve  = {R_wall_nom['R_valve']:+.4f}")
    print("\n=== wall far end (ref), linear (−2 kPa) ===")
    print(f"  R_wall   = {R_wall_lin['R_far']:+.4f}")
    print(f"  R_valve  = {R_wall_lin['R_valve']:+.4f}")
