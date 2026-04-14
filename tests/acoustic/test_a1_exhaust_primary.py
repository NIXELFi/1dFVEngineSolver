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
from bcs.subsonic import fill_subsonic_inflow_left
from bcs.valve import fill_valve_ghost, fill_valve_ghost_characteristic
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
    save_pipe_dump,
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


def _fill_subsonic_inflow_right(pipe, *, rho: float, u: float, p: float, Y: float) -> None:
    """Right-end mirror of bcs/subsonic.py:fill_subsonic_inflow_left.

    Imposes ALL FOUR primitives in the right ghost cells — same
    over-specified subsonic-inflow logic that A2's plenum-far variant
    diagnosed as absorbing. Inlined here only for the asymmetry
    investigation test variant; not used by the main solver.
    """
    from solver.state import I_RHO_A, I_MOM_A, I_E_A, I_Y_A
    ng = pipe.n_ghost
    nc = pipe.n_cells
    gm1 = pipe.gamma - 1.0
    E = p / gm1 + 0.5 * rho * u * u
    for i in range(ng + nc, pipe.n_total):
        A = pipe.area[i]
        pipe.q[i, I_RHO_A] = rho * A
        pipe.q[i, I_MOM_A] = rho * u * A
        pipe.q[i, I_E_A]   = E * A
        pipe.q[i, I_Y_A]   = rho * Y * A


def run_a1(
    *, p_cyl_peak_bar: float, label: str,
    far_end_bc: str = "wall",
    valve_bc_fn=fill_valve_ghost_characteristic,
):
    """Run A1 with a configurable cylinder overpressure.

    ``far_end_bc``: "wall" (default, reflective) or "plenum" (the same
    over-specified stagnation-inflow BC that A2 diagnosed as absorbing).
    The plenum-far variant exists for the asymmetry investigation
    (Phase B → C1 transition): if R_valve(exhaust) measured against a
    plenum reference matches R_valve(exhaust) measured against a wall
    reference, the intake/exhaust asymmetry is regime-dependent
    (Explanation 1) rather than a measurement artifact (Explanation 2).
    """
    pipe = _build_pipe()

    vp, theta_fixed = make_always_open_valve(
        diameter=EXH_VALVE_D, max_lift=EXH_VALVE_MAX_LIFT,
        seat_angle_deg=EXH_VALVE_SEAT_DEG, n_valves=EXH_N_VALVES,
        ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
    )

    p_pulse = p_cyl_peak_bar * 1e5

    if far_end_bc == "wall":
        def _fill_far(_):
            fill_reflective_right(pipe)
    elif far_end_bc == "plenum":
        def _fill_far(_):
            _fill_subsonic_inflow_right(
                pipe, rho=P_ATM / (R_AIR * T_ATM), u=0.0, p=P_ATM, Y=0.0,
            )
    else:
        raise ValueError(f"far_end_bc must be 'wall' or 'plenum', got {far_end_bc!r}")

    def bc_apply(t: float) -> None:
        p_cyl = p_pulse if t < T_PULSE_S else P_ATM
        valve_bc_fn(
            pipe, pipe_end="left", valve_type="exhaust", vp=vp,
            theta_local_deg=theta_fixed,
            p_cyl=p_cyl, T_cyl=T_ATM, xb_cyl=0.0,
        )
        _fill_far(t)

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
    # Relabel the "R_wall" field generically for the plenum-far variant
    # (where it is actually measuring R_plenum, not R_wall).
    R["R_far"]      = R["R_wall"]
    R["R_far_peak"] = R["R_wall_peak"]
    R["c0_m_s"] = c0
    R["arrivals"] = arrivals
    R["label"] = label
    R["p_cyl_peak_bar"] = p_cyl_peak_bar
    R["far_end_bc"] = far_end_bc

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
    # Waterfall vmax scaled to actual measured A1 amplitude so small-
    # amplitude (linear) runs are visible at the same dynamic range as
    # nominal runs.
    vmax_kPa = max(0.5, abs(R["A1_peak"]) / 1000.0 * 1.5)
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

    # Persist the full primitive history as a pipe-state dump for the
    # standalone waterfall viewer (diagnostics/waterfall_viewer.py).
    save_pipe_dump(
        run, "primary",
        DIAG_DIR / f"a1_{label}_dump.npz",
        source=f"tests/acoustic/test_a1_exhaust_primary.py:run_a1({label})",
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
    _, R_nominal = run_a1(p_cyl_peak_bar=3.0,  label="nominal_3bar")
    _, R_linear  = run_a1(p_cyl_peak_bar=1.02, label="linear_1p02bar")
    _write_a1_summary(R_nominal, R_linear)
    return R_nominal, R_linear


@pytest.fixture(scope="module")
def a1_asymmetry_runs():
    """Asymmetry investigation (Phase B → C1 transition).

    Diagnoses the OLD `fill_valve_ghost` (zero-pressure-gradient) BC's
    intake-vs-exhaust asymmetry. Explicitly uses the OLD BC so the
    diagnostic numbers stay stable across the fix; the post-fix
    behavior is verified separately by the main fixture (``a1_runs``).
    """
    _, R_wall_lin   = run_a1(p_cyl_peak_bar=1.02, label="asym_wall_linear",
                              far_end_bc="wall",
                              valve_bc_fn=fill_valve_ghost)
    _, R_plenum_lin = run_a1(p_cyl_peak_bar=1.02, label="asym_plenum_linear",
                              far_end_bc="plenum",
                              valve_bc_fn=fill_valve_ghost)
    _write_asymmetry_note(R_wall_lin, R_plenum_lin)
    return R_wall_lin, R_plenum_lin


def _write_asymmetry_note(R_wall: dict, R_plenum: dict) -> None:
    path = DIAG_DIR / "asymmetry_investigation.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    R_intake_wall_lin_ref = -0.7973  # docs/acoustic_diagnosis/a2_summary.txt

    # Cross-check measurements (computed inline in the test fixture below).
    # Re-run programmatically so the report is reproducible. These also use
    # the OLD BC explicitly so the asymmetry diagnostic numbers are stable.
    _, R_rare_lin = run_a1(p_cyl_peak_bar=0.98, label="asym_rare_wall_linear",
                           far_end_bc="wall",
                           valve_bc_fn=fill_valve_ghost)
    _, R_rare_nom = run_a1(p_cyl_peak_bar=0.7,  label="asym_rare_wall_nominal",
                           far_end_bc="wall",
                           valve_bc_fn=fill_valve_ghost)

    with path.open("w") as f:
        f.write("# Asymmetry investigation: same valve BC, different R\n\n")
        f.write("**Question.** Phase A measured R_valve(exhaust, linear, "
                "wall-far) ≈ −0.10 and R_valve(intake, linear, wall-far) "
                "≈ −0.80. Same `bcs/valve.py:fill_valve_ghost` code path. "
                "Is the asymmetry a real regime-dependent feature of the "
                "broken BC (Explanation 1) or a measurement artifact tied "
                "to the opposite-end BC (Explanation 2)?\n\n")

        f.write("## Initial test: swap the far-end BC\n\n")
        f.write("Swap the wall right-end BC in A1 for the same "
                "`fill_subsonic_inflow_*` plenum BC that A2 uses, and "
                "compare R_valve(exhaust) between configurations.\n\n")
        f.write("| Test | far end | R_far (impulse) | R_valve (impulse) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| A1 exhaust          | wall   | "
                f"{R_wall['R_far']:+.3f} | {R_wall['R_valve']:+.3f} |\n")
        f.write(f"| A1 exhaust          | plenum | "
                f"{R_plenum['R_far']:+.3f} | {R_plenum['R_valve']:+.3f} |\n")
        f.write(f"| A2 intake (Phase A) | wall   | (≈ +0.91) "
                f"| {R_intake_wall_lin_ref:+.3f} |\n\n")
        f.write("**Surprise — but a measurement artifact, not a "
                "diagnosis change.** R_valve(exhaust, plenum-far) reads "
                f"{R_plenum['R_valve']:+.3f}, far from the wall-far "
                f"{R_wall['R_valve']:+.3f}. Inspecting the probe time "
                "series (`a1_asym_plenum_linear_probes.png`) shows that "
                "the wave dies on first contact with the plenum BC and "
                "the pipe then reaches a **steady-state pressure offset** "
                "of about +150 Pa (because both ends are open absorbers "
                "with mismatched reservoir specs). The windowed extremum "
                "at the expected A2 / A3 arrival times then picks up that "
                "static drift, producing a meaningless +1 ratio. The "
                "plenum-far A1 setup cannot measure R_valve cleanly — the "
                "wave doesn't survive long enough to reach the valve a "
                "second time.\n\n")

        f.write("## Cleaner cross-check: same code path, same wall "
                "reference, opposite-sign perturbation\n\n")
        f.write("Force the exhaust valve into the same regime that "
                "intake operates in, by perturbing the cylinder pressure "
                "**downward** instead of upward. This launches a "
                "rarefaction wave at the exhaust valve while keeping the "
                "wall right-end BC and the same `fill_valve_ghost` code "
                "path. If the wave-type (compression vs rarefaction) is "
                "what drives the asymmetry, R_valve(exhaust, rarefaction) "
                "should match R_valve(intake, rarefaction) ≈ −0.80.\n\n")
        f.write("| Test | far end | cyl perturbation | wave type | R_valve |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| A1 exhaust  | wall | +2 kPa     | compression | "
                f"{R_wall['R_valve']:+.3f} |\n")
        f.write(f"| A1 exhaust  | wall | −2 kPa     | rarefaction | "
                f"{R_rare_lin['R_valve']:+.3f} |\n")
        f.write(f"| A1 exhaust  | wall | −300 kPa   | rarefaction | "
                f"{R_rare_nom['R_valve']:+.3f} |\n")
        f.write(f"| A2 intake   | wall | −2 kPa     | rarefaction | "
                f"{R_intake_wall_lin_ref:+.3f} |\n\n")

        f.write("## Diagnosis (refined)\n\n")
        rv_rare = R_rare_lin["R_valve"]
        rv_intake = R_intake_wall_lin_ref
        if abs(abs(rv_rare) - abs(rv_intake)) < 0.15:
            f.write(f"R_valve(exhaust, rarefaction) = {rv_rare:+.3f} "
                    f"matches R_valve(intake, rarefaction) = "
                    f"{rv_intake:+.3f} within measurement noise "
                    f"(≈ 0.15, the MUSCL-HLLC dissipation floor we "
                    f"calibrated against A2's wall-ref control). "
                    f"R_valve(exhaust, compression) = {R_wall['R_valve']:+.3f} "
                    f"is the outlier.\n\n"
                    "**Conclusion.** The asymmetry is between **wave types** "
                    "(compression vs rarefaction), not between intake and "
                    "exhaust valves. The `fill_valve_ghost` code path "
                    "absorbs compressions but mostly-reflects rarefactions. "
                    "Both the intake and exhaust valves would absorb if "
                    "they saw a compression in the engine cycle; both would "
                    "reflect if they saw a rarefaction.\n\n"
                    "**Mechanism.** When a compression wave arrives at an "
                    "exhaust valve with cyl held at atmospheric, "
                    "p_pipe_face >> p_cyl. The orifice equation produces a "
                    "large outflow mdot, which sets the ghost velocity to "
                    "a large magnitude. With p_ghost = p_pipe (zero "
                    "gradient) AND a strongly prescribed velocity, the BC "
                    "is essentially a Dirichlet velocity condition for the "
                    "interior — it imposes the steady-orifice flow rate "
                    "and absorbs whatever acoustic content arrives. When a "
                    "rarefaction arrives, p_pipe_face < p_cyl, the "
                    "orifice runs in the opposite direction with a *small* "
                    "mdot (only 30 % overpressure differential typical), "
                    "so the prescribed velocity at the ghost is small. "
                    "With small u_ghost and p_ghost = p_pipe, the BC "
                    "looks much closer to a wall (u ≈ 0, p matched), "
                    "which reflects cleanly.\n\n"
                    "**Implication for Phase C1.** The characteristic-"
                    "based fix should eliminate the asymmetry. Both wave "
                    "types should produce similar R magnitudes "
                    "(probably both in the −0.5 to −0.9 range, the "
                    "physically correct value for an open valve onto a "
                    "fixed-pressure reservoir). If the post-fix R differs "
                    "by more than 0.2 between compression and rarefaction "
                    "tests, the characteristic formulation has a real "
                    "bug we missed and we stop again.\n")
        else:
            f.write(f"R_valve(exhaust, rarefaction) = {rv_rare:+.3f} "
                    f"vs R_valve(intake, rarefaction) = {rv_intake:+.3f}. "
                    "These do NOT match within the 0.15 measurement floor, "
                    "so the wave-type explanation is incomplete and there "
                    "is some remaining intake/exhaust asymmetry. **Stop "
                    "and investigate before fixing.**\n")


def test_a1_asymmetry_investigation(a1_asymmetry_runs):
    R_wall, R_plenum = a1_asymmetry_runs
    # No hard assertion — this is a diagnostic, not a regression test.
    # Just confirm the measurements completed.
    assert np.isfinite(R_wall["R_valve"])
    assert np.isfinite(R_plenum["R_valve"])
    assert np.isfinite(R_wall["R_far"])
    assert np.isfinite(R_plenum["R_far"])


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
    """+2 kPa linear-acoustic perturbation. The unambiguous diagnostic:
    R_wall must be ≈ +1 (sanity); R_valve is the BC reflection coefficient.

    Phase C1 (post-fix) acceptance: |R_valve| > 0.3 with negative sign.
    The characteristic + orifice BC must produce a partial pressure-
    release reflection (compression → reduced compression with sign-flip
    or attenuated). The exact magnitude depends on the orifice impedance
    (A_eff/A_pipe ratio) — for the SDM26 exhaust valve at max lift this
    sits in the [-0.3, -0.9] band.
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
    # Phase C1 post-fix acceptance bar.
    assert Rv < 0.0, (
        f"R_valve = {Rv:+.3f} is non-negative — the post-fix BC should give "
        f"pressure-release-like reflection (negative sign)."
    )
    assert abs(Rv) > 0.3, (
        f"|R_valve| = {abs(Rv):.3f} ≤ 0.3 in the linear regime — the BC is "
        f"reflecting too weakly relative to the orifice impedance. Expected "
        f"|R| in the [0.3, 0.9] band for an SDM26 exhaust valve at max lift."
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
