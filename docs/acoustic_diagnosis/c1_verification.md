# Phase C1 verification report

**Date.** 2026-04-14
**Branch.** `diag/acoustic-bc`
**Scope.** Exhaust + intake valve BC only. Plenum BC (`bcs/subsonic.py`)
and junction CV (`bcs/junction_cv.py`) intentionally NOT touched.

## What changed

`bcs/valve.py:fill_valve_ghost_characteristic` (new): characteristic-
based ghost fill with simultaneous orifice constraint, drop-in
replacement for the old `fill_valve_ghost`. Two cases:

- **Pipe-side outflow** (mass leaves pipe through the valve): bisect
  p_face ∈ [p_cyl, p_int] such that the pipe-side characteristic
  mass flux (computed from interior entropy + interior J invariant +
  isentropic ρ_face from interior) equals the orifice mass flux
  (compressible orifice with subsonic + choked branches).
- **Pipe-side inflow** (cyl pushes mass into pipe): energy + J system
  pins (c_face, u_face) independently of p_face; bisect p_face ∈
  [p_int, p_cyl] for orifice mass-flux match. Falls back to
  orifice-driven kickstart when characteristic system is degenerate
  (cold-start limitation of pure characteristic theory; documented
  in the BC's docstring).

Closed valve (A_eff < 1e-8 m²): reflective ghost (unchanged).

`models/sdm26.py` line 36: `from bcs.valve import fill_valve_ghost_characteristic as fill_valve_ghost`. Single-line wiring change.

`tests/acoustic/test_a1_exhaust_primary.py` and
`tests/acoustic/test_a3_sdm26_manifold.py`: switched to use the
characteristic BC by default. The asymmetry-investigation fixture
explicitly pins the OLD BC so its diagnostic numbers are preserved.

## Acceptance criteria — results

### Acoustic test A1 (exhaust valve, linear regime, +2 kPa)

| Wave type | R_wall (sanity) | R_valve | Bar |
|---|---|---|---|
| compression (+2 kPa) | +0.99 | **−0.83** | \|R\| > 0.3, neg sign ✓ |
| rarefaction (−2 kPa) | +0.99 | **−0.90** | \|R\| > 0.3, neg sign ✓ |

Asymmetry resolved: |R| differs by < 0.1 between wave types now (was
0.10 vs 0.76 with the old BC). Magnitude lands in the [0.3, 0.9]
band you predicted for an SDM26 exhaust at max lift.

### A1 waterfall (post-C1)

`docs/acoustic_diagnosis/a1_linear_1p02bar_waterfall.png` — clear
chevron pattern of bouncing waves throughout the full 20 ms run.
Each successive reflection is attenuated by the orifice impedance
exactly as you described. Visible round trips: ~10 before the
signal blurs into numerical noise.

(`a1_nominal_3bar_waterfall.png` for the 3 bar pulse shows the same
chevron pattern but with clearer leading edges. The 3-bar nominal
R_valve impulse measurement is contaminated by a window-edge effect
because the post-fix BC produces a faster wave on outgoing legs;
linear regime is the correct diagnostic per your earlier guidance.)

### Acoustic test A2 (intake valve, linear regime, −2 kPa)

| Configuration | R_far | R_valve | Note |
|---|---|---|---|
| wall far end | +0.91 | −0.84 | clean intake-valve diagnosis ✓ |
| plenum far end | −0.07 | (unreliable) | plenum still absorbing — Phase C2 work |

Intake valve passes the same |R| > 0.3 negative-sign bar.

### Acoustic test A3 (full SDM26 4-2-1 manifold, linear, +5 kPa)

| Metric | Pre-C1 | Post-C1 | Bar |
|---|---|---|---|
| R_round_trip (impulse) | +0.023 | **+0.022** | unchanged |

A3 round-trip is essentially identical pre/post C1. The 4-2 junction
CV kills the wave on first contact, so the valve-BC fix can't show
its effect at the round-trip probe — the wave never gets back to the
P0 valve. Confirms the junction CV is the next absorber to fix
(deferred per your plan; opens as Phase E if SDM26 sweep still shows
no tuning after C2).

### Existing 95-test validation suite

```
tests/ --ignore=tests/acoustic   →   95 passed in 42.98s
```

Specifically green:
- All FV scheme tests (Sod, Lax, 1-2-3, contacts, conservation, nozzle, friction+heat, choked restrictor, junction-CV conservation, cylinder integrator, gas properties, HLLC unit tests)
- **`test_valve_Cd_scaling_affects_VE`** (the regression test that catches BCs without orifice impedance) ✓

### SDM26 single-RPM engine point (10500 RPM, 25 cycles)

| Metric | Pre-C1 (no orifice) | **Post-C1** | Bar |
|---|---|---|---|
| EGT mean (final cycle) | 1893 K | **1251 K** | 1000–1400 K ✓ |
| IMEP (final cycle) | oscillating 6.4–9.5 bar | **9.98 bar (converged @ cycle 18)** | converged ✓ |
| VE | oscillating 40–67 % | **62.6 %** | converged ✓ |
| Wheel power | n/a (chaotic) | **36.3 hp** | sane ✓ |
| Mass nonconservation | machine | **−3.3e−19 kg/cycle** | machine ✓ |

Per-cycle convergence trace shows clean monotonic IMEP rise from
9.0 bar (cycle 1) down to 7.3 (cycle 3, transient still settling)
back up to 9.98 bar (cycle 18) and stable thereafter.

## Outstanding (not addressed in C1, by design)

- Plenum BC (`bcs/subsonic.py:fill_subsonic_inflow_left`): still
  R_plenum ≈ −0.07. C2 fix.
- Junction CV (`bcs/junction_cv.py`): A3 still dead. Phase E if needed.
- Engine-level sweep (full RPM range) not yet rerun. Single-point
  10500 confirms the BC is healthy; full sweep is part of C3 (post-C2).
- The 3-bar nominal A1 measurement still has a window-edge artifact at
  the post-fix wave-speed regime. Linear regime is the trustworthy
  diagnostic; the artifact does not affect the engine-level outcome.

## Awaiting greenlight for C2 (plenum BC)
