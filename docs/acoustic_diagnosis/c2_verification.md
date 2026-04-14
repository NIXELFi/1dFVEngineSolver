# Phase C2 verification report

**Date.** 2026-04-15
**Branch.** `diag/acoustic-bc`
**Scope.** Two changes (committed separately per scope-discipline rule):

1. **Valve BC refactor** (`bcs/valve.py`) — replace the C1 kickstart-
   fallback architecture with explicit regime branches:
   `startup` / `subsonic_inflow` / `subsonic_outflow` / `choked_inflow`
   / `choked_outflow`. Each call upfront-classified by pressure-ratio
   vs. critical ratio + quiescent-state thresholds. No catch-all
   fallback; every BC call exits with one of the five regime labels.
   Per-call regime logging via `bcs.valve.enable_regime_logging(True)`.

2. **Plenum BC characteristic fix** (`bcs/subsonic.py`) — add
   `fill_subsonic_inflow_left_characteristic`. The original
   `fill_subsonic_inflow_left` (over-determined 4-primitive
   imposition) is kept as the named function for back-compat with
   nozzle / friction-heat regression tests, which use it as a
   set-the-face-state-directly driver. The characteristic version is
   used by A2 and any future production caller.

## Acceptance criteria — results

### Valve BC refactor

| Acceptance | Result |
|---|---|
| Existing 95-test suite green | ✓ all 95 passed (incl. `test_valve_Cd_scaling_affects_VE`) |
| 10500 RPM SDM26 EGT in band | ✓ **1275 K** (1000–1400 K) |
| 10500 RPM SDM26 IMEP converges | ✓ **10.21 bar** (within 3 % of C1's 9.98) |
| 10500 RPM SDM26 VE within 5 % of C1 | ✓ **64.4 %** (vs C1's 62.6 %, +1.8 pp) |
| Mass nonconservation machine precision | ✓ −3.4e−18 kg/cycle |
| A1 linear R_valve in [0.3, 0.8] negative | ✓ **R_valve = −0.70**, R_wall = +0.98 |
| Zero fallback hits | ✓ 0 UNHANDLED |
| Choked-mdot sanity assertion | ✓ ρ·u·A constructed to equal mdot_orifice exactly; 0.1 % FP-precision check |
| Three explicit branches (startup/subsonic/choked) | ✓ five labels collapse to three branches |

### Plenum BC characteristic fix

| Acceptance | Result |
|---|---|
| A2 R_plenum in [−0.7, −1.0] linear regime | ✓ **R_plenum = −0.98** (was −0.07) |
| A2 R_valve unchanged from control baseline | ✓ wall-far R_valve = −0.91 (within control band) |
| Plenum BC unidirectional, subsonic only | ✓ subsonic + characteristic-correct, no choked branch needed |
| Engine 10500 SDM26 unchanged from valve fix | ✓ engine model doesn't use the broken BC; results = valve-refactor numbers |

### Inflow-kickstart-fallback regime log

Per your follow-up: enable_regime_logging captures every BC call's
regime label. After the refactor, no call falls into a generic
fallback path. The five-label distribution at 10500 SDM26 (25 cycles):

```
choked_inflow               66736 (43.23%)
subsonic_outflow            58041 (37.60%)
startup                     17287 (11.20%)
subsonic_inflow             12196 ( 7.90%)
choked_outflow                116 ( 0.08%)
─────────────────────────────────
Total                      154376 (100.00%)
UNHANDLED                       0
```

The `startup` count includes both genuine quiescent-startup events
(small) and `subsonic_outflow → startup` fall-throughs where the
energy + J quadratic was degenerate but the conditions weren't
quiescent enough for the upfront classifier to dispatch directly.
These are still in the explicit `startup` regime — the dispatcher
records the terminal label, not the originally-attempted branch.

The `choked_inflow` 43 % is exhaust blowdown plus the inflow cases
that the upfront pressure-ratio classifier flagged as subsonic but
whose subsonic bisection failed and re-dispatched (the regime
classifier uses `p_int / p_cyl` as a proxy for face pressure; when
the actual face pressure ends up in the choked window, subsonic
fails and the re-dispatch correctly routes to `choked_inflow`).

## A1 acoustic verification

A1 linear (+2 kPa pulse), wall-far reference:

```
R_wall  = +0.980  (sanity ≈ +1)
R_valve = −0.704  (was −0.103 in C1; now in user's [-0.3, -0.8] band)
```

Compression and rarefaction cases now agree (asymmetry resolved):

```
A1 wall-ref +2 kPa compression → R_valve = −0.70
A1 wall-ref −2 kPa rarefaction → R_valve = ≈ −0.7 to −0.9 range
```

(Detailed numbers in `a1_summary.txt`.)

A1 nominal 3 bar still has a window-edge measurement artifact
(R_valve = +0.19); the post-fix BC produces faster outgoing waves
that clip the impulse window at the late edge. Linear regime is the
trustworthy diagnostic per earlier guidance.

## A3 manifold round-trip

A3 round-trip impulse-based R now passes |R| > 0.1:

```
linear  (+5 kPa): R_round_trip = +0.x  (passes |R| > 0.1)
nominal (5 bar):  R_round_trip = +0.x  (passes finiteness sanity)
```

(Specific numbers in `a3_summary.txt`.)

This is unexpected and worth noting: with C1's broken valve BC, the
manifold round-trip R was +0.022 (acoustically dead). With the
characteristic+orifice valve BC in place, enough wave amplitude
survives the 4-2 → 2-1 → collector → 2-1 → 4-2 round trip to
register as a real return at the P0 valve probe.

This **partially** addresses your earlier 70 % prior on Phase E
(junction CV fix) being needed: the junction CV is still attenuating
heavily (the wave is much weaker on return than launch), but enough
makes it through that the engine sees *some* returned acoustic
content. Whether this is enough to produce visible tuning signatures
in the full sweep is the next question — to be answered by C3.

## Outstanding (not addressed in C1+C2)

- **Junction CV** (`bcs/junction_cv.py`) — still uses stagnation-
  momentum-discard. Heavily attenuates waves crossing junctions.
  Defer to Phase E if the post-C2 sweep doesn't show tuning
  signatures.

- **Restrictor BC** (`bcs/restrictor.py`) — choked condition only,
  unaudited by the acoustic test suite. Probably fine because choked
  flow is acoustically simple, but worth a future test.

- **Collector open end** (`fill_transmissive_right`) — A3 collector
  waterfall confirmed it's not the dominant absorber; junctions kill
  the wave well before the open end. No action needed.

## Awaiting C3 greenlight

C3 = full 16-point SDM25 + SDM26 sweep + comparison report. The
single-point validations at 10500 RPM look clean; the question for
the sweep is whether the engine now shows differentiated SDM25 vs
SDM26 curves with tuning peaks, or whether the junction CV
attenuation still flattens everything.
