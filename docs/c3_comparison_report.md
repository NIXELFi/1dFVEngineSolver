# Phase C3 — Full sweep comparison report

**Date.** 2026-04-15
**Branch.** `diag/acoustic-bc`
**Scope.** 16-point sweep (6000–13500 RPM, 500-RPM steps), 12-cycle minimum, 40-cycle cap, IMEP-convergence stop (0.5 % cycle-to-cycle).

---

## ⚠ Known limitation — tuned-length prediction not yet reliable

**V2 in its current state is NOT reliable for exhaust primary-length
optimization or any decision involving acoustic tuning of the exhaust
manifold.** The Phase C1 + C2 fixes corrected the valve and plenum BCs,
but the junction control-volume model in `bcs/junction_cv.py` still
attenuates incident waves significantly at each junction crossing. The
SDM26 4-2-1 manifold requires four junction transmissions per round
trip, and the measured per-junction transmission of ≈ 0.69 gives a
round-trip survival of (0.69)⁴ ≈ 0.23. That is enough wave amplitude
to keep cylinder breathing dynamics realistic at single-point operation
(EGT correct, IMEP converged, mass conservation at machine precision),
but **not enough for tuning resonances to compound into visible torque
or VE peaks across the sweep**.

The diagnostic measurement that pins this down is the A3 manifold
round-trip reflection coefficient: linear-regime R_round_trip = +0.228
(the bar for clean tuning prediction is ≥ 0.5, equivalent to per-
junction transmission ≥ 0.84). The full audit is in
`docs/acoustic_diagnosis/findings.md`; the Phase E plan that closes
this gap is in `docs/phase_e_plan.md`.

**Phase E (junction CV characteristic-coupling) is scheduled but not
started.** Until Phase E is complete, treat any V2 prediction that
depends on exhaust acoustic tuning as qualitative-only. See the
"Capabilities" table in `README.md` for a question-by-question
breakdown of what V2 can and cannot answer reliably in its current
state.

---

## TL;DR

- Engine results unchanged in shape from pre-fix. Both SDM25 and SDM26 still show **monotone-decreasing torque from 6000 RPM**, no acoustic tuning peaks above 6000 RPM in either configuration.
- SDM25 vs SDM26 curves are still **scaled copies of each other** (shape-diff score: 0.0002 post-fix vs 0.0003 pre-fix — essentially unchanged).
- A3 manifold round-trip R improved **10.3× in linear regime** (0.022 → 0.228) but the engine sweep does not yet reflect this. At nominal blowdown amplitudes (5 bar), R_round_trip = +0.011, essentially unchanged from pre-fix.
- All acceptance bars met: EGT in [800, 1500] K everywhere, machine-precision conservation, zero UNHANDLED BC calls across 2.7M classifications.
- Wall clock per sweep grew ~1.49× (62→93 s SDM25, 67→100 s SDM26) due to the bisection-based BC. Acceptable; the sweep is not in the inner loop of any production usage.

**The data is consistent with Outcome 3** of your three-way decision gate: the valve+plenum BC fixes alone are not enough to recover tuning peaks. The junction CV is the dominant remaining absorber. **Phase E (junction CV characteristic-coupling) is the next step.**

## Sweep tables (post-fix)

### SDM25 (4-1 topology)

| RPM | IMEP bar | VE % | EGT K | Whl Pwr kW | Whl Tq Nm | intake g/cyc | nc kg/cyc | cycles | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 6000 | 13.68 | 85.2 | 1035 | 30.2 | 48.1 | 0.601 | 8.2e-18 | 12 | 7.0 |
| 6500 | 13.36 | 82.9 | 1064 | 31.3 | 45.9 | 0.585 | 8.3e-18 | 12 | 6.5 |
| 7000 | 13.07 | 81.0 | 1075 | 32.2 | 43.9 | 0.571 | 5.3e-18 | 13 | 6.6 |
| 7500 | 12.78 | 79.2 | 1000 | 32.9 | 41.9 | 0.558 | 4.6e-18 | 14 | 6.8 |
| 8000 | 12.49 | 77.5 | 1047 | 33.4 | 39.8 | 0.546 | 6.0e-18 | 15 | 6.9 |
| 8500 | 12.16 | 75.5 | 1095 | 33.5 | 37.6 | 0.533 | 5.5e-18 | 14 | 6.0 |
| 9000 | 11.85 | 73.6 | 1131 | 33.4 | 35.4 | 0.519 | 6.3e-18 | 14 | 5.7 |
| 9500 | 11.56 | 72.0 | 1153 | 33.1 | 33.3 | 0.508 | 6.2e-18 | 15 | 6.0 |
| 10000 | 11.27 | 70.5 | 1155 | 32.6 | 31.1 | 0.498 | 6.1e-18 | 15 | 5.7 |
| 10500 | 10.98 | 69.0 | 1198 | 31.8 | 29.0 | 0.486 | 3.8e-18 | 15 | 5.4 |
| 11000 | 10.69 | 67.3 | 1225 | 30.8 | 26.8 | 0.475 | 4.5e-18 | 15 | 5.2 |
| 11500 | 10.37 | 65.4 | 1256 | 29.4 | 24.4 | 0.461 | 5.2e-18 | 15 | 4.9 |
| 12000 | 10.12 | 63.9 | 1285 | 28.1 | 22.4 | 0.451 | 4.6e-18 | 16 | 5.1 |
| 12500 | 9.86 | 62.5 | 1325 | 26.5 | 20.3 | 0.441 | 4.7e-18 | 16 | 4.9 |
| 13000 | 9.60 | 61.0 | 1350 | 24.6 | 18.1 | 0.430 | 5.1e-18 | 17 | 5.1 |
| 13500 | 9.32 | 59.3 | 1385 | 22.4 | 15.8 | 0.418 | 4.6e-18 | 16 | 4.5 |

### SDM26 (4-2-1 topology)

| RPM | IMEP bar | VE % | EGT K | Whl Pwr kW | Whl Tq Nm | intake g/cyc | nc kg/cyc | cycles | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 6000 | 13.10 | 82.0 | 1040 | 28.6 | 45.5 | 0.579 | 3.0e-18 | 12 | 8.2 |
| 6500 | 12.78 | 80.0 | 1073 | 29.6 | 43.4 | 0.564 | 3.6e-18 | 12 | 7.6 |
| 7000 | 12.46 | 78.0 | 1103 | 30.2 | 41.3 | 0.550 | 4.6e-18 | 12 | 7.0 |
| 7500 | 12.15 | 76.1 | 1123 | 30.8 | 39.2 | 0.537 | 5.2e-18 | 13 | 7.3 |
| 8000 | 11.84 | 74.2 | 1148 | 31.0 | 37.0 | 0.523 | 5.0e-18 | 13 | 6.8 |
| 8500 | 11.51 | 72.2 | 1175 | 31.0 | 34.8 | 0.509 | 5.6e-18 | 13 | 6.4 |
| 9000 | 11.19 | 70.2 | 1199 | 30.7 | 32.5 | 0.495 | 3.4e-18 | 14 | 6.7 |
| 9500 | 10.84 | 68.1 | 1221 | 30.0 | 30.2 | 0.480 | 3.1e-18 | 14 | 6.6 |
| 10000 | 10.54 | 66.2 | 1237 | 29.3 | 28.0 | 0.467 | 3.8e-18 | 14 | 6.0 |
| 10500 | 10.25 | 64.5 | 1253 | 28.4 | 25.8 | 0.455 | 5.9e-18 | 14 | 5.7 |
| 11000 | 9.97 | 62.8 | 1266 | 27.2 | 23.6 | 0.443 | 4.2e-18 | 14 | 5.4 |
| 11500 | 9.72 | 61.2 | 1284 | 26.0 | 21.6 | 0.432 | 4.3e-18 | 15 | 5.6 |
| 12000 | 9.45 | 59.7 | 1307 | 24.5 | 19.5 | 0.421 | 4.1e-18 | 15 | 5.3 |
| 12500 | 9.23 | 58.4 | 1333 | 22.9 | 17.5 | 0.412 | 4.4e-18 | 16 | 5.5 |
| 13000 | 8.99 | 57.1 | 1362 | 21.1 | 15.5 | 0.403 | 3.4e-18 | 15 | 4.9 |
| 13500 | 8.74 | 55.5 | 1382 | 18.9 | 13.3 | 0.392 | 3.3e-18 | 16 | 5.1 |

## Specific callouts (per acceptance criteria)

### Torque peak RPM
- **SDM25 post-fix.** Peak torque 48.1 Nm at **6000 RPM**. Monotone-decreasing from 48.1 Nm at 6000 → 15.8 Nm at 13500. _Pre-fix: peak at 6000 RPM._
- **SDM26 post-fix.** Peak torque 45.5 Nm at **6000 RPM**. Monotone-decreasing from 45.5 Nm at 6000 → 13.3 Nm at 13500. _Pre-fix: peak at 6000 RPM._
- **Both peaks remain at 6000 RPM (the lowest sample point).** The user's qualitative bar from the Phase A prompt is *not met*: "Torque curve has a peak somewhere above 6000 RPM for at least one of the two configurations."

### Wheel power peak
- **SDM25 post-fix.** Peak wheel power 33.5 kW at **8500 RPM** (mechanical peak: torque × RPM, not acoustic).
- **SDM26 post-fix.** Peak wheel power 31.0 kW at **8000 RPM**.

### Curve-shape differentiation (SDM25 vs SDM26)
- **Torque shape-diff score.** Post-fix: 0.0002. Pre-fix: 0.0003. (0 = identical shape after linear scaling; > 0.05 ≈ visibly different.)
- **VE shape-diff score.** Post-fix: 0.0008. Pre-fix: 0.0013.
- **Both scores are essentially zero**, meaning the post-fix curves are still scaled copies of each other (same shape, different absolute level). The user's bar — "SDM25 and SDM26 should produce visually different curve shapes, not just scaled copies" — is *not met*.

### VE wiggles / tuning bumps
- Inspect `c3_plots/ve.png` and `c3_plots/wheel_torque.png`. Both VE and torque curves are smooth monotone curves with no visible local maxima or wiggles in either configuration.

## A3 manifold round-trip — quantitative

(Re-run with the C1+C2 BCs from `tests/acoustic/test_a3_sdm26_manifold.py`.)

| variant | A1 imp [Pa·s] | A2 imp [Pa·s] | R_round_trip (impulse) | R_round_trip (peak) |
|---|---|---|---|---|
| Pre-fix linear (5 kPa) | +3.52 | +0.08 | **+0.022** | +0.039 |
| Pre-fix nominal (5 bar) | +118.5 | −0.04 | −0.0003 | +0.008 |
| Post-fix linear (5 kPa) | +4.76 | +1.08 | **+0.228** | +0.509 |
| Post-fix nominal (5 bar) | +119.8 | +1.32 | +0.011 | +0.021 |

Linear-regime R_round_trip improved **10.3×** (0.022 → 0.228), comfortably above the |R| > 0.1 acceptance bar. Nominal-amplitude (5 bar pulse) is still essentially dead because at shock strengths the junction CV's stagnation-momentum-discard absorbs almost the full incident KE. The linear value is the diagnostic that matters for sustained acoustic tuning effects (which compound at small amplitudes); the nominal value reflects strong-shock dissipation, much of which is correctly physical.

Manifold waterfall PNGs from the latest A3 run are in `docs/acoustic_diagnosis/a3_*_waterfall.png`. The P0 primary's waterfall now shows a clear outbound diagonal followed by a *visibly returning* diagonal at t ≈ 4.6 ms (linear regime), where the pre-fix run showed silence after t ≈ 3 ms.

## Conservation diagnostic

- **SDM25 max nonconservation across all cycles of all 16 RPM points:** 8.35e-18 kg/cycle (machine precision, well below the 1e-12 acceptance bar).
- **SDM26 max nonconservation:** 5.91e-18 kg/cycle.

## Wall-clock comparison

| Sweep | Pre-fix [s] | Post-fix [s] | Ratio |
|---|---|---|---|
| SDM25 | 62 | 93 | 1.49× |
| SDM26 | 67 | 100 | 1.49× |

The ~1.49× slowdown comes from the bisection-based valve BC (60-iteration cap, ~5–10 typical iterations per call). BCs are evaluated once per pipe-end per time step (~9 calls per step in the SDM26 model: 4 intake valves + 4 exhaust valves + 1 collector); the per-step BC cost grew from microseconds to tens of microseconds. Still negligible relative to the MUSCL-Hancock interior update; the absolute wall is fine.

## Regime-log summary (cross-sweep)

### SDM25 (1,330,859 BC calls across 16 RPM points)

| regime | count | % |
|---|---|---|
| choked_inflow | 586,649 | 44.08% |
| subsonic_outflow | 527,986 | 39.67% |
| startup | 155,538 | 11.69% |
| subsonic_inflow | 54,946 | 4.13% |
| choked_outflow | 5,740 | 0.43% |
| **UNHANDLED** | 0 | — |

### SDM26 (1,359,809 BC calls across 16 RPM points)

| regime | count | % |
|---|---|---|
| choked_inflow | 577,312 | 42.46% |
| subsonic_outflow | 544,289 | 40.03% |
| startup | 136,578 | 10.04% |
| subsonic_inflow | 99,578 | 7.32% |
| choked_outflow | 2,052 | 0.15% |
| **UNHANDLED** | 0 | — |

**Zero UNHANDLED BC calls across both sweeps** — every call exits with one of the five regime labels. The startup category captures the genuine cold-start transient at each RPM point's first few cycles plus periodic equilibrium moments mid-cycle (wave nodes, valve transitions where |u_int| < 1 m/s and Δp/p < 1e-4); these are physically real events, not iteration failures.

Per-RPM startup-rate trend (SDM26): 7.3 % at 6000 RPM, rising slightly with RPM to 12.9 % at 11500 RPM, then easing back to 11.5 % at 13500 RPM. The rate does NOT drop to near-zero on later cycles within an RPM point because the engine is producing many physically-quiescent moments per cycle (especially at higher RPM where valve transitions are more frequent in absolute time). This is expected behavior for the new BC, not a sign of unconverged transient.

## Phase E recommendation

The C3 sweep shows that the C1+C2 BC fixes are **necessary but not sufficient** for engine-level acoustic tuning. The valve and plenum BCs are now characteristically correct, the manifold round-trip survives at 22 % in linear regime (10× pre-fix), and the engine produces converged, machine-precision-conservative, EGT-in-band results. But the engine sweep is essentially unchanged in shape from pre-fix:

- Both configurations still show monotone-decreasing torque from 6000 RPM with peak at the lowest sample point.
- SDM25 and SDM26 differ only by an offset in absolute level, not by curve shape.
- VE shows no acoustic wiggles in either configuration.

Per the user's three-way decision gate stated in the C2 review, this lands in **Outcome 3 (~20 % prior, but updated by the data): Phase E necessary**. The junction CV is the dominant remaining absorber, and the SDM26 4-2-1 topology requires four junction transmissions per round trip (two going out, two coming back), so a per-junction transmission of ~70 % gives a round-trip amplitude survival of (0.7)⁴ ≈ 24 % — consistent with the measured 22 %. To get tuning peaks visible in the engine sweep, we need per-junction transmission closer to 0.95+.

Recommended next phase: **characteristic-coupling junction CV** (Winterbone & Pearson § 9.2.3 multi-pipe Riemann junction OR an N-leg-coupled face-state solver). Significant code surgery in `bcs/junction_cv.py` and `models/sdm26.py` (junction wiring), comparable in scope to Phase C1+C2 combined.

## Artifacts inventory

- `docs/sdm25_sweep.json` (post-fix) and `docs/sdm26_sweep.json` (post-fix) — full per-cycle stats per RPM point.
- `docs/sdm25_sweep_prefix.json` and `docs/sdm26_sweep_prefix.json` — pre-fix snapshots for overlay comparison.
- `docs/c3_regime_log_sdm25.json` and `docs/c3_regime_log_sdm26.json` — per-RPM regime classification breakdowns.
- `docs/c3_sweep_meta.json` — wall-clock + nonconservation summary.
- `docs/c3_plots/*.png` — overlay plots (wheel power, wheel torque, VE, EGT, intake mass, IMEP, BMEP/IMEP/FMEP stack).
- `docs/acoustic_diagnosis/a3_*_waterfall.png` — manifold waterfalls (re-run with C1+C2 BCs).
