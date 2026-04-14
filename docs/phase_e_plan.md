# Phase E plan — characteristic-coupling junction model

**Status as of 2026-04-15.** Scheduled, not started. To be picked up
2–4 weeks after the SDM26 design review settles.

## Why this exists

The Phase A acoustic-BC audit (`docs/acoustic_diagnosis/findings.md`)
identified three boundary-condition components in V2 that were
absorbing acoustic content that should have been reflected. Phase C1
fixed `bcs/valve.py:fill_valve_ghost` (zero-pressure-gradient ghost →
characteristic + orifice with explicit subsonic / choked / startup
regimes). Phase C2 fixed `bcs/subsonic.py:fill_subsonic_inflow_left`
(over-determined 4-primitive imposition → standard subsonic-inflow
characteristic BC).

Phase C3 ran the full 16-point sweep with the C1+C2 BCs in place
(`docs/c3_comparison_report.md`). The result: the engine sweep is
**still monotone-decreasing torque from 6000 RPM in both SDM25 and
SDM26**, with the two configs producing curves that are essentially
scaled copies of each other (shape-diff score 0.0002). The acoustic-
test-suite measurement at A3 (full 4-2-1 manifold round trip) shows
the wave is alive but attenuated: linear-regime R_round_trip = +0.228
post-fix vs +0.022 pre-fix (10.3× improvement), but nominal-amplitude
R_round_trip = +0.011 (essentially unchanged from pre-fix +0.022).

The diagnostic math: the SDM26 4-2-1 topology requires four junction
crossings per round trip (two going outbound, two coming back). With
22 % round-trip survival, the per-junction transmission coefficient
is (0.228)^(1/4) ≈ 0.69 — i.e. the junction CV is dropping ~30 % of
incident wave amplitude per crossing. To get tuning peaks visible at
the cylinder valve face, per-junction transmission needs to be ≥ 0.95
(then round-trip survival = 0.95⁴ ≈ 0.81, comparable to a well-tuned
real engine).

The remaining absorber is `bcs/junction_cv.py:JunctionCV.fill_ghosts`,
which uses a stagnation control volume (Winterbone & Pearson "constant-
pressure junction"). The model is correct in the conservation sense
(mass + energy + ρY all balance to machine precision) but discards
directional momentum, which is exactly the absorption mode the audit
diagnosed.

## Approach options

Two textbook approaches, both nontrivial. Both replace the current
stagnation-CV model with something that preserves more of the
incident wave amplitude.

### Option E.1 — Multi-pipe characteristic Riemann junction

Reference: Winterbone & Pearson, *Theory of Engine Manifold Design*
(2000), §9.2.3 *"Characteristic-based junction model"*. Also
described in Corberán & Gascón (1995, IJTS), Pearson & Winterbone
(1996, JSV). Same idea different formulation in Bauer-Habermann
(2018) and the Chalet papers.

Idea: at the junction face, each incident pipe contributes its
**outgoing** Riemann invariants (J⁻ for pipes whose junction-end
flow is subsonic outflow, J⁺ for inflow). A 0-D pressure-and-entropy
state lives at the junction; mass and energy conservation across the
N legs give N − 1 algebraic constraints. Solve the (N+1)-equation
nonlinear system at each step for (p_junction, mdot_per_leg) given
the outgoing characteristic invariants from each pipe. Each leg's
ghost cell is then back-constructed from (p_junction, leg-specific
incoming characteristic = mdot · A / ρ).

Pros:
- Preserves acoustic information in the outgoing characteristics
  (those are the invariants of the wave equation; not absorbed).
- Conservative by construction (mass + energy across the N legs).
- Standard in the engine-1D literature; well-documented and well-
  validated by other 1D codes (GT-Power and WAVE both use variants).
- Drop-in for the existing `JunctionCV` interface (`fill_ghosts` +
  `absorb_fluxes` semantics preserved; the implementation behind
  changes).

Cons:
- Nonlinear solve per step per junction (Newton or fixed-point,
  ~5–10 iterations to convergence at each of the 3 SDM26 junctions).
  Bisection is too slow in 1D for an N-D system.
- Sensitive to the choice of momentum equation in the multi-pipe
  formulation. Several variants exist — Pearson's "stagnation
  pressure equality" works for matched-area junctions; Corberán's
  "loss-coefficient" treatment handles area-mismatched junctions
  better but introduces an empirical loss factor.
- Variable γ across legs (cyl-side vs pipe-side burned/unburned gas)
  needs careful handling at multi-pipe junctions where the legs
  have different gas properties.

### Option E.2 — N-leg coupled face-state solver

Same physics conclusion, different solver structure: instead of
maintaining a 0-D "junction state" object that stores (p, T, Y) and
gets updated, solve a multi-pipe Riemann problem at each junction
face in the same way HLLC handles a 2-pipe Riemann problem at a
cell face. Each leg's face state and the junction-shared face state
are computed simultaneously by solving the multi-leg wave structure.

Pros:
- More aligned with the rest of the V2 architecture (everything
  else is face-flux-based, no separate "0-D objects").
- No separate "junction state" to track or test for conservation —
  the conservation is in the face fluxes, same as cells.
- Easier to extend to non-stagnant junctions (e.g. distributed-loss
  manifolds) later.

Cons:
- More invasive surgery in `bcs/junction_cv.py` AND in the engine
  model's `step()` method (junction.fill_ghosts is currently called
  separately from the per-pipe MUSCL step; an N-leg face solver
  would need to integrate the pipe interior states with the
  multi-pipe Riemann solve).
- Less literature precedent; the engine-1D codes (GT-Power, WAVE,
  AVL Boost) all use Option E.1 or close variants.

**Recommended starting point: Option E.1 with Pearson's stagnation-
pressure-equality formulation.** Match what the engine-1D community
does, get the easy case working, evaluate against A3, decide whether
loss-coefficient extension is needed before declaring done.

## Acceptance criteria

Engine-level (the bar that matters for the team):

- **A3 round-trip R linear ≥ 0.5** (per-junction transmission ≥ 0.84;
  round-trip survival 0.84⁴ ≈ 0.50).
- **SDM26 sweep shows differentiated curve shape from SDM25**, not
  just an offset. Quantitatively: shape-diff score (the cosine-
  distance metric in `diagnostics/c3_report.py:shape_diff_score`)
  > 0.05 for either VE or wheel torque.
- **At least one configuration shows a torque peak above 6000 RPM**.
  Real CBR600RR torque peaks somewhere in the 8000–11000 range.
- **VE has a visible local maximum or wiggle** in at least one
  configuration in the 7000–13000 RPM window.

Regression / sanity (preserving what already works):

- All 102 tests still green (95 existing + 7 acoustic).
- 10500 SDM26 single-point: EGT in [1000, 1400] K, IMEP within ±5 %
  of the C2 result of 10.21 bar, mass nonconservation at machine
  precision.
- A1 acoustic: R_valve ≈ −0.7 unchanged (this is the valve BC, not
  the junction).
- A2 acoustic: R_plenum ≈ −0.98 unchanged (this is the plenum BC,
  not the junction).
- 16-point sweep wall clock within 2× of C3 (currently 100 s SDM26;
  budget up to 200 s for the multi-pipe Riemann solver overhead).

## Known tradeoffs

The Winterbone characteristic-junction is **harder to make strictly
conservative than the stagnation CV**. The stagnation-CV approach
sums face fluxes into the CV state and the CV state is a control
volume by definition — mass + energy balance algebraically. The
characteristic-junction has no such CV; conservation has to be
imposed as one of the algebraic constraints in the nonlinear solve,
and roundoff errors in the iteration can introduce small (O(1e-10))
drifts per junction per step. This is known and documented in the
literature; the standard fix is a "conservation correction" pass
that nudges the converged junction state to enforce strict mass +
energy balance after each iteration.

This is acceptable as long as the conservation-correction keeps the
overall sweep nonconservation below ~1e-10 kg/cycle (still 8 orders
of magnitude better than V1's MOC). If we can't hit that we may need
to fall back to a hybrid approach that uses the stagnation CV for
mass bookkeeping and the characteristic junction only for ghost-cell
fill (decoupling acoustics from conservation accounting).

## Test infrastructure already in place

- `tests/acoustic/test_a3_sdm26_manifold.py` is the single most
  important test for Phase E. It directly measures round-trip
  manifold reflection, which is the integrated effect of the
  junctions plus the collector open-end.
- `tests/test_junction_cv.py` (existing) is the conservation
  regression — must stay green after the rewrite.
- `bcs.valve.enable_regime_logging()` is the diagnostic for
  per-call BC behavior; if the junction restructure changes regime
  distributions in surprising ways, this will catch it.

## Estimated scope

Comparable to Phase C1 + C2 combined (~2 weeks of focused work from
either of us). The math is well-documented; the integration with the
existing FV time-stepping is the part that takes time.

## Next steps when picked up

1. Read Winterbone § 9.2.3 carefully; pick the specific formulation
   variant (Pearson stagnation-pressure-equality vs Corberán loss-
   coefficient).
2. Sketch the nonlinear solve on paper for a 3-leg junction (4-2
   merge in the SDM26 manifold). Verify the conservation algebra.
3. Implement in `bcs/junction_cv.py` alongside the existing
   `JunctionCV` (don't replace yet; keep the old class for back-
   compat with the test suite).
4. Wire into a single junction in `models/sdm26.py` (the 2-1
   junction first — fewer legs = simpler) for incremental
   validation.
5. Run A3 acoustic test against the partial fix; iterate on the
   junction nonlinear solve until A3 round-trip R approaches the
   acceptance bar.
6. Wire into all three junctions; rerun A3 + 10500 single point.
7. Full 16-point sweep; compare against the C3 baseline.
8. Phase E verification report at `docs/phase_e_verification.md`.
