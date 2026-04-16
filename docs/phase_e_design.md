# Phase E1 — Characteristic-coupled junction: design document

**Status:** draft for review. No implementation code written yet.
**Branch:** `phase-e/junction-coupling` (off main at `c3-complete`).
**Supersedes the "Next steps" list in `docs/phase_e_plan.md`.**

---

## 0. Starting position

Two junction implementations exist in the codebase today:

1. **`bcs/junction_cv.py`** — `JunctionCV` (stagnation control volume).
   Currently wired into both SDM25 and SDM26. Sums face fluxes into a 0-D
   reservoir; discards incident momentum; absorbs kinetic energy as
   stagnation enthalpy. Exactly conservative in mass + energy + ρY by
   construction (it is a control volume). Known to attenuate acoustic
   waves: A3 round-trip gives per-junction transmission ≈ 0.69.

2. **`bcs/junction.py`** — `apply_junction` (constant-static-pressure
   characteristic-coupled draft, from Phase 3 WIP, commit `2704bc0`).
   **Never wired into any model, no tests, dormant since Phase 3.**
   Performs Newton-on-p_j with isentropic expansion from each interior
   to the junction pressure. Mass conservation imposed via the iteration
   residual. No choked-leg handling, no energy-conservation diagnostic,
   FD derivative (not analytic). Does *not* solve the full Phase-E
   requirements but the core formulation is sound and this is the
   natural base to build from.

Phase E is **not** starting from scratch. The design below takes
`bcs/junction.py` as a reference point, identifies its gaps against the
Phase E acceptance criteria, and specifies the additions and corrections
needed to lift it into the production `CharacteristicJunction`.

The architectural call from the plan — *parallel alternative, not
in-place replacement* — means the old `JunctionCV` stays in
`bcs/junction_cv.py` untouched. The new code goes into a new file
(`bcs/junction_characteristic.py`), exposing a `CharacteristicJunction`
class with the same life-cycle hooks (`fill_ghosts`, `absorb_fluxes`)
as `JunctionCV` so engine models can swap between them via a
constructor parameter or factory.

---

## 1. Which Winterbone variant

**Recommendation: constant-static-pressure junction, mass + energy
conservation across legs, characteristic compatibility per leg.**

This matches Winterbone & Pearson's "Type 1" multi-pipe junction (the
formulation closest to `bcs/junction.py`'s existing draft) and is the
formulation Corberán & Gascón 1995 showed to be exactly conservative in
mass and approximately so in energy.

Alternatives considered and rejected:

- **Constant-stagnation-pressure (Pearson).** Used for matched-area
  manifolds (4-1 with equal primaries meeting a single collector of
  matched total area). Fails when primaries and secondaries have
  significantly mismatched areas, which is the SDM26 case (4×32 mm into
  2×~42 mm is area-mismatched by ~40 %). Documented to over-recover
  kinetic energy at the merge and therefore overshoot transmission.
  Rejected.

- **Loss-coefficient junction (Corberán 1995).** Adds an empirical
  loss factor based on area ratios. More accurate for strongly
  mismatched junctions but introduces a tunable parameter, which
  violates the project rule against calibration against dyno data or
  expected behaviour. Left as a *post-E extension* if constant-static
  falls short at acceptance.

- **0-D CV with characteristic ghost (hybrid).** Keep the JunctionCV
  mass/energy bookkeeping but fill ghosts using characteristic
  compatibility with the CV state. The problem is that the CV state is
  a stagnation reservoir, not a pressure wave, so the characteristic
  fill reduces to the existing behaviour. Rejected as no real
  improvement over current.

**Uncertainty flagged.** I do not have Winterbone & Pearson on disk.
The formulation below is the constant-static-pressure variant I know
from primary-source papers (Corberán 1995 IJTS, Pearson 1994 PhD
thesis, and the draft in `bcs/junction.py` which cites Winterbone
§9.2.3). Section numbers attributed to Winterbone in this document
should be treated as *likely-correct-but-not-verified*. If the
approach proves unsound for the SDM26 geometry at E2 unit-test time,
we stop and consult the book.

---

## 2. Governing equations

The junction connects N incident pipes. Each pipe i has a face at the
junction end with area A_i. Define sign σ_i such that σ_i > 0 when
positive velocity at the face means flow *into* the junction.

Interior state of pipe i, at the first real cell adjacent to the
junction face: (ρ_i, u_i, p_i, Y_i), with sound speed c_i = √(γ p_i / ρ_i)
and specific entropy s_i = p_i / ρ_i^γ (defined up to a constant).

**Unknowns:** p_j (scalar junction static pressure), and for each leg
the face state (ρ_f,i, u_f,i). All expressed as functions of the single
scalar p_j via the closure equations below, so the only true unknown
is p_j.

### 2a. Closure: characteristic compatibility (per leg)

For each leg, the outgoing Riemann invariant along the characteristic
that carries information *from* the pipe interior *toward* the junction
face is conserved between the interior and the face:

- RIGHT-end junctions (pipe is upstream of junction): outgoing
  invariant is J⁺ = u + 2c/(γ−1).
- LEFT-end junctions (pipe is downstream of junction): outgoing
  invariant is J⁻ = u − 2c/(γ−1).

Entropy is carried along the u-characteristic. For *subsonic outflow*
(interior → face → junction) the entropy at the face equals the
interior entropy:

    s_f,i = s_i     (outflow from leg to junction)

For *subsonic inflow* (junction → face → interior), the entropy at the
face is set by the junction-side reservoir (mass-weighted mix of
upstream-leg entropies, computed after the iteration converges and
mass directions are known):

    s_f,i = s_junction_mixed     (inflow from junction to leg)

Combined with the characteristic equation and p_f,i = p_j:

    Outflow case (s_f,i = s_i):
        c_f,i = c_i · (p_j / p_i)^((γ−1)/(2γ))    [isentropic in the leg]
        u_f,i = u_i + (2/(γ−1)) · (c_i − c_f,i) · s_end
        ρ_f,i = ρ_i · (p_j / p_i)^(1/γ)
    where s_end = +1 at RIGHT-end, −1 at LEFT-end.

    Inflow case: same form but with (ρ_i, p_i, c_i) replaced by the
        junction-mixed reservoir state. First pass, use the interior
        values (lagged-Picard on Y_mixed); after mass directions
        converge, relax Y_mixed and re-solve.

This is equivalent to what `bcs/junction.py` does today, with the
correction that the inflow case should use the junction reservoir's
entropy rather than the leg's interior entropy. Current draft silently
uses the interior entropy for inflow legs, which over-predicts density
recovery at inflow legs. **This is fix #1 for Phase E2.**

### 2b. Constraint: mass conservation

    Σ_i σ_i · ρ_f,i(p_j) · u_f,i(p_j) · A_i = 0

One scalar nonlinear equation in one scalar unknown p_j. Solve by
Newton. The analytic derivative dR/dp_j (where R is the mass residual)
can be computed in closed form from the chain rule applied to the
isentropic closure; I'll write that out in E2 when I implement, not
here.

### 2c. Constraint: energy conservation (diagnostic, not algebraic)

Once p_j is found from mass conservation, compute

    Ė_net = Σ_i σ_i · ρ_f,i · u_f,i · A_i · h0_f,i

where h0 = h + ½u² and h = γ/(γ−1) · p/ρ. This should be ≈ 0 for a
constant-static-pressure junction if the entropy choice in 2a is
consistent. It will not be exactly zero at subsonic-multi-leg because
the static-pressure constraint does not guarantee static-enthalpy
continuity; the residual is the "junction loss" term and is of order
½ρu² · ΔA/A̅ for area-mismatched junctions.

**Design decision:** treat Ė_net as a diagnostic, report it in the
regime log, and fail the unit tests if it exceeds a threshold
(proposed: 1 % of the maximum |σ_i ρ_f u_f h0 A_i| term). This keeps
the formulation non-dissipative for matched-area junctions (where it
will be machine-zero) and lets us quantify the loss for mismatched
ones. If the diagnostic fires above 1 % on realistic SDM26 runs, that
is the trigger to promote to the Corberán loss-coefficient variant.

### 2d. Composition transport

For legs carrying mass *into* the junction (σ_i u_f,i > 0 at the
face), Y is carried from the interior to the junction reservoir.
Mass-weighted mixing gives:

    Y_mixed = Σ_{inflow} (σ_i ρ_f,i u_f,i A_i) · Y_i  /  Σ_{inflow} (σ_i ρ_f,i u_f,i A_i)

For legs carrying mass *out* of the junction (into the leg), Y_f,i =
Y_mixed. This is already correct in `bcs/junction.py` and stays.

---

## 3. Solution strategy

**Newton on p_j, with analytic Jacobian, damping, and bracket
protection.**

```
    p_j^(0) = Σ A_i p_i / Σ A_i                      # area-weighted interior mean
    for k = 0..K_max:
        R, dR_dpj = residual_and_jacobian(p_j^(k))
        if |R| < tol: break
        Δp = -R / dR_dpj
        Δp = clamp(Δp, -0.2 · p_j^(k), +0.2 · p_j^(k))    # 20 % step cap
        p_j^(k+1) = max(p_j^(k) + Δp, p_floor)
    solve Y_mixed from converged mass directions
    relax s_f,i at inflow legs with s_mixed, re-solve p_j (one extra pass)
    fill ghost cells
    return (p_j, iter_count, |R|_final)
```

Expected iteration count: 3–8 Newton steps per junction per time step,
based on what `bcs/junction.py` typically converges in (measured during
Phase 3 WIP before it was retired). p_floor = 1000 Pa; solutions below
that indicate cavitation or iteration runaway and the model does not
support them.

**Startup robustness.** If R does not converge in K_max = 30 steps,
raise an explicit `JunctionConvergenceError` with the leg states
logged. Do not silently fall back. The plan rule is "stop and report
if anything surprises you" — non-convergence is a surprise.

**Performance.** The inner `residual_and_jacobian` is pure arithmetic
on N ≤ 5 legs. Numba `@njit` after correctness is validated. 3
junctions × 8 Newton steps × ~100k time steps per cycle × 25 cycles
per sweep point × 16 points = ~10⁹ Newton evaluations per full sweep.
Budget concern is real — expect 1.5–2× slowdown vs JunctionCV before
Numba, 1.0–1.2× after. Acceptance budget from plan is 2× so we have
margin.

---

## 4. Choked-leg handling

The standard characteristic closure fails if the face flow in any leg
reaches M = 1 at the junction face. At M = 1, the outgoing-information
characteristic coincides with the u-characteristic and the system loses
one degree of freedom per choked leg.

**Physical case where this matters:** exhaust primary during blowdown.
Cylinder pressure can peak above 20 bar right after exhaust-valve
opening; the primary sees a choked flow at its upstream (cylinder)
end, and the wave propagating down the primary can still be subsonic
at the junction end. But if a strong second wave from a neighbouring
primary's blowdown encounters the junction during collapse, the
junction face itself can flirt with choke. A3 linear-amplitude case
(±0.5 % P_atm perturbation) will not hit this. A3 nominal-amplitude
(±10 % P_atm) and the engine-model blowdown case may.

**Detection.** During Newton iteration, compute M_f,i = u_f,i / c_f,i
for each leg. If |M_f,i| > 1 − ε (proposed ε = 0.02), flag the leg as
"near-choke". If |M_f,i| > 1 − ε persists at the converged p_j for any
leg, reject the solution and invoke the choked branch.

**Choked branch.** For any leg i flagged as choked:

- The mass flux at the face is set from the leg's interior stagnation
  state alone (not a function of p_j):
      ρ_f,i u_f,i = ρ0_i · c0_i · ((γ+1)/2)^(-(γ+1)/(2(γ−1)))
  Sign is determined by σ_i and the blowdown direction (known from
  u_i).
- The face static pressure in the choked leg is *not* p_j; it is the
  sonic throat pressure p_* = p0_i · (2/(γ+1))^(γ/(γ−1)).
- Solve the reduced system: p_j is determined by mass conservation
  across the non-choked legs, with the choked-leg contribution fixed.
- Ghost fill for choked legs uses the sonic-throat state, not p_j.

This is the same three-regime dispatch structure C1 used for the valve
BC: an explicit regime classifier at the top of the junction step that
picks between normal-subsonic (Newton on p_j) and choked (reduced
system with one or more legs clamped at sonic). No catch-all fallback.

**Multiple simultaneously choked legs.** Treated the same way: fix
each choked leg's mass flux at its own sonic value, solve for p_j from
the remaining subsonic legs. If *all* legs are choked, the junction is
mass-imbalanced for that step and we have a hard physical
contradiction — raise `JunctionAllChokedError` and stop.

---

## 5. Conservation proof sketch

**Mass.** Exact by construction: the iteration converges on
Σ σ_i ρ_f u_f A_i = 0. At converged state, mass fluxes across the N
legs sum to zero. Ghost cells are filled with (ρ_f,i, u_f,i), so the
HLLC-reconstructed face flux in each leg matches the iteration-imposed
value to O(Δx²) (MUSCL-Hancock second order). The residual is the
reconstruction discrepancy, empirically ~10⁻¹² relative on matched-area
junctions in the existing draft.

**Energy.** Approximate. The constant-static-pressure constraint does
not guarantee energy continuity at mismatched-area junctions. Expected
residual: O(½ ρ u² · ΔA/A̅) per time step, which for SDM26 primaries
into secondaries with 40 % area mismatch and u ≈ 30 m/s, ρ ≈ 0.4
kg/m³, gives ~72 J/m³ × pipe volume ~10⁻⁵ m³ × 10⁴ steps/cycle = 0.007
J/cycle. Fuel energy per cycle at idle ~100 J/cycle. So the junction
energy non-conservation should be 4–5 orders of magnitude below fuel
energy per cycle. Acceptable.

**Composition (ρY).** Exact when mass is exact, because ρY is
transported with the mass flux and the mixed-Y construction is itself
mass-weighted.

**Diagnostic.** `CharacteristicJunction` exposes:
- `last_mass_residual`  — scalar, Σ σ_i ρ_f u_f A_i, should be < 1e-12
- `last_energy_residual` — scalar, Σ σ_i ρ_f u_f A_i h0_f,i
- `last_niter` — Newton iteration count, for performance tracking
- `last_regime` — "subsonic" | "choked_N" where N is count of choked legs

These land in the C3/E4 sweep log so we can catch regressions.

---

## 6. Interface specification

```python
class CharacteristicJunction:
    """Constant-static-pressure characteristic-coupled N-pipe junction.

    Alternative to JunctionCV for higher acoustic transmission.
    Mass-conservative exactly; energy-conservative to ~ΔA/A order.
    Use where wave survival through the junction matters.
    """

    def __init__(
        self,
        legs: List[JunctionLeg],
        *,
        gamma: float = 1.4,
        R_gas: float = 287.0,
        newton_tol: float = 1e-9,
        newton_max_iter: int = 30,
        choke_margin: float = 0.02,       # |M| < 1 - margin → subsonic
    ) -> None: ...

    # --- life-cycle hooks, matches JunctionCV signature ---

    def fill_ghosts(self) -> None:
        """Called BEFORE each pipe MUSCL step. Solves Newton for p_j,
        writes ghost cells on every leg."""

    def absorb_fluxes(self, dt: float) -> None:
        """Called AFTER each pipe MUSCL step. No-op for the
        characteristic junction (mass/energy accounting happens via
        face fluxes directly, no separate CV state to update). Kept
        for interface symmetry so models can swap JunctionCV <->
        CharacteristicJunction without other changes."""

    # --- diagnostics ---

    @property
    def last_p_junction(self) -> float: ...
    @property
    def last_mass_residual(self) -> float: ...
    @property
    def last_energy_residual(self) -> float: ...
    @property
    def last_niter(self) -> int: ...
    @property
    def last_regime(self) -> str: ...
```

**Leg struct reuse.** `JunctionLeg` (from `bcs/junction.py`) has the
right shape (pipe + end + sign). Either reuse it verbatim or move it
to a shared `bcs/_junction_common.py`. Decision deferred to E2 impl.

**Factory for model wiring.** `models/sdm26.py` currently does:

```python
self.j_exh1 = JunctionCV.from_legs([...], ...)
```

After E3, model constructors take a junction type parameter:

```python
SDM26(..., junction_type: str = "stagnation")  # "stagnation" | "characteristic"
```

and dispatch internally. Default remains "stagnation" until E4
acceptance passes, then switches to "characteristic" as the new
default. JunctionCV never removed.

---

## 7. What breaks in the existing `bcs/junction.py` that E2 must fix

1. **Inflow entropy bug.** Inflow legs currently use interior entropy
   instead of junction-mixed entropy. Corrected in E2.
2. **No choked handling.** Add three-regime dispatch like valve C1.
3. **FD derivative.** Replace with analytic Jacobian (closed-form from
   isentropic closure chain rule).
4. **No tests.** Add 8 from plan.
5. **No energy diagnostic.** Add the residual tracking from §2c.
6. **No convergence error.** Current code silently bails after
   max_iter. Replace with explicit raise.
7. **Damping is ad-hoc ±20 %.** Keep (standard Newton damping for
   nonlinear algebraic systems) but document the rationale.
8. **No Numba.** Add `@njit` on the hot inner loop after correctness
   proven.

Items 1–3 are physics. 4–8 are engineering.

---

## 8. Out of scope for Phase E

- **Distributed-loss junctions** (non-zero pressure-loss between legs
  due to turbulent mixing). Would require empirical loss coefficients,
  violates no-tuning rule. Post-E if measured loss is excessive.
- **Variable γ across legs.** Exhaust side is always burned gas at
  γ ≈ 1.33; currently treated as uniform γ = 1.4. This is a known
  issue from Phase 1 audit and is scheduled for a separate phase
  (not E).
- **Non-ideal gas EOS.** Same phase as variable γ.
- **Turbulent mixing kinetics.** Instantaneous mass-weighted mixing is
  the 1D industry standard.

---

## 9. Phase E1 deliverable (this document) — questions for review

Before proceeding to E2 implementation, the following require
confirmation:

**Q1 — Constant-static-pressure formulation OK?** Alternative would
be Pearson's constant-stagnation-pressure, which is simpler but
inappropriate for the area-mismatched SDM26 case. My recommendation
is constant-static. Confirm?

**Q2 — Parallel file (`bcs/junction_characteristic.py`), not
in-place rewrite of `bcs/junction.py`?** The plan says parallel. The
existing `bcs/junction.py` is effectively a Phase-3 WIP that was
abandoned; it has no consumers. Options:
  (a) Leave `bcs/junction.py` alone, add new `junction_characteristic.py`.
  (b) Delete `bcs/junction.py` (unused), put new code in its name.
  (c) Refactor `bcs/junction.py` in place into the new class.
My recommendation is (a) — cleanest; old file is harmless but I won't
be extending it. Confirm?

**Q3 — Energy residual as diagnostic only, not algebraic constraint?**
The formulation (§2c) reports energy residual as a number but does
not enforce it. Acceptance-criterion violation if it exceeds 1 % of
max leg enthalpy flux. This is a conscious deviation from strict
conservation. Confirm OK?

**Q4 — Choked-leg dispatch modeled on valve C1's three-regime
pattern.** Explicit regime classifier, no catch-all fallback. Confirm
OK?

**Q5 — Winterbone sectioning is uncertain.** I do not have the book.
If the implementation produces surprises at E2 unit-test level, I
stop and flag rather than guess. Confirm OK?

**Q6 — Default junction type.** Stays `"stagnation"` until E4
acceptance passes, then switches to `"characteristic"`. This means
any external consumer of SDM25/SDM26 constructed without
specifying the junction type will silently get the new behaviour
after E4 ships. Alternative: keep stagnation as default forever,
make characteristic opt-in. Confirm which?

---

## 10. After this is approved

Proceed to **E2: implementation + 8 unit tests** (branch already
created). Commit after tests pass, before engine wiring. Then E3
single-point, then E4 sweep + report.

No code until these questions are answered.

---

## Addendum — dormant draft sanity check (2026-04-14)

Per the reviewer's request, before starting E2 implementation, ran
`bcs/junction.py:apply_junction` (dormant, untested, Phase-3 WIP)
through the full A3 4-2-1 manifold harness at both amplitudes. Script:
`diagnostics/phase_e_draft_sanity.py`. No modifications to the draft
itself — it was run exactly as it has sat on disk since commit
`2704bc0`.

**Result.**

| amplitude        | draft `apply_junction` | JunctionCV (C3 baseline) |
|------------------|------------------------|--------------------------|
| linear +5 kPa    | **R_rt = +0.698**      | R_rt = +0.228            |
| nominal 5 bar    | R_rt = −0.042          | R_rt = +0.011            |

Linear regime: draft gives per-junction transmission ≈ 0.698^(1/4) =
**0.913**, 32 % above the Phase-E acceptance bar of 0.84
(0.84⁴ ≈ 0.50). The underlying constant-static-pressure
characteristic closure is fundamentally working for small-amplitude
acoustic propagation.

Nominal regime: draft fails, and fails *worse* than the stagnation
CV — including flipping the sign of A2 (returned wave is inverted
and weak). This is almost certainly the missing choked-leg
handling. A 5-bar blowdown drives the primary face well above
M = 1, the Newton iteration is not set up to recognise this, and the
iteration either diverges, hits the 20 % damping wall indefinitely,
or converges to a p_j that is inconsistent with the actual wave
physics. Supports fix #2 in §7 (choked-leg three-regime dispatch) as
the highest-priority fix for E2.

**Verdict: draft is close to correct in the regime Phase-E acceptance
measures, E2 is polish + choked fix, not genuine from-scratch.** This
does not change the implementation plan — all 8 fixes in §7 are still
needed, all 8 unit tests are still required, the new file
`bcs/junction_characteristic.py` is still the delivery — but it
raises confidence that the formulation is correct and the acceptance
bar is realistically achievable.

The 8 enumerated fixes now have a priority ordering, based on which
ones are likely responsible for the nominal-regime failure:

1. **(Highest)** Choked-leg three-regime dispatch — likely root
   cause of the 5-bar failure.
2. Inflow entropy correction — may matter more at nominal amplitude
   where density recovery at inflow legs is larger.
3. Explicit convergence error raise — because the draft may be
   silently hitting max_iter at 5-bar and returning stale state.
4. Analytic Jacobian — correctness now, performance later.
5. Energy diagnostic — needed for acceptance but doesn't affect
   solver correctness.
6–8. Tests, Numba, documentation — engineering work.

Draft plots saved to
`docs/acoustic_diagnosis/phase_e_draft_{linear_5kPa,nominal_5bar}_P0_probes.png`
for visual comparison with the C3 baseline plots at
`docs/acoustic_diagnosis/a3_{linear_5kPa,nominal_5bar}_P0_probes.png`.
