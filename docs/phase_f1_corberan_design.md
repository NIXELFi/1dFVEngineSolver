# Phase F1a — Corberán loss-coefficient junction: design document

## Status: REVERTED (2026-04-15)

**The Corberán reformulation was implemented correctly but does not
serve the intended calibration purpose.** Constant-stagnation-pressure
(Corberán) produces fundamentally different VE from constant-static-
pressure (Phase E CSP) at engine-relevant Mach numbers. Measured:
VE = 58% (Corberán at K≈0) vs 83% (CSP) at 10500 RPM SDM26. The
25-point drop is the inherent difference between the two formulations,
not a K-value effect — it accumulates as O(M²) per junction crossing
compounded over ~10 wave transits × 3 junctions per cycle.

Additionally, the K coefficient in the Corberán formulation acts as a
stagnation-constraint relaxation (higher K → more mass throughput →
higher VE), not as a wave-attenuation knob. This is the opposite of
what the calibration required (reduce spurious tuning peaks).

**Decision:** revert to Phase E CSP junction. Accept the inviscid
91% per-junction transmission as a documented limitation. Proceed
with F2/F3/F4 (Levine-Schwinger, restrictor Cd, Wiebe ramp) which
address the other three identified sim-vs-dyno error sources.

**Future work (Phase G):** a post-solve characteristic-space damping
applied to the outgoing Riemann invariant at the junction face,
with the HLLC flux recomputed against the modified ghost state and
the Newton residual defined in terms of the post-damping HLLC flux
for conservation consistency. Concept documented in the project
planning notes. Estimated 100-150 lines when the math is worked out.

**Original design document preserved below for reference.**

---

**Original status:** draft for review. No implementation code written. F1 attempts
reverted to the Phase-E checkpoint.
**Branch:** `phase-f/calibration` (off main @ `c3-complete`, containing the
committed E2b + E4 + dense-sweep + PDF-report artifacts).
**Supersedes the original "one-scalar loss coefficient" plan of F1.**

---

## 0. Why this exists and what failed before

The original Phase F1 plan specified a scalar loss coefficient α that would
multiply the transmitted wave amplitude at each junction crossing. Three
implementation attempts (damp u perturbation in `_face_from_pj`, damp the
p_j deviation, damp the ghost-state perturbation relative to interior) all
produced the same failure mode: the A3 round-trip reflection coefficient
barely responded to α because Newton's mass-balance iteration compensates
for the damping by adjusting p_j. In the constant-static-pressure
formulation, mass conservation is the *only* constraint, so any scalar
damping gets redistributed rather than dissipated.

The physically correct fix is to introduce a *stagnation pressure drop*
across the junction face, which dissipates acoustic energy. This breaks the
constant-static-pressure assumption and requires reformulating the junction
as Corberán & Gascón's (1995) loss-coefficient variant of the characteristic
junction.

This is not a one-scalar addition. It is a reformulation of the junction
mathematics. Estimated 200–300 lines of change plus extensive testing.

---

## 1. Mathematical formulation

The Corberán junction replaces the constant-static-pressure condition
`p_face,i = p_junction ∀ i` with a **constant-stagnation-pressure**
condition coupled to a per-leg loss term. The five relations per junction
are:

### 1a. Mass conservation (unchanged)

```
Σ σ_i · ρ_face,i · u_face,i · A_i = 0              (1)
```

summed over all legs. σ_i is the into-junction sign (+1 for RIGHT-end
legs, −1 for LEFT-end legs). This is the same global constraint as the
constant-static-pressure formulation.

### 1b. Per-leg stagnation pressure drop (new)

For each leg i:

```
p_0,face,i = p_0,junction + σ_i · u_face,i / |u_face,i| · K_i · ½ρ_face,i · u_face,i²     (2)
          = p_0,junction + sign(σ_i · u_face,i) · K_i · ½ρ_face,i · u_face,i²
```

where:
- `p_0,junction` is the **single shared junction stagnation pressure**
  (replaces `p_junction` as the iteration variable; this is the scalar
  the outer solve iterates on).
- `p_0,face,i = p_face,i + ½ρ_face,i · u_face,i²` is the face stagnation
  pressure on leg i.
- `K_i` is the **loss coefficient** for leg i, nondimensional, typically
  in [0.05, 0.6] depending on geometry and flow direction.
- The sign convention: when fluid flows *into* the junction on leg i
  (σ_i · u_face,i > 0), the stagnation pressure drops from the face to the
  junction reservoir (energy dissipated entering the merge). When fluid
  flows *out* of the junction (σ_i · u_face,i < 0), the junction reservoir
  loses stagnation pressure relative to the face (energy dissipated leaving
  the merge).

**Uncertainty flag.** The exact sign convention in equation (2) is my
recollection from Corberán & Gascón 1995 and standard loss-coefficient
literature. I am confident the magnitude relation is right; the sign
convention may need swapping after cross-checking against the paper.

### 1c. Per-leg characteristic compatibility (unchanged)

For each leg i, the outgoing Riemann invariant along the interior-to-face
characteristic is conserved:

```
u_face,i = u_interior,i + (2/(γ-1)) · (c_interior,i - c_face,i) · s_end,i    (3a)
ρ_face,i = ρ_interior,i · (p_face,i / p_interior,i)^(1/γ)                    (3b)
c_face,i = c_interior,i · (p_face,i / p_interior,i)^((γ-1)/(2γ))             (3c)
```

where `s_end,i` is +1 at RIGHT-end legs (J⁺ incoming) and −1 at LEFT-end
(J⁻ incoming). These relations connect the face primitives to the pipe
interior state via isentropic expansion from `p_interior,i` to
`p_face,i`. **Note that `p_face,i` now differs per leg** — this is what
changes from the constant-static-pressure formulation.

### 1d. Closure

Per leg we have 4 unknowns: (ρ_face,i, u_face,i, p_face,i, c_face,i).
We have 4 relations: (3a), (3b), (3c), and the stagnation-pressure
definition `p_0,face,i = p_face,i + ½ρ_face,i · u_face,i²` combined with
(2). This is a 4-equation-4-unknown nonlinear system in the face
primitives of leg i, parameterized by the scalar `p_0,junction`.

The global constraint (1) closes the scalar unknown `p_0,junction`.

### 1e. Counting

- Global unknown: `p_0,junction` (1 scalar)
- Per-leg unknowns: 4 (ρ, u, p, c at face)
- Global constraints: 1 (mass balance, eq 1)
- Per-leg constraints: 4 (eqs 2, 3a, 3b, 3c)

For an N-leg junction: (1 + 4N) unknowns, (1 + 4N) equations. Well-posed.

### 1f. Backward compatibility — K=0 fast path

When K_i = 0 for all legs, eq (2) reduces to `p_0,face,i = p_0,junction`.
Combined with the stagnation-pressure definition and the characteristic
relations, this Corberán reduction is *approximately but not identically*
equal to the Phase-E constant-static-pressure formulation. They differ
by O(M²) ≈ 1–5% at typical engine exhaust Mach numbers.

**Design decision (revised per review):** use a **K=0 fast path**. The
constructor detects `all(K_i == 0)` once (K values are compile-time
constants, not dynamic) and selects the code path accordingly:

```
if any(K > 0):
    Corberán outer-inner (constant p_0,junction)
else:
    Phase-E constant-static-p (single secant on p_j)
```

Both paths share the characteristic compatibility relations (eqs 3a-c)
and the HLLC-consistent MUSCL-aware Newton residual. Only the top-level
solver branch differs. Estimated additional branching logic: 30–50 lines
in `fill_ghosts()`, not 200+ lines of duplicated formulation.

**Benefit.** Exact bit-for-bit equality with Phase E at K=0. Unit test 10
(K=0 reduction) can assert exact equality rather than approximate. Future
regression debugging is cleaner — a K=0 run either matches Phase E
exactly (formulation correct) or does not (bug to investigate), with no
approximate-equality judgment call.

---

## 2. Loss coefficient values (starting points)

Winterbone & Pearson 1999 chapter 9 tabulates K for various merge
geometries. **I do not have the book on disk and do not recall specific
table numbers reliably.** I will flag every K value below as "starting
estimate from standard-literature range" and refine in F1d against SDM25
dyno.

### 2a. Typical published ranges (from memory + general 1D literature)

- Sharp-edged merging flow (primary → secondary at a 4-2 junction):
  K_incoming ≈ 0.3 – 0.5
- Sharp-edged diverging flow (secondary → primary during return wave):
  K_outgoing ≈ 0.05 – 0.15
- Straight-through flow (in-line connection, not at a merge):
  K ≈ 0 (no loss)
- Right-angle take-off (side branch): higher K, up to 0.8–1.0

The asymmetry K_in >> K_out captures the physical reality that
contraction+mixing into a merge dissipates much more energy than
expansion out of one.

### 2b. Starting values for Phase F1c

For SDM26 (two-stage 4-2-1 manifold):
- **4→2 junction #1** (P0, P3 → S0):
  K_incoming = [0.4, 0.4, 0.4, 0.4]  (only two actually incoming at any
  time, but we list all four primaries because the junction has 4 incident
  legs)
  
  Actually no — this junction has only 3 legs: P0 (in), P3 (in),
  S0 (out). K_incoming = [0.4, 0.4], K_outgoing = [0.1].
- **4→2 junction #2** (P1, P2 → S1): same geometry as #1.
  K_incoming = [0.4, 0.4], K_outgoing = [0.1].
- **2→1 junction** (S0, S1 → C): similar geometry.
  K_incoming = [0.4, 0.4], K_outgoing = [0.1].

For SDM25 (one-stage 4-1 manifold):
- **4→1 junction** (P0, P1, P2, P3 → C): wider merge angle and sharper
  contraction than SDM26's 4→2 merge → more turbulent dissipation per
  the published K trends. Starting K_in = **0.5** (vs 0.4 for SDM26).
  K_incoming = [0.5, 0.5, 0.5, 0.5], K_outgoing = [0.1].

The K_in = 0.5 vs K_in = 0.4 split gives the Corberán formulation a
better chance of producing the right SDM25 power-curve shape on the
first F1c run, which keeps F1d calibration adjustments small and
geometry-attributable.

These are starting values. They get calibrated against SDM25 dyno in F1d.

### 2c. Decision on directionality

Real junctions have DIFFERENT K for flow in different directions (a 4-2
merge loses more during outbound blowdown than during return wave
propagation). Fully general Corberán has **K_ij** for each inlet-outlet
pair, but this becomes unwieldy for a 4-leg junction (6 pairs × 2
directions = 12 coefficients).

**Simplification:** use a single per-leg K that depends only on whether
the leg is currently *acting as* an inlet (σ·u > 0) or outlet (σ·u < 0)
at the converged state. Store two scalars per leg: `K_when_incoming` and
`K_when_outgoing`. The direction of flow is determined by the converged
face state and selects which K applies.

This matches the "typical K_in = 0.4, K_out = 0.1" asymmetry from the
literature and keeps the parameter count manageable (2 scalars per leg,
vs the 12-per-junction general case).

---

## 3. Solver strategy — outer-inner iteration

### 3a. Recommended: outer Newton on p_0,junction, inner per-leg solve

**Outer.** Given current `p_0,junction` guess, each leg's face state is
determined by solving eqs (2) + (3a-c) as a 4-equation-4-unknown
subsystem. The global residual is eq (1) — sum of mass fluxes. Outer
Newton (secant, consistent with Phase E2b) iterates `p_0,junction` until
global mass balance.

**Inner.** For each leg, the 4-unknown subsystem reduces to a scalar
nonlinear equation. **Parameterize by face velocity `u_face,i`**
(revised per review — u has a natural bracket from |u| < c_face = sonic
choking, which is the exact boundary we detect for the choked branch
anyway). Steps:

1. Given `u_face,i` guess, invert (3a) for `c_face,i`:
   ```
   c_face,i = c_interior,i − ((γ−1)/2) · s_end · (u_face,i − u_interior,i)
   ```
2. Compute `p_face,i` from isentropy (3c): `p_face = p_i · (c_face/c_i)^(2γ/(γ−1))`.
3. Compute `ρ_face,i` from (3b): `ρ_face = ρ_i · (p_face/p_i)^(1/γ)`.
4. Compute stagnation pressure `p_0,face,i = p_face + ½ρ_face · u_face²`.
5. Compute expected `p_0,face,i` from (2):
   `p_0,face_expected = p_0,junction + sign(σ·u_face) · K · ½ρ_face · u_face²`
6. Inner residual: `p_0,face_computed − p_0,face_expected`.
7. Inner secant on `u_face,i` to drive residual to zero.

**Why u_face over p_face as the inner variable:**
- Natural bracket: |u_face| < c_face (subsonic limit). The solver
  detects choking cleanly when the iterate tries to push |u| past
  the sonic limit.
- No separate Mach check needed: |u_face / c_face| is directly
  available at each iteration step.
- With p_face as the iterate, the choke check requires back-computing
  u and c at each step — extra arithmetic per inner iteration.

Inner iteration is per-leg and independent, so it can be vectorized /
parallelized if needed. Expect 3–5 inner iterations per leg, 3–8 outer
iterations, total ~20–40 inner iterations per time step per junction.
Similar to Phase E2b's 3–5 secant steps but with more work per step.

### 3b. Rejected: coupled global Newton

Solve the full (1 + 4N)-dim system simultaneously with a vector Newton.
Faster per iteration but harder to make robust, especially for choked
legs (where the system becomes singular in one of the variables).
Outer-inner is ~1.5–2× slower per junction per step but much easier to
debug and stop-gate.

### 3c. MUSCL-aware face reconstruction

The Phase E2b fix (match MUSCL's predictor + half-slope reconstruction
inside the Newton residual for strict mass conservation) still applies.
The inner-solve face state is what we write to the ghost cells, and
MUSCL's interior-side reconstruction runs against those ghosts. The
Newton residual uses HLLC on the MUSCL-reconstructed interior side vs
the converged ghost state, identical mechanism to Phase E2b.

### 3d. Initial guesses

- Outer: `p_0,junction^(0)` = area-weighted mean of interior p_0 (interior
  stagnation pressures). At steady state this is very close to the
  solution.
- Inner: `p_face,i^(0)` = current `p_interior,i` (face starts near
  interior — a reasonable starting point; the characteristic expansion
  will move it).

---

## 4. Choked-leg handling

### 4a. When does the new formulation have a choked leg?

Same criterion as Phase E: the converged face Mach M_f,i exceeds
(1 − choke_margin) for some leg i with flow INTO the junction (σ·u > 0).
The inner solve detects this during iteration; when p_face,i drops below
the critical pressure p* that would give M_f = 1, the inner secant is
clamped at the choke condition.

### 4b. Choked-branch math

Per Phase E design doc §4, the standard treatment is to fix the choked
leg's mass flux at sonic throat:

```
ρ_f,i · u_f,i · A_i = ρ*_i · c*_i · A_i   (sonic throat flux, from
                                          leg-i interior stagnation)
```

The face static pressure in the choked leg is the sonic throat pressure
`p*_i`, which is determined by `p_0,interior,i` and γ:

```
p*_i = p_0,interior,i · (2/(γ+1))^(γ/(γ-1))
```

**Corberán loss in the choked branch.** If the leg is choked, eq (2)
still applies to the *post-throat* state. This is where the Corberán
formulation starts to have edge cases:

- In the inviscid limit, the choked leg has p_face = p*, and the junction
  sees a mass flux that's independent of p_j (choke decouples the leg).
- With loss, the K · ½ρu² term at the throat is applied to convert some
  of the sonic kinetic energy into loss. The junction reservoir pressure
  is reduced accordingly.

**Simplification for F1b.** In the choked branch, treat the choked legs
exactly as Phase E does (sonic throat flux, independent of p_0,junction).
The loss term for choked legs is zero (they don't contribute to junction
stagnation pressure). Non-choked legs see the full Corberán formulation
with the reduced fixed_mdot_sum from the choked legs.

This is a simplification. Physically a choked flow DOES lose stagnation
pressure to the junction, but the magnitude is captured elsewhere (the
sonic throat itself is a known ~10-20% stagnation pressure loss). Adding
another K · ½ρ u*² on top would double-count. Accept the simplification;
document it.

### 4c. All-choked error

Same as Phase E: raise `JunctionAllChokedError` if every leg is choked.
The reduced system has zero subsonic legs to balance mass.

### 4d. Smooth transition at choke boundary

Hard thresholds at M = 1 − `choke_margin` create discontinuous BC
behavior when a leg transitions in or out of choke across a time step.
The discontinuity generates spurious wave reflections as the solver
"snaps" between formulations.

**Fix: blend over the choke_margin band.** Define:
```
M_into = σ_i · u_face,i / c_face,i
φ(M_into) = linear ramp from 0 (at M_into = 1 − choke_margin)
             to 1 (at M_into = 1)
```
with `choke_margin = 0.02` (same value Phase E uses for the hard
threshold).

Per-leg face mass flux in the blended band:
```
F_mass_leg,i = (1 − φ) · F_mass_subsonic_leg,i
             +      φ  · F_mass_choked_leg,i
```

Below the band: fully subsonic (Corberán or Phase E fast path).
Above the band: fully choked (sonic throat flux, no K contribution).
Inside the band: linear blend.

This gives C⁰-continuous BC behavior across the choke transition. The
band is 2% wide in Mach, which translates to ~2% of the relevant
engine cycle duration — wide enough to eliminate spurious reflections,
narrow enough that the blend is not physically meaningful on its own
(neither pure subsonic Corberán nor pure choked is wrong inside the
band; they are both approximations to a transonic regime that 1D
inviscid can't fully capture).

Implementation location: inside the per-leg face-state evaluation, after
the inner secant converges. If the converged u_face has |M_into| in the
blending band, evaluate both branches and combine. If outside the band,
use the appropriate branch directly.

---

## 5. Conservation properties

### 5a. Mass

**Exact to machine precision.** The global constraint (1) is enforced
by outer Newton to tolerance 1e-13 kg/s. Test 9 (non-uniform closed
domain conservation) should pass at the same machine-precision level as
Phase E2b, for any K > 0.

### 5b. Energy

**Not conserved.** The stagnation pressure drop term `K · ½ρu²` represents
energy dissipated to turbulence in the merge. This energy is lost from
the 1D system (not tracked in any reservoir). Expected magnitude:

```
ΔE_loss = K · ½ρ · u² · u · A = K · ½ρ · u³ · A    per leg per second
```

For a typical exhaust blowdown at u = 100 m/s, ρ = 1 kg/m³, A = 8e-4 m²,
K = 0.4:
```
ΔE_loss ≈ 0.4 · 0.5 · 1 · 10⁶ · 8e-4 = 160 W per junction
```

For SDM26 with 3 junctions and peaks at ~40 kW cylinder power, the
junction loss totals ~0.5 kW ≈ 1.2% of indicated power. This is the
physical loss we are introducing; it is what gets converted to exhaust
gas temperature downstream. Appropriate for engine operation; documented
as "junction dissipation" rather than "numerical error."

### 5c. Composition (ρY)

**Exact when mass is exact.** ρY is transported with the mass flux, and
the mass balance closure carries ρY automatically. The mixed-Y
computation logic from Phase E is unchanged.

### 5d. Diagnostic residuals and sweep-log reporting

Per-junction diagnostics (written to the junction instance each
`fill_ghosts` call):
- `last_p_0_junction` — the solved stagnation pressure [Pa]
- `last_mass_residual` — global mass residual at converged state [kg/s]
- `last_energy_dissipation_W` — the K·½ρu³·A loss sum across all legs
  on this junction [W]. Positive = loss; a persistently negative value
  would flag a physics bug.
- `last_inner_iter_max` — max inner-iteration count across legs
- `last_niter` — outer secant iterations

**Sweep-log integration (F1d).** Extend the per-RPM sweep output with
a new column `junction_loss_kW` reporting the cycle-averaged sum of
`last_energy_dissipation_W` across all junctions in the engine model,
converted to kW. For SDM26 this sums 3 junctions; for SDM25 this is
1 junction.

**Sanity expectations.**
- K=0.4, typical blowdown u ~ 100 m/s, ρ ~ 1 kg/m³, A ~ 8e-4 m²:
  per-leg instantaneous loss ~ 0.4 · 0.5 · 1 · 10⁶ · 8e-4 = 160 W.
  Cycle-averaged across 4 primaries: ~50 W per junction peak.
  Summed over 3 SDM26 junctions: ~150 W steady, peaking at a few
  hundred W during blowdown. **Not a few kW.** If the diagnostic
  reports > 1 kW cycle-average, K is too aggressive.
- Very low values (< 10 W cycle-average) suggest K is too small, or
  the characteristic junction is still not generating enough velocity
  perturbation to activate meaningful loss.

The sweep-log column gives us a sanity check on K values before we
even run the dyno comparison — any implausible loss magnitude flags
a problem with the implementation or the K choice.

---

## 6. Backward compatibility with Phase E

### 6a. K = 0 uses the Phase-E constant-static-pressure fast path

Revised per review (see §1f). When all K_i = 0 the constructor
selects the Phase-E code path, giving **exact bit-for-bit equality**
with Phase E at K=0. No O(M²) approximation is accepted.

### 6b. K = 0 reduction test

Unit test 10: run A3 with K = 0 on all legs. Expect R_round_trip
**exactly equal** to the Phase-E baseline (+0.7484) to machine
precision. If the measured value differs at all, the fast-path
selection or shared helpers have a bug.

### 6c. Engine model backward compatibility

Existing model files (`models/sdm25.py`, `models/sdm26.py`) currently
construct `CharacteristicJunction` without any K arguments. The new
dataclass defaults `K_incoming = 0`, `K_outgoing = 0` so pre-F1c engine
runs behave identically to Phase E (modulo the O(M²) reformulation
difference). Model files get updated in F1c to specify real K values.

---

## 7. Interface specification

### 7a. Dataclass changes

```python
@dataclass
class CharacteristicJunction:
    legs: List[JunctionLeg]
    gamma: float = 1.4
    R_gas: float = 287.0
    newton_tol: float = 1e-13
    newton_max_iter: int = 30
    choke_margin: float = 0.02

    # Phase F1 — Corberán loss coefficients.
    # ``K_incoming[i]`` applies on leg i when it is acting as an
    # inflow (fluid flowing from leg into junction, σ·u > 0).
    # ``K_outgoing[i]`` applies when the leg is acting as an
    # outflow (fluid flowing from junction into leg, σ·u < 0).
    # Direction is determined by the converged face state per step.
    # Length of each array must equal len(legs). Default = zeros
    # (inviscid, matches Phase E Corberán-at-K=0 behavior).
    K_incoming: List[float] = field(default_factory=list)
    K_outgoing: List[float] = field(default_factory=list)

    # Internal knobs (unchanged from Phase E)
    _inflow_entropy_picard: bool = field(default=True, repr=False)

    # Diagnostics (extended)
    last_p_0_junction: float = 101325.0        # was last_p_junction
    last_mass_residual: float = 0.0
    last_energy_dissipation_W: float = 0.0     # new
    last_niter: int = 0                         # outer iterations
    last_inner_iter_max: int = 0                # new
    last_regime: str = "subsonic"
    last_y_mixed: float = 0.0
```

`last_p_junction` is renamed to `last_p_0_junction` because the variable
is now stagnation pressure, not static. We'll keep a compatibility
alias that returns `last_p_0_junction` when the old name is read.

### 7b. Constructor validation

In `__post_init__`:
- If `K_incoming` and/or `K_outgoing` are empty, default to zeros of
  length `len(legs)` (matches existing no-argument calls).
- If non-empty, require length == `len(legs)` else `ValueError`.
- Require each K ≥ 0 else `ValueError`.

### 7c. Factory

`make_junction` gets optional `K_incoming`, `K_outgoing` parameters:

```python
def make_junction(
    junction_type: str, legs, *,
    gamma=1.4, R_gas=287.0, p_init=..., T_init=..., Y_init=...,
    K_incoming=None, K_outgoing=None,   # new
):
    ...
    if junction_type == "characteristic":
        return CharacteristicJunction(
            legs=legs, gamma=gamma, R_gas=R_gas,
            K_incoming=K_incoming or [0.0] * len(legs),
            K_outgoing=K_outgoing or [0.0] * len(legs),
        )
```

### 7d. Engine model integration

`SDM26Engine.__init__` gets optional `junction_K_incoming` /
`junction_K_outgoing` dicts keyed by junction label, or more simply a
single pair of floats that applies to all junctions:

```python
SDM26Engine(
    cfg,
    junction_type="characteristic",
    junction_K_in=0.4,   # applied to all incoming legs on all junctions
    junction_K_out=0.1,  # applied to all outgoing legs
)
```

Per-junction overrides can be added in F1d if needed after calibration.
For F1c we use the uniform K_in, K_out approach.

---

## 8. Phased delivery

Following the F1 rescope prompt:

- **F1a** (this doc): design document. Stop for review.
- **F1b**: implementation + unit tests 1–12 (9 existing + 3 new: K=0
  reduction, K scaling monotonicity, K asymmetry).
- **F1c**: engine integration. Update model files with K=[0.4, 0.1]
  starting values. 10500 SDM26 single point. Stop for review.
- **F1d**: K calibration against SDM25 dyno. Full sweep with tuned K.
  Acceptance gates from the plan (r_P > 0.85, r_T > 0.4, RMSE_P < 5 kW,
  RMSE_T < 8 Nm).

Each sub-phase is its own commit.

---

## 9. Review outcomes (resolved 2026-04-15)

The six design questions were resolved in review. Summary of final
decisions now reflected in the sections above:

- **Q1 (K values).** Starting estimates confirmed: K_in = 0.4,
  K_out = 0.1 for SDM26 junctions. **K_in = 0.5** for the SDM25 4→1
  junction (wider merge angle, sharper contraction → more
  dissipation). Final values tuned in F1d within the [0.3, 0.6] /
  [0.05, 0.2] published ranges.
- **Q2 (sign convention).** Confirmed: `sign(σ·u_face)` with inflow
  giving p_0,face > p_0,junction.
- **Q3 (choked-branch).** Confirmed: zero K contribution from choked
  legs (sonic throat dissipates). Documented in §4b that this
  targets the Corberán loss correction at late-exhaust / intake-
  overlap regimes where tuned-length physics matters most.
- **Q4 (K=0 backward compat).** **Revised:** use K=0 fast path for
  exact Phase-E equality (see §1f and §6a). Implementation cost
  ~30-50 lines, benefit is exact bit-for-bit regression.
- **Q5 (inner unknown).** **Revised:** iterate on `u_face,i` (not
  `p_face,i`). Natural bracket from sonic limit, no separate Mach
  check needed. See §3a.
- **Q6 (sweep-log reporting).** Confirmed: `junction_loss_kW`
  column added to per-RPM sweep output. Cycle-averaged sum across
  all junctions in the engine model. See §5d.

Two additional specifications added:

- **Smooth choke transition** (§4d): blend subsonic and choked
  branches linearly over a 2% Mach band around 1.0 to avoid
  discontinuous BC behavior and spurious reflections across
  choke transitions.
- **F1d expectations calibrated to reality** (§10): the aspirational
  gates (r_T > 0.4, RMSE_P < 5 kW) may or may not all clear — what
  matters is substantial improvement and K values inside the
  Winterbone range. No over-tuning to hit metrics artificially.

---

## 10. F1d calibration — realistic expectations

The F1d acceptance gates were set aspirationally in the original plan.
Realistic expectations based on the pre-F1 residual error structure:

- **r_P (power shape correlation)**: pre-F1 = 0.75. Target 0.85.
  Achievable range 0.85–0.90.
- **r_T (torque shape correlation)**: pre-F1 = **−0.30**. Target > 0.4.
  The negative correlation was driven by the spurious 5100 RPM
  junction-overshoot peak. Corberán with reasonable K should collapse
  that peak, turning r_T positive. Achievable range 0.3–0.5. May not
  clear 0.4 if there are other residual shape-mismatch sources
  (combustion model, wall heat).
- **RMSE_P**: pre-F1 = 9.5 kW. Target < 5 kW. Achievable range 4–6 kW.
- **RMSE_T**: pre-F1 = 13.8 Nm. Target < 8 Nm. Achievable range 7–10 Nm.

**If all gates clear** with K inside the published range: commit as
calibrated baseline, proceed to F2–F4.

**If substantial improvement is achieved but some gates miss**:
commit K values, document residual error sources, proceed to F2–F4.
F2 (Levine-Schwinger) and F4 (Wiebe ramp) may close some of the
remaining gaps, particularly on r_T and torque-band errors.

**Do not over-tune K** to clear the gates. K stays inside the
Winterbone published range.

---

## 11. After this is committed

Proceed to **F1b** on the same `phase-f/calibration` branch. This
design doc gets committed as a separate commit before any code.

F1b sub-plan:
1. Extend `CharacteristicJunction` dataclass with K_incoming,
   K_outgoing arrays + new diagnostics.
2. Implement K=0 fast path (preserve current Phase E solver unchanged).
3. Implement K>0 Corberán outer-inner solver (secant on p_0,junction,
   per-leg inner secant on u_face,i).
4. Implement smooth choke-transition blending.
5. Write 3 new unit tests (K=0 reduction at exact equality, K scaling
   monotonicity, K asymmetry with mass-conservation check).
6. Verify all 12 unit tests pass.
7. Stop for review before F1c engine integration.
