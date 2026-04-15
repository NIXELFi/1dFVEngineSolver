# Phase F1a — Corberán loss-coefficient junction: design document

**Status:** draft for review. No implementation code written. F1 attempts
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

### 1f. Backward compatibility

When K_i = 0 for all legs, eq (2) reduces to `p_0,face,i = p_0,junction`.
Combined with the stagnation-pressure definition and the characteristic
relations, this implies `p_face,i = p_0,junction − ½ρ_face,i · u_face,i²`.

The constant-static-pressure formulation has `p_face,i = p_junction` where
p_junction is the static pressure. The Corberán formulation with K = 0
uses a common stagnation pressure instead. **These two are equivalent at
the incompressible / low-Mach limit** where ½ρu² is negligible compared to
p. They differ at higher Mach number by O(M²). For typical engine exhaust
Mach numbers (up to ~0.3 at blowdown peak), this O(M²) = 0.09 discrepancy
is small but not zero.

**Design decision on K=0 backward compatibility.** The Corberán
formulation with K=0 is *approximately* but not *identically* equal to
the constant-static-pressure formulation. To make the K=0 reduction
*exactly* match the Phase E baseline, we would need to keep both
formulations and switch between them. This doubles the solver code.

**Proposal:** accept the O(M²) difference at K=0 and document it. The
Phase-E-vs-Phase-F A3 baseline will shift by O(1%) even at K=0, which is
a known consequence of swapping formulations. All downstream tests
adjust their baselines accordingly.

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
- **4→1 junction** (P0, P1, P2, P3 → C): wider angle merge, sharper
  contraction, expect slightly higher K_in.
  K_incoming = [0.4, 0.4, 0.4, 0.4], K_outgoing = [0.1].

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
nonlinear equation. Parameterize by face static pressure `p_face,i`
(alternative: by `u_face,i` — either works; I'll use `p_face,i`). Steps:

1. Given `p_face,i` guess, compute `ρ_face,i` and `c_face,i` from (3b),
   (3c) — isentropic from interior.
2. Compute `u_face,i` from (3a).
3. Compute `p_0,face,i = p_face,i + ½ρ_face,i · u_face,i²`.
4. Compute expected `p_0,face,i` from (2):
   `p_0,face_expected = p_0,junction + sign(σ·u_face) · K · ½ρ_face · u_face²`
5. Inner residual: `p_0,face_computed − p_0,face_expected`.
6. Inner secant on `p_face,i` to drive residual to zero.

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

### 5d. Diagnostic residuals

Add to the junction's diagnostic log:
- `last_p_0_junction` — the solved stagnation pressure
- `last_mass_residual` — global mass residual at converged state
- `last_energy_dissipation_W` — the K · ½ρu³ · A loss sum across all
  legs (positive = loss; negative would flag a physics bug)
- `last_inner_iter_max` — max inner-iteration count across legs

Report `last_energy_dissipation_W` in the sweep log so we can track
junction losses across the RPM grid.

---

## 6. Backward compatibility with Phase E

### 6a. K = 0 reduces to approximately constant-static-pressure

Discussed in §1f. Not *identically* equal, but matches within O(M²) ≈ 1-5%
of the wave amplitude at typical engine operating conditions. Acceptable.

### 6b. K = 0 reduction test

Unit test 10 (from the plan): run A3 with K = 0 on all legs. Expect
R_round_trip within 5% of the Phase E baseline (+0.7484). If the
measured value is significantly different (say > 10% off), the
formulation or implementation has a bug.

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

## 9. Open questions for review

**Q1** — K_in / K_out values. Do you have Winterbone on hand or a prior
dyno-validated K for SDM-class manifolds? If yes, supply values. If no,
default to my starting estimates (K_in=0.4, K_out=0.1) with the plan to
tune in F1d.

**Q2** — Sign convention in eq (2). I wrote `sign(σ·u_face)` such that
inflow (σu > 0) gives positive loss (p_0,face > p_0,junction). Please
sanity-check — Corberán may use the reverse convention.

**Q3** — Choked-branch loss term. §4b proposes zero K contribution from
choked legs (the sonic throat itself dissipates). Confirm this is the
right simplification or flag if you want the Corberán loss applied at
the throat as well.

**Q4** — K=0 backward compatibility. Discussed in §1f and §6a. The
Corberán formulation with K=0 is approximately but not identically
equal to Phase E's constant-static-pressure. Accept the ~1-5% A3
baseline shift at K=0 vs exactly reproducing Phase E? The alternative
is to maintain both formulations side-by-side, which doubles solver
code.

**Q5** — Inner-solve parameter. I proposed iterating on `p_face,i` as
the inner unknown. Alternative: iterate on `u_face,i`. Either works
mathematically. Any preference?

**Q6** — Energy loss reporting. §5d proposes `last_energy_dissipation_W`
as a new diagnostic. Want this reported in the sweep log per point
alongside the existing conservation diagnostics?

---

## 10. After this is approved

Proceed to **F1b** on the same `phase-f/calibration` branch. Commit the
design doc separately before touching the code.

No code until these questions are answered.
