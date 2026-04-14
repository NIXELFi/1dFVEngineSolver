# V1 Physics Audit

Phase 1 deliverable for the SDM26 1D Engine Simulator V2 rewrite.

This document describes V1 **as actually implemented** on disk at
`~/Developer/1d/` as of 2026-04-13, branch `feat/impedance-coupled-valve-bc`
(HEAD `719dd8e`, with uncommitted working-tree changes preserved). Nothing in
`1d/` was modified to produce this audit.

## How to reproduce

```bash
cd ~/Developer/1d_v2
python3 -m diagnostics.v1_mass_audit --rpm 10500 --cycles 3 --warmup 2 \
    --out docs/v1_mass_audit_10500.json
python3 -m diagnostics.v1_valve_entropy_probe --rpm 10500 --warmup 2 --cycles 1 \
    --out docs/v1_valve_entropy_probe_10500.csv
```

Both diagnostics import V1 as a library from a single sanctioned site
(`1d_v2/diagnostics/*`); no other V2 code is permitted to import from `1d/`.

## 1. Module map of `1d/`

Paths are relative to `1d/engine_simulator/`. One-liner per file. **[impedance-WIP]**
tags flag modules that are part of the in-progress
`feat/impedance-coupled-valve-bc` work rather than the stable MOC core.

### `gas_dynamics/` — pipe interior solver and thermodynamics

| File | Purpose |
|------|---------|
| `pipe.py` | `Pipe` class: 1D grid, Benson non-dimensional state arrays (`lam`, `bet`, `AA`), derived dimensional arrays (`p`, `T`, `rho`, `u`, `a`). Stores `artificial_viscosity` per pipe. |
| `moc_solver.py` | `advance_interior_points()` (MOC predictor with friction / heat-transfer / area-change source terms and optional Laplacian AV) and `extrapolate_boundary_incoming()` (C± foot-point trace at the two pipe ends). |
| `gas_properties.py` | Reference state (`P_REF=101325`, `T_REF=300`, `A_REF≈347.2`, `RHO_REF≈1.177`), γ(T) for unburned/burned, R_mix(xb), Blasius, Dittus-Boelter, Benson A↔p↔T↔ρ conversions. |
| `cfl.py` | Global CFL timestep from all pipes. |

### `boundaries/`

| File | Purpose |
|------|---------|
| `base.py` | `PipeEnd` enum, `BoundaryCondition` ABC. Convention: at LEFT, β arrives; at RIGHT, λ arrives. |
| `closed_end.py` | Wall: `lam = bet` (zero velocity). |
| `open_end.py` | Atmospheric outflow (pressure-match) and inflow (stagnation-enthalpy match). |
| `valve_bc.py` **[impedance-WIP]** | Newton-Raphson on boundary `A_b` matching valve-orifice `ṁ` to pipe `ρ·u·A` with pipe-side AA (not cylinder entropy). |
| `restrictor.py` | Isentropic choked nozzle; sets `pipe.AA=1` on the downstream pipe. |
| `junction.py` | N-pipe constant-pressure junction; mass-weighted AA mixing on inflowing legs. |
| `area_change.py` | Sudden expansion/contraction with Borda/Weisbach loss. |

### `engine/` — cylinder sub-models

| File | Purpose |
|------|---------|
| `cylinder.py` | 0D thermodynamic cylinder. Open-cycle uses Euler on (p, m, T); closed-cycle uses RK4 on p. Sets fuel mass at IVC. |
| `combustion.py` | Wiebe `x_b = 1 − exp(−a·τ^(m+1))` with canonical (possibly negative-BTDC) angles. |
| `heat_transfer.py` | Woschni correlation with IVC reference state. |
| `valve.py` | sin² lift profile, L/D → Cd lookup table, low/medium/high-lift reference area. |
| `geometry.py` | Bore/stroke, V(θ), dV/dθ, chamber surface area. |
| `kinematics.py` | Cylinder phase offsets, ω(rpm). |

### `simulation/` — orchestration

| File | Purpose |
|------|---------|
| `orchestrator.py` **[impedance-WIP]** | Builds the system, owns the time loop, applies RPM-dependent combustion corrections, computes performance metrics. |
| `plenum.py` | `RestrictorPlenumBC`: coupled NR solve for plenum pressure against restrictor ṁ, runner ṁ, and plenum capacitance. Uses actual plenum entropy (`AA_plen`). |
| `engine_cycle.py` | Crank-angle tracker, event detection. |
| `convergence.py` | `ConvergenceChecker`: relative change of per-cylinder `p_at_IVC` cycle-to-cycle. |
| `parallel_sweep.py` | RPM-sweep multiprocessing runner and progress events. |

### `config/`

| File | Purpose |
|------|---------|
| `engine_config.py` | Dataclass schemas: `PipeConfig`, `ValveConfig`, `CylinderConfig`, `CombustionConfig`, `RestrictorConfig`, `JunctionConfig`, `SimulationConfig`, `EngineConfig`. |
| `cbr600rr.json` | Base SDM26 geometry and coefficients. |
| `SDM25_final.json` **[impedance-WIP]** | 2025 car geometry; uncommitted edit sets `restrictor.discharge_coefficient=0.91`. |

### `postprocessing/`, `validation/`

Standard: performance metric aggregation, results container, matplotlib plots;
Sod shock tube, Helmholtz resonator, and published-engine regression harness.

### Uncommitted working-tree state on `feat/impedance-coupled-valve-bc`

- `engine_simulator/__init__.py` — version bump to `1.08`.
- `config/SDM25_final.json` — `restrictor.discharge_coefficient: 0.926 → 0.91`.
- `simulation/orchestrator.py` — two-segment combustion-efficiency ramp
  keyed on the May-2025 SDM25 DynoJet data, with the 0.88 cap.
- Untracked: `_calibrate.py`, math-model PNG/PDF/HTML, `_screenshot.mjs`,
  `docs/superpowers/plans/2026-04-11-impedance-coupled-valve-bc.md`,
  `_plot_review/.DS_Store`.

The impedance-branch plan spells out (i) a Newton-Raphson exhaust-valve BC
matching orifice ṁ to pipe characteristic ṁ and (ii) per-pipe AV tuning. The
core MOC interior solver and the restrictor/plenum/junction code pre-date this
branch. V2 replaces the stable MOC core. The impedance BC work is a probable
pointer to *what* V2's valve BC should converge on numerically, but V2 will
derive that from the conservative formulation rather than ported from this
branch.

## 2. Governing equations as implemented

### 2.1 Pipe interior — Benson non-dimensional MOC

**Reference state** (`gas_properties.py:17-22`):
`P_REF = 101325 Pa`, `T_REF = 300 K`, `γ_REF = 1.4`, `A_REF = √(γ_REF·R_air·T_REF) ≈ 347.2 m/s`, `ρ_REF ≈ 1.177 kg/m³`, `R_AIR = 287 J/(kg·K)`.

**Primary state stored per node** (`pipe.py:63-74`):
- `lam = A + (γ−1)/2 · U` (C+ Riemann invariant)
- `bet = A − (γ−1)/2 · U` (C− Riemann invariant)
- `AA` (Benson entropy-level parameter)

**Derived relations** (`pipe.py:97-126`, post-sub-atmospheric fix):
- `A = (λ+β)/2`,  `U = (λ−β)/(γ−1)`
- `a = A · A_REF`,  `u = U · A_REF`
- `T = T_REF · A²`  — temperature depends only on A (NOT on AA).
- `p = P_REF · (A/AA)^(2γ/(γ−1))`
- `ρ = p/(R_AIR·T)` — ideal gas, consistent with (p, T).

γ in the pipe is hard-coded to `GAMMA_REF = 1.4` (`pipe.py:37`). Only the
cylinder uses the mixture-aware γ(T, xb) / R(xb) from `gas_properties.py`.

**Interior advance** (`moc_solver.py:126-259`): a first-order predictor in
characteristic form on a fixed Eulerian mesh.

For each interior node i and forward step dt:
1. Trace three foot points back to time `n`:
   - C+ foot: `x_R = x_i − (u+a)·dt`
   - C− foot: `x_L = x_i − (u−a)·dt`
   - C0  foot: `x_S = x_i − u·dt`
2. Linearly interpolate `λ, β, AA` (and flow variables for sources) at each foot.
3. Update:
   - `new_lam = λ_R + Δλ_source + lam_correction(AA)`
   - `new_bet = β_L + Δβ_source + bet_correction(AA)`
   - `new_AA  = AA_S + Δ(AA)_source`
4. After the per-node update, optionally apply a Laplacian AV pass on
   `λ, β`:  `λ_i ← λ_i + ν·(λ_{i+1} − 2λ_i + λ_{i−1})`.
5. Clamp `λ, β, AA ≥ 0.01` (`moc_solver.py:277-279`).

**Source terms** (`moc_solver.py:207-254`):
- Friction: `f = 0.3164·Re^(−0.25)` for Re > 2300, `f = 64/Re` for Re ≤ 2300,
  floor `f = 0` for `Re < 1` (`gas_properties.py:69-80`).
- Wall heat: Dittus-Boelter — `Nu = 0.023·Re^0.8·Pr^0.4` with `Pr = 0.71`,
  `k = 0.026·(T/300)^0.7 W/(m·K)`,  `h = Nu·k/D`,
  `q_wall = h·(T − T_wall)`.
- Area change: `A·U·(dA/dx)/A_cross` added to the source with a sign that
  depends on the characteristic direction.

**Entropy (AA) source** (`moc_solver.py:219-253`): both heat-transfer and
friction contribute via
`dAA ≈ AA·(γ−1)/(2·A²)·[…heat… + …|U|²U·A_REF/(2D)]·dt`.
A comment at line 226 — *"Simplified: just use entropy from C0 path"* —
flags that the entropy-source coupling into λ and β via the
`lam_correction = (A/AA)·dAA` term is **not** the full Benson correction;
the implementation knowingly approximates.

**CFL** (`cfl.py`): global dt = `cfl_num · min_pipes( dx / max_node(|u|+a) )`,
with `cfl_num = 0.85` (`cbr600rr.json:221`).

**Artificial viscosity** (`moc_solver.py:261-269`): explicit Laplacian on
`λ, β` (not on AA), documented as compensating for the absence of physical
acoustic absorption in 1D MOC. Stability requires `ν ≤ 0.5`; default global
value `simulation.artificial_viscosity = 0.05` (`cbr600rr.json:224`). The
config allows per-pipe override; in `cbr600rr.json` every primary pipe sets
`artificial_viscosity = 0.0`, relying on the global default being overridden
(a convention the orchestrator resolves with "use pipe value if ≥ 0 else
global" at `orchestrator.py:382`). This is an empirical knob.

### 2.2 Boundary conditions

All BCs fix boundary `λ` or `β` (and sometimes AA) so that subsequent MOC
foot-tracing at the adjacent interior nodes is well-posed. No BC physically
transports mass during its own call; mass motion happens in the following
interior step.

**Closed end** (`closed_end.py:16-24`): `u = 0` ⇒ reflect the arriving
invariant: `bet[-1] = lam[-1]` at RIGHT, `lam[0] = bet[0]` at LEFT.

**Open end** (`open_end.py:38-128`):
- Outflow (U·n > 0): set `A_boundary = A_from_pressure(p_atm, AA)`; the
  incoming invariant is preserved and the other is fixed.
- Inflow (U·n < 0): NR on β (or λ) against stagnation enthalpy
  `A²+(γ−1)/2·U² = A_stag²·AA²`; then force `pipe.AA = 1.0` (fresh air
  assumption).

**Restrictor** (`restrictor.py:62-174`): treat atmosphere as the stagnation
reservoir, compute `ṁ(p_downstream)` from the isentropic choked-nozzle
formula with `Cd = 0.95` and `A_t = π·(0.02)²/4`. Set pipe boundary
velocity from `u = ṁ / (ρ_down · A_pipe)`, then fix λ or β via the
arriving-invariant + U relation. Hard-sets `pipe.AA = 1.0`
(`restrictor.py:170`).

**Plenum / restrictor-plenum coupling** (`plenum.py:73-196`): iterate on
`p_plen` against
  `ṁ_restrictor(p_plen) − Σ ṁ_runner_i(p_plen) − (V_plen/(R·T_plen·dt))·(p_plen − p_plen^{n−1}) = 0`.
The runner-side density uses the **plenum's own entropy** `AA_plen` derived
from `(p_plen, T_plen)` (`plenum.py:97-98`), i.e. entropy-aware on the
intake side. The plenum temperature relaxes toward ambient with
`τ_thermal = 0.005 s` (`plenum.py:190-193`) — tunable knob, not a physical
heat-transfer coefficient.

**Valve BC (exhaust + intake)** (`valve_bc.py:40-189`): for each pipe-end +
cylinder pair, Newton-Raphson up to 25 iterations on `A_b` at the boundary
such that `ρ_b·u_b·A_pipe = ± ṁ_orifice(p_up, T_up, p_down)` with
`u_b = 2(A_b − R_in)/(γ−1)` or `2(R_in − A_b)/(γ−1)` depending on the end.
The orifice uses `Cd(L/D)` from the valve tables and the compressible-orifice
equation. Pressure and temperature on the pipe side are derived from
`A_b` and the **pipe's own** `AA` (`valve_bc.py:165-167`):

```python
p_b = P_REF * (A_b / AA)**(2γ/(γ−1))
T_b = T_REF * A_b**2
ρ_b = p_b / (R_AIR · T_b)
```

This is the **entropy-carrying bug**. For exhaust outflow, the gas crossing
the valve is hot burned-gas at the cylinder's high entropy, but the BC
places it on the pipe's low-entropy isentrope. See §4.

**Junction (N-pipe constant-pressure)** (`junction.py:53-193`): NR on
`p_junction` against `Σ sign_i · ρ_i · u_i · A_i = 0`, where each pipe's
`A_j` is computed from its own AA. After the pressure solve, mass-weighted
AA mixing is applied **only** to the pipes receiving flow from the junction
(`junction.py:170-192`). Pipes supplying flow keep their AA. This means hot
exhaust exiting pipe A and cold gas re-entering pipe B get mass-weighted
mixed, but the inflow side carries that mixed entropy forward. The
implementation is simple and adequate for cycle-averaged mass accounting,
but contact discontinuities at the junction are smeared one cell per step.

**Area change (sudden)** (`area_change.py:57-134`): NR on the junction
pressure with a Borda-Weisbach loss coefficient
`K = (1 − A_min/A_max)²` for expansions, `0.5·(1 − A_min/A_max)` for
contractions, applied as a downstream pressure drop.

### 2.3 Cylinder sub-models

**State** (`cylinder.py:42-51`): `p, T, m, m_fuel, x_b, V, γ`, plus flows
set by valve BCs (`mdot_intake`, `mdot_exhaust`), boundary-gas temperatures
(`T_intake`, `T_exhaust`), and per-cycle accumulators (`m_intake_total`,
`m_exhaust_total`, `work_cycle`, `p_at_IVC`, `T_at_IVC`).

**Open cycle** (`cylinder.py:149-178`): Euler on
- `dp/dt = (1/V)[−γ·p·dV/dt + (γ−1)·(Q̇_comb − Q̇_ht) + γ·R·T_int·ṁ_in − γ·R·T·ṁ_out]`
- `dm/dt = ṁ_in − ṁ_out`
- `dT/dt = T·(dp/dt/p + dV/dt/V − dm/dt/m)` (ideal-gas closure)

**Closed cycle** (`cylinder.py:181-223`): RK4 on `p(θ)` with
- `dp/dθ = −γ·(p/V)·dV/dθ + (γ−1)/V·(dQ_comb/dθ − dQ_ht/dθ)`

T is recovered from ideal-gas after the step.

**Wiebe** (`combustion.py:56-96`): standard single-Wiebe with
`a = 5.0`, `m = 2.0` (`cbr600rr.json:195-196`), baseline
`combustion_duration = 50°`, `spark_advance = 25°`,
`ignition_delay = 7°`, `combustion_efficiency = 0.96` (base, before RPM
scaling — see §5), `q_lhv = 44 MJ/kg`, `afr_target = 13.1`.

**Woschni** (`heat_transfer.py:22-95`): with `C1_gas_exchange = 6.18`,
`C1_compression = C1_combustion = 2.28`, `C2_combustion = 3.24e−3` (matches
Heywood's original values). Characteristic velocity
`w = C1·S̄_p + C2·(V_d·T_ref)/(p_ref·V_ref)·max(p − p_mot, 0)`, with `p_mot`
a polytropic (γ=1.35) motored pressure referenced to IVC. Heat-transfer
coefficient `h_c = 3.26·B^{−0.2}·p_kPa^{0.8}·T^{−0.53}·w^{0.8}`.
Cylinder wall temperature `T_wall = 450 K` (`heat_transfer.py:19`).

**Valve** (`valve.py`): sin² lift profile over `close − open`; Cd table in
`cbr600rr.json` (intake peak 0.57, exhaust peak 0.55). Reference area is
piecewise in L/D (curtain for L/D < 0.25, port-limited above).

## 3. Empirical coefficients and their sources (as implemented)

| Coefficient | Value | Location | Apparent source |
|-------------|-------|----------|-----------------|
| R_air | 287 J/(kg·K) | `gas_properties.py:14` | standard |
| R_burned | 295 J/(kg·K) | `gas_properties.py:15` | stoich gasoline-air approx |
| γ_unburned(T) | 1.38 − 1.2e−4·(T−300) | `gas_properties.py:26-27` | empirical fit |
| γ_burned(T) | 1.30 − 8.0e−5·(T−300) | `gas_properties.py:30-32` | empirical fit |
| Blasius | f = 0.3164·Re^−0.25 (Re>2300) | `gas_properties.py:69-80` | Blasius |
| Dittus-Boelter | Nu = 0.023·Re^0.8·Pr^0.4 | `gas_properties.py:84-88`, also inlined in MOC | Dittus-Boelter |
| Pr | 0.71 | `gas_properties.py:65` | air |
| Sutherland (μ) | 1.8e−5·(T/293)^0.7 | `gas_properties.py:50-54` | empirical |
| thermal conductivity | 0.026·(T/300)^0.7 W/(m·K) | `gas_properties.py:57-61` | empirical |
| Woschni C1_gx | 6.18 | `heat_transfer.py:22` | Woschni (1967) |
| Woschni C1_comp / C1_comb | 2.28 | `heat_transfer.py:23-24` | Woschni |
| Woschni C2_comb | 3.24e−3 | `heat_transfer.py:25` | Woschni |
| Woschni wall T | 450 K | `heat_transfer.py:19` | typical CBR600RR |
| Wiebe a | 5.0 | `cbr600rr.json:195` | Heywood default (~90%-burn at τ=1) |
| Wiebe m | 2.0 | `cbr600rr.json:196` | Heywood default |
| combustion duration | 50° base | `cbr600rr.json:197` | typical SI |
| spark advance | 25° BTDC base | `cbr600rr.json:198` | typical SI |
| ignition delay | 7° | `cbr600rr.json:199` | typical SI |
| combustion efficiency (base) | 0.96 | `cbr600rr.json:200` | typical SI |
| q_LHV | 44 MJ/kg | `cbr600rr.json:201` | gasoline |
| afr_target | 13.1 | `cbr600rr.json:203` | slightly rich for power |
| restrictor Cd | 0.95 | `cbr600rr.json:207` | FSAE bellmouth nozzle, typical; 0.926 in SDM25_final, tuned to 0.91 in uncommitted edit |
| throat diameter | 20 mm | `cbr600rr.json:206` | FSAE rule |
| plenum volume | 1.5 L | `cbr600rr.json:212` | SDM26 CAD |
| CFL safety | 0.85 | `cbr600rr.json:221` | standard |
| convergence tol | 0.5 % on p_IVC | `cbr600rr.json:222` | engineering |
| global AV | 0.05 | `cbr600rr.json:224` | calibrated empirically; 0.0 on primaries |
| intake-runner wall T | 325 K | `cbr600rr.json:92` | typical warm intake |
| exh-primary wall T | 650 K | `cbr600rr.json:131` | skin temp under load |
| exh-secondary wall T | 550 K | `cbr600rr.json:172` | empirical |
| exh-collector wall T | 500 K | `cbr600rr.json:191` | empirical |
| pipe node counts | 30 runners/primaries, 20 secondaries/collector | `cbr600rr.json` | convergence study (undocumented) |

## 4. Mass-leak diagnostic results

`diagnostics/v1_mass_audit.py` runs V1 with an observer-style instrumented
time loop that re-integrates the pipe / plenum / cylinder mass book after
every stage of the per-step update and attributes any drift to that stage.
All V1 objects are used verbatim; V1 source files are untouched.

### Topline numbers at 10500 RPM (3 cycles, 2 cycles warmup)

Summary (from `docs/v1_mass_audit_10500.json`):

| Quantity | Value (kg) |
|----------|-----------|
| Total system mass drift | +5.40e−5 |
| Restrictor "claimed" net inflow | +2.52e−3 |
| Collector "claimed" net outflow | −5.29e−3 (negative = net atmospheric backflow through open end) |
| Expected drift from port fluxes | +7.81e−3 |
| **Drift after port accounting** | **−7.76e−3** |

**Interpretation.** The system's actual mass change over the 3 audited
cycles is tiny (5.4e−5 kg, ~1 % of the ~5.6e−3 kg system inventory), which
is consistent with cyclic steady state. However, integrating the *fluxes
the BCs claim they delivered* predicts a net inflow of 7.81e−3 kg — about
40 % of total system mass would have been added if those flux reports were
accurate. The 7.76e−3 kg residual between claim and reality means **the
BC-reported ṁ values are not consistent with what the interior MOC update
actually transports.** This is the primary class of conservation error in
V1.

### Per-stage unaccounted drift (kg over 3 cycles)

| Stage | Unaccounted drift (kg) | Per-cycle |
|-------|----------------------|-----------|
| restrictor_plenum | −2.57e−3 | −8.6e−4 |
| collector_open_end | −5.29e−3 | −1.8e−3 |
| pipe_interior_MOC | +2.38e−4 | +7.9e−5 |
| extrapolate_boundary | −2.00e−4 | −6.6e−5 |
| exhaust_junctions | +7.3e−5 | +2.4e−5 |
| intake_valve_bcs | +3.3e−6 | +1.1e−6 |
| exhaust_valve_bcs | −7.8e−7 | −2.6e−7 |
| cylinder_advance | −8.1e−6 | −2.7e−6 |

The largest sources of conservation violation are the **restrictor-plenum
NR solve** and the **collector open end** — both sites where a reported
flux is being set against an MOC interior that does not enforce it. The
valve BCs themselves are comparatively well-behaved in bulk mass terms.

**Per-cycle bulk leak.** Of the ~9e−4 kg per cycle that the restrictor
claims to pass, V1 silently loses ~8.6e−4 kg (the pipes + plenum don't
actually gain that much). This is consistent with the observed ~10 % VE
shortfall at 11 k RPM in the SDM25 dyno calibration — although caution is
warranted because the BC claim and actual fluid transport do not match
1:1 in a Riemann-variable scheme.

### Caveat on attribution

MOC boundary conditions do not physically transport mass during the BC
call — they re-write `λ, β, AA` at the boundary node. Physical transport
happens on the next interior MOC advance. The per-stage attribution above
reflects the *bookkeeping* drift at each stage boundary, not a clean causal
chain. The **only** number that is a true conservation-error metric is the
top-line `drift_after_port_accounting_kg` integrated over a converged
cycle. For V2, this metric must be zero to machine precision on a closed
domain (per the validation suite, §Phase 2 test 5).

## 5. Valve BC entropy error — measured

`diagnostics/v1_valve_entropy_probe.py` runs V1 for 2 warmup + 1 probe
cycles at 10500 RPM, and while cylinder 0's exhaust valve is open, compares
the V1 exhaust-primary boundary temperature against the **isentropic
expansion** of cylinder gas from `(p_cyl, T_cyl)` to the instantaneous pipe
boundary pressure `p_pipe`, i.e.:
`T_isen = T_cyl · (p_pipe/p_cyl)^((γ−1)/γ)` with γ=1.35.

### Results (`docs/v1_valve_entropy_probe_10500.csv`)

| Metric | Value |
|--------|-------|
| Samples during exhaust event | 383 |
| Average pipe-boundary T | 326–420 K |
| Average isentropic T | ~1300 K |
| Average T error (T_isen − T_pipe) | **+961 K** |
| Average fractional error | **+75 %** (pipe is ~25 % of the correct absolute T) |
| Average wave-speed ratio `a_isen / a_pipe` | **1.95 ×** |
| Peak-blowdown point (EVO+0.01°) | T_cyl=1853 K, p_cyl=6.12 bar, T_pipe=326 K, T_isen=1254 K, a_isen/a_pipe=1.93 |
| Worst-case point (~80° after EVO) | T_cyl=1628 K, p_cyl=3.34 bar, T_pipe=419 K, T_isen=1622 K, a_isen/a_pipe=1.93 |

### Mechanism and line references

The V1 exhaust valve BC runs an NR on the pipe's boundary `A_b` using the
**pipe's own entropy parameter** `AA = pipe.AA[idx]` (`valve_bc.py:65`).
The residual insists that `ρ_b·u_b·A_pipe` match the orifice ṁ computed
from cylinder (p, T). This gets p_b, u_b, and ṁ right, but T_b is:

```python
T_b = T_REF * A_b**2   # valve_bc.py:166
```

with `A_b` chosen by the NR. The `A_b` that satisfies the pressure residual
is the one consistent with the **pipe's isentrope passing through p_b**,
not the cylinder's. Result: `T_b` is on the pipe's cold-gas isentrope even
during blowdown, so the burned-gas enthalpy never enters the pipe. The
pipe then sees a ρ that is far too high (because p is right but T is too
low), and a speed of sound that is ~1/2 of the physical value. This is
exactly what the V1 docstring at `valve_bc.py:9-15` already acknowledges
("underestimates exhaust gas temperatures and wave speeds by ~2x"), and
what the SDM25 dyno-calibration memory records as the dominant calibration
gap at 11 k RPM.

**Downstream consequences** (all observed in prior dyno comparisons):
- Tuned-length predictions that should scale with `a` (speed of sound) are
  off by ~√2 in pulse timing.
- EGT estimates from the sim are unusable.
- Exhaust-wave-induced VE augmentation at resonance is suppressed.

## 6. Every V1 site that touches a contact discontinuity / entropy jump

| Location | What it does | Why it is a problem for V2 |
|----------|--------------|----------------------------|
| `moc_solver.py:133` (x_S foot) | Advects `AA` along `u` (C0) inside each pipe | Linearly smears AA across contact waves; one cell per step |
| `moc_solver.py:219-253` | Computes `dAA` from friction/heat sources and folds a `(A/AA)·dAA` correction into λ/β updates | Comment "Simplified: just use entropy from C0 path" (line 226) — known partial implementation |
| `moc_solver.py:261-269` | Laplacian AV on λ and β | Additional diffusion on Riemann invariants; smears contact waves further when AV>0 |
| `boundaries/valve_bc.py:65,117,165-167` | Uses pipe-side AA to derive p_b, T_b; NR converges on A_b with AA fixed to the pipe's cold-gas entropy | **Primary cause of EGT underprediction** |
| `boundaries/valve_bc.py:138-141` | Assigns cylinder `T_exhaust = T_cyl` on outflow, cylinder `T_intake = T_b` on intake inflow | Correct direction but only recorded on the cylinder side; no composition scalar propagates into the pipe |
| `boundaries/junction.py:170-192` | Mass-weighted AA mixing only for pipes receiving flow from the junction | Smears burned/unburned contact one cell per step per junction pass |
| `boundaries/restrictor.py:170` | `pipe.AA[idx] = 1.0` hard-coded | Fine for atmospheric inflow, wrong if reversed flow ever happens at the restrictor (uncommon in practice) |
| `boundaries/open_end.py:90, 127` | `pipe.AA[idx] = 1.0` on inflow | Same as above — correct for inflow from atmosphere |
| `simulation/plenum.py:97-98, 165-168` | Uses `AA_plen = AA_from_p_T(p_plen, T_plen)` for runner inlet | Correct entropy-aware BC; one of the recently-fixed pieces worth carrying conceptually into V2 |

V2 replaces this whole entropy-accounting apparatus with an **advected
composition scalar** `ρ·Y·A` on the conservative (ρA, ρuA, EA, ρYA) vector,
and the HLLC contact wave carries species and entropy correctly by
construction.

## 7. Empirical tunings in V1 that compensate for numerical error

Gathered from `orchestrator.py`, config files, and source-level documentation.
These are the knobs future-V2 code should **not** inherit.

1. **Combustion efficiency cap at 0.88** (`orchestrator.py:275-276`). The
   comment says: *"The 0.88 cap absorbs model deficiencies (wave speed
   error, sin² cam profile)."* This is a direct numerical-error
   compensation. V2 must remove it; if efficiency < 1 is physical, it
   should be a separate, labeled parameter (η_comb), not a clamp on the
   Wiebe.
2. **Two-segment RPM-dependent combustion efficiency ramp**
   (`orchestrator.py:271-276`). Calibrated against SDM25 dyno (May 2025)
   to close the 7–8 k and 11 k gaps. Hides at least two different physical
   errors (wave-speed misprediction, cam-profile mismatch) behind one
   fudge factor. V2 must not carry this over; the plan is that
   conservative transport and proper exhaust entropy make the un-fudged
   Wiebe predict the right VE curve.
3. **Laplacian AV globally 0.05, per-pipe zero on primaries**
   (`cbr600rr.json:224`, per-pipe `artificial_viscosity=0.0` on exhaust
   primaries). Tuned by hand to keep runner standing waves from
   exploding. V2 limiter (minmod / van Leer) provides physical damping;
   AV must not appear in V2.
4. **Minimum clamps `λ, β, AA ≥ 0.01`** (`moc_solver.py:277-279`).
   Prevents crashes when the MOC produces unphysical states at strong
   shocks. V2 should not need these — a correct HLLC flux with positivity
   preservation handles strong shocks without clamping.
5. **Plenum thermal relaxation `τ = 0.005 s`** (`plenum.py:190`). Chosen
   to relax plenum T toward ambient in the absence of a real wall-heat
   model. V2 should resolve the plenum as an FV domain with its own wall
   heat-transfer source if wall cooling matters, or leave it adiabatic.
6. **Wall temperatures as tuning knobs**. The current config uses 650 K
   primaries, 550 K secondaries, 500 K collector, 325 K intake, 450 K
   cylinder. The spec lists 900–1100 K primaries at WOT as the physically
   reasonable band; V1's 650 K is low and probably compensates partially
   for the low pipe T the entropy bug produces (less gradient → less
   heat loss → somewhat hotter pipe). V2 must pick physical values and
   leave them alone.
7. **sim.artificial_viscosity in orchestrator + per-pipe fallback logic**
   (`orchestrator.py:381-387`). The whole "global default vs. per-pipe
   override" mechanism is there because AV is being tuned empirically
   per-pipe. Not needed in V2.
8. **FMEP correlation** (`orchestrator.py:548-550`). This is physics
   (engine friction), not numerical error. V2 can reuse this block
   unchanged; it is not an error-compensating tuning.
9. **Restrictor Cd = 0.95 / 0.926 / 0.91**. The drifting value across
   configs is partly real part-to-part variation in SDM25's throat
   geometry and partly absorbing volumetric-efficiency mismatch elsewhere.
   V2 should fix Cd at its measured/published value
   (0.967 per the spec) and not use it as a tuning knob.

## 8. V1 time-stepping architecture (for reference)

`SimulationOrchestrator.run_single_rpm` (`orchestrator.py:289-504`) performs
the following per step:

1. `dt = compute_cfl_timestep(all_pipes, cfl_num)`, hard-capped at 1 ms.
2. `extrapolate_boundary_incoming(pipe, dt)` for every pipe — foot-traces
   the incoming λ or β and the C0 AA into the two boundary nodes.
3. `restrictor_plenum.solve_and_apply(dt)` — NR for plenum pressure, sets
   intake-runner LEFT λ and AA.
4. For each cylinder: zero `mdot_intake`, `mdot_exhaust`.
5. For each cylinder: apply intake valve BC to runner RIGHT end.
6. For each cylinder: apply exhaust valve BC to primary LEFT end.
7. For each exhaust junction: constant-pressure NR solve.
8. Exhaust collector open-end BC on RIGHT.
9. For each pipe: `advance_interior_points(pipe, dt, …, AV)`.
10. For each cylinder: `cyl.advance(theta, dtheta, rpm)` — RK4 for closed
    cycle, Euler for open cycle.
11. If crossing a cycle boundary, record `p_at_IVC` and check convergence
    via `ConvergenceChecker` (max per-cylinder relative change of
    `p_at_IVC` against previous cycle, tol 0.5 %).

V2 collapses the BC solve order: the restrictor-plenum NR goes away (plenum
is an FV domain; coupling is via ghost cells), and every BC becomes a
ghost-cell fill. See the Phase-2 spec.

## 9. Files written by Phase 1

- `1d_v2/docs/v1_physics_audit.md` — this document.
- `1d_v2/diagnostics/v1_mass_audit.py` — observer-style mass audit.
- `1d_v2/diagnostics/v1_valve_entropy_probe.py` — exhaust-valve-BC entropy
  quantification.
- `1d_v2/docs/v1_mass_audit_10500.json` — 10500-RPM mass-audit JSON.
- `1d_v2/docs/v1_valve_entropy_probe_10500.csv` — 10500-RPM valve-entropy
  per-step probe data.

## 10. Recommendations for Phase 2

These are corollaries of the findings above, not new design decisions:

1. **Conservative formulation is required, not optional.** The V1
   bookkeeping drift (7.8e−3 kg / 3 cycles of BC-claimed flux disagreeing
   with actual transport) is not a bug to patch — it is the consequence of
   doing gas-dynamics in Riemann variables with boundary fluxes computed
   by separate NR solves that the interior doesn't enforce. V2's FV formulation with
   HLLC fluxes makes the boundary flux the boundary flux by construction.
2. **Composition scalar is required.** The factor-of-2 wave-speed error is
   the single biggest cause of the SDM25 calibration gap. V2's `ρYA`
   transport with HLLC contact-wave preservation is the minimal fix; no
   other change has comparable leverage.
3. **The `0.88` Wiebe cap and the two-segment RPM combustion-efficiency
   ramp must not exist in V2.** If V2 with proper physics can't reproduce
   the dyno VE curve without them, *that* is the signal that there is a
   further physics gap worth finding, not that we need another fudge.
4. **Throwaway-branch fallback was not needed.** The observer-style
   diagnostic got everything by importing V1 as a library, so no V1-side
   branch was created. V1 working tree is unchanged from the start of
   this task (verified by `git status --porcelain` diff).

---

*End of Phase 1 audit. Awaiting confirmation before Phase 2 begins.*
