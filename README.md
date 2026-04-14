# 1D FV Engine Solver

A 1D compressible-flow engine simulator built around a finite-volume scheme
with an HLLC Riemann solver and advected composition scalar. Targets the
Honda CBR600RR I4 with the FSAE 20 mm intake restrictor as used on the
Sun Devil Motorsports SDM26 car.

This is a parallel rewrite of an earlier MOC-based solver (see
[`1dMOCEngineSolver`](https://github.com/NIXELFi/1dMOCEngineSolver)) that
had a structural entropy-transport limitation at the exhaust valve
boundary, underpredicting exhaust gas temperatures by ~1000 K and
mispredicting exhaust wave speeds by ~2×. This code fixes those
limitations by construction: conservative FV + HLLC resolves contact
waves correctly, composition is transported as a conservative scalar
ρY on the HLLC contact wave, and 0D junction control volumes replace
the characteristic-based junction coupling.

See [`docs/v2_vs_v1_comparison.md`](docs/v2_vs_v1_comparison.md) for a
direct comparison against the MOC solver, including plots and the
profiling breakdown.

## Scope

- 1D quasi-compressible Euler with friction, wall heat transfer, and
  composition transport (unburned air, burned stoichiometric
  gasoline-air at x_b=1).
- Finite volume, second-order MUSCL-Hancock, HLLC Riemann solver with
  Einfeldt-Batten wave speeds.
- Cylinder 0D model with Wiebe combustion and Woschni wall heat.
- Ghost-cell boundary conditions for restrictor, valves, and junctions.
- Numba `@njit` on the interior kernel; serial execution.

Not in scope: multi-dimensional effects, detailed chemistry beyond a
two-species frozen transport model, spray/droplet physics, turbo-
charging, or anything that requires more than 1D acoustic propagation.

## Not production validated

**This is an educational and research code.** It has been validated
against analytic shock-tube solutions (Sod, Lax, 123), isentropic
choked-nozzle mass flow, closed-domain conservation, and quasi-1D
nozzle steady state, all to the tolerances documented in
`tests/`. It has NOT been validated against a physical dyno measurement
of the target engine yet.

**Anyone using this code for actual engine design decisions should
validate the relevant predictions against their own dyno data.** The
scheme is conservative to machine precision and the physics is correct
by construction, but coefficients like combustion efficiency, friction
correlations, and valve discharge coefficients carry empirical
uncertainty. A 1D simulation predicts trends well and absolute numbers
approximately; treating it as ground truth for geometry decisions
without physical validation is a way to blow up an engine.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `numba`, `pytest`, `matplotlib`.
Python 3.9+ tested.

## Quick start

```python
from models.sdm26 import SDM26Engine, SDM26Config

engine = SDM26Engine(SDM26Config())
result = engine.run_single_rpm(10500, n_cycles=25, verbose=True)
for stats in result["cycle_stats"]:
    print(f"cycle {stats['cycle']}: IMEP={stats['imep_bar']:.2f} bar, "
          f"EGT={stats['EGT_mean']:.0f} K, "
          f"nonconservation={stats['nonconservation']:.2e}")
```

Full RPM sweep:

```bash
.venv/bin/python3 -m models.sweep docs/v2_sweep.json
```

V1 comparison (runs V1 separately via the diagnostics-exception
sanctioned import) and full report:

```bash
.venv/bin/python3 -m diagnostics.run_v1_sweep docs/v1_sweep.json
.venv/bin/python3 -m diagnostics.make_comparison
```

## Repository layout

```
solver/            FV scheme — HLLC, MUSCL-Hancock, sources, state
bcs/               Ghost-cell boundary conditions
cylinder/          0D cylinder model — gas properties, Wiebe, Woschni, valves
models/            Full engine assemblies and sweep drivers
geometry/          (reserved for future parametric geometry)
diagnostics/       V1 audits and comparison scripts — the sole V1-import site
docs/              Physics audit, conservation discipline, comparison report
tests/             Validation suite — Sod/Lax/123, contacts, conservation,
                   nozzle, friction+heat, choked restrictor, junction CV
configs/           (reserved for config files)
```

## Tests

```bash
.venv/bin/pytest tests/
```

40+ tests, all passing. Mass, energy, and composition are conserved to
machine precision on closed-domain sealed runs. HLLC preserves stationary
and moving contact discontinuities. MUSCL-Hancock gives second-order L1
convergence on smooth regions of Sod.

## Conservation metric discipline

See [`docs/conservation_metrics.md`](docs/conservation_metrics.md). TL;DR:

- `nonconservation_residual_kg` is the real conservation metric.
  It should be at machine precision (O(1e-18)) always.
- `raw_cycle_drift_kg` is a convergence diagnostic, not a conservation
  metric. Nonzero during startup transient is physics, not a bug.

Gate on the first. Diagnose cyclic convergence with the second.

## References

- Toro, E.F., *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer 2009. Chapters 10 (HLLC), 14 (MUSCL-Hancock),
  16 (source terms).
- Winterbone, D.E. & Pearson, R.J., *Design Techniques for Engine
  Manifolds*, Wiley 1999. Junction-cell formulation.
- LeVeque, R.J., *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge 2002.
- Heywood, J.B., *Internal Combustion Engine Fundamentals*, McGraw-Hill
  1988. Wiebe and Woschni coefficients.

## License

MIT (see `LICENSE`).

## Acknowledgements

Built for Sun Devil Motorsports (SDM26, Arizona State University, FSAE).
