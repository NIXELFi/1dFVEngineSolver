# SDM26 configuration reference

Every physical parameter in the SDM26 model is a field on `SDM26Config`.
Default construction reproduces the baseline CBR600RR geometry used in
the Phase 3 comparison. Any field can be overridden by keyword argument:

```python
from models.sdm26 import SDM26Config, SDM26Engine
cfg = SDM26Config(primary_length=0.35, primary_diameter_out=0.038)
engine = SDM26Engine(cfg)
engine.run_single_rpm(10500, n_cycles=25, stop_at_convergence=True)
```

All validation runs at `SDM26Config.__post_init__`, so an invalid value
raises immediately rather than producing silent garbage downstream.

## Pipe taper

Each FV pipe (runner, primary, secondary, collector) accepts a
`*_diameter_in` and an optional `*_diameter_out`. If `_out` is `None`
the pipe is straight. If `_out` is set the pipe is a linearly-tapered
cone (diameter varies linearly with x; area varies as D²).

Aggressive tapers (diameter ratio > 3×) emit a `UserWarning` because
they can produce strong wave reflections at the taper and shock
formation under supersonic flow. The simulator still runs; the warning
is advisory.

## Parameter surface

### Engine geometry
`bore`, `stroke`, `con_rod`, `CR`, `n_cylinders`, `firing_order`,
`firing_interval`.

Validation: bore/stroke/con_rod > 0; CR > 1; con_rod ≥ stroke/2
(slider-crank realisability).

### Intake runners (×`n_cylinders`)
Symmetric case — all runners identical:
`runner_length`, `runner_diameter_in`, `runner_diameter_out`,
`runner_n_cells`, `runner_wall_T`.

Asymmetric — per-cylinder lists (length = `n_cylinders`):
`runner_lengths`, `runner_diameters_in`, `runner_diameters_out`,
`runner_wall_Ts`. A list of `None` for `_out` means that runner is
straight.

### Exhaust primaries (×`n_cylinders`)
Same pattern as runners:
`primary_length`, `primary_diameter_in`, `primary_diameter_out`,
`primary_n_cells`, `primary_wall_T`,
and per-cylinder lists
`primary_lengths`, `primary_diameters_in`, `primary_diameters_out`,
`primary_wall_Ts`.

### Exhaust secondaries (×2)
`secondary_length`, `secondary_diameter_in`, `secondary_diameter_out`,
`secondary_n_cells`, `secondary_wall_T`,
plus per-secondary lists (length 2) for asymmetry.

### Exhaust collector (×1)
`collector_length`, `collector_diameter_in`, `collector_diameter_out`,
`collector_n_cells`, `collector_wall_T`.

### Plenum
1D FV representation of matched volume. The geometry is a straight pipe
of length `plenum_length` and area `plenum_volume / plenum_length`.
Fields: `plenum_volume`, `plenum_length`, `plenum_n_cells`,
`plenum_wall_T`.

### Restrictor
`restrictor_throat_diameter`, `restrictor_Cd`. Cd must be in (0, 1].

### Ambient
`p_ambient`, `T_ambient`.

### Combustion (Wiebe + physics)
`wiebe_a`, `wiebe_m`, `combustion_duration`, `spark_advance`,
`ignition_delay`, `eta_comb`, `q_lhv`, `afr_target`,
`T_wall_cylinder`.

Validation: `eta_comb` in (0, 1] (no 0.88 cap); `wiebe_a > 0`;
`wiebe_m ≥ 0`; all the rest > 0.

### Woschni coefficients
`woschni_C1_gas_exchange`, `woschni_C1_compression`,
`woschni_C1_combustion`, `woschni_C2_combustion`.

Defaults are Heywood/Woschni 1967.

### Intake valve
`intake_valve_diameter`, `intake_valve_max_lift`,
`intake_valve_open_angle`, `intake_valve_close_angle`,
`intake_valve_seat_angle`, `intake_n_valves`,
`intake_ld_table`, `intake_cd_table`.

### Exhaust valve
Symmetric: `exhaust_*` versions of the above.

Validation: ld_table and cd_table must be the same length; cd in [0,1];
ld monotone increasing; open/close window > 0° and ≤ 360°; L/D at max
lift ≤ 0.5 (warning if higher).

### Numerics
`cfl`, `limiter`. `cfl` in (0, 1]. `limiter` is `LIMITER_MINMOD` (0),
`LIMITER_VAN_LEER` (1), or `LIMITER_SUPERBEE` (2).

## Wall-temperature policy

Per the Phase 1 audit recommendation, wall temperatures are physical
values (not tuned to compensate for solver error). Defaults:

| Pipe | T_wall [K] |
|---|---:|
| Intake runners | 325 |
| Plenum | 320 |
| Exhaust primaries | 1000 |
| Exhaust secondaries | 800 |
| Exhaust collector | 700 |
| Cylinder (Woschni) | 450 |

Validation: `200 K ≤ T_wall ≤ 1500 K`.

## Parametric sweeps

`models.parameter_sweep` provides one-call-per-field sweep helpers:

```python
from models.sdm26 import SDM26Config
from models.parameter_sweep import sweep_parameter, taper_primary

cfg = SDM26Config()

# Primary-length sweep at 10500 RPM (tuned-length study)
rows = sweep_parameter(cfg, "primary_length",
                       [0.20, 0.25, 0.30, 0.35, 0.40],
                       rpm=10500.0)

# Collector taper sweep (diverging megaphone)
rows = taper_primary(cfg, [None, 0.036, 0.040, 0.044], rpm=10500.0)
```

Each returned row has `imep_bar`, `ve_atm`, `EGT_valve_K`,
`indicated_power_kW`, `converged_cycle`, `nonconservation_max`. Costs
~3 s per point after the first (Numba warms up on the first call).

## Common sweeps, typical ranges

| Parameter | Typical range | Units | Physical effect |
|---|---|---|---|
| `primary_length` | 0.15 – 0.45 | m | exhaust wave-timing / tuned length |
| `primary_diameter_in` | 0.028 – 0.040 | m | blowdown flow capacity |
| `primary_diameter_out` | in × 1.0 – 1.3 | m | diffuser recovery |
| `runner_length` | 0.15 – 0.35 | m | intake ram-tuning |
| `runner_diameter_in` | 0.030 – 0.045 | m | intake velocity / VE trade |
| `collector_diameter_out` | 0.045 – 0.065 | m | megaphone / back-pressure |
| `restrictor_Cd` | 0.92 – 0.97 | — | measured in a flow bench |
| `plenum_volume` | 0.001 – 0.003 | m³ | intake capacitance |
| `spark_advance` | 15 – 35 | deg BTDC | MBT timing |
| `eta_comb` | 0.90 – 0.98 | — | combustion efficiency |

## What validation does not check

The simulator does not verify:

- That your geometry is physically buildable (e.g. primaries all fit
  without crossing each other).
- That valve timing is compatible with piston motion (interference).
- Fuel rail sizing, injector response, or any of the EFI-side
  concerns.
- Structural integrity or thermal stress on the chosen geometry.

Those are outside the scope of a 1D gas-dynamics simulator. The
simulator will tell you what the gas does for a given geometry. You
tell the machinist whether the geometry is buildable.
