# Conservation metrics in the V2 engine simulator — read this before worrying

There are two quantities that look similar but mean entirely different things.
Do not confuse them.

## `nonconservation_residual_kg` — the real conservation metric

Defined as: `raw_cycle_drift − (mass_in_restrictor − mass_out_collector)`
integrated over one cycle. Equivalently, the sum of all face-flux "creation"
and "destruction" that is not accounted for by the external ports (restrictor
and open-end collector) and the internal conservative couplings (valve
cylinder-pipe flux, junction CV flux).

**This is what the "1e-8 kg/cycle" stop-gate was trying to measure.** In V2,
this residual is at machine precision (±O(1e-18)) every cycle at every RPM.
It is zero by construction of the FV scheme: every face flux appears with
equal and opposite signs in the two cells it connects, so the only way the
system can gain or lose mass is through the restrictor (inflow) or the
collector (outflow), both of which are tracked.

**Stop-gate rule:** if `nonconservation_residual_kg` exceeds 1e-15 kg/cycle
at any RPM in any cycle, there is a bug. Stop and investigate. Do not
relax the tolerance — this is a roundoff-order metric and anything larger
is a real error.

## `raw_cycle_drift_kg` — a convergence diagnostic, NOT a conservation metric

Defined as: total system mass at end of cycle − total system mass at start
of cycle, where "total system mass" = Σ(pipe real-cell ρA dx) + Σ(cylinder m)
+ Σ(junction CV M).

This is physically nonzero during the startup transient because the engine
is out of cyclic balance: e.g. cylinders fire their first combustion events
before the intake side has built up flow, so the collector sees mass leaving
before the restrictor can replenish it. The raw drift decays monotonically
as the engine approaches its cyclic attractor.

**Do not treat nonzero raw drift as a bug.** It is physics. At cycle 20 of
a cold-start run at 10500 RPM the raw drift is typically 8e-7 kg, decaying
by a factor of ~1.5 per cycle. By cycle 30–40 it is below 1e-8, not because
the FV scheme got more conservative (it was always machine-precision
conservative) but because the engine finally equilibrated so that
ṁ_restrictor ≈ ṁ_collector when integrated over a cycle.

**Interpretation rule:** raw drift is a convergence diagnostic. When its
magnitude stops decreasing cycle-to-cycle, the engine has reached cyclic
steady state. When it oscillates at a small value (tens of μg per cycle),
that is the residual cycle-to-cycle variation of a multi-cylinder engine
where the four firing events do not exactly repeat because of coupled
acoustic dynamics.

## The two together

- `nonconservation_residual ≈ 1e-18`, `raw_drift ≈ 1e-3` → the scheme is
  conservative; the engine is deep in startup transient. Let it run longer.
- `nonconservation_residual ≈ 1e-18`, `raw_drift ≈ 1e-8` → converged.
- `nonconservation_residual ≈ 1e-3`, `raw_drift ≈ 1e-3` → real leak,
  probably at a BC. Stop and find it.
- `nonconservation_residual ≈ 1e-3`, `raw_drift ≈ 1e-18` → two equal and
  opposite bugs cancelling. Unusual, but possible. Stop and find both.

Track both. Gate on the first. Diagnose convergence with the second.
