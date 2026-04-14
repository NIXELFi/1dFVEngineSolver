# Asymmetry investigation: same valve BC, different R

**Question.** Phase A measured R_valve(exhaust, linear, wall-far) ≈ −0.10 and R_valve(intake, linear, wall-far) ≈ −0.80. Same `bcs/valve.py:fill_valve_ghost` code path. Is the asymmetry a real regime-dependent feature of the broken BC (Explanation 1) or a measurement artifact tied to the opposite-end BC (Explanation 2)?

## Initial test: swap the far-end BC

Swap the wall right-end BC in A1 for the same `fill_subsonic_inflow_*` plenum BC that A2 uses, and compare R_valve(exhaust) between configurations.

| Test | far end | R_far (impulse) | R_valve (impulse) |
|---|---|---|---|
| A1 exhaust          | wall   | +1.222 | -0.103 |
| A1 exhaust          | plenum | +0.229 | +0.997 |
| A2 intake (Phase A) | wall   | (≈ +0.91) | -0.797 |

**Surprise — but a measurement artifact, not a diagnosis change.** R_valve(exhaust, plenum-far) reads +0.997, far from the wall-far -0.103. Inspecting the probe time series (`a1_asym_plenum_linear_probes.png`) shows that the wave dies on first contact with the plenum BC and the pipe then reaches a **steady-state pressure offset** of about +150 Pa (because both ends are open absorbers with mismatched reservoir specs). The windowed extremum at the expected A2 / A3 arrival times then picks up that static drift, producing a meaningless +1 ratio. The plenum-far A1 setup cannot measure R_valve cleanly — the wave doesn't survive long enough to reach the valve a second time.

## Cleaner cross-check: same code path, same wall reference, opposite-sign perturbation

Force the exhaust valve into the same regime that intake operates in, by perturbing the cylinder pressure **downward** instead of upward. This launches a rarefaction wave at the exhaust valve while keeping the wall right-end BC and the same `fill_valve_ghost` code path. If the wave-type (compression vs rarefaction) is what drives the asymmetry, R_valve(exhaust, rarefaction) should match R_valve(intake, rarefaction) ≈ −0.80.

| Test | far end | cyl perturbation | wave type | R_valve |
|---|---|---|---|---|
| A1 exhaust  | wall | +2 kPa     | compression | -0.103 |
| A1 exhaust  | wall | −2 kPa     | rarefaction | -0.761 |
| A1 exhaust  | wall | −300 kPa   | rarefaction | -0.553 |
| A2 intake   | wall | −2 kPa     | rarefaction | -0.797 |

## Diagnosis (refined)

R_valve(exhaust, rarefaction) = -0.761 matches R_valve(intake, rarefaction) = -0.797 within measurement noise (≈ 0.15, the MUSCL-HLLC dissipation floor we calibrated against A2's wall-ref control). R_valve(exhaust, compression) = -0.103 is the outlier.

**Conclusion.** The asymmetry is between **wave types** (compression vs rarefaction), not between intake and exhaust valves. The `fill_valve_ghost` code path absorbs compressions but mostly-reflects rarefactions. Both the intake and exhaust valves would absorb if they saw a compression in the engine cycle; both would reflect if they saw a rarefaction.

**Mechanism.** When a compression wave arrives at an exhaust valve with cyl held at atmospheric, p_pipe_face >> p_cyl. The orifice equation produces a large outflow mdot, which sets the ghost velocity to a large magnitude. With p_ghost = p_pipe (zero gradient) AND a strongly prescribed velocity, the BC is essentially a Dirichlet velocity condition for the interior — it imposes the steady-orifice flow rate and absorbs whatever acoustic content arrives. When a rarefaction arrives, p_pipe_face < p_cyl, the orifice runs in the opposite direction with a *small* mdot (only 30 % overpressure differential typical), so the prescribed velocity at the ghost is small. With small u_ghost and p_ghost = p_pipe, the BC looks much closer to a wall (u ≈ 0, p matched), which reflects cleanly.

**Implication for Phase C1.** The characteristic-based fix should eliminate the asymmetry. Both wave types should produce similar R magnitudes (probably both in the −0.5 to −0.9 range, the physically correct value for an open valve onto a fixed-pressure reservoir). If the post-fix R differs by more than 0.2 between compression and rarefaction tests, the characteristic formulation has a real bug we missed and we stop again.
