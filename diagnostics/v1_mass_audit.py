"""V1 mass conservation diagnostic — observer-style.

EXCEPTIONAL FILE: this is the ONLY place in 1d_v2/ that is allowed to import
from the V1 codebase at 1d/. The diagnostic is conceptually part of the Phase
1 audit, not part of the V2 solver. V2 solver code (solver/, bcs/, cylinder/,
geometry/, models/, tests/) MUST NOT import from 1d/. If you are tempted to
add another cross-repo import, stop and reconsider.

Approach: the diagnostic rebuilds the exact same simulation state V1 does, but
runs its own instrumented time loop that mirrors SimulationOrchestrator.run_single_rpm.
After each stage (BC, interior advance, cylinder advance) it re-integrates the
pipe / plenum / cylinder mass book and attributes any bookkeeping drift to the
stage that produced it. V1 source files are never modified.

Run:
    python -m diagnostics.v1_mass_audit --rpm 10500 --cycles 2

Outputs a JSON summary to stdout (and the path passed to --out, if any) with:
  - total system mass drift per cycle
  - drift attributable to: restrictor_plenum, intake_valves, exhaust_valves,
    junctions, collector_open_end, pipe_interior (MOC), cylinder_advance
  - per-pipe mass-flux residual at each interior step
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add the V1 repo root to sys.path so `import engine_simulator...` resolves.
# V1 lives alongside 1d_v2/ at ~/Developer/1d/.
_V1_ROOT = Path(__file__).resolve().parents[2] / "1d"
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

# ---- Imports from V1 (the ONE sanctioned cross-repo import site) ---------
from engine_simulator.boundaries.base import PipeEnd  # noqa: E402
from engine_simulator.config.engine_config import EngineConfig  # noqa: E402
from engine_simulator.gas_dynamics.cfl import compute_cfl_timestep  # noqa: E402
from engine_simulator.gas_dynamics.moc_solver import (  # noqa: E402
    advance_interior_points,
    extrapolate_boundary_incoming,
)
from engine_simulator.simulation.engine_cycle import EngineCycleTracker  # noqa: E402
from engine_simulator.simulation.orchestrator import SimulationOrchestrator  # noqa: E402
# --------------------------------------------------------------------------


def pipe_mass(pipe) -> float:
    """Integrate mass in a pipe using trapezoidal rule on the node grid.

    rho and area are defined at nodes; dx is uniform. Mass = ∫ρA dx.
    """
    rho = pipe.rho
    area = pipe.area
    dx = pipe.dx
    # Trapezoidal rule over n nodes
    integrand = rho * area
    return float(dx * (0.5 * integrand[0] + integrand[1:-1].sum() + 0.5 * integrand[-1]))


def system_mass(sim: SimulationOrchestrator) -> dict[str, float]:
    """Tally mass in every component of the system.

    Returns a dict with keys for each pipe name, plus plenum and cylinders.
    """
    book = {}
    for p in sim.all_pipes:
        book[f"pipe::{p.name}"] = pipe_mass(p)
    book["plenum"] = float(sim.restrictor_plenum.m)
    for i, c in enumerate(sim.cylinders):
        book[f"cylinder::{i}"] = float(c.m)
    return book


def sum_book(book: dict[str, float]) -> float:
    return float(sum(book.values()))


@dataclass
class StageDelta:
    """Mass change attributable to one stage of the update."""

    stage: str
    delta_pipes: float = 0.0  # kg
    delta_plenum: float = 0.0
    delta_cyl: float = 0.0
    reported_mdot_dt: float = 0.0  # what the BC *claims* to have moved
    # Mass attributed to this BC's inflow/outflow at its ports. Positive = into system.
    port_inflow_dt: float = 0.0

    @property
    def delta_total(self) -> float:
        return self.delta_pipes + self.delta_plenum + self.delta_cyl

    @property
    def unaccounted(self) -> float:
        """Drift that is not explained by the BC's reported port flux.

        Pipe-interior and cylinder-advance stages should have zero port flux,
        so any delta is pure numerical leak. BC stages should have
        delta_total ≈ port_inflow_dt.
        """
        return self.delta_total - self.port_inflow_dt


@dataclass
class StepReport:
    t: float
    theta: float
    dt: float
    total_before: float
    total_after: float
    stages: list[StageDelta] = field(default_factory=list)


@dataclass
class AuditRollup:
    """Aggregated drift per stage over the entire run."""

    per_stage: dict[str, float] = field(default_factory=dict)
    per_stage_abs: dict[str, float] = field(default_factory=dict)
    total_drift: float = 0.0
    total_mass_flowed_through_restrictor: float = 0.0
    n_steps: int = 0

    def record(self, step: StepReport) -> None:
        self.n_steps += 1
        self.total_drift += step.total_after - step.total_before
        for st in step.stages:
            self.per_stage[st.stage] = self.per_stage.get(st.stage, 0.0) + st.unaccounted
            self.per_stage_abs[st.stage] = self.per_stage_abs.get(st.stage, 0.0) + abs(st.unaccounted)


def run_audit(
    config_name: str | None = None,
    rpm: float = 10500.0,
    n_cycles: int = 2,
    warmup_cycles: int = 0,
    verbose: bool = True,
    cfl_override: float | None = None,
) -> dict[str, Any]:
    """Run V1 with instrumented time loop and accumulate a mass audit.

    Args:
        config_name: one of "default" (cbr600rr.json built-in) or a path-like
            recognized by V1's EngineConfig. If None, uses the built-in default.
        rpm: target engine speed (rpm).
        n_cycles: number of cycles to run (exclusive of warmup).
        warmup_cycles: cycles to run before starting to accumulate stats.
        cfl_override: if set, overrides the CFL safety factor.
    """
    cfg = EngineConfig() if config_name in (None, "default") else EngineConfig.load(config_name)
    sim = SimulationOrchestrator(cfg)
    sim._reinitialize(rpm)

    cfl_num = cfl_override if cfl_override is not None else cfg.simulation.cfl_number
    tracker = EngineCycleTracker(rpm)

    rollup = AuditRollup()
    step_reports: list[StepReport] = []  # keep a thin tail for final report
    cycle_counter = 0
    t = 0.0

    max_steps = int(1e6)
    step = 0
    t_wall_start = time.time()

    # Running totals for port flux into/out of the overall system
    mass_in_restrictor = 0.0
    mass_out_collector = 0.0

    while cycle_counter < warmup_cycles + n_cycles and step < max_steps:
        step += 1
        dt = compute_cfl_timestep(sim.all_pipes, cfl_num)
        dt = min(dt, 1e-3)
        dtheta = tracker.advance(dt)
        theta = tracker.theta

        # ---- Snapshot before step ------------------------------------------
        book_before = system_mass(sim)
        total_before = sum_book(book_before)
        step_report = StepReport(t=t, theta=theta, dt=dt, total_before=total_before, total_after=total_before)

        # ---- Stage 0: extrapolate incoming Riemann vars (no mass change) ---
        for pipe in sim.all_pipes:
            extrapolate_boundary_incoming(pipe, dt)
        # The extrapolation changes boundary values but we haven't called update_derived;
        # pipe.rho at nodes hasn't been refreshed yet in general. To keep the accounting
        # honest, force a refresh now so the "post-extrap" book reflects the actual rho.
        for pipe in sim.all_pipes:
            pipe.update_derived()
        book_post_extrap = system_mass(sim)
        step_report.stages.append(StageDelta(
            stage="extrapolate_boundary",
            delta_pipes=sum(book_post_extrap[k] - book_before[k]
                             for k in book_post_extrap if k.startswith("pipe::")),
        ))

        # ---- Stage 1: restrictor-plenum-runner coupled solve ---------------
        book_pre = system_mass(sim)
        sim.restrictor_plenum.solve_and_apply(dt)
        # Refresh derived on runners (apply sets lam/AA but not necessarily p/T/rho)
        for p in sim.intake_runners:
            p.update_derived()
        book_post = system_mass(sim)
        mdot_restr = sim.restrictor_plenum.last_mdot_restrictor

        stage = StageDelta(
            stage="restrictor_plenum",
            reported_mdot_dt=mdot_restr * dt,
            port_inflow_dt=mdot_restr * dt,  # mass enters system from atmosphere via restrictor
        )
        for k in book_pre:
            if k.startswith("pipe::intake"):
                stage.delta_pipes += book_post[k] - book_pre[k]
            elif k == "plenum":
                stage.delta_plenum += book_post[k] - book_pre[k]
        step_report.stages.append(stage)
        mass_in_restrictor += mdot_restr * dt

        # ---- Stage 2: intake valve BCs (runner RIGHT -> cylinder) ----------
        for c in sim.cylinders:
            c.mdot_intake = 0.0
            c.mdot_exhaust = 0.0
        book_pre = system_mass(sim)
        for i in range(cfg.n_cylinders):
            sim.intake_valve_bcs[i].apply(
                sim.intake_runners[i], PipeEnd.RIGHT, dt, theta_deg=theta, rpm=rpm,
            )
        for p in sim.intake_runners:
            p.update_derived()
        book_post = system_mass(sim)
        # intake valve BCs only update pipe boundary values; cylinder mass update
        # happens during cyl.advance(). The reported mdot_intake is still 0 here in
        # system-mass terms until the cylinder integrates it. We record the pipe-side
        # delta and the claimed flux so we can compare later.
        claimed_intake_flux = 0.0
        for c in sim.cylinders:
            # mdot_intake has been accumulated by intake valve BC
            claimed_intake_flux += c.mdot_intake * dt
        stage = StageDelta(
            stage="intake_valve_bcs",
            reported_mdot_dt=claimed_intake_flux,
            port_inflow_dt=0.0,  # purely internal transfer pipe->cylinder; net system inflow 0
        )
        for k in book_pre:
            if k.startswith("pipe::intake"):
                stage.delta_pipes += book_post[k] - book_pre[k]
        step_report.stages.append(stage)

        # ---- Stage 3: exhaust valve BCs (cylinder -> primary LEFT) ---------
        book_pre = system_mass(sim)
        for i in range(cfg.n_cylinders):
            sim.exhaust_valve_bcs[i].apply(
                sim.exhaust_primaries[i], PipeEnd.LEFT, dt, theta_deg=theta, rpm=rpm,
            )
        for p in sim.exhaust_primaries:
            p.update_derived()
        book_post = system_mass(sim)
        claimed_exhaust_flux = 0.0
        for c in sim.cylinders:
            claimed_exhaust_flux += c.mdot_exhaust * dt
        stage = StageDelta(
            stage="exhaust_valve_bcs",
            reported_mdot_dt=claimed_exhaust_flux,
            port_inflow_dt=0.0,
        )
        for k in book_pre:
            if k.startswith("pipe::exhaust_primary"):
                stage.delta_pipes += book_post[k] - book_pre[k]
        step_report.stages.append(stage)

        # ---- Stage 4: exhaust junctions -----------------------------------
        book_pre = system_mass(sim)
        for junc in sim.exhaust_junctions:
            junc.apply(dt)
        # Update derived on all exhaust pipes
        for p in sim.exhaust_primaries + sim.exhaust_secondaries + [sim.exhaust_collector]:
            p.update_derived()
        book_post = system_mass(sim)
        stage = StageDelta(
            stage="exhaust_junctions",
            port_inflow_dt=0.0,  # internal redistribution
        )
        for k in book_pre:
            if k.startswith("pipe::exhaust"):
                stage.delta_pipes += book_post[k] - book_pre[k]
        step_report.stages.append(stage)

        # ---- Stage 5: collector open end ----------------------------------
        book_pre = system_mass(sim)
        sim.exhaust_open_bc.apply(sim.exhaust_collector, PipeEnd.RIGHT, dt)
        sim.exhaust_collector.update_derived()
        book_post = system_mass(sim)
        # Estimate collector exit flux from the collector's RIGHT-end state
        coll = sim.exhaust_collector
        mdot_coll = float(coll.rho[-1] * coll.u[-1] * coll.area[-1])
        # Outflow is positive u at RIGHT end
        stage = StageDelta(
            stage="collector_open_end",
            reported_mdot_dt=mdot_coll * dt,
            port_inflow_dt=-mdot_coll * dt,  # mass leaves system
        )
        stage.delta_pipes = book_post[f"pipe::{coll.name}"] - book_pre[f"pipe::{coll.name}"]
        step_report.stages.append(stage)
        mass_out_collector += mdot_coll * dt

        # ---- Stage 6: pipe interior (MOC advance) --------------------------
        book_pre = system_mass(sim)
        global_av = cfg.simulation.artificial_viscosity
        for pipe in sim.all_pipes:
            av = pipe.artificial_viscosity if pipe.artificial_viscosity >= 0 else global_av
            advance_interior_points(pipe, dt, include_sources=True, artificial_viscosity=av)
        book_post = system_mass(sim)
        stage = StageDelta(stage="pipe_interior_MOC", port_inflow_dt=0.0)
        for k in book_pre:
            if k.startswith("pipe::"):
                stage.delta_pipes += book_post[k] - book_pre[k]
        step_report.stages.append(stage)

        # ---- Stage 7: cylinder advance -------------------------------------
        book_pre = system_mass(sim)
        for cyl in sim.cylinders:
            cyl.advance(theta, dtheta, rpm)
        book_post = system_mass(sim)
        # Cylinder advance moves mass between pipes (via claimed intake/exhaust
        # mdot) and cylinder; net system mass change should be ~0 if cylinders
        # apply mdot consistent with what valve BCs claimed.
        stage = StageDelta(stage="cylinder_advance", port_inflow_dt=0.0)
        for k in book_pre:
            if k == "plenum":
                continue
            if k.startswith("cylinder::"):
                stage.delta_cyl += book_post[k] - book_pre[k]
            elif k.startswith("pipe::"):
                stage.delta_pipes += book_post[k] - book_pre[k]
        step_report.stages.append(stage)

        # ---- End of step ---------------------------------------------------
        total_after = sum_book(system_mass(sim))
        step_report.total_after = total_after
        rollup.record(step_report)
        # Keep only a tail of step records
        if len(step_reports) > 2000:
            step_reports = step_reports[-1000:]
        step_reports.append(step_report)

        t += dt

        new_cycle = int(theta / 720.0)
        if new_cycle > cycle_counter:
            if verbose:
                drift_abs = total_after - rollup.total_drift  # (redundant; just debug)
                print(f"  cycle {new_cycle}: total={total_after:.6e} kg, cum_drift={rollup.total_drift:.3e} kg, "
                      f"mass_in={mass_in_restrictor:.3e}, mass_out={mass_out_collector:.3e}")
            cycle_counter = new_cycle

    rollup.total_mass_flowed_through_restrictor = mass_in_restrictor

    # ---- Assemble summary -----------------------------------------------------
    summary = {
        "rpm": rpm,
        "n_cycles": n_cycles,
        "warmup_cycles": warmup_cycles,
        "n_steps": rollup.n_steps,
        "wall_time_s": time.time() - t_wall_start,
        "total_mass_drift_kg": rollup.total_drift,
        "mass_in_restrictor_kg": mass_in_restrictor,
        "mass_out_collector_kg": mass_out_collector,
        "expected_drift_from_ports_kg": mass_in_restrictor - mass_out_collector,
        "drift_after_port_accounting_kg": rollup.total_drift - (mass_in_restrictor - mass_out_collector),
        "per_stage_unaccounted_kg": dict(rollup.per_stage),
        "per_stage_unaccounted_abs_kg": dict(rollup.per_stage_abs),
    }

    # Leak rate per BC, normalized by total mass flowed through restrictor
    if abs(mass_in_restrictor) > 0:
        summary["per_stage_leak_fraction_of_throughput"] = {
            k: v / mass_in_restrictor for k, v in rollup.per_stage.items()
        }

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rpm", type=float, default=10500.0)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="path to write JSON summary")
    ap.add_argument("--cfl", type=float, default=None)
    args = ap.parse_args()

    t0 = time.time()
    summary = run_audit(
        rpm=args.rpm,
        n_cycles=args.cycles,
        warmup_cycles=args.warmup,
        cfl_override=args.cfl,
    )
    summary["wall_time_total_s"] = time.time() - t0

    txt = json.dumps(summary, indent=2, default=float)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
