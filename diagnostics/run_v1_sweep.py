"""Run V1 RPM sweep and save results to JSON.

EXCEPTIONAL FILE: imports V1 as a library, same exception as v1_mass_audit.py.
V2 solver/BC/cylinder code MUST NOT import from 1d/.

Produces docs/v1_sweep.json with the same schema as docs/v2_sweep.json for
direct comparison in docs/v2_vs_v1_comparison.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_V1_ROOT = Path(__file__).resolve().parents[2] / "1d"
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from engine_simulator.boundaries.base import PipeEnd  # noqa: E402
from engine_simulator.config.engine_config import EngineConfig  # noqa: E402
from engine_simulator.gas_dynamics.cfl import compute_cfl_timestep  # noqa: E402
from engine_simulator.gas_dynamics.moc_solver import (  # noqa: E402
    advance_interior_points, extrapolate_boundary_incoming,
)
from engine_simulator.simulation.engine_cycle import EngineCycleTracker  # noqa: E402
from engine_simulator.simulation.orchestrator import SimulationOrchestrator  # noqa: E402


def _pipe_mass(pipe) -> float:
    return float(pipe.dx * (
        0.5 * pipe.rho[0] * pipe.area[0]
        + (pipe.rho[1:-1] * pipe.area[1:-1]).sum()
        + 0.5 * pipe.rho[-1] * pipe.area[-1]
    ))


def _system_mass(sim: SimulationOrchestrator) -> float:
    m = 0.0
    for p in sim.all_pipes:
        m += _pipe_mass(p)
    m += float(sim.restrictor_plenum.m)
    for c in sim.cylinders:
        m += float(c.m)
    return m


def _egt_at_primary_valve(sim: SimulationOrchestrator) -> float:
    """Approximate V1 EGT at the exhaust primary LEFT node (valve side) —
    average across the 4 primaries. V1 stores T at nodes, so just read T[0]."""
    vals = [float(p.T[0]) for p in sim.exhaust_primaries]
    return float(np.mean(vals))


def run_v1_one_rpm(rpm: float, n_cycles_max: int = 20,
                   convergence_tol_imep: float = 0.005,
                   convergence_min_cycles: int = 8):
    """Run V1 at a single RPM with instrumented time loop (mirrors V1's
    orchestrator) so we can record EGT and system mass for the comparison."""
    cfg = EngineConfig()
    sim = SimulationOrchestrator(cfg)
    sim._reinitialize(rpm)

    tracker = EngineCycleTracker(rpm)
    cfl_num = cfg.simulation.cfl_number

    cycle_stats = []
    prev_cycle = 0
    step_count = 0
    last_mass = _system_mass(sim)
    t_start = time.time()
    converged_cycle = -1

    while True:
        dt = compute_cfl_timestep(sim.all_pipes, cfl_num)
        dt = min(dt, 1e-3)
        dtheta = tracker.advance(dt)
        theta = tracker.theta
        step_count += 1

        for pipe in sim.all_pipes:
            extrapolate_boundary_incoming(pipe, dt)
        sim.restrictor_plenum.solve_and_apply(dt)
        for c in sim.cylinders:
            c.mdot_intake = 0.0
            c.mdot_exhaust = 0.0
        for i in range(cfg.n_cylinders):
            sim.intake_valve_bcs[i].apply(
                sim.intake_runners[i], PipeEnd.RIGHT, dt, theta_deg=theta, rpm=rpm,
            )
            sim.exhaust_valve_bcs[i].apply(
                sim.exhaust_primaries[i], PipeEnd.LEFT, dt, theta_deg=theta, rpm=rpm,
            )
        for junc in sim.exhaust_junctions:
            junc.apply(dt)
        sim.exhaust_open_bc.apply(sim.exhaust_collector, PipeEnd.RIGHT, dt)

        global_av = cfg.simulation.artificial_viscosity
        for pipe in sim.all_pipes:
            av = pipe.artificial_viscosity if pipe.artificial_viscosity >= 0 else global_av
            advance_interior_points(pipe, dt, include_sources=True, artificial_viscosity=av)
        for cyl in sim.cylinders:
            cyl.advance(theta, dtheta, rpm)

        new_cycle = int(theta / 720.0)
        if new_cycle > prev_cycle:
            m_now = _system_mass(sim)
            V_d_total = sim.cylinders[0].geometry.V_d * cfg.n_cylinders
            total_work = float(sum(c.work_cycle for c in sim.cylinders))
            total_intake = float(sum(c.m_intake_total for c in sim.cylinders))
            imep_bar = (total_work / V_d_total) / 1e5 if V_d_total > 0 else 0.0
            rho_atm = cfg.p_ambient / (287.0 * cfg.T_ambient)
            ve_atm = total_intake / (rho_atm * V_d_total) if V_d_total > 0 else 0.0
            indicated_power_kW = total_work * rpm / 120.0 / 1000.0
            egt = _egt_at_primary_valve(sim)
            cycle_stats.append({
                "cycle": new_cycle,
                "mass_total": m_now,
                "mass_drift": m_now - last_mass,
                "imep_bar": imep_bar,
                "ve_atm": ve_atm,
                "EGT_mean": egt,
                "indicated_power_kW": indicated_power_kW,
            })
            last_mass = m_now

            # Convergence check (IMEP-based, V1-style)
            if (len(cycle_stats) >= convergence_min_cycles + 1
                    and converged_cycle < 0):
                prev_imep = cycle_stats[-2]["imep_bar"]
                this_imep = cycle_stats[-1]["imep_bar"]
                if abs(prev_imep) > 1e-6:
                    rel = abs(this_imep - prev_imep) / abs(prev_imep)
                    if rel < convergence_tol_imep:
                        converged_cycle = new_cycle

            for c in sim.cylinders:
                c.m_intake_total = 0.0
                c.m_exhaust_total = 0.0
                c.work_cycle = 0.0
            prev_cycle = new_cycle

            if converged_cycle > 0 and new_cycle >= converged_cycle + 1:
                break
            if new_cycle >= n_cycles_max:
                break

    wall = time.time() - t_start
    last = cycle_stats[-1]
    return {
        "rpm": rpm,
        "converged_cycle": converged_cycle,
        "n_cycles_run": len(cycle_stats),
        "imep_bar": last["imep_bar"],
        "ve_atm": last["ve_atm"],
        "EGT_valve_K": last["EGT_mean"],
        "indicated_power_kW": last["indicated_power_kW"],
        "mass_drift_last": last["mass_drift"],
        "wall_time_s": wall,
        "step_count": step_count,
    }


def main():
    rpms = [6000.0 + 500.0 * i for i in range(16)]
    out_path = sys.argv[1] if len(sys.argv) > 1 else "docs/v1_sweep.json"
    t_all = time.time()
    results = []
    for rpm in rpms:
        r = run_v1_one_rpm(rpm)
        results.append(r)
        print(
            f"{rpm:6.0f} RPM  cycles={r['n_cycles_run']:2d} conv@{r['converged_cycle']}  "
            f"IMEP={r['imep_bar']:5.2f}  VE={r['ve_atm']*100:5.1f}%  "
            f"EGT={r['EGT_valve_K']:5.0f}K  P={r['indicated_power_kW']:5.1f}kW  "
            f"drift_last={r['mass_drift_last']:+.2e}  wall={r['wall_time_s']:5.1f}s"
        )
    Path(out_path).write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  V1 sweep total wall: {time.time() - t_all:.1f} s")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
