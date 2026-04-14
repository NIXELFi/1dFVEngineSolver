"""V1 exhaust valve BC entropy error probe — observer-style.

EXCEPTIONAL FILE: same cross-repo-import exemption as v1_mass_audit.py.
V2 solver code MUST NOT import from 1d/.

Purpose: quantify the temperature mismatch between (a) the cylinder gas state
during blowdown and (b) the exhaust primary pipe boundary state V1 sets after
the exhaust valve BC runs. V1's valve_bc docstring states the error is ~2x;
this probe measures it on a representative exhaust event at a chosen RPM.

Output: CSV of (theta_deg, cyl_T, cyl_p, pipe_T_boundary, pipe_p_boundary,
pipe_a_boundary, cyl_a, T_isentropic_expansion) for every step within the
exhaust event of cylinder 0, plus a summary of the worst-case error.

Run:
    python -m diagnostics.v1_valve_entropy_probe --rpm 10500 --out docs/v1_valve_entropy_probe.csv
"""

from __future__ import annotations

import argparse
import csv
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
from engine_simulator.gas_dynamics.gas_properties import R_AIR  # noqa: E402
from engine_simulator.gas_dynamics.moc_solver import (  # noqa: E402
    advance_interior_points,
    extrapolate_boundary_incoming,
)
from engine_simulator.simulation.engine_cycle import EngineCycleTracker  # noqa: E402
from engine_simulator.simulation.orchestrator import SimulationOrchestrator  # noqa: E402


def speed_of_sound(T: float, gamma: float = 1.35, R: float = R_AIR) -> float:
    return float(np.sqrt(gamma * R * max(T, 1.0)))


def isentropic_T_expansion(T_cyl: float, p_cyl: float, p_pipe: float,
                           gamma: float = 1.35) -> float:
    """Isentropic expansion temperature from cylinder to pipe pressure.

    T_pipe / T_cyl = (p_pipe / p_cyl) ** ((gamma-1)/gamma)
    """
    if p_cyl <= 0:
        return T_cyl
    ratio = max(p_pipe / p_cyl, 1e-6)
    return T_cyl * ratio ** ((gamma - 1.0) / gamma)


def run_probe(rpm: float = 10500.0, warmup_cycles: int = 2, probe_cycles: int = 1,
              out_path: str | None = None) -> dict:
    cfg = EngineConfig()
    sim = SimulationOrchestrator(cfg)
    sim._reinitialize(rpm)

    tracker = EngineCycleTracker(rpm)
    cfl_num = cfg.simulation.cfl_number

    cyl0 = sim.cylinders[0]
    pipe0 = sim.exhaust_primaries[0]
    evo = cyl0.exhaust_valve.open_angle % 720.0
    evc = cyl0.exhaust_valve.close_angle % 720.0

    rows: list[dict] = []

    cycle_counter = 0
    t_wall = time.time()
    step = 0

    while cycle_counter < warmup_cycles + probe_cycles and step < int(1e6):
        step += 1
        dt = compute_cfl_timestep(sim.all_pipes, cfl_num)
        dt = min(dt, 1e-3)
        dtheta = tracker.advance(dt)
        theta = tracker.theta
        theta_local_cyl0 = cyl0.local_theta(theta)

        for p in sim.all_pipes:
            extrapolate_boundary_incoming(p, dt)
        for p in sim.all_pipes:
            p.update_derived()
        sim.restrictor_plenum.solve_and_apply(dt)
        for p in sim.intake_runners:
            p.update_derived()
        for c in sim.cylinders:
            c.mdot_intake = 0.0
            c.mdot_exhaust = 0.0
        for i in range(cfg.n_cylinders):
            sim.intake_valve_bcs[i].apply(
                sim.intake_runners[i], PipeEnd.RIGHT, dt, theta_deg=theta, rpm=rpm,
            )
        for p in sim.intake_runners:
            p.update_derived()
        for i in range(cfg.n_cylinders):
            sim.exhaust_valve_bcs[i].apply(
                sim.exhaust_primaries[i], PipeEnd.LEFT, dt, theta_deg=theta, rpm=rpm,
            )
        for p in sim.exhaust_primaries:
            p.update_derived()
        for junc in sim.exhaust_junctions:
            junc.apply(dt)
        for p in sim.exhaust_primaries + sim.exhaust_secondaries + [sim.exhaust_collector]:
            p.update_derived()
        sim.exhaust_open_bc.apply(sim.exhaust_collector, PipeEnd.RIGHT, dt)
        sim.exhaust_collector.update_derived()

        # Record state right after exhaust valve BC set pipe[0] of primary 1,
        # when exhaust valve of cyl 0 is open and blowdown is active
        ex_valve = cyl0.exhaust_valve
        is_open = ex_valve.is_open(theta_local_cyl0)
        if cycle_counter >= warmup_cycles and is_open:
            T_cyl = cyl0.T
            p_cyl = cyl0.p
            a_cyl = speed_of_sound(T_cyl, gamma=1.35)
            T_pipe = float(pipe0.T[0])
            p_pipe = float(pipe0.p[0])
            rho_pipe = float(pipe0.rho[0])
            u_pipe = float(pipe0.u[0])
            a_pipe = float(pipe0.a[0])
            AA_pipe = float(pipe0.AA[0])
            T_isen = isentropic_T_expansion(T_cyl, p_cyl, p_pipe, gamma=1.35)
            a_isen = speed_of_sound(T_isen, gamma=1.35)
            mdot_ex = float(cyl0.mdot_exhaust)

            rows.append({
                "theta": theta,
                "theta_local": theta_local_cyl0,
                "t": t_wall - t_wall,  # filler; we don't track sim time separately
                "cyl_T": T_cyl,
                "cyl_p": p_cyl,
                "cyl_a": a_cyl,
                "pipe_T": T_pipe,
                "pipe_p": p_pipe,
                "pipe_rho": rho_pipe,
                "pipe_u": u_pipe,
                "pipe_a": a_pipe,
                "pipe_AA": AA_pipe,
                "T_isentropic": T_isen,
                "a_isentropic": a_isen,
                "mdot_exhaust": mdot_ex,
                "T_error_K": T_isen - T_pipe,
                "T_error_frac": (T_isen - T_pipe) / max(T_isen, 1.0),
                "a_ratio_isentropic_to_pipe": a_isen / max(a_pipe, 1e-6),
            })

        global_av = cfg.simulation.artificial_viscosity
        for p in sim.all_pipes:
            av = p.artificial_viscosity if p.artificial_viscosity >= 0 else global_av
            advance_interior_points(p, dt, include_sources=True, artificial_viscosity=av)
        for c in sim.cylinders:
            c.advance(theta, dtheta, rpm)

        new_cycle = int(theta / 720.0)
        if new_cycle > cycle_counter:
            cycle_counter = new_cycle

    # Summary: find the peak-blowdown row (max cyl_p) and worst error
    if not rows:
        return {"error": "no rows captured"}

    # Filter to the blowdown phase (first ~60 deg of exhaust open, high p_cyl)
    peak = max(rows, key=lambda r: r["cyl_p"])
    worst = max(rows, key=lambda r: r["T_error_K"])
    avg_T_err = float(np.mean([r["T_error_K"] for r in rows]))
    avg_T_frac = float(np.mean([r["T_error_frac"] for r in rows]))
    avg_a_ratio = float(np.mean([r["a_ratio_isentropic_to_pipe"] for r in rows]))

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return {
        "rpm": rpm,
        "evo_deg": evo,
        "evc_deg": evc,
        "n_rows": len(rows),
        "peak_blowdown": peak,
        "worst_T_error_row": worst,
        "avg_T_error_K": avg_T_err,
        "avg_T_error_frac": avg_T_frac,
        "avg_a_ratio_isentropic_to_pipe": avg_a_ratio,
        "wall_time_s": time.time() - t_wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rpm", type=float, default=10500.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    summary = run_probe(
        rpm=args.rpm,
        warmup_cycles=args.warmup,
        probe_cycles=args.cycles,
        out_path=args.out,
    )
    import json
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
