"""SDM26 engine assembly — Honda CBR600RR 599 cc I4 with FSAE 20 mm restrictor.

Geometry and coefficients copied from V1 at `1d/engine_simulator/config/cbr600rr.json`
(copy date 2026-04-13). Differences vs V1:
  - exhaust-pipe wall temperatures raised to physical values (V1 ran cold
    to compensate for the entropy-BC bug; V2 corrects the physics so wall
    T should reflect the real skin temperature at WOT).
  - plenum is an FV domain (1D pipe of matched volume, 0.3 m × ~50 cm²)
    rather than a lumped 0-D volume.
  - valve BCs are entropy-aware ghost-cell fills (V2's main fix).
  - no V1 imports.

Per the spec's stop-gates for Phase 3 single-RPM verification:
  EGT at exhaust-primary valve face must be 1000-1300 K at 10500 RPM WOT.
  Cycle-to-cycle mass drift must be < 1e-8 kg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from solver.state import make_pipe_state, set_uniform, PipeState, I_RHO_A, I_MOM_A, I_E_A, I_Y_A
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from solver.sources import apply_sources

from bcs.restrictor import fill_choked_restrictor_left
from bcs.junction import apply_junction, JunctionLeg, LEFT, RIGHT
from bcs.valve import fill_valve_ghost
from bcs.simple import fill_transmissive_right

from cylinder.combustion import WiebeParams
from cylinder.cylinder import CylinderModel, CylinderGeom
from cylinder.heat_transfer import WoschniParams
from cylinder.kinematics import cylinder_phase_offsets, omega_from_rpm
from cylinder.valve import (
    ValveParams, INTAKE_LD_TABLE, INTAKE_CD_TABLE,
    EXHAUST_LD_TABLE, EXHAUST_CD_TABLE,
)


# -------------------- Config (SDM26 defaults) --------------------

@dataclass
class SDM26Config:
    # Engine geometry — CBR600RR
    bore: float = 0.067
    stroke: float = 0.0425
    con_rod: float = 0.0963
    CR: float = 12.2
    n_cylinders: int = 4
    firing_order: tuple = (1, 2, 4, 3)
    firing_interval: float = 180.0

    # Intake runners (4x identical)
    runner_length: float = 0.245
    runner_diameter: float = 0.038
    runner_n_cells: int = 30
    runner_wall_T: float = 325.0

    # Exhaust primaries (4x, slight D variation in V1 — use 0.032 m here)
    primary_length: float = 0.308
    primary_diameter: float = 0.032
    primary_n_cells: int = 30
    primary_wall_T: float = 1000.0  # V2 physical value (V1 ran 650 K cold)

    # Exhaust secondaries (2x)
    secondary_length: float = 0.392
    secondary_diameter: float = 0.038
    secondary_n_cells: int = 20
    secondary_wall_T: float = 800.0

    # Exhaust collector
    collector_length: float = 0.1
    collector_diameter: float = 0.05
    collector_n_cells: int = 20
    collector_wall_T: float = 700.0

    # Plenum
    plenum_volume: float = 0.0015     # m³
    plenum_length: float = 0.3        # m, somewhat arbitrary 1D stand-in
    plenum_n_cells: int = 20
    plenum_wall_T: float = 320.0

    # Restrictor
    restrictor_throat_diameter: float = 0.020
    restrictor_Cd: float = 0.967       # spec target (V1's 0.926-0.95 was tuned)

    # Ambient
    p_ambient: float = 101325.0
    T_ambient: float = 300.0

    # Combustion (physics only; no V1 fudge ramps)
    wiebe_a: float = 5.0
    wiebe_m: float = 2.0
    combustion_duration: float = 50.0
    spark_advance: float = 25.0
    ignition_delay: float = 7.0
    eta_comb: float = 0.96
    q_lhv: float = 44.0e6
    afr_target: float = 13.1
    T_wall_cylinder: float = 450.0

    # Numerics
    cfl: float = 0.85
    limiter: int = LIMITER_MINMOD


# -------------------- Engine model --------------------

class SDM26Engine:
    """One SDM26 simulation state.

    Advance with `run_single_rpm(rpm, n_cycles)` and read results off the
    cylinders' per-cycle accumulators.
    """

    def __init__(self, cfg: SDM26Config):
        self.cfg = cfg

        # Plenum pipe (constant area matching plenum_volume/plenum_length)
        A_plen = cfg.plenum_volume / cfg.plenum_length
        self.plenum = make_pipe_state(
            cfg.plenum_n_cells, cfg.plenum_length,
            area_fn=lambda x: A_plen,
            gamma=1.4, R_gas=287.0, wall_T=cfg.plenum_wall_T, n_ghost=2,
        )
        # Initialise with ambient static state
        set_uniform(self.plenum, rho=cfg.p_ambient / (287.0 * cfg.T_ambient),
                    u=0.0, p=cfg.p_ambient, Y=0.0)

        # Intake runners
        A_runner = 0.25 * np.pi * cfg.runner_diameter ** 2
        self.runners: List[PipeState] = []
        for _ in range(4):
            s = make_pipe_state(
                cfg.runner_n_cells, cfg.runner_length,
                area_fn=lambda x: A_runner,
                gamma=1.4, R_gas=287.0, wall_T=cfg.runner_wall_T, n_ghost=2,
            )
            set_uniform(s, rho=cfg.p_ambient / (287.0 * cfg.T_ambient),
                        u=0.0, p=cfg.p_ambient, Y=0.0)
            self.runners.append(s)

        # Exhaust primaries
        A_primary = 0.25 * np.pi * cfg.primary_diameter ** 2
        self.primaries: List[PipeState] = []
        for _ in range(4):
            s = make_pipe_state(
                cfg.primary_n_cells, cfg.primary_length,
                area_fn=lambda x: A_primary,
                gamma=1.4, R_gas=287.0, wall_T=cfg.primary_wall_T, n_ghost=2,
            )
            set_uniform(s, rho=cfg.p_ambient / (287.0 * cfg.T_ambient),
                        u=0.0, p=cfg.p_ambient, Y=0.0)
            self.primaries.append(s)

        # Exhaust secondaries
        A_sec = 0.25 * np.pi * cfg.secondary_diameter ** 2
        self.secondaries: List[PipeState] = []
        for _ in range(2):
            s = make_pipe_state(
                cfg.secondary_n_cells, cfg.secondary_length,
                area_fn=lambda x: A_sec,
                gamma=1.4, R_gas=287.0, wall_T=cfg.secondary_wall_T, n_ghost=2,
            )
            set_uniform(s, rho=cfg.p_ambient / (287.0 * cfg.T_ambient),
                        u=0.0, p=cfg.p_ambient, Y=0.0)
            self.secondaries.append(s)

        # Collector
        A_col = 0.25 * np.pi * cfg.collector_diameter ** 2
        self.collector = make_pipe_state(
            cfg.collector_n_cells, cfg.collector_length,
            area_fn=lambda x: A_col,
            gamma=1.4, R_gas=287.0, wall_T=cfg.collector_wall_T, n_ghost=2,
        )
        set_uniform(self.collector, rho=cfg.p_ambient / (287.0 * cfg.T_ambient),
                    u=0.0, p=cfg.p_ambient, Y=0.0)

        self.all_pipes: List[PipeState] = [
            self.plenum,
            *self.runners,
            *self.primaries,
            *self.secondaries,
            self.collector,
        ]

        # Cylinders
        offsets = cylinder_phase_offsets(
            cfg.n_cylinders, list(cfg.firing_order), cfg.firing_interval,
        )
        geom = CylinderGeom(cfg.bore, cfg.stroke, cfg.con_rod, cfg.CR)
        wiebe = WiebeParams(
            a=cfg.wiebe_a, m=cfg.wiebe_m,
            duration_deg=cfg.combustion_duration,
            spark_advance_deg=cfg.spark_advance,
            ignition_delay_deg=cfg.ignition_delay,
            eta_comb=cfg.eta_comb,
            q_lhv=cfg.q_lhv, afr_target=cfg.afr_target,
        )
        woschni = WoschniParams(bore=cfg.bore, stroke=cfg.stroke, T_wall=cfg.T_wall_cylinder)
        intake_valve = ValveParams(
            diameter=0.0275, max_lift=0.00856,
            open_angle_deg=350.0, close_angle_deg=585.0,
            seat_angle_deg=45.0, n_valves=2,
            ld_table=INTAKE_LD_TABLE, cd_table=INTAKE_CD_TABLE,
        )
        exhaust_valve = ValveParams(
            diameter=0.023, max_lift=0.00735,
            open_angle_deg=140.0, close_angle_deg=365.0,
            seat_angle_deg=45.0, n_valves=2,
            ld_table=EXHAUST_LD_TABLE, cd_table=EXHAUST_CD_TABLE,
        )
        self.cylinders: List[CylinderModel] = []
        for i in range(cfg.n_cylinders):
            cyl = CylinderModel(
                geom=geom, wiebe=wiebe, woschni=woschni,
                intake_valve=intake_valve, exhaust_valve=exhaust_valve,
                phase_offset_deg=offsets[i + 1],
            )
            cyl.initialize(p=cfg.p_ambient, T=cfg.T_ambient, theta_global_deg=0.0)
            self.cylinders.append(cyl)

    # ---- BC helpers ----

    def _apply_intake_junction(self) -> float:
        """5-leg junction between plenum RIGHT and all 4 runners' LEFT ends."""
        legs = [JunctionLeg(self.plenum, RIGHT, sign=+1)]
        for r in self.runners:
            legs.append(JunctionLeg(r, LEFT, sign=-1))
        return apply_junction(legs)

    def _apply_exhaust_junctions(self):
        """Firing pairs (1,4)→secondary_0, (2,3)→secondary_1; secondaries → collector."""
        j1 = apply_junction([
            JunctionLeg(self.primaries[0], RIGHT, sign=+1),
            JunctionLeg(self.primaries[3], RIGHT, sign=+1),
            JunctionLeg(self.secondaries[0], LEFT, sign=-1),
        ])
        j2 = apply_junction([
            JunctionLeg(self.primaries[1], RIGHT, sign=+1),
            JunctionLeg(self.primaries[2], RIGHT, sign=+1),
            JunctionLeg(self.secondaries[1], LEFT, sign=-1),
        ])
        j3 = apply_junction([
            JunctionLeg(self.secondaries[0], RIGHT, sign=+1),
            JunctionLeg(self.secondaries[1], RIGHT, sign=+1),
            JunctionLeg(self.collector, LEFT, sign=-1),
        ])
        return j1, j2, j3

    # ---- Time step ----

    def _ensure_scratch(self, pipe: PipeState):
        """Lazily allocate per-pipe scratch buffers so we can inspect face
        fluxes after the MUSCL step."""
        n = pipe.n_total
        buf = getattr(pipe, "_scratch", None)
        if buf is None or buf["w"].shape[0] != n:
            buf = {
                "w":      np.zeros((n, 4)),
                "slopes": np.zeros((n, 4)),
                "wL":     np.zeros((n, 4)),
                "wR":     np.zeros((n, 4)),
                "flux":   np.zeros((n + 1, 4)),
            }
            pipe._scratch = buf
        return buf

    def step(self, theta_deg: float, dt: float, rpm: float):
        cfg = self.cfg
        gamma = 1.4
        A_t = 0.25 * np.pi * cfg.restrictor_throat_diameter ** 2

        # 1. Apply BCs (ghost-cell fills) — these use current state before
        #    this step's MUSCL advance.
        fill_choked_restrictor_left(
            self.plenum, cfg.p_ambient, cfg.T_ambient, A_t, cfg.restrictor_Cd,
        )
        self._apply_intake_junction()

        for i, cyl in enumerate(self.cylinders):
            theta_local = cyl.local_theta(theta_deg)
            fill_valve_ghost(
                self.runners[i], RIGHT, "intake",
                cyl.intake_valve, theta_local,
                cyl.state.p, cyl.state.T, cyl.state.x_b,
            )
            fill_valve_ghost(
                self.primaries[i], LEFT, "exhaust",
                cyl.exhaust_valve, theta_local,
                cyl.state.p, cyl.state.T, cyl.state.x_b,
            )

        self._apply_exhaust_junctions()
        fill_transmissive_right(self.collector)

        # 2. MUSCL-Hancock step for every pipe; keep the flux array so we can
        #    read the boundary face flux directly.
        for pipe in self.all_pipes:
            buf = self._ensure_scratch(pipe)
            muscl_hancock_step(
                pipe.q, pipe.area, pipe.area_f, pipe.dx, dt,
                gamma, pipe.n_ghost, LIMITER_MINMOD,
                buf["w"], buf["slopes"], buf["wL"], buf["wR"], buf["flux"],
            )

        # 3. Read the ACTUAL HLLC flux at the valve faces for cylinder bookkeeping.
        #    Sign convention for the cylinder:
        #       mdot_intake = + into cylinder. At runner's RIGHT end, flux
        #       direction "into cylinder" corresponds to positive flux at the
        #       face between last real cell and first right-ghost (j = ng+nc).
        #       mdot_exhaust = + out of cylinder. At primary's LEFT end, flux
        #       direction "out of cylinder into pipe" is positive at the face
        #       between last left-ghost and first real cell (j = ng).
        intake_flux = np.zeros(cfg.n_cylinders)
        intake_flux_T = np.zeros(cfg.n_cylinders)
        intake_energy_flux = np.zeros(cfg.n_cylinders)
        exhaust_flux = np.zeros(cfg.n_cylinders)
        for i in range(cfg.n_cylinders):
            r = self.runners[i]
            j = r.n_ghost + r.n_cells
            f = r._scratch["flux"][j]
            intake_flux[i] = f[0]       # kg/s, + = into cylinder
            intake_energy_flux[i] = f[2]  # J/s
            # Approximate enthalpy T for cylinder's T_intake bookkeeping:
            # (E+p)u·A = m_flux · h, so h = f[2] / m_flux; T = h · (γ-1)/γ · 1/R_air
            if abs(f[0]) > 1e-20:
                h = f[2] / f[0]
                T_est = h * (gamma - 1.0) / gamma / 287.0
                intake_flux_T[i] = max(T_est, 100.0)
            else:
                intake_flux_T[i] = self.cylinders[i].state.T_intake

            p_pipe = self.primaries[i]
            j = p_pipe.n_ghost
            f = p_pipe._scratch["flux"][j]
            exhaust_flux[i] = f[0]      # kg/s, + = out of cylinder into pipe

        # 4. Source-step (friction + wall heat).
        for pipe in self.all_pipes:
            apply_sources(
                pipe.q, pipe.area, pipe.hydraulic_D, dt, gamma, 287.0,
                pipe.wall_T, pipe.n_ghost,
                apply_friction=True, apply_heat=True,
            )

        # 5. Advance each cylinder. mdot_intake and mdot_exhaust are the SIGNED
        #    HLLC face fluxes — this is the shared-flux conservative coupling.
        dtheta = dt * (180.0 / np.pi) * omega_from_rpm(rpm)
        for i, cyl in enumerate(self.cylinders):
            cyl.state.mdot_intake = intake_flux[i]
            cyl.state.mdot_exhaust = exhaust_flux[i]
            cyl.state.T_intake = intake_flux_T[i]
            cyl.advance(theta_deg, dtheta, rpm, dt)

    # ---- Run loop ----

    def run_single_rpm(self, rpm: float, n_cycles: int = 5,
                       verbose: bool = False) -> dict:
        cfg = self.cfg
        omega = omega_from_rpm(rpm)
        theta = 0.0
        target_theta = n_cycles * 720.0

        # Per-cycle bookkeeping
        cycle_stats = []
        prev_cycle = 0
        last_mass_total = self._system_mass()
        step_count = 0
        while theta < target_theta:
            dt = cfl_dt(self.plenum.q, self.plenum.area, self.plenum.dx, 1.4,
                        cfg.cfl, self.plenum.n_ghost)
            for p in self.all_pipes:
                d = cfl_dt(p.q, p.area, p.dx, 1.4, cfg.cfl, p.n_ghost)
                if 0.0 < d < dt:
                    dt = d
            if dt <= 0.0:
                raise RuntimeError(f"positivity failure at θ={theta:.1f}°")
            dt = min(dt, 1e-4)

            self.step(theta, dt, rpm)
            step_count += 1
            theta += dt * (180.0 / np.pi) * omega

            new_cycle = int(theta / 720.0)
            if new_cycle > prev_cycle:
                m_now = self._system_mass()
                stats = {
                    "cycle": new_cycle,
                    "mass_total": m_now,
                    "mass_drift": m_now - last_mass_total,
                    "cylinders": [
                        {
                            "m": c.state.m,
                            "p": c.state.p,
                            "T": c.state.T,
                            "m_intake": c.state.m_intake_total,
                            "m_exhaust": c.state.m_exhaust_total,
                            "work": c.state.work_cycle,
                        }
                        for c in self.cylinders
                    ],
                    "EGT_mean": float(np.mean([
                        _primary_entrance_T(p, 1.4) for p in self.primaries
                    ])),
                }
                cycle_stats.append(stats)
                last_mass_total = m_now
                if verbose:
                    print(f"  cycle {new_cycle}: mass={m_now:.4e} kg, EGT={stats['EGT_mean']:.0f} K")
                # Reset per-cycle accumulators
                for c in self.cylinders:
                    c.state.m_intake_total = 0.0
                    c.state.m_exhaust_total = 0.0
                    c.state.work_cycle = 0.0
                prev_cycle = new_cycle

        return {
            "rpm": rpm,
            "n_cycles": n_cycles,
            "step_count": step_count,
            "cycle_stats": cycle_stats,
        }

    def _system_mass(self) -> float:
        total = 0.0
        for p in self.all_pipes:
            s = p.real_slice()
            total += float(p.dx * p.q[s, I_RHO_A].sum())
        for c in self.cylinders:
            total += float(c.state.m)
        return total


def _primary_entrance_T(pipe: PipeState, gamma: float) -> float:
    idx = pipe.n_ghost
    A = pipe.area[idx]
    rho = pipe.q[idx, I_RHO_A] / A
    u = pipe.q[idx, I_MOM_A] / (rho * A)
    E = pipe.q[idx, I_E_A] / A
    p = max((gamma - 1.0) * (E - 0.5 * rho * u * u), 1.0)
    return p / (rho * 287.0)
