"""Run the Phase C3 post-fix sweeps for SDM25 (4-1) and SDM26 (4-2-1).

Runs each at the standard 16 RPM points (6000–13500 in 500-RPM steps),
12 cycles minimum with extension up to 40 if convergence not reached,
with bcs.valve regime logging enabled across the whole sweep.

Outputs:
  docs/sdm25_sweep.json            (post-fix, overwritten)
  docs/sdm26_sweep.json            (post-fix, overwritten)
  docs/c3_regime_log_sdm25.json    (regime distribution + per-cycle breakdown)
  docs/c3_regime_log_sdm26.json
  docs/c3_sweep_meta.json          (wall-clock + nonconservation diagnostics)

Pre-fix data should already be backed up to:
  docs/sdm25_sweep_prefix.json
  docs/sdm26_sweep_prefix.json

Aborts immediately and reports if EGT goes outside [800, 1500] K, mass
nonconservation exceeds 1e-12 kg/cycle, or any UNHANDLED BC call is
logged. The acceptance bars are checked per-RPM-point so a regression
is caught before wasting compute on the rest of the sweep.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import List

from configs.config_loader import load_v1_json
from models.sdm26 import SDM26Config, SDM26Engine
import bcs.valve

# Standard 16-point sweep
RPMS = [6000.0 + 500.0 * i for i in range(16)]
N_CYCLES_MAX = 40

# Acceptance bars (regression catches)
EGT_BAND = (800.0, 1500.0)            # K
NC_MAX_PER_CYCLE = 1.0e-12            # kg/cycle absolute


def run_one_sweep(cfg: SDM26Config, label: str, log_path: Path) -> dict:
    out = {"label": label, "points": []}
    cycle1_regime: Counter = Counter()
    laterCycles_regime: Counter = Counter()
    total_regime: Counter = Counter()
    t_all = time.time()
    print(f"\n=== {label} ===")
    for rpm in RPMS:
        eng = SDM26Engine(cfg)
        bcs.valve.enable_regime_logging(True)
        t0 = time.time()
        r = eng.run_single_rpm(
            rpm, n_cycles=N_CYCLES_MAX, stop_at_convergence=True,
            convergence_tol_imep=0.005, convergence_min_cycles=8,
            verbose=False,
        )
        wall = time.time() - t0

        # Snapshot regime log for this RPM
        log = bcs.valve.get_regime_log()
        rpm_regime = Counter(e["regime"] for e in log)
        total_regime.update(rpm_regime)

        # Convergence diagnostics
        last = r["cycle_stats"][-1]
        nc_max = max(abs(s["nonconservation"]) for s in r["cycle_stats"])
        raw_drift_final = last["mass_drift"]

        # Acceptance gates
        if last["EGT_mean"] < EGT_BAND[0] or last["EGT_mean"] > EGT_BAND[1]:
            print(f"  REGRESSION: {rpm:5.0f} RPM EGT={last['EGT_mean']:.0f}K out of band")
            raise RuntimeError(f"EGT regression at {rpm} RPM: {last['EGT_mean']:.0f} K")
        if nc_max > NC_MAX_PER_CYCLE:
            print(f"  REGRESSION: {rpm:5.0f} RPM nc_max={nc_max:.2e} > {NC_MAX_PER_CYCLE:.0e}")
            raise RuntimeError(f"Conservation regression at {rpm}: {nc_max:.2e}")
        if rpm_regime.get("UNHANDLED", 0) > 0:
            n = rpm_regime["UNHANDLED"]
            print(f"  REGRESSION: {rpm:5.0f} RPM UNHANDLED BC calls = {n}")
            raise RuntimeError(f"Unhandled BC at {rpm}: {n} events")

        pt = dict(last)
        pt.update({
            "rpm": rpm,
            "converged_cycle": r["converged_cycle"],
            "n_cycles_run": r["n_cycles_run"],
            "step_count": r["step_count"],
            "wall_time_s": wall,
            "nonconservation_max": nc_max,
            "raw_drift_final": raw_drift_final,
            "regime_counts": dict(rpm_regime),
        })
        out["points"].append(pt)
        print(
            f"  {rpm:5.0f} RPM  cycles={r['n_cycles_run']:2d}  "
            f"IMEP={last['imep_bar']:5.2f}  VE={last['ve_atm']*100:5.1f}%  "
            f"EGT={last['EGT_mean']:5.0f}K  P_whl={last['wheel_power_kW']:5.1f}kW  "
            f"T_whl={last['wheel_torque_Nm']:5.1f}Nm  "
            f"nc={nc_max:.1e}  wall={wall:5.1f}s"
        )

    out["total_wall_s"] = time.time() - t_all
    out["regime_total"] = dict(total_regime)
    print(f"  Sweep total wall time: {out['total_wall_s']:.1f} s")

    # Persist regime log summary
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "label": label,
        "regime_total": dict(total_regime),
        "per_rpm": [
            {"rpm": pt["rpm"], "regime_counts": pt["regime_counts"]}
            for pt in out["points"]
        ],
    }, indent=2))
    return out


def main():
    sdm25_cfg = load_v1_json(Path(__file__).parent.parent / "configs/sdm25.json")
    sdm26_cfg = load_v1_json(Path(__file__).parent.parent / "configs/sdm26.json")

    docs = Path(__file__).parent.parent / "docs"

    sdm25 = run_one_sweep(sdm25_cfg, "SDM25", docs / "c3_regime_log_sdm25.json")
    sdm26 = run_one_sweep(sdm26_cfg, "SDM26", docs / "c3_regime_log_sdm26.json")

    (docs / "sdm25_sweep.json").write_text(json.dumps(sdm25, indent=2, default=float))
    (docs / "sdm26_sweep.json").write_text(json.dumps(sdm26, indent=2, default=float))

    meta = {
        "branch": "diag/acoustic-bc",
        "sdm25_total_wall_s": sdm25["total_wall_s"],
        "sdm26_total_wall_s": sdm26["total_wall_s"],
        "sdm25_max_nonconservation": max(p["nonconservation_max"] for p in sdm25["points"]),
        "sdm26_max_nonconservation": max(p["nonconservation_max"] for p in sdm26["points"]),
        "egt_band_K": list(EGT_BAND),
        "nc_max_per_cycle_bar_kg": NC_MAX_PER_CYCLE,
        "rpms": RPMS,
    }
    (docs / "c3_sweep_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nWritten: sdm25_sweep.json, sdm26_sweep.json, c3_regime_log_*.json, c3_sweep_meta.json")
    print(f"\nSDM25: peak power = {max(pt['wheel_power_kW'] for pt in sdm25['points']):.1f} kW")
    print(f"SDM26: peak power = {max(pt['wheel_power_kW'] for pt in sdm26['points']):.1f} kW")


if __name__ == "__main__":
    main()
