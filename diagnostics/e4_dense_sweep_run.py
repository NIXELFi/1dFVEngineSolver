"""Phase E4 dense sweep — 100 RPM resolution.

Matches the dyno's 100 RPM resolution (3600-12900 RPM for SDM25).
Engine model is fine down to ~4000 RPM for both configs; below that
the combustion model's low-RPM regime kicks in and convergence slows.

Writes sdm{25,26}_sweep_e4_dense.json. Per-point failures are logged
and the sweep continues (do not lose 60 min of compute to one bad
point).
"""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter
from pathlib import Path

from configs.config_loader import load_v1_json
from models.sdm26 import SDM26Config, SDM26Engine
import bcs.valve


# 100 RPM resolution from 4000 to 13500 = 96 points
RPMS = [4000.0 + 100.0 * i for i in range(96)]
N_CYCLES_MAX = 40

EGT_BAND = (900.0, 1600.0)           # K, wider for dense sweep edge cases
NC_MAX_PER_CYCLE = 1.0e-4            # kg/cycle absolute (lenient for dense)


def run_one_sweep(cfg: SDM26Config, label: str, out_path: Path, log_path: Path):
    out = {"label": label, "junction_type": "characteristic", "points": [],
           "failed_points": []}
    total_regime: Counter = Counter()
    t_all = time.time()
    print(f"\n=== {label} (characteristic, 100 RPM dense) ===", flush=True)
    for i, rpm in enumerate(RPMS):
        try:
            eng = SDM26Engine(cfg, junction_type="characteristic")
            bcs.valve.enable_regime_logging(True)
            t0 = time.time()
            r = eng.run_single_rpm(
                rpm, n_cycles=N_CYCLES_MAX, stop_at_convergence=True,
                convergence_tol_imep=0.005, convergence_min_cycles=8,
                verbose=False,
            )
            wall = time.time() - t0

            log = bcs.valve.get_regime_log()
            rpm_regime = Counter(e["regime"] for e in log)
            total_regime.update(rpm_regime)

            last = r["cycle_stats"][-1]
            nc_max = max(abs(s["nonconservation"]) for s in r["cycle_stats"])
            raw_drift_final = last["mass_drift"]

            # Soft bar: log out-of-band but don't abort the sweep.
            warn_egt = (last["EGT_mean"] < EGT_BAND[0]
                        or last["EGT_mean"] > EGT_BAND[1])
            warn_nc = nc_max > NC_MAX_PER_CYCLE
            warn_unhandled = rpm_regime.get("UNHANDLED", 0) > 0

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
                "warn_egt": warn_egt,
                "warn_nc": warn_nc,
                "warn_unhandled": warn_unhandled,
            })
            out["points"].append(pt)
            flag = ""
            if warn_egt:       flag += " [EGT-band]"
            if warn_nc:        flag += " [nc]"
            if warn_unhandled: flag += " [UNHANDLED]"
            print(
                f"  [{i+1:3d}/{len(RPMS)}] {rpm:5.0f} RPM  "
                f"cyc={r['n_cycles_run']:2d}  "
                f"IMEP={last['imep_bar']:5.2f}  VE={last['ve_atm']*100:5.1f}%  "
                f"EGT={last['EGT_mean']:5.0f}K  P={last['wheel_power_kW']:5.1f}kW  "
                f"T={last['wheel_torque_Nm']:5.1f}Nm  "
                f"nc={nc_max:.1e}  wall={wall:5.1f}s{flag}",
                flush=True,
            )
        except Exception as ex:
            print(f"  [{i+1:3d}/{len(RPMS)}] {rpm:5.0f} RPM  FAILED: "
                  f"{type(ex).__name__}: {ex}", flush=True)
            out["failed_points"].append({
                "rpm": rpm, "error": f"{type(ex).__name__}: {ex}",
                "traceback": traceback.format_exc(),
            })

        # Incremental write so a crash doesn't lose partial progress
        if (i + 1) % 10 == 0 or i + 1 == len(RPMS):
            out["total_wall_s"] = time.time() - t_all
            out["regime_total"] = dict(total_regime)
            out_path.write_text(json.dumps(out, indent=2, default=float))

    out["total_wall_s"] = time.time() - t_all
    out["regime_total"] = dict(total_regime)
    print(f"  Sweep total wall time: {out['total_wall_s']:.1f} s", flush=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "label": label,
        "regime_total": dict(total_regime),
        "per_rpm": [
            {"rpm": pt["rpm"], "regime_counts": pt["regime_counts"]}
            for pt in out["points"]
        ],
    }, indent=2))


def main():
    sdm25_cfg = load_v1_json(Path(__file__).parent.parent / "configs/sdm25.json")
    sdm26_cfg = load_v1_json(Path(__file__).parent.parent / "configs/sdm26.json")
    docs = Path(__file__).parent.parent / "docs"

    run_one_sweep(sdm25_cfg, "SDM25",
                  docs / "sdm25_sweep_e4_dense.json",
                  docs / "e4_dense_regime_log_sdm25.json")
    run_one_sweep(sdm26_cfg, "SDM26",
                  docs / "sdm26_sweep_e4_dense.json",
                  docs / "e4_dense_regime_log_sdm26.json")
    print("\nWritten: sdm{25,26}_sweep_e4_dense.json")


if __name__ == "__main__":
    main()
