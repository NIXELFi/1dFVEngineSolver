"""Parameter-sensitivity study at a fixed RPM.

Sweeps every interesting SDM26Config field through a physically-reasonable
range at one operating point, collects converged IMEP / VE / EGT /
indicated power for each value, and produces:

  docs/sensitivity_<group>.png       — 4-panel plots per parameter
  docs/sensitivity_tornado.png       — relative sensitivity ranking
  docs/sensitivity_summary.png       — compact grid of all sweeps
  docs/sensitivity_results.json      — raw data

Purpose: confirm visually that every exposed parameter actually moves the
physics in the expected direction and by a plausible magnitude. This is
the "does the simulator actually respond to changes" demo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.sdm26 import SDM26Config
from models.parameter_sweep import sweep_parameter


# (field, values, xlabel, unit, group)
SWEEPS = [
    # Exhaust geometry
    ("primary_length",             np.linspace(0.20, 0.45, 6),   "primary length",        "m",  "exhaust_geom"),
    ("primary_diameter_in",        np.linspace(0.028, 0.040, 6), "primary ID",            "m",  "exhaust_geom"),
    ("primary_diameter_out",       np.linspace(0.030, 0.044, 6), "primary OD (taper)",    "m",  "exhaust_geom"),
    ("collector_diameter_in",      np.linspace(0.040, 0.065, 6), "collector ID",          "m",  "exhaust_geom"),
    # Intake geometry
    ("runner_length",              np.linspace(0.15, 0.35, 6),   "runner length",         "m",  "intake_geom"),
    ("runner_diameter_in",         np.linspace(0.030, 0.045, 6), "runner ID",             "m",  "intake_geom"),
    ("plenum_volume",              np.linspace(0.001, 0.0030, 6),"plenum volume",         "m³", "intake_geom"),
    # Restrictor
    ("restrictor_Cd",              np.linspace(0.90, 0.98, 5),   "restrictor Cd",         "—",  "restrictor"),
    # Combustion
    ("spark_advance",              np.linspace(15.0, 35.0, 6),   "spark advance (BTDC)",  "°",  "combustion"),
    ("combustion_duration",        np.linspace(30.0, 70.0, 6),   "combustion duration",   "°",  "combustion"),
    ("eta_comb",                   np.linspace(0.85, 0.99, 6),   "η_comb",                "—",  "combustion"),
    ("wiebe_m",                    np.linspace(1.0, 3.0, 5),     "Wiebe m",               "—",  "combustion"),
    # Valves
    ("intake_valve_max_lift",      np.linspace(0.006, 0.011, 6), "intake max lift",       "m",  "valves"),
    ("exhaust_valve_max_lift",     np.linspace(0.005, 0.010, 6), "exhaust max lift",      "m",  "valves"),
    # Thermal
    ("T_wall_cylinder",            np.linspace(350.0, 600.0, 6), "cylinder T_wall",       "K",  "thermal"),
    ("primary_wall_T",             np.linspace(700.0, 1200.0, 6),"exhaust primary T_wall","K",  "thermal"),
]


def run_all(rpm: float = 10500.0, verbose: bool = True) -> dict:
    base = SDM26Config()
    out = {"rpm": rpm, "sweeps": []}
    t0_all = time.time()
    for field_name, values, xlabel, unit, group in SWEEPS:
        t0 = time.time()
        rows = sweep_parameter(base, field_name, [float(v) for v in values],
                               rpm=rpm, verbose=False)
        wall = time.time() - t0
        if verbose:
            ve_range = max(r["ve_atm"] for r in rows) - min(r["ve_atm"] for r in rows)
            imep_range = max(r["imep_bar"] for r in rows) - min(r["imep_bar"] for r in rows)
            print(f"  {field_name:32s}  {len(rows)} pts, "
                  f"ΔIMEP={imep_range:+5.2f} bar  ΔVE={ve_range*100:+5.2f}%  "
                  f"wall={wall:4.1f}s")
        out["sweeps"].append({
            "field": field_name, "xlabel": xlabel, "unit": unit,
            "group": group,
            "rows": rows,
            "wall_time_s": wall,
        })
    out["total_wall_time_s"] = time.time() - t0_all
    return out


def plot_each_param(data: dict, out_dir: Path):
    for s in data["sweeps"]:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
        xs = [r["value"] for r in s["rows"]]
        imep = [r["imep_bar"] for r in s["rows"]]
        ve = [r["ve_atm"] * 100 for r in s["rows"]]
        egt = [r["EGT_valve_K"] for r in s["rows"]]
        p = [r["indicated_power_kW"] for r in s["rows"]]
        for ax, y, lbl, clr in zip(
            axes, [imep, ve, egt, p],
            ["IMEP [bar]", "VE atm [%]", "EGT [K]", "P_ind [kW]"],
            ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"],
        ):
            ax.plot(xs, y, "o-", color=clr, linewidth=1.8)
            ax.set_xlabel(f"{s['xlabel']} [{s['unit']}]")
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"SDM26 sensitivity @ {data['rpm']:.0f} RPM: {s['field']}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"sensitivity_{s['field']}.png", dpi=110)
        plt.close(fig)


def plot_tornado(data: dict, out_path: Path):
    """Horizontal bar chart: for each parameter, show the range of
    indicated-power change it produces when swept across its test range,
    expressed as % of the baseline power. Sorted by magnitude."""
    rows = []
    for s in data["sweeps"]:
        vals = [r["indicated_power_kW"] for r in s["rows"]]
        if not vals:
            continue
        p_min, p_max = min(vals), max(vals)
        p_mid = 0.5 * (p_min + p_max)
        rel_range = (p_max - p_min) / p_mid if p_mid > 0 else 0.0
        rows.append((s["field"], rel_range * 100.0, p_min, p_max))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    colors = ["#1f77b4"] * len(rows)
    ax.barh(names, deltas, color=colors, alpha=0.85)
    ax.set_xlabel("Δ indicated power across swept range [% of midrange]")
    ax.set_title(f"Parameter sensitivity @ {data['rpm']:.0f} RPM")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_summary_grid(data: dict, out_path: Path):
    """Compact grid: one small plot per parameter showing IMEP vs parameter.
    Makes it easy to see at a glance which parameters matter."""
    n = len(data["sweeps"])
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows))
    axes = axes.flatten()
    for ax, s in zip(axes, data["sweeps"]):
        xs = [r["value"] for r in s["rows"]]
        imep = [r["imep_bar"] for r in s["rows"]]
        ax.plot(xs, imep, "o-", color="#1f77b4", linewidth=1.6, markersize=4)
        ax.set_title(f"{s['field']}", fontsize=9)
        ax.set_xlabel(f"{s['xlabel']} [{s['unit']}]", fontsize=8)
        ax.set_ylabel("IMEP [bar]", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    # Hide unused axes
    for ax in axes[len(data["sweeps"]):]:
        ax.axis("off")
    fig.suptitle(f"SDM26 IMEP sensitivity @ {data['rpm']:.0f} RPM", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    print("Running sensitivity study (may take a few minutes; Numba warms up on first call)…")
    data = run_all(rpm=10500.0, verbose=True)
    (out_dir / "sensitivity_results.json").write_text(
        json.dumps(data, indent=2, default=float)
    )
    print("\nPlotting individual parameters...")
    plot_each_param(data, out_dir)
    plot_tornado(data, out_dir / "sensitivity_tornado.png")
    plot_summary_grid(data, out_dir / "sensitivity_summary.png")
    print(f"Wrote {len(data['sweeps'])} per-param plots + tornado + summary grid.")
    print(f"Total wall: {data['total_wall_time_s']:.1f} s")


if __name__ == "__main__":
    main()
