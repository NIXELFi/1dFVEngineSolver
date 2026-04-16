"""Phase E4 cross-cylinder coupling waterfall.

Runs one SDM26 sim at 8000 RPM (the VE peak), capturing per-step
state snapshots of primary 0, and renders an x-t waterfall of the
exhaust primary covering one converged cycle.

If cross-cylinder coupling is active, primary 0's probe should see:
  - its own blowdown pulse once per 720° crank revolution
  - three smaller pulses from cylinders 1/2/3's blowdowns transmitted
    through the 4-2-1 manifold

The waterfall shows these as diagonal streaks at the sound speed.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config_loader import load_v1_json
from models.sdm26 import SDM26Engine
from solver.state import I_RHO_A, I_MOM_A, I_E_A
from solver.muscl import cfl_dt


DOCS = Path(__file__).parent.parent / "docs"
OUT = DOCS / "e4_plots"
OUT.mkdir(parents=True, exist_ok=True)


def run_with_dump(rpm: float, n_cycles: int = 15):
    cfg = load_v1_json(DOCS.parent / "configs/sdm26.json")
    eng = SDM26Engine(cfg, junction_type="characteristic")
    # We need theta-indexed snapshots of primary 0 during the last cycle.
    # Monkey-patch the step function to record state per step into a
    # buffer. Easier: run to convergence, then run one extra cycle with
    # recording enabled manually via a thin wrapper.

    # Warm up to convergence
    eng.run_single_rpm(
        rpm, n_cycles=n_cycles, stop_at_convergence=True,
        convergence_tol_imep=0.005, convergence_min_cycles=10,
        verbose=False,
    )

    # Now run one more cycle (720° crank) with recording.
    from cylinder.kinematics import omega_from_rpm
    omega = omega_from_rpm(rpm)
    theta = 0.0
    target = 720.0
    p0 = eng.primaries[0]
    p0_hist = []  # list of (t, p_array)
    t = 0.0
    while theta < target:
        dt = cfl_dt(
            eng.plenum.q, eng.plenum.area, eng.plenum.dx, 1.4,
            cfg.cfl, eng.plenum.n_ghost,
        )
        for pipe in eng.all_pipes:
            d = cfl_dt(pipe.q, pipe.area, pipe.dx, 1.4, cfg.cfl, pipe.n_ghost)
            if 0.0 < d < dt:
                dt = d
        dt = min(dt, 1e-4)
        eng.step(theta, dt, rpm)
        t += dt
        theta += dt * (180.0 / np.pi) * omega

        # Record primary 0 pressure profile on real cells
        s = p0.real_slice()
        A = p0.area[s]
        rho = p0.q[s, I_RHO_A] / A
        mom = p0.q[s, I_MOM_A] / A
        E = p0.q[s, I_E_A] / A
        u = mom / rho
        p = (p0.gamma - 1.0) * (E - 0.5 * rho * u * u)
        p0_hist.append((t, p.copy()))

    return p0, p0_hist


def plot_waterfall(p0, p0_hist, rpm: float, out_path: Path):
    times = np.array([h[0] for h in p0_hist])
    data = np.array([h[1] for h in p0_hist])
    x = np.linspace(0.0, p0.n_cells * p0.dx, p0.n_cells)

    P_atm = 101325.0
    dev = (data - P_atm) / 1000.0   # kPa

    # Downsample to ~500 rows
    max_rows = 500
    if data.shape[0] > max_rows:
        stride = max(1, data.shape[0] // max_rows)
        idxs = list(range(0, data.shape[0], stride))
        if idxs[-1] != data.shape[0] - 1:
            idxs.append(data.shape[0] - 1)
        dev = dev[idxs]
        times = times[idxs]

    vmax = max(float(np.max(np.abs(dev))), 1.0)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        dev, aspect="auto", origin="upper",
        extent=[x[0] * 1000, x[-1] * 1000,
                times[-1] * 1000, times[0] * 1000],
        cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("x along primary 0 [mm]")
    ax.set_ylabel("time [ms]")
    ax.set_title(
        f"SDM26 exhaust primary 0 — one converged cycle at {rpm:.0f} RPM\n"
        "Cross-cylinder coupling: pulses from cyl 0 (self) + cyl 1/2/3 via 4-2-1"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("p − P_atm  [kPa]")
    fig.text(
        0.01, 0.005,
        f"shape: {dev.shape[0]}×{dev.shape[1]}   "
        f"L: {x[-1]*1000:.0f} mm   "
        f"t: 0 → {times[-1]*1000:.2f} ms   "
        f"vmax: ±{vmax:.1f} kPa",
        fontsize=7, family="monospace", color="#444",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(DOCS.parent)}")


def main():
    for rpm in (8000.0, 11500.0):
        print(f"\n=== SDM26 @ {rpm:.0f} RPM ===")
        p0, hist = run_with_dump(rpm)
        plot_waterfall(
            p0, hist, rpm,
            out_path=OUT / f"sdm26_primary0_cycle_{int(rpm)}rpm.png",
        )


if __name__ == "__main__":
    main()
