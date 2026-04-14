"""Standalone x-t waterfall viewer for pipe-state dumps.

A minimal visualization tool for inspecting the pressure / velocity /
temperature / Mach / Y wave structure inside a single 1D pipe over time.
Consumes the .npz dump format written by
``tests/acoustic/_helpers.py:save_pipe_dump`` (see that function's
docstring for the field list).

Usage:

    # CLI
    python -m diagnostics.waterfall_viewer \\
        --dump path/to/dump.npz --field pressure --output waterfall.png

    # Library
    from diagnostics.waterfall_viewer import render_waterfall
    render_waterfall("dump.npz", field="pressure", output_path="out.png")

Out of scope (deferred to Phase 5 visualization work):
  - engine network plots (multi-pipe / junction layouts)
  - interactive controls
  - animation
  - web rendering
  - cylinder P-V loop integration
  - real-time solver coupling
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Reference values used for diverging-colormap centering
P_ATM = 101325.0     # Pa
GAMMA_DEFAULT = 1.4
R_AIR = 287.0        # J / (kg K)

# Output downsample target (rows in the rendered image)
DEFAULT_MAX_ROWS = 500

# Field configuration: cmap, reference centre (None → sequential),
# auto-vmax fraction (multiplier on per-frame max(|x − ref|))
FIELD_CONFIGS = {
    "pressure":    {"cmap": "RdBu_r",   "ref": P_ATM, "unit": "Pa",  "scale": 1e-3, "scale_unit": "kPa"},
    "velocity":    {"cmap": "RdBu_r",   "ref": 0.0,   "unit": "m/s", "scale": 1.0,  "scale_unit": "m/s"},
    "temperature": {"cmap": "inferno",  "ref": None,  "unit": "K",   "scale": 1.0,  "scale_unit": "K"},
    "mach":        {"cmap": "viridis",  "ref": None,  "unit": "-",   "scale": 1.0,  "scale_unit": "Mach"},
    "Y":           {"cmap": "magma",    "ref": None,  "unit": "-",   "scale": 1.0,  "scale_unit": "Y"},
    "density":     {"cmap": "viridis",  "ref": None,  "unit": "kg/m³","scale": 1.0, "scale_unit": "kg/m³"},
}


def _load_dump(dump_path: Path) -> dict:
    """Load and lightly validate a pipe-state .npz dump."""
    data = np.load(dump_path, allow_pickle=False)
    required = {"time", "x", "pressure", "velocity", "temperature", "density", "Y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"dump {dump_path} missing required arrays: {sorted(missing)}"
        )
    out = {k: data[k] for k in data.files}
    # 0-d string scalars come back as numpy 0-d arrays; cast to plain str
    for key in ("pipe_name", "source"):
        if key in out and out[key].ndim == 0:
            out[key] = str(out[key])
    return out


def _downsample_rows(arr: np.ndarray, t: np.ndarray, max_rows: int):
    """Stride-decimate the leading axis to at most max_rows; preserve last row."""
    n = arr.shape[0]
    if n <= max_rows:
        return arr, t
    stride = max(1, n // max_rows)
    idxs = list(range(0, n, stride))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return arr[idxs], t[idxs]


def _compute_field(dump: dict, field: str, gamma: float) -> np.ndarray:
    """Return the (n_rows, n_cols) array for the requested field."""
    if field in ("pressure", "velocity", "temperature", "density", "Y"):
        return dump["pressure" if field == "pressure" else field
                    if field != "Y" else "Y"]  # passthrough lookup
    if field == "mach":
        T = dump["temperature"]
        u = dump["velocity"]
        c = np.sqrt(gamma * R_AIR * np.maximum(T, 1.0))
        return u / c
    raise ValueError(
        f"unknown field {field!r}; valid options: "
        f"{sorted(FIELD_CONFIGS.keys())}"
    )


def render_waterfall(
    dump_path: str | Path, *,
    field: str = "pressure",
    output_path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    title: Optional[str] = None,
    vmax_override: Optional[float] = None,
    figsize: tuple = (9, 7),
    dpi: int = 130,
) -> Path:
    """Render an x-t waterfall image from a pipe-state dump.

    Returns the output path on success. Raises ValueError if the dump
    is malformed or the requested field is unknown.
    """
    dump_path = Path(dump_path)
    output_path = Path(output_path)

    if field not in FIELD_CONFIGS:
        raise ValueError(
            f"unknown field {field!r}; valid options: "
            f"{sorted(FIELD_CONFIGS.keys())}"
        )
    cfg = FIELD_CONFIGS[field]

    dump = _load_dump(dump_path)
    gamma = float(dump.get("gamma", GAMMA_DEFAULT))
    pipe_name = dump.get("pipe_name", dump_path.stem)
    source = dump.get("source", "")
    length_m = float(dump.get("length", dump["x"][-1] + (dump["x"][1] - dump["x"][0])))

    arr = _compute_field(dump, field, gamma)
    arr, t = _downsample_rows(arr, dump["time"], max_rows)
    x = dump["x"]

    # Color scaling
    ref = cfg["ref"]
    scale = cfg["scale"]
    if ref is not None:
        dev = (arr - ref) * scale
        if vmax_override is not None:
            vmax_scaled = vmax_override * scale
        else:
            absmax = float(np.max(np.abs(dev)))
            vmax_scaled = max(absmax * 1.0, 1e-12)
        vmin, vmax = -vmax_scaled, +vmax_scaled
        plot_arr = dev
        cbar_label = (f"{field} − ref  [{cfg['scale_unit']}]"
                      if cfg["scale_unit"] != "Mach"
                      else "Mach number")
    else:
        plot_arr = arr * scale
        if vmax_override is not None:
            vmin, vmax = (arr.min() * scale, vmax_override * scale)
        else:
            vmin, vmax = float(plot_arr.min()), float(plot_arr.max())
        cbar_label = f"{field}  [{cfg['scale_unit']}]"

    fig, ax = plt.subplots(figsize=figsize)
    extent = [x[0] * 1000.0, x[-1] * 1000.0, t[-1] * 1000.0, t[0] * 1000.0]
    im = ax.imshow(
        plot_arr, aspect="auto", origin="upper", extent=extent,
        cmap=cfg["cmap"], vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )

    # Title + metadata header
    nice_title = title or (
        f"Waterfall — pipe '{pipe_name}', field = {field}"
        + (f"  ({source})" if source else "")
    )
    ax.set_title(nice_title, fontsize=11)
    ax.set_xlabel("position x  [mm]")
    ax.set_ylabel("time  [ms]")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)

    # Footer text with provenance
    n_rows, n_cols = arr.shape
    footer = (
        f"source: {dump_path.name}   pipe: {pipe_name}   "
        f"field: {field}   shape: {n_rows}×{n_cols}   "
        f"t: 0 → {t[-1]*1000:.2f} ms   L: {length_m*1000:.0f} mm   "
        f"vmax: ±{abs(vmax):.3g} ({cfg['scale_unit']})"
    )
    fig.text(0.01, 0.005, footer, fontsize=7, family="monospace", color="#444")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m diagnostics.waterfall_viewer",
        description="Render an x-t waterfall image from a pipe-state .npz dump.",
    )
    p.add_argument("--dump", required=True, type=Path,
                   help="Path to the .npz dump file.")
    p.add_argument("--field", default="pressure",
                   choices=sorted(FIELD_CONFIGS.keys()),
                   help="Field to plot (default: pressure).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output PNG path.")
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                   help="Downsample to at most this many time-axis rows "
                        f"(default: {DEFAULT_MAX_ROWS}).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Override automatic colormap vmax (in source units).")
    p.add_argument("--title", type=str, default=None,
                   help="Override the figure title.")
    p.add_argument("--dpi", type=int, default=130,
                   help="PNG resolution (default: 130).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    out = render_waterfall(
        args.dump, field=args.field, output_path=args.output,
        max_rows=args.max_rows, vmax_override=args.vmax,
        title=args.title, dpi=args.dpi,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
