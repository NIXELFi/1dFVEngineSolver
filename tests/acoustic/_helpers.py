"""Shared driver + diagnostics for acoustic reflection tests.

Provides:
  - ``AcousticRun``: container for one single-pipe run (state, time series,
    waterfall buffer).
  - ``step_pipe_no_sources``: one MUSCL-Hancock step with NO friction/heat
    source terms (those would add numerical dissipation that confuses
    reflection-coefficient measurement).
  - ``run_acoustic``: top-level driver that handles BC fill, stepping, and
    probe/waterfall recording, with downsampling.
  - ``arrival_peaks`` + ``reflection_coefficients``: extract successive
    signed-peak arrivals at a probe and compute per-reflection R values.
  - ``save_waterfall_png`` + ``save_timeseries_png``: plotting utilities
    that write into docs/acoustic_diagnosis/.

The reflection-coefficient convention follows the user-specified design:
given successive signed peak amplitudes A_1, A_2, A_3, ... at an interior
probe, with each successive arrival separated by one boundary reflection,
R_per_reflection = A_{n+1} / A_n. When the wave bounces between two ends
of known type (e.g. a perfect wall with R ≈ +1 and a diagnosis-target
valve BC), the even/odd-index ratios isolate each end's R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from solver.muscl import LIMITER_MINMOD, cfl_dt, muscl_hancock_step
from solver.state import (
    I_E_A,
    I_MOM_A,
    I_RHO_A,
    I_Y_A,
    PipeState,
)

# Where all acoustic diagnosis artifacts are written.
DIAG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "acoustic_diagnosis"

# Standard atmosphere used everywhere in these tests.
P_ATM = 101325.0
T_ATM = 300.0
GAMMA = 1.4
R_AIR = 287.0
RHO_ATM = P_ATM / (R_AIR * T_ATM)


# -----------------------------------------------------------------------------
# Pipe scratch + stepping
# -----------------------------------------------------------------------------


def ensure_scratch(pipe: PipeState) -> Dict[str, np.ndarray]:
    """Allocate (or reuse) the MUSCL scratch buffers on a pipe."""
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


def step_pipe_no_sources(pipe: PipeState, dt: float, gamma: float = GAMMA) -> None:
    """One MUSCL-Hancock + HLLC step. No friction, no heat transfer.

    Ghost cells MUST be filled by the caller before invoking.
    """
    buf = ensure_scratch(pipe)
    muscl_hancock_step(
        pipe.q, pipe.area, pipe.area_f, pipe.dx, dt, gamma,
        pipe.n_ghost, LIMITER_MINMOD,
        buf["w"], buf["slopes"], buf["wL"], buf["wR"], buf["flux"],
    )


# -----------------------------------------------------------------------------
# Probe + waterfall recording
# -----------------------------------------------------------------------------


def cell_index_at(pipe: PipeState, x_target: float) -> int:
    """Return the real-cell array index closest to x_target along the pipe.

    x=0 is the LEFT end of the real domain, x=L is the RIGHT end.
    """
    i_real = int(round(x_target / pipe.dx - 0.5))
    i_real = max(0, min(pipe.n_cells - 1, i_real))
    return pipe.n_ghost + i_real


def pipe_pressure(pipe: PipeState) -> np.ndarray:
    """Pressure in every real cell (length n_cells)."""
    s = pipe.real_slice()
    A = pipe.area[s]
    rho = pipe.q[s, I_RHO_A] / A
    mom = pipe.q[s, I_MOM_A] / A
    E = pipe.q[s, I_E_A] / A
    u = mom / rho
    gm1 = pipe.gamma - 1.0
    return gm1 * (E - 0.5 * rho * u * u)


def pipe_velocity(pipe: PipeState) -> np.ndarray:
    s = pipe.real_slice()
    A = pipe.area[s]
    rho = pipe.q[s, I_RHO_A] / A
    mom = pipe.q[s, I_MOM_A] / A
    return mom / rho


def pipe_temperature(pipe: PipeState) -> np.ndarray:
    s = pipe.real_slice()
    A = pipe.area[s]
    rho = pipe.q[s, I_RHO_A] / A
    mom = pipe.q[s, I_MOM_A] / A
    E = pipe.q[s, I_E_A] / A
    u = mom / rho
    gm1 = pipe.gamma - 1.0
    p = gm1 * (E - 0.5 * rho * u * u)
    return p / (rho * R_AIR)


# -----------------------------------------------------------------------------
# Generic driver
# -----------------------------------------------------------------------------


@dataclass
class ProbeRecord:
    x: float             # target position [m]
    cell: int            # actual cell index used
    t: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)
    u: List[float] = field(default_factory=list)
    T: List[float] = field(default_factory=list)


@dataclass
class AcousticRun:
    """Result of one acoustic test run."""
    pipes: Dict[str, PipeState]
    t_end: float
    time: np.ndarray                 # (n_rows,) downsampled timestamps [s]
    waterfalls: Dict[str, np.ndarray]  # name → (n_rows, n_cells) pressure [Pa]
    probes: Dict[str, Dict[str, ProbeRecord]]  # pipe name → probe label → record
    dt_history: List[float]
    n_steps: int
    # Full primitive history for each pipe — populated alongside waterfalls.
    # Each value dict has keys 'p', 'u', 'T', 'rho', 'Y' each → (n_rows, n_cells).
    # Used by diagnostics/waterfall_viewer.py via save_pipe_dump.
    state_history: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


def run_acoustic(
    *,
    pipes: Dict[str, PipeState],
    bc_apply: Callable[[float], None],
    t_end: float,
    probes_spec: Dict[str, Dict[str, float]],   # pipe_name -> {label: x_target}
    waterfall_rows: int = 500,
    cfl: float = 0.5,
    post_step_hook: Optional[Callable[[float, float], None]] = None,
    gamma: float = GAMMA,
) -> AcousticRun:
    """Advance one or more pipes in lockstep with a single BC-fill callback.

    Parameters
    ----------
    pipes
        Named pipes. All are stepped with the same dt.
    bc_apply
        Called each step BEFORE the MUSCL update with the current time t. It
        should populate all ghost cells (valve, wall, reservoir, junction).
    post_step_hook
        Optional: called AFTER the step with (t, dt). Use for junction CV
        ``absorb_fluxes`` or any other post-step bookkeeping.
    probes_spec
        For each pipe, a dict of ``{probe_label: x_target_m}``.
    """
    # Initialize scratch on every pipe
    for pipe in pipes.values():
        ensure_scratch(pipe)

    # Resolve probe cell indices
    probes: Dict[str, Dict[str, ProbeRecord]] = {}
    for name, spec in probes_spec.items():
        pipe = pipes[name]
        probes[name] = {
            label: ProbeRecord(x=x, cell=cell_index_at(pipe, x))
            for label, x in spec.items()
        }

    # Waterfall buffers — preallocate an over-sized list per pipe then truncate.
    n_cells = {name: pipe.n_cells for name, pipe in pipes.items()}
    wf_rows: Dict[str, List[np.ndarray]] = {name: [] for name in pipes}
    # Full-state buffers (per-step primitive snapshots) — used by save_pipe_dump
    # for the standalone waterfall viewer (diagnostics/waterfall_viewer.py).
    state_rows: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"p": [], "u": [], "T": [], "rho": [], "Y": []} for name in pipes
    }
    t_rows: List[float] = []
    dt_history: List[float] = []

    def _snapshot_state(name: str, pipe: PipeState) -> None:
        s = pipe.real_slice()
        A = pipe.area[s]
        rho = pipe.q[s, I_RHO_A] / A
        mom = pipe.q[s, I_MOM_A] / A
        E = pipe.q[s, I_E_A] / A
        u = mom / rho
        gm1 = pipe.gamma - 1.0
        p = gm1 * (E - 0.5 * rho * u * u)
        T = p / (rho * R_AIR)
        Y = pipe.q[s, I_Y_A] / (rho * A)
        state_rows[name]["p"].append(p.copy())
        state_rows[name]["u"].append(u.copy())
        state_rows[name]["T"].append(T.copy())
        state_rows[name]["rho"].append(rho.copy())
        state_rows[name]["Y"].append(Y.copy())

    t = 0.0
    n_steps = 0

    # Record initial state (row 0)
    t_rows.append(t)
    for name, pipe in pipes.items():
        wf_rows[name].append(pipe_pressure(pipe).copy())
        _snapshot_state(name, pipe)
    for name, recs in probes.items():
        pipe = pipes[name]
        for label, rec in recs.items():
            i = rec.cell
            A = pipe.area[i]
            rho = pipe.q[i, I_RHO_A] / A
            mom = pipe.q[i, I_MOM_A] / A
            E = pipe.q[i, I_E_A] / A
            u_i = mom / rho
            p_i = (pipe.gamma - 1.0) * (E - 0.5 * rho * u_i * u_i)
            T_i = p_i / (rho * R_AIR)
            rec.t.append(t); rec.p.append(p_i); rec.u.append(u_i); rec.T.append(T_i)

    # We'll compute a stride for downsampling the waterfall to waterfall_rows
    # rows AFTER the run (we record every step into wf_rows and decimate).

    while t < t_end:
        # Compute dt BEFORE applying BCs so that characteristic-junction
        # BCs (which need dt for their MUSCL-aware residual) can be
        # filled correctly. cfl_dt only reads real cells so it does not
        # need valid ghost state.
        dt = min(
            cfl_dt(pipe.q, pipe.area, pipe.dx, gamma, cfl, pipe.n_ghost)
            for pipe in pipes.values()
        )
        if dt <= 0.0:
            raise RuntimeError(f"Positivity violated at t={t:.6e}")
        if t + dt > t_end:
            dt = t_end - t
        dt_history.append(dt)

        bc_apply(t, dt)

        for pipe in pipes.values():
            step_pipe_no_sources(pipe, dt, gamma=gamma)

        if post_step_hook is not None:
            post_step_hook(t, dt)

        t += dt
        n_steps += 1

        # Probes: record every step (cheap)
        for name, recs in probes.items():
            pipe = pipes[name]
            for label, rec in recs.items():
                i = rec.cell
                A = pipe.area[i]
                rho = pipe.q[i, I_RHO_A] / A
                mom = pipe.q[i, I_MOM_A] / A
                E = pipe.q[i, I_E_A] / A
                u_i = mom / rho
                p_i = (pipe.gamma - 1.0) * (E - 0.5 * rho * u_i * u_i)
                T_i = p_i / (rho * R_AIR)
                rec.t.append(t); rec.p.append(p_i); rec.u.append(u_i); rec.T.append(T_i)

        # Waterfall rows: record every step (decimate later)
        t_rows.append(t)
        for name, pipe in pipes.items():
            wf_rows[name].append(pipe_pressure(pipe).copy())
            _snapshot_state(name, pipe)

    # Decimate waterfall rows to at most `waterfall_rows`
    n_total_rows = len(t_rows)
    stride = max(1, n_total_rows // waterfall_rows)
    idxs = list(range(0, n_total_rows, stride))
    if idxs[-1] != n_total_rows - 1:
        idxs.append(n_total_rows - 1)
    time_arr = np.array([t_rows[i] for i in idxs])
    waterfalls = {
        name: np.vstack([wf_rows[name][i] for i in idxs])
        for name in pipes
    }
    state_history = {
        name: {
            field_key: np.vstack([state_rows[name][field_key][i] for i in idxs])
            for field_key in ("p", "u", "T", "rho", "Y")
        }
        for name in pipes
    }

    return AcousticRun(
        pipes=pipes,
        t_end=t_end,
        time=time_arr,
        waterfalls=waterfalls,
        probes=probes,
        dt_history=dt_history,
        n_steps=n_steps,
        state_history=state_history,
    )


# -----------------------------------------------------------------------------
# Pipe-state dump format (consumed by diagnostics/waterfall_viewer.py)
# -----------------------------------------------------------------------------

def save_pipe_dump(
    run: "AcousticRun", pipe_name: str, out_path: Path,
    *, source: str = "",
) -> Path:
    """Persist a pipe's full primitive-state history to an .npz file.

    Format:
      time     — (n_rows,) timestamps [s]
      x        — (n_cols,) cell-centre positions [m]
      pressure — (n_rows, n_cols) [Pa]
      velocity — (n_rows, n_cols) [m/s]
      temperature — (n_rows, n_cols) [K]
      density  — (n_rows, n_cols) [kg/m³]
      Y        — (n_rows, n_cols) burned-mass fraction
      pipe_name (string scalar)
      source   (string scalar; free-form provenance)
      gamma    (float scalar)
      length   (float scalar) [m]

    Consumed by ``diagnostics.waterfall_viewer.render_waterfall``.
    """
    pipe = run.pipes[pipe_name]
    sh = run.state_history[pipe_name]
    x = np.array([(i + 0.5) * pipe.dx for i in range(pipe.n_cells)])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        time=run.time,
        x=x,
        pressure=sh["p"],
        velocity=sh["u"],
        temperature=sh["T"],
        density=sh["rho"],
        Y=sh["Y"],
        pipe_name=str(pipe_name),
        source=str(source),
        gamma=float(pipe.gamma),
        length=float(pipe.dx * pipe.n_cells),
    )
    return out_path


# -----------------------------------------------------------------------------
# Reflection coefficient extraction
# -----------------------------------------------------------------------------


@dataclass
class Arrival:
    t: float       # time of peak
    amp: float     # signed (p - P_ATM) at peak
    idx: int       # index into probe time series


def arrival_peaks(
    t: np.ndarray, p: np.ndarray, *,
    min_separation_s: float,
    min_abs_amp: float = 500.0,
    reference: float = P_ATM,
) -> List[Arrival]:
    """Find signed pressure-perturbation peaks in a probe time series.

    A "peak" is any local extremum of (p - reference) whose magnitude
    exceeds ``min_abs_amp`` and which is separated from adjacent peaks by
    at least ``min_separation_s`` seconds. The sign of ``Arrival.amp`` is
    the sign of (p - reference) at the peak (positive for compression,
    negative for rarefaction).

    Implementation: first-derivative sign change on (p - reference),
    filter by |amp| >= min_abs_amp, then greedy-thin to enforce the
    minimum temporal separation.
    """
    t = np.asarray(t)
    p = np.asarray(p)
    pp = p - reference
    n = len(pp)
    raw: List[int] = []
    for k in range(1, n - 1):
        a, b, c = pp[k - 1], pp[k], pp[k + 1]
        # Local max on (p - ref): b > a and b >= c, or local min: b < a and b <= c.
        is_max = b > a and b >= c
        is_min = b < a and b <= c
        if is_max or is_min:
            if abs(b) >= min_abs_amp:
                raw.append(k)
    # Greedy min-separation enforcement — keep the stronger of nearby peaks.
    thinned: List[int] = []
    for k in raw:
        if not thinned:
            thinned.append(k)
            continue
        if t[k] - t[thinned[-1]] < min_separation_s:
            if abs(pp[k]) > abs(pp[thinned[-1]]):
                thinned[-1] = k
        else:
            thinned.append(k)
    return [Arrival(t=float(t[k]), amp=float(pp[k]), idx=k) for k in thinned]


def successive_ratios(arrivals: Sequence[Arrival]) -> List[float]:
    """Signed ratio A_{n+1} / A_n for consecutive arrivals."""
    out: List[float] = []
    for i in range(1, len(arrivals)):
        prev = arrivals[i - 1].amp
        curr = arrivals[i].amp
        if abs(prev) < 1e-12:
            continue
        out.append(curr / prev)
    return out


def windowed_signed_extremum(
    t: np.ndarray, p: np.ndarray,
    t_start: float, t_end: float,
    reference: float = P_ATM,
) -> Tuple[float, float]:
    """Signed extremum of (p − reference) inside [t_start, t_end].

    Returns (signed_value, time_of_extremum). If the window is empty,
    returns (0.0, 0.5*(t_start+t_end)).
    """
    t = np.asarray(t); p = np.asarray(p)
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return 0.0, 0.5 * (t_start + t_end)
    pp = (p - reference)[mask]
    tt = t[mask]
    k = int(np.argmax(np.abs(pp)))
    return float(pp[k]), float(tt[k])


def windowed_signed_impulse(
    t: np.ndarray, p: np.ndarray,
    t_start: float, t_end: float,
    reference: float = P_ATM,
) -> float:
    """Signed impulse ∫(p − reference) dt over [t_start, t_end].

    For a pulse of roughly rectangular shape with amplitude A and duration
    T_PULSE at the probe, the impulse is ≈ A·T_PULSE. Ratios of impulses
    between successive arrivals are therefore directly the reflection
    coefficients, robust to dispersion-induced peak-position shifts that
    bias windowed-extremum measurements.
    """
    t = np.asarray(t); p = np.asarray(p)
    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    pp = (p - reference)[mask]
    if len(tt) < 2:
        return 0.0
    return float(np.trapz(pp, tt))


@dataclass
class WindowedArrival:
    n: int           # arrival index, 1-based
    t_center: float  # nominal window center [s]
    t_peak: float    # actual time of signed extremum
    amp: float       # signed (p − reference) at extremum
    impulse: float   # signed ∫(p − reference) dt over the window [Pa·s]
    t_window: Tuple[float, float]


def arrivals_at_probe(
    t: np.ndarray, p: np.ndarray,
    *,
    length_m: float,
    c0_m_s: float,
    pulse_width_s: float,
    probe_x_m: float,
    max_arrivals: int = 3,
    reference: float = P_ATM,
) -> List[WindowedArrival]:
    """Build a list of WindowedArrivals at an interior probe.

    Wave model: a rectangular compression pulse launched from the LEFT end
    of a pipe of length ``length_m``, bouncing between the left BC
    (diagnosis target) and the right BC (reflective wall).

    The n-th arrival at the probe is a wave that has traversed
    (2n − 1)·L/2 + (L/2 − probe_x_m if n even else probe_x_m) … actually,
    for a probe at x = x_p, the n-th arrival (1-based, alternating
    directions) occupies the probe during

        [ t_n_start,  t_n_start + pulse_width ]

    where t_n_start depends on n and the probe position:

        n = 1 (outbound):        t_1_start = x_p / c
        n = 2 (wall reflection): t_2_start = (2L − x_p) / c
        n = 3 (valve reflection):t_3_start = (2L + x_p) / c
        n = 4 (wall, again):     t_4_start = (4L − x_p) / c
        …  general: n odd  → (2(n-1)/2 * L + x_p)/c ;  n even → ((n/2)·2L − x_p)/c

    The windowed extremum is taken over [t_n_start, t_n_start + pulse_width].
    """
    out: List[WindowedArrival] = []
    L = length_m
    xp = probe_x_m
    c = c0_m_s
    for n in range(1, max_arrivals + 1):
        # Number of one-way traversals this arrival has done:
        #   outbound (n=1): xp (just the first traversal)
        #   n=2 (wall bounce): 2L − xp
        #   n=3 (valve bounce): 2L + xp
        #   n=4 (wall bounce #2): 4L − xp
        #   n=5 (valve bounce #2): 4L + xp
        #   …  pattern:  n=2k:   2kL − xp ;   n=2k+1:  2kL + xp
        if n == 1:
            dist = xp
        elif n % 2 == 0:
            k = n // 2
            dist = 2 * k * L - xp
        else:
            k = (n - 1) // 2
            dist = 2 * k * L + xp
        t_start = dist / c
        t_end = t_start + pulse_width_s
        amp, t_peak = windowed_signed_extremum(t, p, t_start, t_end, reference)
        impulse = windowed_signed_impulse(t, p, t_start, t_end, reference)
        out.append(WindowedArrival(
            n=n, t_center=0.5 * (t_start + t_end), t_peak=t_peak,
            amp=amp, impulse=impulse, t_window=(t_start, t_end),
        ))
    return out


def reflection_from_windowed(
    arrivals: Sequence[WindowedArrival],
) -> Dict[str, float]:
    """Compute per-bounce reflection coefficients from windowed arrivals.

    Convention (see arrivals_at_probe docstring):
      A_1 = outbound pulse (no reflection, calibrates launch amplitude)
      A_2 / A_1 = R_wall          (right end)
      A_3 / A_2 = R_valve         (left end — the diagnosis target)

    Two measurement variants are reported:

    **Impulse-based (primary).** Signed ∫(p − P_atm) dt over each arrival
    window. This is robust to dispersion-induced peak shifts and pulse-
    shape distortions; the integral over a moving pulse is conserved to
    first order under pure translation, so the ratio of successive window
    impulses gives R directly.

    **Peak-based (secondary, for cross-check).** Signed extremum of
    (p − P_atm) in the window. Biased by leading-edge vs plateau sampling
    but closer to what a pressure probe would read in a single snapshot.

    Returns a dict with both variants and the individual window values.
    Missing (zero-impulse) ratios are reported as NaN.
    """
    def _ratio(num, den):
        if abs(den) < 1e-20:
            return float("nan")
        return num / den

    A_peak = [a.amp for a in arrivals]
    A_imp = [a.impulse for a in arrivals]

    result: Dict[str, float] = {
        "A1_peak": A_peak[0] if len(A_peak) > 0 else float("nan"),
        "A2_peak": A_peak[1] if len(A_peak) > 1 else float("nan"),
        "A3_peak": A_peak[2] if len(A_peak) > 2 else float("nan"),
        "A1_imp": A_imp[0] if len(A_imp) > 0 else float("nan"),
        "A2_imp": A_imp[1] if len(A_imp) > 1 else float("nan"),
        "A3_imp": A_imp[2] if len(A_imp) > 2 else float("nan"),
    }
    # Peak-based (kept for back-compat + cross-check)
    result["R_wall_peak"]  = _ratio(A_peak[1], A_peak[0]) if len(A_peak) >= 2 else float("nan")
    result["R_valve_peak"] = _ratio(A_peak[2], A_peak[1]) if len(A_peak) >= 3 else float("nan")
    # Impulse-based (primary)
    result["R_wall"]  = _ratio(A_imp[1], A_imp[0]) if len(A_imp) >= 2 else float("nan")
    result["R_valve"] = _ratio(A_imp[2], A_imp[1]) if len(A_imp) >= 3 else float("nan")
    return result


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_waterfall_png(
    run: AcousticRun, pipe_name: str, length_m: float, *,
    out_path: Path,
    title: str,
    vmax_kPa: float = 300.0,
) -> None:
    """Write an x-t pressure waterfall for the named pipe.

    x = 0 (pipe left end) is at the LEFT edge of the image, x = L at the
    RIGHT edge. Time runs DOWNWARD from t=0 at the top.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wf = run.waterfalls[pipe_name]
    dev_kPa = (wf - P_ATM) / 1000.0
    n_rows, n_cols = dev_kPa.shape

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0.0, length_m * 1000.0, run.time[-1] * 1000.0, 0.0]  # time downward
    im = ax.imshow(
        dev_kPa, aspect="auto", origin="upper", extent=extent,
        cmap="RdBu_r", vmin=-vmax_kPa, vmax=+vmax_kPa,
        interpolation="nearest",
    )
    ax.set_xlabel("x  [mm]  (0 = pipe left end, L = right)")
    ax.set_ylabel("time  [ms]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("p − P_atm  [kPa]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_timeseries_png(
    run: AcousticRun, pipe_name: str, *,
    out_path: Path,
    title: str,
) -> None:
    """Plot p(t) at every probe of the named pipe on one axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, rec in run.probes[pipe_name].items():
        t_ms = np.array(rec.t) * 1000.0
        dev_kPa = (np.array(rec.p) - P_ATM) / 1000.0
        ax.plot(t_ms, dev_kPa, label=f"{label}  (x={rec.x*1000:.0f} mm)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time  [ms]")
    ax.set_ylabel("p − P_atm  [kPa]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Ghost-cell helpers shared across tests
# -----------------------------------------------------------------------------


def set_uniform_atmosphere(pipe: PipeState) -> None:
    """Initialize every cell (real + ghost) to P_atm, T_atm, u=0, Y=0."""
    from solver.state import set_uniform
    rho = P_ATM / (R_AIR * T_ATM)
    set_uniform(pipe, rho=rho, u=0.0, p=P_ATM, Y=0.0)


def make_always_open_valve(
    *, diameter: float, max_lift: float, seat_angle_deg: float,
    n_valves: int, ld_table: Sequence[float], cd_table: Sequence[float],
):
    """Return (ValveParams, theta_fixed_deg) that produces maximum lift.

    We set open=0, close=720 so the sin² lift profile peaks at θ=360.
    """
    from cylinder.valve import ValveParams
    vp = ValveParams(
        diameter=diameter, max_lift=max_lift,
        open_angle_deg=0.0, close_angle_deg=720.0,
        seat_angle_deg=seat_angle_deg, n_valves=n_valves,
        ld_table=np.array(ld_table, dtype=np.float64),
        cd_table=np.array(cd_table, dtype=np.float64),
    )
    return vp, 360.0
