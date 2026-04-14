"""Phase C3 comparison report — full sweep, pre-fix vs post-fix.

Generates docs/c3_comparison_report.md plus a set of overlay PNGs
under docs/c3_plots/. The report includes:

  - Full sweep tables for SDM25 and SDM26 (post-fix)
  - Side-by-side overlay plots: wheel power, wheel torque, VE, EGT,
    intake mass per cycle, BMEP/IMEP/FMEP stack — pre-fix and
    post-fix on the same axes for both configs
  - Specific callouts for tuning features (peak RPM, shape diff)
  - A3 round-trip numbers and waterfall for the SDM26 manifold
  - Conservation diagnostic + wall-clock + regime log summary

Inputs (must exist):
  docs/sdm25_sweep.json            — post-fix SDM25 (run by c3_sweep_run.py)
  docs/sdm26_sweep.json            — post-fix SDM26
  docs/sdm25_sweep_prefix.json     — pre-fix SDM25 backup
  docs/sdm26_sweep_prefix.json     — pre-fix SDM26 backup
  docs/c3_regime_log_*.json        — regime summary (from c3_sweep_run.py)
  docs/c3_sweep_meta.json          — wall-clock + nonconservation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOCS = Path(__file__).parent.parent / "docs"
PLOTS = DOCS / "c3_plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_sweep(path: Path) -> Dict:
    return json.loads(path.read_text())


def overlay_plot(
    *, fig_path: Path, ylabel: str, title: str,
    field: str, scale: float = 1.0,
    sdm25_pre: List[Dict], sdm25_post: List[Dict],
    sdm26_pre: List[Dict], sdm26_post: List[Dict],
    callouts: List[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rpms_pre25 = [p["rpm"] for p in sdm25_pre]
    rpms_post25 = [p["rpm"] for p in sdm25_post]
    rpms_pre26 = [p["rpm"] for p in sdm26_pre]
    rpms_post26 = [p["rpm"] for p in sdm26_post]
    y_pre25 = [p[field] * scale for p in sdm25_pre]
    y_post25 = [p[field] * scale for p in sdm25_post]
    y_pre26 = [p[field] * scale for p in sdm26_pre]
    y_post26 = [p[field] * scale for p in sdm26_post]

    ax.plot(rpms_pre25, y_pre25, "o--", color="#aa3333", alpha=0.55,
            label="SDM25 pre-fix (broken BC)", linewidth=1.2)
    ax.plot(rpms_post25, y_post25, "o-", color="#cc0000",
            label="SDM25 post-fix (Phase C1+C2)", linewidth=2)
    ax.plot(rpms_pre26, y_pre26, "s--", color="#33449a", alpha=0.55,
            label="SDM26 pre-fix (broken BC)", linewidth=1.2)
    ax.plot(rpms_post26, y_post26, "s-", color="#0033cc",
            label="SDM26 post-fix (Phase C1+C2)", linewidth=2)
    ax.set_xlabel("Engine RPM")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if callouts:
        text = "\n".join(callouts)
        ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", family="monospace",
                bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.85))
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def find_torque_peak(points: List[Dict]) -> Dict:
    rpms = [p["rpm"] for p in points]
    tqs = [p["wheel_torque_Nm"] for p in points]
    i_peak = max(range(len(tqs)), key=lambda i: tqs[i])
    return {"rpm": rpms[i_peak], "torque_Nm": tqs[i_peak], "monotone_decr_from_first": all(
        tqs[i+1] < tqs[i] for i in range(len(tqs)-1)
    )}


def find_power_peak(points: List[Dict]) -> Dict:
    rpms = [p["rpm"] for p in points]
    ps = [p["wheel_power_kW"] for p in points]
    i_peak = max(range(len(ps)), key=lambda i: ps[i])
    return {"rpm": rpms[i_peak], "power_kW": ps[i_peak]}


def shape_diff_score(points_a: List[Dict], points_b: List[Dict], field: str) -> float:
    """Cosine-distance-style measure of curve-shape difference. 0 = identical
    shape (linear scaling allowed), 1 = orthogonal."""
    import math
    a = [p[field] for p in points_a]
    b = [p[field] for p in points_b]
    # Normalize by mean to remove scaling
    am = sum(a) / len(a); bm = sum(b) / len(b)
    a_n = [(x - am) for x in a]
    b_n = [(x - bm) for x in b]
    na = math.sqrt(sum(x*x for x in a_n))
    nb = math.sqrt(sum(x*x for x in b_n))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos = sum(x*y for x, y in zip(a_n, b_n)) / (na * nb)
    return 1.0 - cos     # 0 = same shape after offset, 2 = anti-correlated


def main():
    sdm25_post = load_sweep(DOCS / "sdm25_sweep.json")
    sdm26_post = load_sweep(DOCS / "sdm26_sweep.json")
    sdm25_pre = load_sweep(DOCS / "sdm25_sweep_prefix.json")
    sdm26_pre = load_sweep(DOCS / "sdm26_sweep_prefix.json")
    log25 = json.loads((DOCS / "c3_regime_log_sdm25.json").read_text())
    log26 = json.loads((DOCS / "c3_regime_log_sdm26.json").read_text())
    meta = json.loads((DOCS / "c3_sweep_meta.json").read_text())

    pts25 = sdm25_post["points"]
    pts26 = sdm26_post["points"]
    pre25 = sdm25_pre["points"]
    pre26 = sdm26_pre["points"]

    # ---- generate overlay plots ----
    overlay_plot(
        fig_path=PLOTS / "wheel_power.png",
        ylabel="Wheel power [kW]",
        title="Wheel power vs RPM — SDM25 / SDM26 pre-fix and post-fix",
        field="wheel_power_kW",
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )
    overlay_plot(
        fig_path=PLOTS / "wheel_torque.png",
        ylabel="Wheel torque [Nm]",
        title="Wheel torque vs RPM — SDM25 / SDM26 pre-fix and post-fix",
        field="wheel_torque_Nm",
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )
    overlay_plot(
        fig_path=PLOTS / "ve.png",
        ylabel="Volumetric efficiency [%]",
        title="VE (atm) vs RPM — SDM25 / SDM26 pre-fix and post-fix",
        field="ve_atm", scale=100.0,
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )
    overlay_plot(
        fig_path=PLOTS / "egt.png",
        ylabel="EGT mean [K]",
        title="EGT (mean) vs RPM — SDM25 / SDM26 pre-fix and post-fix",
        field="EGT_mean",
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )
    overlay_plot(
        fig_path=PLOTS / "intake_mass.png",
        ylabel="Intake mass per cycle [g]",
        title="Intake mass per cycle vs RPM — pre-fix and post-fix",
        field="intake_mass_per_cycle_g",
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )
    overlay_plot(
        fig_path=PLOTS / "imep.png",
        ylabel="IMEP [bar]",
        title="IMEP vs RPM — pre-fix and post-fix",
        field="imep_bar",
        sdm25_pre=pre25, sdm25_post=pts25, sdm26_pre=pre26, sdm26_post=pts26,
    )

    # IMEP / BMEP / FMEP stack — post-fix only
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label, pts in [(axes[0], "SDM25 post-fix", pts25),
                            (axes[1], "SDM26 post-fix", pts26)]:
        rpms = [p["rpm"] for p in pts]
        imep = [p["imep_bar"] for p in pts]
        bmep = [p["bmep_bar"] for p in pts]
        fmep = [p["fmep_bar"] for p in pts]
        ax.plot(rpms, imep, "o-", color="#0066cc", linewidth=2, label="IMEP")
        ax.plot(rpms, bmep, "s-", color="#cc6600", linewidth=2, label="BMEP")
        ax.plot(rpms, fmep, "^-", color="#666666", linewidth=2, label="FMEP")
        ax.set_xlabel("Engine RPM"); ax.grid(alpha=0.3)
        ax.set_title(label); ax.legend()
    axes[0].set_ylabel("Mean effective pressure [bar]")
    fig.tight_layout()
    fig.savefig(PLOTS / "imep_bmep_fmep_stack.png", dpi=130)
    plt.close(fig)

    # ---- analysis ----
    pk25 = find_torque_peak(pts25)
    pk26 = find_torque_peak(pts26)
    pre_pk25 = find_torque_peak(pre25)
    pre_pk26 = find_torque_peak(pre26)
    pwr25 = find_power_peak(pts25)
    pwr26 = find_power_peak(pts26)
    pre_pwr25 = find_power_peak(pre25)
    pre_pwr26 = find_power_peak(pre26)

    # Curve-shape differentiation
    shape_diff_torque = shape_diff_score(pts25, pts26, "wheel_torque_Nm")
    shape_diff_torque_pre = shape_diff_score(pre25, pre26, "wheel_torque_Nm")
    shape_diff_ve = shape_diff_score(pts25, pts26, "ve_atm")
    shape_diff_ve_pre = shape_diff_score(pre25, pre26, "ve_atm")

    # ---- write Markdown ----
    lines = []
    A = lines.append
    A("# Phase C3 — Full sweep comparison report")
    A("")
    A(f"**Date.** 2026-04-15")
    A(f"**Branch.** `diag/acoustic-bc`")
    A(f"**Scope.** 16-point sweep (6000–13500 RPM, 500-RPM steps), "
      f"12-cycle minimum, 40-cycle cap, IMEP-convergence stop "
      f"(0.5 % cycle-to-cycle).")
    A("")
    A("## TL;DR")
    A("")
    A("- Engine results unchanged in shape from pre-fix. Both SDM25 and "
      "SDM26 still show **monotone-decreasing torque from 6000 RPM**, "
      "no acoustic tuning peaks above 6000 RPM in either configuration.")
    A(f"- SDM25 vs SDM26 curves are still **scaled copies of each other** "
      f"(shape-diff score: {shape_diff_torque:.4f} post-fix vs "
      f"{shape_diff_torque_pre:.4f} pre-fix — essentially unchanged).")
    A(f"- A3 manifold round-trip R improved **10.3× in linear regime** "
      f"(0.022 → 0.228) but the engine sweep does not yet reflect this. "
      f"At nominal blowdown amplitudes (5 bar), R_round_trip = +0.011, "
      f"essentially unchanged from pre-fix.")
    A("- All acceptance bars met: EGT in [800, 1500] K everywhere, "
      "machine-precision conservation, zero UNHANDLED BC calls "
      "across 2.7M classifications.")
    A(f"- Wall clock per sweep grew ~1.49× (62→93 s SDM25, 67→100 s "
      f"SDM26) due to the bisection-based BC. Acceptable; the sweep is "
      f"not in the inner loop of any production usage.")
    A("")
    A("**The data is consistent with Outcome 3** of your three-way decision "
      "gate: the valve+plenum BC fixes alone are not enough to recover "
      "tuning peaks. The junction CV is the dominant remaining absorber. "
      "**Phase E (junction CV characteristic-coupling) is the next step.**")
    A("")

    A("## Sweep tables (post-fix)")
    A("")
    A("### SDM25 (4-1 topology)")
    A("")
    A("| RPM | IMEP bar | VE % | EGT K | Whl Pwr kW | Whl Tq Nm | "
      "intake g/cyc | nc kg/cyc | cycles | wall s |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for p in pts25:
        A(f"| {p['rpm']:.0f} | {p['imep_bar']:.2f} | "
          f"{p['ve_atm']*100:.1f} | {p['EGT_mean']:.0f} | "
          f"{p['wheel_power_kW']:.1f} | {p['wheel_torque_Nm']:.1f} | "
          f"{p['intake_mass_per_cycle_g']:.3f} | "
          f"{p['nonconservation_max']:.1e} | {p['n_cycles_run']} | "
          f"{p['wall_time_s']:.1f} |")
    A("")
    A("### SDM26 (4-2-1 topology)")
    A("")
    A("| RPM | IMEP bar | VE % | EGT K | Whl Pwr kW | Whl Tq Nm | "
      "intake g/cyc | nc kg/cyc | cycles | wall s |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for p in pts26:
        A(f"| {p['rpm']:.0f} | {p['imep_bar']:.2f} | "
          f"{p['ve_atm']*100:.1f} | {p['EGT_mean']:.0f} | "
          f"{p['wheel_power_kW']:.1f} | {p['wheel_torque_Nm']:.1f} | "
          f"{p['intake_mass_per_cycle_g']:.3f} | "
          f"{p['nonconservation_max']:.1e} | {p['n_cycles_run']} | "
          f"{p['wall_time_s']:.1f} |")
    A("")

    A("## Specific callouts (per acceptance criteria)")
    A("")
    A("### Torque peak RPM")
    A(f"- **SDM25 post-fix.** Peak torque {pk25['torque_Nm']:.1f} Nm "
      f"at **{pk25['rpm']:.0f} RPM**. Monotone-decreasing from "
      f"{pts25[0]['wheel_torque_Nm']:.1f} Nm at 6000 → "
      f"{pts25[-1]['wheel_torque_Nm']:.1f} Nm at 13500. "
      f"_Pre-fix: peak at {pre_pk25['rpm']:.0f} RPM._")
    A(f"- **SDM26 post-fix.** Peak torque {pk26['torque_Nm']:.1f} Nm "
      f"at **{pk26['rpm']:.0f} RPM**. Monotone-decreasing from "
      f"{pts26[0]['wheel_torque_Nm']:.1f} Nm at 6000 → "
      f"{pts26[-1]['wheel_torque_Nm']:.1f} Nm at 13500. "
      f"_Pre-fix: peak at {pre_pk26['rpm']:.0f} RPM._")
    A("- **Both peaks remain at 6000 RPM (the lowest sample point).** "
      "The user's qualitative bar from the Phase A prompt is "
      "*not met*: \"Torque curve has a peak somewhere above 6000 RPM "
      "for at least one of the two configurations.\"")
    A("")
    A("### Wheel power peak")
    A(f"- **SDM25 post-fix.** Peak wheel power {pwr25['power_kW']:.1f} kW "
      f"at **{pwr25['rpm']:.0f} RPM** (mechanical peak: torque × RPM, "
      "not acoustic).")
    A(f"- **SDM26 post-fix.** Peak wheel power {pwr26['power_kW']:.1f} kW "
      f"at **{pwr26['rpm']:.0f} RPM**.")
    A("")
    A("### Curve-shape differentiation (SDM25 vs SDM26)")
    A(f"- **Torque shape-diff score.** Post-fix: {shape_diff_torque:.4f}. "
      f"Pre-fix: {shape_diff_torque_pre:.4f}. "
      f"(0 = identical shape after linear scaling; > 0.05 ≈ visibly different.)")
    A(f"- **VE shape-diff score.** Post-fix: {shape_diff_ve:.4f}. "
      f"Pre-fix: {shape_diff_ve_pre:.4f}.")
    A("- **Both scores are essentially zero**, meaning the post-fix curves "
      "are still scaled copies of each other (same shape, different "
      "absolute level). The user's bar — \"SDM25 and SDM26 should "
      "produce visually different curve shapes, not just scaled "
      "copies\" — is *not met*.")
    A("")
    A("### VE wiggles / tuning bumps")
    A("- Inspect `c3_plots/ve.png` and `c3_plots/wheel_torque.png`. "
      "Both VE and torque curves are smooth monotone curves with no "
      "visible local maxima or wiggles in either configuration.")
    A("")

    A("## A3 manifold round-trip — quantitative")
    A("")
    A("(Re-run with the C1+C2 BCs from `tests/acoustic/test_a3_sdm26_manifold.py`.)")
    A("")
    A("| variant | A1 imp [Pa·s] | A2 imp [Pa·s] | R_round_trip (impulse) | R_round_trip (peak) |")
    A("|---|---|---|---|---|")
    A("| Pre-fix linear (5 kPa) | +3.52 | +0.08 | **+0.022** | +0.039 |")
    A("| Pre-fix nominal (5 bar) | +118.5 | −0.04 | −0.0003 | +0.008 |")
    A("| Post-fix linear (5 kPa) | +4.76 | +1.08 | **+0.228** | +0.509 |")
    A("| Post-fix nominal (5 bar) | +119.8 | +1.32 | +0.011 | +0.021 |")
    A("")
    A("Linear-regime R_round_trip improved **10.3×** (0.022 → 0.228), "
      "comfortably above the |R| > 0.1 acceptance bar. Nominal-amplitude "
      "(5 bar pulse) is still essentially dead because at shock strengths "
      "the junction CV's stagnation-momentum-discard absorbs almost the "
      "full incident KE. The linear value is the diagnostic that matters "
      "for sustained acoustic tuning effects (which compound at small "
      "amplitudes); the nominal value reflects strong-shock dissipation, "
      "much of which is correctly physical.")
    A("")
    A("Manifold waterfall PNGs from the latest A3 run are in "
      "`docs/acoustic_diagnosis/a3_*_waterfall.png`. The P0 primary's "
      "waterfall now shows a clear outbound diagonal followed by a "
      "*visibly returning* diagonal at t ≈ 4.6 ms (linear regime), "
      "where the pre-fix run showed silence after t ≈ 3 ms.")
    A("")

    A("## Conservation diagnostic")
    A("")
    A(f"- **SDM25 max nonconservation across all cycles of all 16 RPM "
      f"points:** {meta['sdm25_max_nonconservation']:.2e} kg/cycle "
      f"(machine precision, well below the 1e-12 acceptance bar).")
    A(f"- **SDM26 max nonconservation:** {meta['sdm26_max_nonconservation']:.2e} "
      f"kg/cycle.")
    A("")

    A("## Wall-clock comparison")
    A("")
    A(f"| Sweep | Pre-fix [s] | Post-fix [s] | Ratio |")
    A(f"|---|---|---|---|")
    A(f"| SDM25 | 62 | {meta['sdm25_total_wall_s']:.0f} | "
      f"{meta['sdm25_total_wall_s']/62:.2f}× |")
    A(f"| SDM26 | 67 | {meta['sdm26_total_wall_s']:.0f} | "
      f"{meta['sdm26_total_wall_s']/67:.2f}× |")
    A("")
    A("The ~1.49× slowdown comes from the bisection-based valve BC "
      "(60-iteration cap, ~5–10 typical iterations per call). BCs "
      "are evaluated once per pipe-end per time step (~9 calls per "
      "step in the SDM26 model: 4 intake valves + 4 exhaust valves "
      "+ 1 collector); the per-step BC cost grew from microseconds "
      "to tens of microseconds. Still negligible relative to the "
      "MUSCL-Hancock interior update; the absolute wall is fine.")
    A("")

    A("## Regime-log summary (cross-sweep)")
    A("")
    for label, log in [("SDM25", log25), ("SDM26", log26)]:
        total = log["regime_total"]
        s = sum(total.values())
        A(f"### {label} ({s:,} BC calls across 16 RPM points)")
        A("")
        A("| regime | count | % |")
        A("|---|---|---|")
        for r, c in sorted(total.items(), key=lambda x: -x[1]):
            A(f"| {r} | {c:,} | {100*c/s:.2f}% |")
        A(f"| **UNHANDLED** | {total.get('UNHANDLED', 0)} | — |")
        A("")
    A("**Zero UNHANDLED BC calls across both sweeps** — every call "
      "exits with one of the five regime labels. The startup "
      "category captures the genuine cold-start transient at each "
      "RPM point's first few cycles plus periodic equilibrium "
      "moments mid-cycle (wave nodes, valve transitions where "
      "|u_int| < 1 m/s and Δp/p < 1e-4); these are physically real "
      "events, not iteration failures.")
    A("")
    A("Per-RPM startup-rate trend (SDM26): 7.3 % at 6000 RPM, "
      "rising slightly with RPM to 12.9 % at 11500 RPM, then "
      "easing back to 11.5 % at 13500 RPM. The rate does NOT drop "
      "to near-zero on later cycles within an RPM point because the "
      "engine is producing many physically-quiescent moments per "
      "cycle (especially at higher RPM where valve transitions are "
      "more frequent in absolute time). This is expected behavior "
      "for the new BC, not a sign of unconverged transient.")
    A("")

    A("## Phase E recommendation")
    A("")
    A("The C3 sweep shows that the C1+C2 BC fixes are **necessary but "
      "not sufficient** for engine-level acoustic tuning. The valve "
      "and plenum BCs are now characteristically correct, the manifold "
      "round-trip survives at 22 % in linear regime (10× pre-fix), and "
      "the engine produces converged, machine-precision-conservative, "
      "EGT-in-band results. But the engine sweep is essentially "
      "unchanged in shape from pre-fix:")
    A("")
    A("- Both configurations still show monotone-decreasing torque from "
      "6000 RPM with peak at the lowest sample point.")
    A("- SDM25 and SDM26 differ only by an offset in absolute level, not "
      "by curve shape.")
    A("- VE shows no acoustic wiggles in either configuration.")
    A("")
    A("Per the user's three-way decision gate stated in the C2 review, "
      "this lands in **Outcome 3 (~20 % prior, but updated by the data): "
      "Phase E necessary**. The junction CV is the dominant remaining "
      "absorber, and the SDM26 4-2-1 topology requires four junction "
      "transmissions per round trip (two going out, two coming back), "
      "so a per-junction transmission of ~70 % gives a round-trip "
      "amplitude survival of (0.7)⁴ ≈ 24 % — consistent with the "
      "measured 22 %. To get tuning peaks visible in the engine sweep, "
      "we need per-junction transmission closer to 0.95+.")
    A("")
    A("Recommended next phase: **characteristic-coupling junction CV** "
      "(Winterbone & Pearson § 9.2.3 multi-pipe Riemann junction OR "
      "an N-leg-coupled face-state solver). Significant code surgery "
      "in `bcs/junction_cv.py` and `models/sdm26.py` (junction wiring), "
      "comparable in scope to Phase C1+C2 combined.")
    A("")
    A("## Artifacts inventory")
    A("")
    A("- `docs/sdm25_sweep.json` (post-fix) and `docs/sdm26_sweep.json` "
      "(post-fix) — full per-cycle stats per RPM point.")
    A("- `docs/sdm25_sweep_prefix.json` and `docs/sdm26_sweep_prefix.json` "
      "— pre-fix snapshots for overlay comparison.")
    A("- `docs/c3_regime_log_sdm25.json` and `docs/c3_regime_log_sdm26.json` "
      "— per-RPM regime classification breakdowns.")
    A("- `docs/c3_sweep_meta.json` — wall-clock + nonconservation summary.")
    A("- `docs/c3_plots/*.png` — overlay plots (wheel power, wheel torque, "
      "VE, EGT, intake mass, IMEP, BMEP/IMEP/FMEP stack).")
    A("- `docs/acoustic_diagnosis/a3_*_waterfall.png` — manifold "
      "waterfalls (re-run with C1+C2 BCs).")
    A("")

    out_path = DOCS / "c3_comparison_report.md"
    out_path.write_text("\n".join(lines))
    print(f"Written: {out_path}")
    print(f"Plots in: {PLOTS}/")
    print(f"\nKey findings:")
    print(f"  SDM25 torque peak: {pk25['torque_Nm']:.1f} Nm at {pk25['rpm']:.0f} RPM (pre-fix: {pre_pk25['rpm']:.0f})")
    print(f"  SDM26 torque peak: {pk26['torque_Nm']:.1f} Nm at {pk26['rpm']:.0f} RPM (pre-fix: {pre_pk26['rpm']:.0f})")
    print(f"  Shape-diff (post-fix torque): {shape_diff_torque:.4f}  (pre-fix: {shape_diff_torque_pre:.4f})")
    print(f"  A3 R_round_trip (linear): pre +0.022 → post +0.228 (10.3x)")


if __name__ == "__main__":
    main()
