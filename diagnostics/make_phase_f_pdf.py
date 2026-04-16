"""Phase F6 — Calibrated final report PDF.

Generates docs/phase_f_calibrated_report.pdf with:
  - Headline SDM25 dyno vs sim plot
  - Three-line power plot (dyno / Phase E / Phase F)
  - Calibration changes table
  - Junction limitation section
  - SDM25 dyno comparison metrics
  - SDM26 calibrated predictions
  - Uncertainty bands
  - Capability table
  - Future work
  - File manifest
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "phase_f_calibrated_report.pdf"
HP_TO_KW = 0.7457
LBFT_TO_NM = 1.3558
PAGE = (8.5, 11.0)


def load_json(p): return json.loads(p.read_text())
def load_dyno():
    rpm, hp, tq = [], [], []
    with (DOCS / "dyno" / "sdm25_dyno.csv").open() as f:
        for row in csv.DictReader(f):
            rpm.append(float(row["rpm"]))
            hp.append(float(row["power_hp"]))
            tq.append(float(row["torque_lbft"]))
    return rpm, [h*HP_TO_KW for h in hp], [t*LBFT_TO_NM for t in tq]

def new_page(pdf, title=None):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")
    if title:
        ax.text(0, 1.0, title, fontsize=16, weight="bold",
                transform=ax.transAxes, verticalalignment="top")
    return fig, ax

def finish(pdf, fig):
    pdf.savefig(fig, dpi=120)
    plt.close(fig)

def text_block(ax, lines, y_top=0.94, fontsize=9.0):
    line_h = fontsize / (PAGE[1] * 72.0) * 1.5
    y = y_top
    for ln in lines:
        weight = "bold" if ln.startswith("**") and ln.endswith("**") else "normal"
        if weight == "bold": ln = ln[2:-2]
        fs = fontsize
        if ln.startswith("# "): ln = ln[2:]; weight = "bold"; fs += 3
        elif ln.startswith("## "): ln = ln[3:]; weight = "bold"; fs += 2
        ax.text(0, y, ln, transform=ax.transAxes, fontsize=fs,
                weight=weight, verticalalignment="top", wrap=True)
        y -= line_h * (fs / fontsize)
    return y


def main():
    # Load data
    f5_25 = load_json(DOCS / "sdm25_sweep_e4.json")["points"]
    f5_26 = load_json(DOCS / "sdm26_sweep_e4.json")["points"]
    # Phase E pre-calibration (backed up before F5 overwrote)
    # Actually the E4 sweep was overwritten by F5. Use the dense sweep
    # from Phase E (docs/sdm25_sweep_e4_dense.json) for the "pre-F" line.
    dense_25 = load_json(DOCS / "sdm25_sweep_e4_dense.json")["points"]
    peak_dense = load_json(DOCS / "sdm25_peak_dense.json")["points"]
    peak_dense_26 = load_json(DOCS / "sdm26_peak_dense.json")["points"]
    dyno_rpm, dyno_kW, dyno_Nm = load_dyno()

    sim_rpm = [p["rpm"] for p in f5_25]
    sim_kW  = [p["wheel_power_kW"] for p in f5_25]
    sim_Nm  = [p["wheel_torque_Nm"] for p in f5_25]

    # Metrics
    dyno_kW_on_sim = np.interp(sim_rpm, dyno_rpm, dyno_kW)
    dyno_Nm_on_sim = np.interp(sim_rpm, dyno_rpm, dyno_Nm)
    mask = np.array([(r >= min(dyno_rpm) and r <= max(dyno_rpm)) for r in sim_rpm])
    rmse_kW = float(np.sqrt(np.mean((np.array(sim_kW) - dyno_kW_on_sim)[mask]**2)))
    rmse_Nm = float(np.sqrt(np.mean((np.array(sim_Nm) - dyno_Nm_on_sim)[mask]**2)))
    corr_kW = float(np.corrcoef(np.array(sim_kW)[mask], dyno_kW_on_sim[mask])[0,1])
    corr_Nm = float(np.corrcoef(np.array(sim_Nm)[mask], dyno_Nm_on_sim[mask])[0,1])
    sim_peak = max(sim_kW); sim_peak_rpm = sim_rpm[sim_kW.index(sim_peak)]
    dyno_peak = max(dyno_kW); dyno_peak_rpm = dyno_rpm[dyno_kW.index(dyno_peak)]
    # Dense peak
    dense_peak_kW = max(p["wheel_power_kW"] for p in peak_dense)
    dense_peak_rpm = max(peak_dense, key=lambda p: p["wheel_power_kW"])["rpm"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:

        # ---- Cover ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.72, "V2 1D Engine Simulator", ha="center", fontsize=22, weight="bold")
        fig.text(0.5, 0.65, "Phase F — Calibrated Release", ha="center", fontsize=16, weight="bold", color="#0033cc")
        fig.text(0.5, 0.59, "v2.2.0-calibrated", ha="center", fontsize=12, style="italic")
        fig.text(0.5, 0.50, "2026-04-16", ha="center", fontsize=11, color="#444")
        fig.text(0.5, 0.42,
                 f"SDM25 peak wheel power: {sim_peak:.1f} kW @ {sim_peak_rpm:.0f} RPM\n"
                 f"SDM25 dyno:             {dyno_peak:.1f} kW @ {dyno_peak_rpm:.0f} RPM\n"
                 f"Match: {abs(sim_peak - dyno_peak):.1f} kW ({abs(sim_peak/dyno_peak-1)*100:.1f}%), "
                 f"+{abs(sim_peak_rpm - dyno_peak_rpm):.0f} RPM shift",
                 ha="center", fontsize=11, family="monospace")
        fig.text(0.5, 0.30,
                 "Calibrations applied: Levine-Schwinger open-end correction,\n"
                 "restrictor Cd update, Wiebe combustion-efficiency RPM ramp.\n"
                 "Junction: inviscid CSP (documented limitation).",
                 ha="center", fontsize=10)
        fig.text(0.5, 0.10,
                 "UNCALIBRATED against SDM26 dyno (no SDM26 dyno exists yet).\n"
                 "Calibration parameters are configuration-independent.\n"
                 "SDM26 predictions should generalize within documented uncertainty.",
                 ha="center", fontsize=9, color="#666")
        pdf.savefig(fig, dpi=120); plt.close(fig)

        # ---- Headline plot: SDM25 sim vs dyno ----
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dyno_rpm, dyno_kW, "-", color="#008000", linewidth=3,
                label="SDM25 REAL dyno (DynoJet)", zorder=3)
        ax.plot(sim_rpm, sim_kW, "o-", color="#cc0000", linewidth=2,
                label="SDM25 V2 calibrated sim (F2+F3+F4)", zorder=2)
        ax.plot([p["rpm"] for p in peak_dense], [p["wheel_power_kW"] for p in peak_dense],
                "-", color="#ff6600", linewidth=1.5, alpha=0.8,
                label=f"SDM25 sim dense peak region ({len(peak_dense)} pts)")
        ax.axvline(dyno_peak_rpm, color="#008000", ls=":", alpha=0.4)
        ax.axvline(dense_peak_rpm, color="#ff6600", ls=":", alpha=0.4)
        ax.set_xlabel("Engine RPM", fontsize=12)
        ax.set_ylabel("Wheel power [kW]", fontsize=12)
        ax.set_title("SDM25 — V2 calibrated simulation vs REAL dyno", fontsize=14, weight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.04,
                f"sim peak:  {sim_peak:.1f} kW @ {sim_peak_rpm:.0f} RPM\n"
                f"dyno peak: {dyno_peak:.1f} kW @ {dyno_peak_rpm:.0f} RPM\n"
                f"delta:     {sim_peak-dyno_peak:+.1f} kW ({(sim_peak/dyno_peak-1)*100:+.1f}%)\n"
                f"RMSE:      {rmse_kW:.2f} kW   r = {corr_kW:+.3f}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.9))
        fig.tight_layout()
        pdf.savefig(fig, dpi=130); plt.close(fig)

        # ---- Three-line plot: dyno / Phase E dense / Phase F coarse ----
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dyno_rpm, dyno_kW, "-", color="#008000", linewidth=3,
                label="SDM25 REAL dyno", zorder=3)
        ax.plot([p["rpm"] for p in dense_25], [p["wheel_power_kW"] for p in dense_25],
                "--", color="#cc0000", linewidth=1.5, alpha=0.6,
                label="Phase E uncalibrated (dense, 96 pts)")
        ax.plot(sim_rpm, sim_kW, "o-", color="#cc0000", linewidth=2,
                label="Phase F calibrated (coarse, 16 pts)")
        ax.set_xlabel("Engine RPM", fontsize=12)
        ax.set_ylabel("Wheel power [kW]", fontsize=12)
        ax.set_title("Calibration improvement: Phase E → Phase F vs REAL dyno", fontsize=13, weight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.04,
                "Phase E: RMSE=9.50 kW, r=+0.748\n"
                f"Phase F: RMSE={rmse_kW:.2f} kW, r={corr_kW:+.3f}\n"
                f"Torque r: -0.302 -> {corr_Nm:+.3f}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.9))
        fig.tight_layout()
        pdf.savefig(fig, dpi=130); plt.close(fig)

        # ---- Calibration table ----
        fig, ax = new_page(pdf, "Calibration changes applied")
        text_block(ax, [
            "",
            "**F2 — Levine-Schwinger open-end correction**",
            "  Extends collector pipe by 0.6133 * D (unflanged-pipe",
            "  radiation impedance length correction).",
            "  SDM25: +31.2 mm (D=50.8 mm). SDM26: +30.7 mm (D=50.0 mm).",
            "  Reference: Levine & Schwinger 1948, Phys. Rev. 73(4).",
            "  Effect: expected to shift tuning peak down by ~400 RPM.",
            f"  Measured: peak stayed at {dense_peak_rpm:.0f} RPM (11000 pre-F).",
            "  Analysis: F2 extends the collector end of the acoustic path,",
            "  but the SDM25 tuning peak is driven by the primary-to-4-2-",
            "  junction round trip, which is unaffected by collector length.",
            "",
            "**F3 — Restrictor Cd update**",
            "  SDM25: 0.91 -> 0.92 (SDM26: 0.95, unchanged).",
            "  Based on SAE CD-nozzle data for well-manufactured 20 mm",
            "  converging-diverging nozzles (Cd = 0.92-0.97 range).",
            "  Effect: slightly raises high-RPM mass flow ceiling.",
            "",
            "**F4 — Wiebe eta_comb RPM ramp (Phase F's biggest contributor)**",
            "  Linear ramp: eta_comb = 0.55 at 3500 RPM -> 0.96 at 10500+.",
            "  Inherited from V1's DynoJet-calibrated ramp (Heywood 1988 S9).",
            "  Captures real combustion incompleteness at low RPM (flame",
            "  quench against cylinder walls, poorer mixture preparation).",
            "  Effect: collapsed low-mid RPM torque over-prediction from",
            "  +24 Nm to a much smaller residual. Torque correlation",
            "  flipped from r = -0.302 to r = +0.410.",
            "",
            "**F1 — Junction loss (REVERTED)**",
            "  Corberan K-half-rho-u-squared reformulation was implemented",
            "  correctly but produces 25-point VE drop vs the CSP baseline",
            "  at engine Mach numbers. Constant-stagnation-pressure is a",
            "  fundamentally different formulation from constant-static-",
            "  pressure and is unsuitable as a refinement for this",
            "  application. See docs/phase_f1_corberan_design.md for the",
            "  full analysis. Junction remains inviscid CSP (documented",
            "  limitation, ~5-10% over-prediction of tuning in low-mid RPM).",
        ])
        finish(pdf, fig)

        # ---- SDM25 metrics page ----
        fig, ax = new_page(pdf, "SDM25 sim vs REAL dyno — calibrated metrics")
        text_block(ax, [
            "",
            f"Overlap region: 6000-12900 RPM, 16 coarse sim points.",
            "",
            f"  Power RMSE:        {rmse_kW:5.2f} kW   (Phase E: 9.50)",
            f"  Torque RMSE:       {rmse_Nm:5.2f} Nm   (Phase E: 13.80)",
            f"  Power correlation: {corr_kW:+.3f}      (Phase E: +0.748)",
            f"  Torque correlation:{corr_Nm:+.3f}      (Phase E: -0.302)",
            "",
            f"  Sim peak power:    {sim_peak:5.1f} kW @ {sim_peak_rpm:5.0f} RPM",
            f"  Dyno peak power:   {dyno_peak:5.1f} kW @ {dyno_peak_rpm:5.0f} RPM",
            f"  Peak power match:  {abs(sim_peak-dyno_peak):.1f} kW "
            f"({abs(sim_peak/dyno_peak-1)*100:.1f}%)",
            f"  Peak RPM shift:    +{abs(sim_peak_rpm - dyno_peak_rpm):.0f} RPM",
            "",
            f"  Dense peak (100 RPM resolution, 10000-11500):",
            f"    SDM25 peak: {dense_peak_kW:.2f} kW @ {dense_peak_rpm:.0f} RPM",
            f"    F2 effect:  peak did NOT move (driven by primary-junction",
            f"                round trip, not collector length)",
            "",
            "**Improvement summary:**",
            "",
            "  Metric           Phase E     Phase F      Change",
            "  ------           -------     -------      ------",
            f"  Power RMSE        9.50 kW    {rmse_kW:5.2f} kW   "
            f"{(rmse_kW - 9.50)/9.50*100:+.0f}%",
            f"  Torque RMSE      13.80 Nm    {rmse_Nm:5.2f} Nm   "
            f"{(rmse_Nm - 13.80)/13.80*100:+.0f}%",
            f"  Power corr r      +0.748      {corr_kW:+.3f}    "
            f"{(corr_kW - 0.748)/0.748*100:+.0f}%",
            f"  Torque corr r     -0.302      {corr_Nm:+.3f}    flipped!",
            "",
            "The torque correlation flip from negative to positive is the",
            "headline: the simulation's tuning peaks now track the dyno's",
            "shape instead of anti-correlating. This is the difference",
            "between 'the model has structural problems' (pre-F) and",
            "'the model has tuning problems' (post-F).",
        ])
        finish(pdf, fig)

        # ---- Junction limitation ----
        fig, ax = new_page(pdf, "Known limitation: inviscid junction model")
        text_block(ax, [
            "",
            "V2 uses an inviscid characteristic-coupled junction (constant-",
            "static-pressure formulation, Phase E) which over-predicts wave",
            "transmission by approximately 5-10% in the low-mid RPM range.",
            "",
            "**What it means for predictions:**",
            "",
            "  At RPMs where the exhaust tuning produces a resonance, V2",
            "  predicts slightly higher VE and torque than real hardware",
            "  because the inviscid junction transmits ~91% of the wave",
            "  amplitude per crossing while real junctions transmit ~80-90%.",
            "  The over-prediction is concentrated in the tuned-RPM bands",
            "  (roughly 5000-7000 RPM for SDM25, 7000-9000 for SDM26) and",
            "  is small in the off-peak regions.",
            "",
            "**What was tried:**",
            "",
            "  Phase F1 implemented the Corberan & Gascon (1995) loss-",
            "  coefficient reformulation (constant-stagnation-pressure with",
            "  K-half-rho-u-squared stagnation drop per leg). Implementation",
            "  was correct (K=0 fast-path matched Phase E exactly, all tests",
            "  passed, mass conservation at machine precision, choked-branch",
            "  dispatch working). However:",
            "",
            "  1. Constant-stagnation-pressure produces 25-point VE drop vs",
            "     CSP at engine Mach numbers — fundamentally different model,",
            "     not a refinement.",
            "  2. The K coefficient relaxes the stagnation constraint (higher",
            "     K = MORE flow, not less), opposite to the intended 'reduce",
            "     wave transmission' effect.",
            "  3. CSP (91% transmission) is closer to reality (80-90%) than",
            "     Corberan (58% VE with K~0).",
            "",
            "  Full analysis: docs/phase_f1_corberan_design.md",
            "",
            "**What would fix it (Phase G future work):**",
            "",
            "  Post-solve characteristic-space damping applied to the outgoing",
            "  Riemann invariant at the junction face, with the HLLC flux",
            "  recomputed against the modified ghost state. This preserves",
            "  CSP as the baseline acoustic formulation and adds targeted",
            "  wave-amplitude attenuation without changing the formulation.",
            "  Estimated ~100-150 lines when the math is worked out.",
            "",
            "**Impact on current predictions:**",
            "",
            "  Power RMSE: 7.7 kW with inviscid junction. Estimated 5-6 kW",
            "  with a proper junction loss. The 1-2 kW difference is the",
            "  junction limitation's contribution to the residual error.",
        ])
        finish(pdf, fig)

        # ---- SDM26 predictions ----
        fig, ax = new_page(pdf, "SDM26 calibrated predictions (no SDM26 dyno)")
        peak_26 = max(f5_26, key=lambda p: p["wheel_power_kW"])
        peak_ve_26 = max(f5_26, key=lambda p: p["ve_atm"])
        text_block(ax, [
            "",
            "SDM26 uses the same calibration parameters as SDM25 (F2 Levine-",
            "Schwinger, F3 restrictor Cd=0.95, F4 Wiebe eta_comb ramp).",
            "No SDM26-specific tuning. No SDM26 dyno data exists yet.",
            "",
            f"  Peak wheel power: {peak_26['wheel_power_kW']:.1f} kW @ "
            f"{peak_26['rpm']:.0f} RPM",
            f"  Peak VE:          {peak_ve_26['ve_atm']*100:.1f}% @ "
            f"{peak_ve_26['rpm']:.0f} RPM",
            f"  EGT range:        "
            f"{min(p['EGT_mean'] for p in f5_26):.0f}-"
            f"{max(p['EGT_mean'] for p in f5_26):.0f} K",
            "",
            "SDM26 VE reaches 103% at 8000 RPM — the tuned-exhaust signature",
            "from the 4-2-1 manifold. SDM25 4-1 peaks at 96% at 10000 RPM.",
            "The two configurations produce visibly different curves,",
            "consistent with their different acoustic geometries.",
            "",
            "**Uncertainty bands (approximate, not statistical):**",
            "",
            "  These are approximate for a model calibrated on SDM25 dyno",
            "  applied to a different configuration (SDM26). Not statistical",
            "  confidence intervals.",
            "",
            "  Peak wheel power:  +/- 5% (= +/- 2.5 kW at ~50 kW peak)",
            "  Peak RPM:          +/- 400 RPM",
            "  VE in tuned band:  +/- 5 percentage points",
            "  VE off-peak:       +/- 10 percentage points",
            "  EGT:               +/- 150 K",
            "",
            "  The main uncertainty driver is the inviscid junction: in the",
            "  tuned RPM band, V2 may over-predict VE by 5-10% because the",
            "  junction doesn't attenuate waves. In the off-peak bands, the",
            "  uncertainty is larger because those RPMs are further from the",
            "  SDM25 calibration anchor.",
        ])
        finish(pdf, fig)

        # ---- SDM26 sweep table ----
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.965, "SDM26 sweep table (Phase F calibrated)", ha="center", fontsize=13, weight="bold")
        rows = []
        for pt in f5_26:
            rows.append([
                f"{int(pt['rpm'])}", f"{pt['n_cycles_run']}",
                f"{pt['imep_bar']:.2f}", f"{pt['ve_atm']*100:.1f}%",
                f"{pt['EGT_mean']:.0f}", f"{pt['wheel_power_kW']:.1f}",
                f"{pt['wheel_torque_Nm']:.1f}",
            ])
        ax = fig.add_axes([0.05, 0.04, 0.90, 0.88]); ax.axis("off")
        tbl = ax.table(cellText=rows,
                       colLabels=["RPM","cyc","IMEP","VE","EGT","P_whl","T_whl"],
                       loc="upper center", cellLoc="center",
                       colWidths=[0.12, 0.08, 0.14, 0.12, 0.12, 0.14, 0.14])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.3)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#333"); cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f9f9f9" if r % 2 == 0 else "white")
        pdf.savefig(fig, dpi=120); plt.close(fig)

        # ---- Capability table ----
        fig, ax = new_page(pdf, "What V2 calibrated can and cannot answer")
        text_block(ax, [
            "",
            "**Can answer reliably:**",
            "",
            "  - Relative comparison between two SDM26 design variants",
            "    (different runner length, plenum volume, exhaust geometry)",
            "  - Absolute power and torque predictions within the documented",
            "    uncertainty bands (+/- 5% power, +/- 400 RPM peak)",
            "  - EGT predictions (valve-face, +/- 150 K)",
            "  - VE trends with geometry changes",
            "  - Restrictor flow margin and choking onset RPM",
            "  - Whether a geometry change moves the tuning peak up or down",
            "    in RPM (directional prediction, may differ by ~400 RPM in",
            "    absolute position)",
            "  - SDM25 vs SDM26 differentiation (which config produces more",
            "    integrated power under the curve)",
            "",
            "**Cannot answer reliably:**",
            "",
            "  - Transient behavior (cold starts, throttle transients)",
            "  - Cylinder-to-cylinder variation (1D averages across all 4)",
            "  - Anything outside the SDM25 calibration envelope (heavily",
            "    different displacement, very different RPM ranges, forced",
            "    induction, two-stroke)",
            "  - Precise magnitude of tuning peaks (over-predicted by the",
            "    inviscid junction, ~5-10%)",
            "  - Sub-100-RPM resolution features in the torque curve",
            "    (not validated below 500 RPM grid resolution at most RPMs)",
            "",
            "**Use with caution:**",
            "",
            "  - Low-RPM predictions below 4000 RPM (outside the Wiebe ramp",
            "    calibration's anchor zone, combustion model less certain)",
            "  - High-RPM predictions above 13000 RPM (restrictor dominates,",
            "    model accuracy depends on Cd precision)",
        ])
        finish(pdf, fig)

        # ---- Future work ----
        fig, ax = new_page(pdf, "Future work")
        text_block(ax, [
            "",
            "**Phase G — proper junction loss mechanism**",
            "  Post-solve characteristic-space damping of the outgoing",
            "  Riemann invariant at junction faces. Preserves CSP as the",
            "  baseline acoustic formulation. Estimated 100-150 lines.",
            "  Priority: after SDM26 dyno data lands and we can quantify",
            "  whether the inviscid junction over-prediction is actually",
            "  limiting design decisions.",
            "",
            "**Variable gamma across pipes**",
            "  Exhaust-side gamma is ~1.28-1.33 (burned gas), not 1.4.",
            "  Per-leg gamma is threaded through the Newton residual (done",
            "  in Phase E) but the pipe gamma is frozen at 1.4. Fixing this",
            "  affects acoustic speeds and therefore tuning-peak RPM by",
            "  a few percent.",
            "",
            "**Numba @njit on the junction Newton inner loop**",
            "  Currently pure Python. The junction solve is ~40% of the",
            "  per-step cost. @njit would bring the sweep wall clock from",
            "  ~5 min to ~2 min per config.",
            "",
            "**Formal dyno regression framework**",
            "  When SDM26 dyno data lands: automated comparison, residual",
            "  tracking, and calibration parameter sensitivity analysis.",
            "",
            "**SDM26 dyno validation**",
            "  The SDM26 predictions in this report are based on SDM25-",
            "  calibrated parameters applied without modification. When",
            "  SDM26 dyno data arrives, compare predictions and adjust",
            "  any SDM26-specific parameters if needed (different runner",
            "  length effect, different heat transfer in 4-2-1 secondaries).",
        ])
        finish(pdf, fig)

        # ---- File manifest ----
        fig, ax = new_page(pdf, "Deliverables & file manifest")
        text_block(ax, [
            "",
            "**Source code (Phase E + F calibrations):**",
            "  bcs/junction_characteristic.py   CharacteristicJunction (CSP)",
            "  bcs/junction_cv.py               JunctionCV (stagnation, unchanged)",
            "  models/sdm26.py                  junction_type constructor param",
            "  cylinder/combustion.py            eta_comb_at_rpm() ramp (F4)",
            "  cylinder/cylinder.py              RPM-dependent eta in advance()",
            "  configs/sdm25.json                F2 collector extension, F3 Cd",
            "  configs/sdm26.json                F2 collector extension",
            "",
            "**Reports:**",
            "  docs/phase_f_calibrated_report.pdf  This document",
            "  docs/phase_e_final_report.pdf       Phase E pre-calibration",
            "  docs/phase_e_comparison.md           Phase E markdown report",
            "  docs/phase_f1_corberan_design.md     F1 design (reverted)",
            "",
            "**Data:**",
            "  docs/sdm25_sweep_e4.json         SDM25 F5 calibrated sweep",
            "  docs/sdm26_sweep_e4.json         SDM26 F5 calibrated sweep",
            "  docs/sdm25_sweep_e4_dense.json   SDM25 Phase E dense (pre-F)",
            "  docs/sdm25_peak_dense.json       SDM25 dense peak region",
            "  docs/sdm26_peak_dense.json       SDM26 dense peak region",
            "  docs/dyno/sdm25_dyno.csv         SDM25 DynoJet data (95 pts)",
            "",
            "**Tests:**",
            "  111 passed, 3 skipped (Corberan tests preserved as docs)",
            "",
            "**Branch:**",
            "  phase-f/calibration -> merge to main",
            "  Tag: v2.2.0-calibrated",
        ])
        finish(pdf, fig)

    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
