"""Tests for SDM26Config parameterization:
- every field is exposed and reaches the simulator
- __post_init__ validation catches bad inputs
- taper (linear-diameter cone) runs and preserves conservation
- per-cylinder asymmetry works
- warnings fire on aggressive tapers
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from models.sdm26 import SDM26Engine, SDM26Config, linear_diameter_area


# ----- validation (hard errors) -----

def test_negative_bore_rejected():
    with pytest.raises(ValueError, match="bore"):
        SDM26Config(bore=-0.01)


def test_CR_leq_1_rejected():
    with pytest.raises(ValueError, match="CR"):
        SDM26Config(CR=1.0)


def test_con_rod_too_short_rejected():
    with pytest.raises(ValueError, match="con_rod"):
        SDM26Config(stroke=0.1, con_rod=0.04)  # rod < stroke/2


def test_zero_length_rejected():
    with pytest.raises(ValueError, match="primary_length"):
        SDM26Config(primary_length=0.0)


def test_zero_diameter_rejected():
    with pytest.raises(ValueError, match="primary_diameter_in"):
        SDM26Config(primary_diameter_in=0.0)


def test_negative_diameter_out_rejected():
    with pytest.raises(ValueError, match="primary_diameter_out"):
        SDM26Config(primary_diameter_out=-0.001)


def test_diameter_out_none_is_straight():
    cfg = SDM26Config(primary_diameter_in=0.032, primary_diameter_out=None)
    L, D_in, D_out, n, wT = cfg.primary_spec(0)
    assert D_in == D_out == 0.032


def test_n_cells_too_small_rejected():
    with pytest.raises(ValueError, match="runner_n_cells"):
        SDM26Config(runner_n_cells=3)


def test_eta_comb_above_one_rejected():
    with pytest.raises(ValueError, match="eta_comb"):
        SDM26Config(eta_comb=1.1)


def test_wall_T_out_of_range_rejected():
    with pytest.raises(ValueError, match="primary_wall_T"):
        SDM26Config(primary_wall_T=100.0)  # below 200 K floor


def test_restrictor_Cd_out_of_range():
    with pytest.raises(ValueError, match="restrictor_Cd"):
        SDM26Config(restrictor_Cd=1.5)


def test_per_cylinder_list_wrong_length():
    with pytest.raises(ValueError, match="primary_lengths"):
        SDM26Config(primary_lengths=[0.3, 0.3, 0.3])  # 3 values, needs 4


def test_valve_cd_ld_table_mismatch():
    with pytest.raises(ValueError, match="same length"):
        SDM26Config(intake_ld_table=(0.1, 0.2), intake_cd_table=(0.3, 0.4, 0.5))


def test_valve_cd_out_of_range():
    with pytest.raises(ValueError, match="cd_table"):
        SDM26Config(intake_cd_table=(0.1, 0.2, 1.5, 0.4, 0.5, 0.6))


def test_non_monotone_ld_table():
    with pytest.raises(ValueError, match="monotone"):
        SDM26Config(intake_ld_table=(0.1, 0.05, 0.15, 0.20, 0.25, 0.30))


# ----- validation (soft warnings) -----

def test_large_taper_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SDM26Config(runner_diameter_in=0.02, runner_diameter_out=0.08)
        msgs = [str(x.message) for x in w]
        assert any("aggressive" in m for m in msgs), msgs


def test_no_taper_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SDM26Config()
        msgs = [str(x.message) for x in w]
        assert not any("aggressive" in m for m in msgs)


# ----- taper physics -----

def test_linear_diameter_area_endpoints():
    fn = linear_diameter_area(length=1.0, D_in=0.04, D_out=0.06)
    # At x=0 → D=0.04, A = π·0.04²/4
    # At x=1 → D=0.06, A = π·0.06²/4
    A0 = fn(0.0)
    A1 = fn(1.0)
    assert abs(A0 - 0.25 * np.pi * 0.04 ** 2) < 1e-12
    assert abs(A1 - 0.25 * np.pi * 0.06 ** 2) < 1e-12
    # Midpoint: D=0.05 → A = π·0.05²/4
    Amid = fn(0.5)
    assert abs(Amid - 0.25 * np.pi * 0.05 ** 2) < 1e-12


def test_linear_diameter_area_straight_pipe():
    fn = linear_diameter_area(length=1.0, D_in=0.04, D_out=0.04)
    # Area should be constant
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert abs(fn(x) - 0.25 * np.pi * 0.04 ** 2) < 1e-12


# ----- engine runs with taper -----

def test_tapered_collector_runs_and_conserves():
    """Collector tapered 38→50 mm (diffuser). Run 10 cycles; nonconservation
    residual stays machine precision."""
    cfg = SDM26Config(
        collector_diameter_in=0.038,
        collector_diameter_out=0.050,
    )
    eng = SDM26Engine(cfg)
    r = eng.run_single_rpm(10500, n_cycles=10, verbose=False)
    for stats in r["cycle_stats"]:
        assert abs(stats["nonconservation"]) < 1e-15, (
            f"cycle {stats['cycle']}: nonconservation = {stats['nonconservation']:.2e}"
        )


def test_tapered_primary_runs():
    """Primary diverging 32→38 mm (common exhaust header design)."""
    cfg = SDM26Config(
        primary_diameter_in=0.032,
        primary_diameter_out=0.038,
    )
    eng = SDM26Engine(cfg)
    r = eng.run_single_rpm(10500, n_cycles=5, verbose=False)
    last = r["cycle_stats"][-1]
    assert np.isfinite(last["imep_bar"])
    assert last["EGT_mean"] > 500.0  # some exhaust heating at cycle 5 transient
    assert abs(last["nonconservation"]) < 1e-15


def test_per_cylinder_primary_asymmetry():
    """V1's config had primaries 0 & 1 at D=0.032 m and primaries 2 & 3 at
    D=0.034 m. Reproduce that asymmetry via the per-cylinder list."""
    cfg = SDM26Config(
        primary_diameters_in=[0.032, 0.032, 0.034, 0.034],
    )
    eng = SDM26Engine(cfg)
    # Check the constructed pipes have the right areas at x=0
    A0 = eng.primaries[0].area[eng.primaries[0].n_ghost]
    A2 = eng.primaries[2].area[eng.primaries[2].n_ghost]
    expected_0 = 0.25 * np.pi * 0.032 ** 2
    expected_2 = 0.25 * np.pi * 0.034 ** 2
    # Cell-centre area at the first real cell uses x = 0.5·dx, slightly off;
    # just verify it's closer to the per-cyl value than to the other
    assert abs(A0 - expected_0) < abs(A0 - expected_2)
    assert abs(A2 - expected_2) < abs(A2 - expected_0)


# ----- parameter changes actually move the simulation -----

def test_longer_primary_changes_EGT_or_imep():
    """Spot check: changing primary length by 50 % should change at least one
    of IMEP, VE, EGT by more than 0.5 %. If it does not, parameters are not
    reaching the simulator."""
    cfg_short = SDM26Config(primary_length=0.2)
    cfg_long = SDM26Config(primary_length=0.4)
    r_short = SDM26Engine(cfg_short).run_single_rpm(10500, n_cycles=15, stop_at_convergence=True)
    r_long = SDM26Engine(cfg_long).run_single_rpm(10500, n_cycles=15, stop_at_convergence=True)
    s = r_short["cycle_stats"][-1]
    l = r_long["cycle_stats"][-1]
    rel = max(
        abs(s["imep_bar"] - l["imep_bar"]) / abs(s["imep_bar"]),
        abs(s["ve_atm"] - l["ve_atm"]) / abs(s["ve_atm"]),
        abs(s["EGT_mean"] - l["EGT_mean"]) / abs(s["EGT_mean"]),
    )
    assert rel > 0.005, f"primary_length change did not affect sim: max rel Δ = {rel:.4f}"


def test_larger_restrictor_increases_VE():
    """A larger restrictor throat should increase the converged VE
    (more air can pass per cycle). Sanity check on the restrictor BC."""
    cfg_small = SDM26Config(restrictor_throat_diameter=0.020)
    cfg_large = SDM26Config(restrictor_throat_diameter=0.028)
    r_small = SDM26Engine(cfg_small).run_single_rpm(10500, n_cycles=15, stop_at_convergence=True)
    r_large = SDM26Engine(cfg_large).run_single_rpm(10500, n_cycles=15, stop_at_convergence=True)
    ve_s = r_small["cycle_stats"][-1]["ve_atm"]
    ve_l = r_large["cycle_stats"][-1]["ve_atm"]
    assert ve_l > ve_s, f"larger throat did not increase VE ({ve_s:.3f} vs {ve_l:.3f})"


def test_valve_Cd_scaling_affects_VE():
    """Scaling the intake Cd table down should reduce VE."""
    cfg_hi = SDM26Config()
    cfg_lo = SDM26Config(intake_cd_table=tuple(0.5 * c for c in SDM26Config().intake_cd_table))
    r_hi = SDM26Engine(cfg_hi).run_single_rpm(10500, n_cycles=25, stop_at_convergence=True,
                                              convergence_min_cycles=8)
    r_lo = SDM26Engine(cfg_lo).run_single_rpm(10500, n_cycles=25, stop_at_convergence=True,
                                              convergence_min_cycles=8)
    ve_hi = r_hi["cycle_stats"][-1]["ve_atm"]
    ve_lo = r_lo["cycle_stats"][-1]["ve_atm"]
    assert ve_lo < ve_hi, f"halving Cd did not reduce VE ({ve_hi:.3f} → {ve_lo:.3f})"
