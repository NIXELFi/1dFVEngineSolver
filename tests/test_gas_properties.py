"""Unit tests for the ported gas_properties module.

These are spot checks confirming the ported polynomials evaluate to the same
values as V1's implementation. They also verify @njit-compatibility (if the
module imports and these run, the @njit compile worked).
"""

from __future__ import annotations

import numpy as np

from cylinder.gas_properties import (
    R_AIR, R_BURNED,
    gamma_unburned, gamma_burned, gamma_mixture, R_mixture,
    speed_of_sound,
)


def test_gamma_unburned_at_300K():
    assert abs(gamma_unburned(300.0) - 1.38) < 1e-12


def test_gamma_unburned_at_900K():
    assert abs(gamma_unburned(900.0) - (1.38 - 1.2e-4 * 600.0)) < 1e-12


def test_gamma_unburned_clamps_below_300K():
    assert gamma_unburned(200.0) == gamma_unburned(300.0)


def test_gamma_unburned_clamps_above_900K():
    assert gamma_unburned(1500.0) == gamma_unburned(900.0)


def test_gamma_burned_at_300K():
    assert abs(gamma_burned(300.0) - 1.30) < 1e-12


def test_gamma_burned_at_3000K():
    assert abs(gamma_burned(3000.0) - (1.30 - 8.0e-5 * 2700.0)) < 1e-12


def test_gamma_mixture_endpoints():
    T = 1500.0
    assert abs(gamma_mixture(T, 0.0) - gamma_unburned(T)) < 1e-12
    assert abs(gamma_mixture(T, 1.0) - gamma_burned(T)) < 1e-12


def test_gamma_mixture_interpolates_linearly():
    T = 1200.0
    xb = 0.4
    expected = 0.6 * gamma_unburned(T) + 0.4 * gamma_burned(T)
    assert abs(gamma_mixture(T, xb) - expected) < 1e-12


def test_R_mixture_endpoints():
    assert R_mixture(0.0) == R_AIR
    assert R_mixture(1.0) == R_BURNED


def test_R_mixture_linear_interpolation():
    xb = 0.3
    expected = 0.7 * R_AIR + 0.3 * R_BURNED
    assert abs(R_mixture(xb) - expected) < 1e-12


def test_speed_of_sound_air_at_300K():
    a = speed_of_sound(1.4, 287.0, 300.0)
    # sqrt(1.4 * 287 * 300) = 347.19 m/s
    assert abs(a - 347.19) < 0.1
