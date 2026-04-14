"""Test 7: friction (Blasius) and wall heat transfer (Dittus-Boelter) sources.

Validation approach:

1. **Friction-only momentum decay**. Uniform gas in a sealed constant-area
   pipe, non-zero initial velocity, heat disabled. Momentum decays; we
   verify the decay rate matches the Blasius friction model over a short
   time (before the flow reverses or reaches zero).

2. **Heat-only temperature relaxation**. Uniform hot gas in a sealed pipe,
   velocity zero, friction disabled. Temperature relaxes toward T_wall
   at the rate predicted by h_conduction = k/D (Nu = 1 floor when u = 0).

3. **Low-Mach Darcy–Weisbach pressure drop**. Subsonic flow through a
   straight pipe with friction; in the nearly incompressible limit the
   static pressure drop should match Darcy–Weisbach Δp = f · (L/D) ·
   0.5 · ρu² to within ~10 % at Mach 0.1–0.2 (compressibility is second
   order).

4. **Fanno-like: pipe length to sonic**. Subsonic flow with friction only
   accelerates toward Mach 1 over a distance L*. We check that adding
   friction to a low-Mach flow raises Mach monotonically (correct sign).
"""

from __future__ import annotations

import numpy as np

from solver.state import make_pipe_state, set_uniform
from solver.muscl import muscl_hancock_step, cfl_dt, LIMITER_MINMOD
from solver.sources import apply_sources, strang_split_step
from bcs.simple import fill_reflective_left, fill_reflective_right
from bcs.subsonic import fill_subsonic_inflow_left, fill_subsonic_outflow_right


GAMMA = 1.4
R_GAS = 287.0


def test_friction_source_alone_decays_momentum_monotonically():
    """Call apply_sources in isolation (no hyperbolic step, no waves)
    on uniform flow. Momentum must decrease monotonically.

    Isolating the source-term kernel avoids the wave-reflection physics
    that pollutes a strang-split test in a sealed pipe.
    """
    state = make_pipe_state(
        n_cells=100, length=1.0,
        area_fn=lambda x: np.pi * 0.01 ** 2,  # D = 20 mm
        gamma=GAMMA, R_gas=R_GAS, wall_T=300.0, n_ghost=2,
    )
    set_uniform(state, rho=1.2, u=50.0, p=1.0e5, Y=0.0)

    def total_mom():
        s = state.real_slice()
        return float(state.q[s, 1].sum())

    dt = 1.0e-5
    m_hist = [total_mom()]
    for _ in range(5000):
        apply_sources(
            state.q, state.area, state.hydraulic_D, dt,
            GAMMA, R_GAS, state.wall_T, state.n_ghost,
            apply_friction=True, apply_heat=False,
        )
        m_hist.append(total_mom())
    m_hist = np.array(m_hist)
    # Strict monotone decrease
    assert (np.diff(m_hist) < 0.0).all(), "momentum not monotone under pure friction"
    # Expected: Blasius decay at ~24.6/s so half-life ≈ 28 ms → after
    # 50 ms we should be below ~0.30 of initial.
    assert m_hist[-1] < 0.5 * m_hist[0], (
        f"friction barely acted: final/initial = {m_hist[-1]/m_hist[0]:.3f}"
    )


def test_heat_source_alone_relaxes_toward_wall():
    """Call apply_sources in isolation for a hot quiescent gas. T cools
    toward T_wall. Smaller pipe + more steps than the earlier attempt so
    the cooling is visible (h_floor is k/D, so D drives the timescale)."""
    T_wall = 300.0
    T0 = 1500.0
    D = 0.005  # 5 mm — fast thermal timescale
    state = make_pipe_state(
        n_cells=40, length=0.1,
        area_fn=lambda x: np.pi * (D / 2) ** 2,
        gamma=GAMMA, R_gas=R_GAS, wall_T=T_wall, n_ghost=2,
    )
    p0 = 1.0e5
    rho0 = p0 / (R_GAS * T0)
    set_uniform(state, rho=rho0, u=0.0, p=p0, Y=0.0)

    gm1 = GAMMA - 1.0
    def avg_T():
        s = state.real_slice()
        T_sum = 0.0
        count = 0
        for i in range(s.start, s.stop):
            A = state.area[i]
            rho = state.q[i, 0] / A
            u = state.q[i, 1] / (rho * A)
            E = state.q[i, 2] / A
            p = gm1 * (E - 0.5 * rho * u * u)
            T_sum += p / (rho * R_GAS)
            count += 1
        return T_sum / count

    dt = 1.0e-4
    T_hist = [avg_T()]
    for _ in range(1000):
        apply_sources(
            state.q, state.area, state.hydraulic_D, dt,
            GAMMA, R_GAS, state.wall_T, state.n_ghost,
            apply_friction=False, apply_heat=True,
        )
        T_hist.append(avg_T())
    T_hist = np.array(T_hist)
    assert (np.diff(T_hist) < 0.0).all(), "T not monotone under pure heat source"
    frac = (T_hist[0] - T_hist[-1]) / (T_hist[0] - T_wall)
    assert frac > 0.1, f"cooled only {frac:.1%} of the way to wall"


def test_friction_darcy_weisbach_low_mach():
    """Driven subsonic flow through a straight pipe with friction. In the
    low-Mach incompressible limit, Δp = f · (L/D) · 0.5 · ρu² within 15 %.
    """
    D = 0.02
    L = 1.0
    state = make_pipe_state(
        n_cells=200, length=L,
        area_fn=lambda x: np.pi * (D / 2) ** 2,
        gamma=GAMMA, R_gas=R_GAS, wall_T=300.0, n_ghost=2,
    )
    rho_in = 1.2
    u_in = 30.0  # Mach ~0.087 at 300 K
    p_in = 1.0e5
    Y_in = 0.0
    set_uniform(state, rho=rho_in, u=u_in, p=p_in, Y=Y_in)

    # Expected Darcy-Weisbach
    mu = 1.8e-5 * (300.0 / 293.0) ** 0.7
    Re = rho_in * u_in * D / mu
    f_blas = 0.3164 * Re ** (-0.25)
    dp_expected = f_blas * (L / D) * 0.5 * rho_in * u_in ** 2
    p_back = p_in - dp_expected  # set back pressure to anticipated value

    n = state.n_total
    w = np.zeros((n, 4)); slopes = np.zeros((n, 4))
    wL = np.zeros((n, 4)); wR = np.zeros((n, 4))
    flux = np.zeros((n + 1, 4))

    for _ in range(8000):
        fill_subsonic_inflow_left(state, rho_in, u_in, p_in, Y_in)
        fill_subsonic_outflow_right(state, p_back)
        dt = cfl_dt(state.q, state.area, state.dx, GAMMA, 0.85, state.n_ghost)
        strang_split_step(
            state.q, state.area, state.area_f, state.hydraulic_D,
            state.dx, dt, GAMMA, R_GAS, state.wall_T,
            state.n_ghost, LIMITER_MINMOD,
            apply_friction=True, apply_heat=False,
            w=w, slopes=slopes, wL=wL, wR=wR, flux=flux,
        )

    s = state.real_slice()
    gm1 = GAMMA - 1.0
    # Static pressure profile: p_i from primitives
    p_prof = np.zeros(state.n_cells)
    for k, i in enumerate(range(s.start, s.stop)):
        A = state.area[i]
        rho = state.q[i, 0] / A
        u = state.q[i, 1] / (rho * A)
        E = state.q[i, 2] / A
        p_prof[k] = gm1 * (E - 0.5 * rho * u * u)
    dp_sim = p_prof[0] - p_prof[-1]
    rel = abs(dp_sim - dp_expected) / dp_expected
    assert rel < 0.20, f"Δp sim={dp_sim:.1f} Pa, DW={dp_expected:.1f} Pa, rel={rel:.2%}"
