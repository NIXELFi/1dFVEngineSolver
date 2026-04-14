"""Exact Riemann solver for the 1D Euler equations with ideal gas.

Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
3rd ed., 2009, Chapter 4. We implement the two-rarefaction iterative solver
for p* (Toro 4.82, 4.83) with Newton iteration, then sample the solution at
(x/t) to produce the primitive state at any post-shock wave-fan location.

This is test-utility code, not part of the production solver. It supports
the Sod / Lax / 123 / contact validation tests.
"""

from __future__ import annotations

import numpy as np


def _pressure_function(p, p_K, rho_K, a_K, gamma):
    """Toro 4.6: f_K(p) = (p - p_K)·A_K/(p + B_K) if p > p_K (shock)
                    = (2 a_K / (γ-1))·((p/p_K)^((γ-1)/(2γ)) - 1) if p ≤ p_K (rarefaction)
    and its derivative.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    if p > p_K:
        A_K = 2.0 / (gp1 * rho_K)
        B_K = gm1 / gp1 * p_K
        sq = np.sqrt(A_K / (p + B_K))
        f = (p - p_K) * sq
        df = sq * (1.0 - 0.5 * (p - p_K) / (p + B_K))
    else:
        f = (2.0 * a_K / gm1) * ((p / p_K) ** (gm1 / (2.0 * gamma)) - 1.0)
        df = (1.0 / (rho_K * a_K)) * (p / p_K) ** (-(gp1) / (2.0 * gamma))
    return f, df


def solve_star(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=1.4, tol=1e-10, max_iter=50):
    """Return (p_star, u_star) for the Riemann problem."""
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)
    # Two-rarefaction guess (Toro 4.46)
    gm1 = gamma - 1.0
    exp1 = gm1 / (2.0 * gamma)
    p_guess = ((a_L + a_R - 0.5 * gm1 * (u_R - u_L)) /
               (a_L / p_L**exp1 + a_R / p_R**exp1)) ** (1.0 / exp1)
    p = max(p_guess, tol)
    for _ in range(max_iter):
        fL, dfL = _pressure_function(p, p_L, rho_L, a_L, gamma)
        fR, dfR = _pressure_function(p, p_R, rho_R, a_R, gamma)
        F = fL + fR + (u_R - u_L)
        dF = dfL + dfR
        dp = -F / dF
        p_new = p + dp
        if p_new < 0.0:
            p_new = 0.5 * p  # fallback safeguard
        if abs(p_new - p) < tol * (abs(p_new) + abs(p)):
            p = p_new
            break
        p = p_new
    fL, _ = _pressure_function(p, p_L, rho_L, a_L, gamma)
    fR, _ = _pressure_function(p, p_R, rho_R, a_R, gamma)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (fR - fL)
    return p, u_star


def sample(xi, rho_L, u_L, p_L, rho_R, u_R, p_R, p_star, u_star, gamma=1.4):
    """Sample the exact Riemann solution at self-similar coordinate xi = x/t.

    Returns (ρ, u, p) at xi.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)
    if xi <= u_star:
        # Left of contact
        if p_star <= p_L:
            # Left rarefaction
            a_star_L = a_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))
            S_HL = u_L - a_L
            S_TL = u_star - a_star_L
            if xi <= S_HL:
                return rho_L, u_L, p_L
            if xi >= S_TL:
                rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
                return rho_star_L, u_star, p_star
            # Inside fan
            u_fan = (2.0 / gp1) * (a_L + 0.5 * gm1 * u_L + xi)
            a_fan = (2.0 / gp1) * (a_L + 0.5 * gm1 * (u_L - xi))
            rho_fan = rho_L * (a_fan / a_L) ** (2.0 / gm1)
            p_fan = p_L * (a_fan / a_L) ** (2.0 * gamma / gm1)
            return rho_fan, u_fan, p_fan
        else:
            # Left shock
            S_L = u_L - a_L * np.sqrt(((gp1) / (2.0 * gamma)) * (p_star / p_L) + (gm1 / (2.0 * gamma)))
            if xi <= S_L:
                return rho_L, u_L, p_L
            rho_star_L = rho_L * ((p_star / p_L + gm1 / gp1) / (gm1 / gp1 * (p_star / p_L) + 1.0))
            return rho_star_L, u_star, p_star
    else:
        # Right of contact
        if p_star <= p_R:
            # Right rarefaction
            a_star_R = a_R * (p_star / p_R) ** (gm1 / (2.0 * gamma))
            S_TR = u_star + a_star_R
            S_HR = u_R + a_R
            if xi >= S_HR:
                return rho_R, u_R, p_R
            if xi <= S_TR:
                rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)
                return rho_star_R, u_star, p_star
            u_fan = (2.0 / gp1) * (-a_R + 0.5 * gm1 * u_R + xi)
            a_fan = (2.0 / gp1) * (a_R - 0.5 * gm1 * (u_R - xi))
            rho_fan = rho_R * (a_fan / a_R) ** (2.0 / gm1)
            p_fan = p_R * (a_fan / a_R) ** (2.0 * gamma / gm1)
            return rho_fan, u_fan, p_fan
        else:
            # Right shock
            S_R = u_R + a_R * np.sqrt(((gp1) / (2.0 * gamma)) * (p_star / p_R) + (gm1 / (2.0 * gamma)))
            if xi >= S_R:
                return rho_R, u_R, p_R
            rho_star_R = rho_R * ((p_star / p_R + gm1 / gp1) / (gm1 / gp1 * (p_star / p_R) + 1.0))
            return rho_star_R, u_star, p_star


def sample_array(x, t, x0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=1.4):
    """Sample solution at array of x values for given t and initial jump at x0."""
    p_star, u_star = solve_star(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)
    n = len(x)
    rho = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    for i, xi_val in enumerate(x):
        xi = (xi_val - x0) / t
        r, uu, pp = sample(xi, rho_L, u_L, p_L, rho_R, u_R, p_R, p_star, u_star, gamma)
        rho[i] = r
        u[i] = uu
        p[i] = pp
    return rho, u, p, p_star, u_star
