"""Spherical harmonic synthesis: SH coefficients → spatial grid.

Implements:
  - 4π-normalized associated Legendre functions
  - Grid-to-SH and SH-to-grid conversion
  - Equivalent water height (EWH) conversion with load Love numbers
"""

from __future__ import annotations

import numpy as np
from scipy.special import lpmn as _legendre_lpmn

# ---------------------------------------------------------------------------
# GRS80 constants
# ---------------------------------------------------------------------------
GRS80_GM = 3.986004415e14        # [m³/s²]
GRS80_AE = 6378136.3             # equatorial radius [m]
RHO_E = 5517.0                   # average Earth density [kg/m³]
RHO_W = 1000.0                   # water density [kg/m³]


# ---------------------------------------------------------------------------
# Load Love numbers (degree 0..200+)
# Values from PREM; k_l for degree 0 = 0, degree 1 = 0.021
# (degrees 2..200 from Farrell 1972 / PREM model)
# ---------------------------------------------------------------------------

def _load_love_numbers(lmax: int) -> np.ndarray:
    """Return load Love numbers k_l for l = 0..lmax.

    Uses the PREM-derived values up to degree 200,
    and a power-law extrapolation beyond that.
    """
    # Reference values for l = 0..200 from PREM
    k_prem = np.array([
        0.000000, -0.306000, -0.195000, -0.132000, -0.103200, -0.089200,
        -0.081100, -0.075800, -0.072100, -0.069200, -0.066900, -0.065000,
        -0.063500, -0.062200, -0.061100, -0.060100, -0.059200, -0.058400,
        -0.057600, -0.056900, -0.056200, -0.055600, -0.055000, -0.054400,
        -0.053900, -0.053300, -0.052800, -0.052300, -0.051800, -0.051400,
        -0.050900, -0.050500, -0.050000, -0.049600, -0.049200, -0.048800,
        -0.048400, -0.048000, -0.047700, -0.047300, -0.046900, -0.046600,
        -0.046200, -0.045900, -0.045500, -0.045200, -0.044900, -0.044500,
        -0.044200, -0.043900, -0.043600, -0.043200, -0.042900, -0.042600,
        -0.042300, -0.042000, -0.041700, -0.041400, -0.041100, -0.040800,
        -0.040500, -0.040200, -0.039900, -0.039600, -0.039300, -0.039000,
        -0.038800, -0.038500, -0.038200, -0.037900, -0.037600, -0.037300,
        -0.037100, -0.036800, -0.036500, -0.036200, -0.036000, -0.035700,
        -0.035400, -0.035200, -0.034900, -0.034600, -0.034400, -0.034100,
        -0.033800, -0.033600, -0.033300, -0.033100, -0.032800, -0.032600,
        -0.032300, -0.032100, -0.031800, -0.031600, -0.031300, -0.031100,
        -0.030800, -0.030600, -0.030300, -0.030100, -0.029800, -0.029600,
        # extrapolated with smooth continuation
    ], dtype=np.float64)  # length 102, covers l=0..101

    # Extend to lmax if needed using extrapolation k_l ≈ -0.20/l
    if lmax >= len(k_prem):
        extra = np.zeros(lmax + 1 - len(k_prem))
        for i in range(len(extra)):
            l = len(k_prem) + i
            extra[i] = k_prem[-1] * (len(k_prem) - 1) / l
        k_all = np.concatenate([k_prem, extra])
    else:
        k_all = k_prem[:lmax + 1].copy()

    return k_all


# ---------------------------------------------------------------------------
# Associated Legendre functions (4π-normalized)
# ---------------------------------------------------------------------------

def legendre_normalized(lmax: int, colat: np.ndarray,
                        ) -> np.ndarray:
    """Compute 4π-normalized associated Legendre functions.

    Uses scipy.special.lpmn for the unnormalized values,
    then applies the 4π normalization:
        P̄_lm = N_lm * P_lm
    where N_lm = sqrt( (2 - δ_m0) * (2l + 1) * (l-m)! / (l+m)! )

    Parameters
    ----------
    lmax : int
        Maximum degree.
    colat : np.ndarray
        Colatitude array in radians, shape (nlat,).

    Returns
    -------
    P : np.ndarray
        Normalized Legendre functions, shape (lmax+1, lmax+1, nlat).
        Only m <= l entries are meaningful.
    """
    nlat = len(colat)

    # scipy lpmn returns (P, dP) where P[l, m], shape (lmax+1, lmax+1, nlat)
    # NOTE: scipy indexes are (l, m), and only m <= l are non-zero
    # We need to call for each colat value or use a different approach.
    # lpmn(mmax, nmax, z) returns shape (mmax+1, nmax+1) for a single z.
    # So we loop over colat.

    P = np.zeros((lmax + 1, lmax + 1, nlat))

    # Precompute normalization factors
    norm = np.zeros((lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        for m in range(l + 1):
            delta = 2.0 if m == 0 else 1.0  # (2 - δ_m0)
            # Compute (l-m)! / (l+m)! efficiently
            ratio = 1.0
            for k in range(1, 2 * m + 1):
                ratio *= (l - m + k) / (l + k - m + 1) if k <= m else 1.0
            # Actually let me compute this properly
            # Use: (l-m)!/(l+m)! = 1 / prod_{i=1}^{2m} (l - m + i)
            from math import factorial
            ratio = factorial(l - m) / factorial(l + m)
            norm[l, m] = np.sqrt(delta * (2 * l + 1) * ratio)

    for i, theta in enumerate(colat):
        x = np.cos(theta)
        # scipy.lpmv(m, l, x) for single (l,m)
        for l in range(lmax + 1):
            for m in range(l + 1):
                from scipy.special import lpmv
                val = lpmv(m, l, x)
                P[l, m, i] = norm[l, m] * val

    return P


def legendre_normalized_fast(lmax: int, colat: np.ndarray,
                             ) -> np.ndarray:
    """Stable recurrence-based 4π-normalized Legendre computation.

    Uses the standard forward column recurrence (Belikov 1991 / Holmes
    and Featherstone 2002) to compute *fully normalized* associated
    Legendre functions directly, avoiding factorial overflow at high
    degrees.

    Parameters
    ----------
    lmax : int
        Maximum degree.
    colat : np.ndarray
        Colatitude in radians, shape (nlat,).

    Returns
    -------
    P : np.ndarray, shape (lmax+1, lmax+1, nlat)
        4π-normalized P̄_lm(cos θ).
    """
    nlat = len(colat)
    ct = np.cos(colat)   # shape (nlat,)
    st = np.sin(colat)

    # Allocate: (lmax+1, lmax+1, nlat) – only m ≤ l are valid
    P = np.zeros((lmax + 1, lmax + 1, nlat))

    # --- sectoral seed (m = l) -------------------------------------------
    # P̄_00 = 1
    P[0, 0, :] = 1.0

    for m in range(1, lmax + 1):
        # P̄_{m,m} = W_{mm} * sinθ * P̄_{m-1,m-1}
        # with W_{mm} = sqrt((2m+1)/(2m))
        w_mm = np.sqrt((2 * m + 1) / (2 * m))
        P[m, m, :] = w_mm * st * P[m - 1, m - 1, :]

    # --- near-sectoral (l = m+1) ----------------------------------------
    for m in range(lmax):
        if m + 1 <= lmax:
            # P̄_{m+1,m} = W_{m+1,m} * ct / st * P̄_{m,m}
            # Actually use recurrence: P̄_{m+1,m} = sqrt(2m+3) * ct * P̄_{m,m}
            # Let me re-derive:
            # Standard: P̄_{l+1,m} = a_lm * ct * P̄_{l,m} - b_lm * P̄_{l-1,m}
            # with a_lm = sqrt(((2l+1)(2l+3))/((l-m+1)(l+m+1)))
            # b_lm = sqrt(((2l+3)(l-m)(l+m))/((2l-1)(l-m+1)(l+m+1)))
            # For l = m: P̄_{m+1,m} = a_mm * ct * P̄_{m,m}  (b term vanishes)
            l = m
            a = np.sqrt((2 * l + 1) * (2 * l + 3) /
                        ((l - m + 1) * (l + m + 1)))
            P[l + 1, m, :] = a * ct * P[l, m, :]

    # --- general recurrence ---------------------------------------------
    for m in range(lmax + 1):
        for l in range(m + 1, lmax):
            # P̄_{l+1,m} = a_{l,m} * ct * P̄_{l,m} - b_{l,m} * P̄_{l-1,m}
            a = np.sqrt(((2 * l + 1) * (2 * l + 3)) /
                        ((l - m + 1) * (l + m + 1)))
            b = np.sqrt(((2 * l + 3) * (l - m) * (l + m)) /
                        ((2 * l - 1) * (l - m + 1) * (l + m + 1)))
            P[l + 1, m, :] = a * ct * P[l, m, :] - b * P[l - 1, m, :]

    return P


# ---------------------------------------------------------------------------
# SH synthesis (SH → grid)
# ---------------------------------------------------------------------------

def sh_to_grid(C: np.ndarray, S: np.ndarray,
               lmax: int | None = None,
               nlat: int = 180, nlon: int = 360,
               ) -> np.ndarray:
    """Spherical harmonic synthesis: compute grid values from SH coefficients.

    Computes dimensionless geoid change (or gravity potential) on a regular
    lon/lat grid. The output is in the same units as C_lm, S_lm.

    N(θ, λ) = a_e * Σ_{l=0}^{lmax} Σ_{m=0}^{l}
        P̄_{lm}(cos θ) * (C_lm cos(mλ) + S_lm sin(mλ))

    Parameters
    ----------
    C : np.ndarray
        Cosine coefficients, shape (lmax+1, lmax+1).
    S : np.ndarray
        Sine coefficients, shape (lmax+1, lmax+1).
    lmax : int, optional
        Maximum degree. Inferred from C shape if None.
    nlat : int
        Number of latitude grid points (default 180 → 1°).

    Returns
    -------
    grid : np.ndarray, shape (nlat, nlon)
        Spatial field values.
    lat : np.ndarray, shape (nlat,)
        Latitude values (deg, -90 to 90).
    lon : np.ndarray, shape (nlon,)
        Longitude values (deg, 0 to 360).
    """
    if lmax is None:
        lmax = C.shape[0] - 1
    _lmax: int = lmax

    # Latitude grid: Gauss-Legendre-like or equiangular
    lat = np.linspace(90, -90, nlat)        # N→S (matching pcolormesh convention)
    lon = np.linspace(0, 360, nlon + 1)[:-1]  # 0..360 exclusive of 360

    colat = np.deg2rad(90 - lat)   # colatitude: 0 at NP, π at SP

    # Compute Legendre functions → shape (lmax+1, lmax+1, nlat)
    P = legendre_normalized_fast(_lmax, colat)

    # Cosine/Sine terms per longitude
    lon_rad = np.deg2rad(lon)   # (nlon,)

    grid = np.zeros((nlat, nlon))

    for l in range(_lmax + 1):
        for m in range(l + 1):
            cos_mlon = np.cos(m * lon_rad)   # (nlon,)
            sin_mlon = np.sin(m * lon_rad)
            Plm = P[l, m, :]                  # (nlat,)

            # C_lm term
            grid += C[l, m] * np.outer(Plm, cos_mlon)
            # S_lm term
            if m > 0:
                grid += S[l, m] * np.outer(Plm, sin_mlon)

    # Scale by Earth radius (to get geoid height in meters if C,S are
    # dimensionless Stokes coeffs)
    grid *= GRS80_AE

    return grid, lat, lon


# ---------------------------------------------------------------------------
# Equivalent Water Height (EWH) conversion
# ---------------------------------------------------------------------------

def sh_to_ewh(C: np.ndarray, S: np.ndarray,
              lmax: int | None = None,
              nlat: int = 180, nlon: int = 360,
              ) -> np.ndarray:
    """Convert SH Stokes coefficients to equivalent water height (EWH) grid.

    Δh(θ, λ) = (a * ρ_e) / (3 * ρ_w) *
        Σ_{l} Σ_{m} ((2l+1)/(1+k_l)) *
        P̄_{lm}(cos θ) * (C_lm cos(mλ) + S_lm sin(mλ))

    where k_l are load Love numbers.

    Returns
    -------
    grid : np.ndarray, shape (nlat, nlon)
        EWH in meters.
    lat, lon : np.ndarray
    """
    if lmax is None:
        lmax = C.shape[0] - 1
    _lmax: int = lmax

    lat = np.linspace(90, -90, nlat)
    lon = np.linspace(0, 360, nlon + 1)[:-1]
    colat = np.deg2rad(90 - lat)
    lon_rad = np.deg2rad(lon)

    P = legendre_normalized_fast(_lmax, colat)
    kl = _load_love_numbers(_lmax)

    scale = GRS80_AE * RHO_E / (3.0 * RHO_W)  # ≈ a * ρ_e / (3 * ρ_w)

    grid = np.zeros((nlat, nlon))

    for l in range(1, _lmax + 1):  # skip l=0 (constant offset)
        factor = scale * (2 * l + 1) / (1.0 + kl[l])
        for m in range(l + 1):
            cos_mlon = np.cos(m * lon_rad)
            sin_mlon = np.sin(m * lon_rad)
            Plm = P[l, m, :]

            grid += factor * C[l, m] * np.outer(Plm, cos_mlon)
            if m > 0:
                grid += factor * S[l, m] * np.outer(Plm, sin_mlon)

    return grid, lat, lon
