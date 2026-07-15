"""Visualization module for spherical harmonic data.

Provides:
  - Kaula / degree-RMS curves
  - Global map (geoid / EWH) with coastlines
  - North-south stripe noise comparison (raw vs filtered)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .sh_io import SHCoeffs, degree_rms
from .sh_synthesis import sh_to_ewh, sh_to_grid

# Use non-interactive backend for script output
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
})


# ---------------------------------------------------------------------------
# Colourmaps
# ---------------------------------------------------------------------------
def _diverging_cmap():
    """Diverging colormap for anomaly plots."""
    return plt.cm.RdBu_r


def _sequential_cmap():
    return plt.cm.viridis


# ---------------------------------------------------------------------------
# Coastline helper
# ---------------------------------------------------------------------------
def _draw_coastlines(ax, lon, lat):
    """Draw simplified world coastlines using cartopy if available,
    otherwise draw a simple outline.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # If cartopy is available but we're using plain axes, skip
        return False
    except ImportError:
        pass

    # Draw a simple continental outline (very rough)
    # Simple rectangle from -180 to 180, -90 to 90
    # For a proper solution, use cartopy
    ax.axhline(y=0, color="gray", linewidth=0.3, alpha=0.5)
    return False


# ---------------------------------------------------------------------------
# Plot 1: Kaula / Degree-RMS curves
# ---------------------------------------------------------------------------

def plot_kaula(sh_list: list[SHCoeffs],
               title: str = "Degree-RMS (Kaula) Curves",
               month_idx: int = 0,
               save_path: str | Path | None = None,
               ) -> plt.Figure:
    """Plot per-degree RMS power spectrum for one or more coefficient sets.

    Parameters
    ----------
    sh_list : list[SHCoeffs]
        Coefficient sets to compare.
    title : str
        Plot title.
    month_idx : int
        Which month index to use from each SHCoeffs.
    save_path : str or Path, optional
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sh_list), 10)))

    for i, sh in enumerate(sh_list):
        rms = degree_rms(sh, month_idx)
        l_vals = np.arange(sh.lmax + 1)
        label = sh.label or f"Set {i + 1}"
        ax.loglog(l_vals[1:], rms[1:], color=colors[i], linewidth=1.0,
                  label=label, marker=".", markersize=2)

    ax.set_xlabel("Spherical Harmonic Degree L")
    ax.set_ylabel("Degree-RMS")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot 2: Global map (EWH or geoid)
# ---------------------------------------------------------------------------

def plot_global_map(sh: SHCoeffs,
                    month_idx: int = 0,
                    field_type: str = "ewh",
                    title: str | None = None,
                    vmin: float | None = None,
                    vmax: float | None = None,
                    cmap=None,
                    save_path: str | Path | None = None,
                    ) -> plt.Figure:
    """Plot a global map from spherical harmonic coefficients.

    Parameters
    ----------
    sh : SHCoeffs
        Coefficient set.
    month_idx : int
        Month index.
    field_type : str
        "ewh" for equivalent water height, "geoid" for geoid height.
    title : str, optional
        Plot title.
    vmin, vmax : float, optional
        Colour scale limits.
    cmap : colormap, optional
    save_path : str or Path, optional

    Returns
    -------
    fig : plt.Figure
    """
    C = sh.C[month_idx]
    S = sh.S[month_idx]

    if field_type == "ewh":
        grid, lat, lon = sh_to_ewh(C, S, lmax=sh.lmax)
        cbar_label = "Equivalent Water Height (m)"
    else:
        grid, lat, lon = sh_to_grid(C, S, lmax=sh.lmax)
        cbar_label = "Geoid Height (m)"

    if cmap is None:
        cmap = _diverging_cmap() if field_type == "ewh" else _sequential_cmap()

    # Auto-scale: symmetric around 0 for diverging
    if vmin is None and vmax is None:
        vabs = np.percentile(np.abs(grid), 98)
        vmin, vmax = -vabs, vabs

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", rasterized=True)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08,
                      shrink=0.75)
    cb.set_label(cbar_label)
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)

    if title is None:
        ts = ""
        if sh.t is not None:
            ts = f"  ({sh.t[month_idx]:.3f})"
        title = f"{sh.label}{ts}"

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot 3: NS stripe noise comparison (raw vs filtered)
# ---------------------------------------------------------------------------

def plot_noise_comparison(sh_raw: SHCoeffs,
                          sh_filtered: SHCoeffs,
                          month_idx: int = 0,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          save_path: str | Path | None = None,
                          ) -> plt.Figure:
    """Side-by-side comparison of raw vs filtered coefficients,
    plus the noise (difference) map.

    Parameters
    ----------
    sh_raw, sh_filtered : SHCoeffs
        Raw and filtered coefficients. Must have same lmax.
    month_idx : int
        Month index.
    save_path : str or Path, optional

    Returns
    -------
    fig : plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    lmax = min(sh_raw.lmax, sh_filtered.lmax)

    Cr = sh_raw.C[month_idx]
    Sr = sh_raw.S[month_idx]
    Cf = sh_filtered.C[month_idx]
    Sf = sh_filtered.S[month_idx]

    # Compute EWH grids
    grid_raw, lat, lon = sh_to_ewh(Cr, Sr, lmax=lmax)
    grid_filt, _, _ = sh_to_ewh(Cf, Sf, lmax=lmax)
    grid_noise = grid_raw - grid_filt

    # Auto colour scale
    if vmin is None and vmax is None:
        vabs = np.percentile(np.abs(grid_raw), 98)
        vmin, vmax = -vabs, vabs

    cmap = _diverging_cmap()

    titles = ["Raw (Unfiltered)", f"Filtered ({sh_filtered.label})", "Noise (Raw - Filtered)"]
    grids = [grid_raw, grid_filt, grid_noise]

    im = None
    for ax, grid, title in zip(axes, grids, titles):
        vmn, vmx = vmin, vmax
        if "Noise" in title:
            vn = np.percentile(np.abs(grid), 99)
            vmn, vmx = -vn, vn
        im = ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=vmn, vmax=vmx,
                          shading="auto", rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lon (°)")
        ax.set_ylabel("Lat (°)")
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

    # Unified colourbar
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.1,
                      shrink=0.5)
    cb.set_label("EWH (m)")

    fig.suptitle(f"Noise Comparison: {sh_raw.label}", fontsize=11, y=1.02)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.90, wspace=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot 4: Degree-RMS difference plot
# ---------------------------------------------------------------------------

def plot_degree_rms_diff(sh_raw: SHCoeffs, sh_filtered: SHCoeffs,
                         month_idx: int = 0,
                         save_path: str | Path | None = None,
                         ) -> plt.Figure:
    """Plot degree-RMS curves: raw vs filtered, plus reduction ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    lmax = min(sh_raw.lmax, sh_filtered.lmax)
    l_vals = np.arange(lmax + 1)

    rms_raw = degree_rms(sh_raw, month_idx)
    rms_filt = degree_rms(sh_filtered, month_idx)

    # Left: both curves
    ax1.loglog(l_vals[1:], rms_raw[1:], "b-", linewidth=1, marker=".",
               markersize=2, label="Raw")
    ax1.loglog(l_vals[1:], rms_filt[1:], "r-", linewidth=1, marker=".",
               markersize=2, label="Filtered")
    ax1.set_xlabel("Degree L")
    ax1.set_ylabel("Degree-RMS")
    ax1.set_title("Degree-RMS Comparison")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: ratio filtered/raw
    ratio = np.zeros_like(rms_filt)
    mask = rms_raw > 0
    ratio[mask] = rms_filt[mask] / rms_raw[mask]
    ax2.semilogy(l_vals[1:], ratio[1:], "k-", linewidth=1, marker=".",
                 markersize=2)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Degree L")
    ax2.set_ylabel("Filtered / Raw RMS Ratio")
    ax2.set_title("Filtered-to-Raw RMS Ratio")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
