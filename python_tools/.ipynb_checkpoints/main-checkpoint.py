#!/usr/bin/env python
"""SSAS-GRACE-filter Python visualisation demo.

Reads MATLAB .mat and ASCII GSM data, computes spatial fields via
spherical harmonic synthesis, and generates:
  1. Kaula (degree-RMS) power spectrum curves
  2. Global equivalent-water-height maps
  3. Raw vs filtered north-south stripe-noise comparison

Usage
-----
    python main.py

Output
------
    All figures are saved to python_tools/output/ by default.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Ensure the project directory is on sys.path so we can import python_tools
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import matplotlib.pyplot as plt

from python_tools.sh_io import (
    SHCoeffs,
    read_mat_csh,
    read_ascii_gsm,
    read_zip_gsm,
)
from python_tools.sh_viz import (
    plot_kaula,
    plot_global_map,
    plot_noise_comparison,
    plot_degree_rms_diff,
)

# ---------------------------------------------------------------------------
# Paths relative to python_tools/ parent (SSAS-GRACE-filter/)
# ---------------------------------------------------------------------------
DATA_DIR = _PROJ / "data"
OUT_DIR = _PROJ / "python_tools" / "output"
MAT_FILE = DATA_DIR / "csr06_gsm_2004-12_2005-02.mat"
ZIP_FILE = _PROJ / "SSAS_final_result" / "CSR06_200204-202106.zip"
GSM_DIR = _PROJ / "ex_read-SH-files"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Helpers
# ===================================================================

def _extract_single(sh: SHCoeffs, idx: int, label_suffix: str = "") -> SHCoeffs:
    """Extract a single month as a new SHCoeffs object."""
    t_val = sh.t[idx:idx+1] if sh.t is not None else None
    label = sh.label + label_suffix if label_suffix else sh.label
    return SHCoeffs(
        C=sh.C[idx:idx+1],
        S=sh.S[idx:idx+1],
        t=t_val,
        lmax=sh.lmax,
        label=label,
    )


def _safe_t(sh: SHCoeffs, idx: int = 0) -> str:
    """Format timestamp safely, returning '?' if not available."""
    if sh.t is not None and idx < len(sh.t):
        return f"{sh.t[idx]:.3f}"
    return "?"


# ===================================================================
def demo_mat_data():
    """Demo 1: read MATLAB .mat, plot Kaula and global EWH maps."""
    print("=" * 60)
    print("Demo 1: MATLAB .mat CSR GSM data")
    print("=" * 60)

    sh = read_mat_csh(MAT_FILE)
    print(f"  Loaded {sh.C.shape[0]} months, lmax={sh.lmax}")
    print(f"  Times: {sh.t}")

    # --- Kaula curves for each month in one plot ---
    months = [_extract_single(sh, i, f" ({_safe_t(sh, i)})") for i in range(sh.C.shape[0])]
    plot_kaula(months, title="CSR GSM Degree-RMS (3 months)",
               save_path=OUT_DIR / "01_kaula_csr_mat.png")
    plt.close("all")

    # --- Global EWH map for the middle month ---
    sh_mid = months[1]
    plot_global_map(sh_mid, month_idx=0, field_type="ewh",
                    title=f"CSR GSM EWH — {_safe_t(sh_mid)}",
                    save_path=OUT_DIR / "02_map_ewh_mat.png")
    plt.close("all")

    print("  Done.\n")
    return sh


# ===================================================================
def demo_ascii_gsm():
    """Demo 2: read ASCII GSM files, plot Kaula comparison."""
    print("=" * 60)
    print("Demo 2: ASCII GSM files")
    print("=" * 60)

    gsm_files = sorted(GSM_DIR.glob("GSM-2_*"))
    if not gsm_files:
        print("  No GSM files found — skipping.")
        return

    print(f"  Found {len(gsm_files)} files")

    sh_list = []
    for f in gsm_files:
        try:
            sh = read_ascii_gsm(f)
            sh_list.append(sh)
            print(f"    {f.name}: lmax={sh.lmax}")
        except Exception as e:
            print(f"    {f.name}: ERROR: {e}")

    if sh_list:
        plot_kaula(sh_list, title="GSM ASCII Files: Degree-RMS",
                   save_path=OUT_DIR / "03_kaula_ascii.png")
        plt.close("all")

    print("  Done.\n")


# ===================================================================
def demo_zip_filtered():
    """Demo 3: read ZIP archive, compare RAW vs FILTERED."""
    print("=" * 60)
    print("Demo 3: ZIP archive — SSAS filtered results")
    print("=" * 60)

    if not ZIP_FILE.exists():
        print(f"  ZIP not found at {ZIP_FILE} — skipping.")
        return None

    # Read first month of filtered data (2002-04)
    sh_filt = read_zip_gsm(ZIP_FILE, month_indices=[0])
    print(f"  Filtered (1st month): lmax={sh_filt.lmax}, t={_safe_t(sh_filt)}")

    # Plot Kaula for filtered coefficients
    plot_kaula([sh_filt], title="CSR RL06 SSAS-Filtered: Degree-RMS (2002-04)",
               save_path=OUT_DIR / "04_kaula_filtered.png")
    plt.close("all")

    # Global EWH map
    plot_global_map(sh_filt, month_idx=0, field_type="ewh",
                    title=f"CSR RL06 SSAS-Filtered EWH — {_safe_t(sh_filt)}",
                    save_path=OUT_DIR / "05_map_ewh_filtered.png")
    plt.close("all")

    # Read multiple months from ZIP and plot Kaula comparison
    n_months = min(12, 196)  # first year
    sh_series = read_zip_gsm(ZIP_FILE, month_indices=list(range(n_months)))
    indices = [0, n_months // 2, n_months - 1]
    multi = [
        _extract_single(sh_series, idx, f" ({_safe_t(sh_series, idx)})")
        for idx in indices
    ]
    plot_kaula(multi, title="CSR RL06 SSAS-Filtered: Degree-RMS (3 epochs)",
               save_path=OUT_DIR / "06_kaula_filtered_timeseries.png")
    plt.close("all")

    print("  Done.\n")
    return sh_filt


# ===================================================================
def demo_noise_comparison():
    """Demo 4: raw (.mat) vs filtered (ZIP) noise comparison."""
    print("=" * 60)
    print("Demo 4: Raw vs Filtered noise comparison")
    print("=" * 60)

    if not MAT_FILE.exists() or not ZIP_FILE.exists():
        print("  Missing data — skipping.")
        return

    sh_raw = read_mat_csh(MAT_FILE)
    sh_raw_month = _extract_single(sh_raw, 0)

    # Find matching month in ZIP (2004-12)
    with zipfile.ZipFile(str(ZIP_FILE)) as zf:
        names = sorted(zf.namelist())

    # 2004-12: index 0 = 2002-04, so 2004-12 = 32 months later
    idx = 32
    if idx >= len(names):
        idx = len(names) - 1

    sh_filt = read_zip_gsm(ZIP_FILE, month_indices=[idx])

    print(f"  Raw:       {sh_raw_month.label} t={_safe_t(sh_raw_month)}")
    print(f"  Filtered:  {sh_filt.label} t={_safe_t(sh_filt)}")

    # Plot noise comparison
    plot_noise_comparison(
        sh_raw_month, sh_filt, month_idx=0,
        save_path=OUT_DIR / "07_noise_comparison.png",
    )
    plt.close("all")

    # Plot degree-RMS difference
    plot_degree_rms_diff(
        sh_raw_month, sh_filt, month_idx=0,
        save_path=OUT_DIR / "08_degree_rms_diff.png",
    )
    plt.close("all")

    print("  Done.\n")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SSAS-GRACE-filter Python Visualisation Tools")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print()

    demo_mat_data()
    demo_ascii_gsm()
    demo_zip_filtered()
    demo_noise_comparison()

    print("=" * 60)
    print(f"All figures saved to {OUT_DIR}")
    print("=" * 60)
