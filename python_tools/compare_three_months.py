"""Visualize raw vs SSAS-filtered vs noise for the 3 months in data/.

Generates a 3×3 figure comparing raw GRACE EWH, SSAS-filtered EWH,
and the removed noise (raw - filtered) for 2004-12, 2005-01, 2005-02.
"""

import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from python_tools.sh_io import read_mat_csh, read_ascii_gsm, SHCoeffs
from python_tools.sh_synthesis import sh_to_ewh

# ── paths ────────────────────────────────────────────────────────
MAT_FILE = _PROJ / "data" / "csr06_gsm_2004-12_2005-02.mat"
FILT_DIR = _PROJ / "SSAS_final_result" / "CSR06_200204-202106"
OUT_DIR = _PROJ / "python_tools" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load raw data (3 months) ─────────────────────────────────────
print("Loading raw .mat data ...")
sh_raw_all = read_mat_csh(MAT_FILE)
print(f"  {sh_raw_all.C.shape[0]} months, lmax={sh_raw_all.lmax}")

# ── load filtered data for matching months ───────────────────────
month_labels = [
    ("2004-12", "CSR06_200412.txt"),
    ("2005-01", "CSR06_200501.txt"),
    ("2005-02", "CSR06_200502.txt"),
]

print("Loading filtered data ...")
sh_filt_list = []
for label, fname in month_labels:
    fp = FILT_DIR / fname
    if fp.exists():
        sh = read_ascii_gsm(fp)
        # store with matching label
        sh = SHCoeffs(C=sh.C, S=sh.S, t=None, lmax=sh.lmax, label=f"SSAS {label}")
        sh_filt_list.append(sh)
        print(f"  {fname}: lmax={sh.lmax}")
    else:
        print(f"  {fname}: NOT FOUND — skipping")
        sh_filt_list.append(None)

# ── synthesize EWH grids ─────────────────────────────────────────
print("Synthesizing EWH grids (this takes ~30s per map) ...")

lmax = min(sh_raw_all.lmax, 60)
nlat, nlon = 180, 360
lat = np.linspace(90, -90, nlat)
lon = np.linspace(0, 360, nlon + 1)[:-1]

raw_grids = []
filt_grids = []
noise_grids = []

for i in range(3):
    label, _ = month_labels[i]

    # Raw
    C_raw = sh_raw_all.C[i]
    S_raw = sh_raw_all.S[i]
    print(f"  Raw {label} ...")
    grid_raw, _, _ = sh_to_ewh(C_raw, S_raw, lmax=sh_raw_all.lmax, nlat=nlat, nlon=nlon)
    raw_grids.append(grid_raw)

    # Filtered
    if sh_filt_list[i] is not None:
        C_filt = sh_filt_list[i].C[0]
        S_filt = sh_filt_list[i].S[0]
        print(f"  Filtered {label} ...")
        grid_filt, _, _ = sh_to_ewh(C_filt, S_filt, lmax=sh_filt_list[i].lmax,
                                    nlat=nlat, nlon=nlon)
    else:
        # fallback: use raw as placeholder
        grid_filt = grid_raw.copy()
    filt_grids.append(grid_filt)

    # Noise = raw - filtered
    noise_grids.append(grid_raw - grid_filt)

# ── global colour scale ──────────────────────────────────────────
all_raw = np.concatenate([g.ravel() for g in raw_grids])
all_noise = np.concatenate([g.ravel() for g in noise_grids])
vmax_raw = np.percentile(np.abs(all_raw), 99)
vmax_noise = np.percentile(np.abs(all_noise), 99)
vmax_raw = max(vmax_raw, 0.01)  # avoid zero
vmax_noise = max(vmax_noise, 0.001)

print(f"  Raw colour range: ±{vmax_raw:.4f} m EWH")
print(f"  Noise colour range: ±{vmax_noise:.4f} m EWH")

# ── plot ─────────────────────────────────────────────────────────
print("Plotting ...")
cmap = plt.cm.RdBu_r

fig, axes = plt.subplots(3, 3, figsize=(16, 13))

col_titles = ["Raw (Unfiltered)", "SSAS Filtered", "Noise (Raw − Filtered)"]
row_labels = [f"2004-12\n({sh_raw_all.t[0]:.3f})",
              f"2005-01\n({sh_raw_all.t[1]:.3f})",
              f"2005-02\n({sh_raw_all.t[2]:.3f})"]

im_last = None
for row in range(3):
    for col in range(3):
        ax = axes[row, col]

        if col == 0:
            grid = raw_grids[row]
            vmn, vmx = -vmax_raw, vmax_raw
        elif col == 1:
            grid = filt_grids[row]
            vmn, vmx = -vmax_raw, vmax_raw
        else:
            grid = noise_grids[row]
            vmn, vmx = -vmax_noise, vmax_noise

        im = ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=vmn, vmax=vmx,
                           shading="auto", rasterized=True)

        # Column titles (top row only)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=11, fontweight="bold", pad=8)

        # Row labels
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=10, rotation=0,
                          labelpad=35, va="center")

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)" if row == 2 else "")
        ax.set_ylabel("Latitude (°)" if col > 0 else "")

        # Save last im for colorbars
        im_last = im

# ── colourbars ───────────────────────────────────────────────────
# Raw/Filtered colourbar (shared scale)
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.012, 0.32])
cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax_raw, vmax_raw), 
                                          cmap=cmap),
                    cax=cbar_ax1)
cb1.set_label("EWH (m) — Raw & Filtered", fontsize=9)
cb1.formatter = ticker.ScalarFormatter(useMathText=True)
cb1.formatter.set_powerlimits((-2, 2))
cb1.update_ticks()

# Noise colourbar (separate scale)
cbar_ax2 = fig.add_axes([0.92, 0.12, 0.012, 0.32])
cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax_noise, vmax_noise),
                                          cmap=cmap),
                    cax=cbar_ax2)
cb2.set_label("EWH (m) — Noise", fontsize=9)
cb2.formatter = ticker.ScalarFormatter(useMathText=True)
cb2.formatter.set_powerlimits((-2, 2))
cb2.update_ticks()

fig.suptitle("GRACE CSR RL06 — Raw vs SSAS-Filtered Equivalent Water Height\n"
             "2004-12 / 2005-01 / 2005-02",
             fontsize=13, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.06,
                    hspace=0.15, wspace=0.08)

out_path = OUT_DIR / "09_three_months_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# ── RMS stats ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Per-Month Noise Statistics (Ocean RMS proxy via global std)")
print("=" * 60)
for i, (label, _) in enumerate(month_labels):
    raw_std = np.std(raw_grids[i])
    filt_std = np.std(filt_grids[i])
    noise_std = np.std(noise_grids[i])
    reduction = (1 - noise_std / raw_std) * 100 if raw_std > 0 else 0
    print(f"  {label}:  Raw σ={raw_std:.4f} m  |  Filtered σ={filt_std:.4f} m  |  "
          f"Noise σ={noise_std:.4f} m  |  Noise/Raw={noise_std/raw_std*100:.1f}%")

print("\nDone.")
