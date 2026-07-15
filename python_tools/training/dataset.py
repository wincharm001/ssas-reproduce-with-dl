"""Training dataset: raw GRACE → SSAS-filtered pairs as EWH patches."""

import zipfile
import re as _re
import sys as _sys
from pathlib import Path as _Path
from typing import Tuple as _Tuple

import numpy as _np
import torch
from torch.utils.data import Dataset as _Dataset

# Allow importing sibling modules
_PROJ = _Path(__file__).resolve().parent.parent.parent
_sys.path.insert(0, str(_PROJ))

from python_tools.sh_io import read_mat_csh as _read_mat, read_ascii_gsm as _read_gsm
from python_tools.sh_synthesis import sh_to_ewh as _sh_to_ewh


# ── Paths ─────────────────────────────────────────────────────────
_RAW_DIR  = _PROJ / "data" / "raw_gsm_rl06"
_FILT_DIR = _PROJ / "SSAS_final_result" / "CSR06_200204-202106"
_MAT_FILE = _PROJ / "data" / "csr06_gsm_2004-12_2005-02.mat"
CACHE_DIR = _PROJ / "data" / "cache_ewh"   # cached pre-synthesized grids

# ── Grid config ───────────────────────────────────────────────────
GRID_SHAPE = (180, 360)   # nlat, nlon
LMAX       = 60
N_CHANNELS = 1            # EWH only (could add lat-weight channel)


def _build_index() -> list[dict]:
    """Build a list of {yyyymm, raw_path, filt_path} entries."""
    # Get filtered months
    filt_files = sorted(_FILT_DIR.glob("CSR06_*.txt"))
    months = {}
    for fp in filt_files:
        m = _re.match(r"CSR06_(\d{6})\.txt", fp.name)
        if m:
            months[m[1]] = fp

    # Get raw files, map by approximate month
    raw_files = {}
    for fp in sorted(_RAW_DIR.glob("GSM-2_*")):
        m = _re.match(r"GSM-2_(\d{4})(\d{3})-(\d{4})(\d{3})_", fp.name)
        if m:
            y1, d1 = int(m[1]), int(m[2])
            from datetime import date, timedelta
            mid = date(y1, 1, 1) + timedelta(days=d1 - 1) + timedelta(days=15)
            key = f"{mid.year:04d}{mid.month:02d}"
            raw_files[key] = fp

    # Pair them
    index = []
    for yyyymm in sorted(months.keys()):
        # Find closest raw file
        best = None
        for key, fp in raw_files.items():
            if key == yyyymm:
                best = fp
                break
        if best is None:
            # fallback: nearest month
            target = int(yyyymm)
            best_dist = 9999
            for key, fp in raw_files.items():
                dist = abs(int(key) - target)
                if dist < best_dist:
                    best_dist = dist
                    best = fp
        if best:
            index.append({"yyyymm": yyyymm, "raw": best, "filt": months[yyyymm]})

    return index


class GRACEDenoiseDataset(_Dataset):
    """Patch-based dataset for GRACE stripe-noise denoising.

    Each sample = (raw_ewh_patch, noise_patch) where noise = raw - filtered.
    """

    def __init__(self, split: str = "train", val_year: int = 2015,
                 patch_size: int = 64, patches_per_grid: int = 16,
                 cache_grids: bool = True):
        """
        Args:
            split: 'train' or 'val'
            val_year: years >= this go to validation set
            patch_size: square patch side length
            patches_per_grid: how many random patches to yield per grid
            cache_grids: precompute & cache EWH grids to disk
        """
        self.split = split
        self.val_year = val_year
        self.patch_size = patch_size
        self.patches_per_grid = patches_per_grid

        index = _build_index()

        # Compute ensemble mean of all raw SH coefficients (static field)
        # The GSM files from GFZ ISDC include the static background;
        # we remove it to get the time-variable part that matches SSAS output.
        self._raw_mean = None
        self._compute_raw_mean(index[:30])  # Use first 30 months as estimate

        # Split by year
        train_list, val_list = [], []
        for entry in index:
            year = int(entry["yyyymm"][:4])
            if year >= val_year:
                val_list.append(entry)
            else:
                train_list.append(entry)

        self.items = train_list if split == "train" else val_list
        self._grids = {}  # lazy cache

        if cache_grids:
            self._precache()

    def _compute_raw_mean(self, sample_entries: list):
        """Estimate static gravity field from sample of raw files."""
        C_sum, S_sum, n = None, None, 0
        for entry in sample_entries:
            try:
                sh = _read_gsm(entry["raw"])
                if C_sum is None:
                    C_sum = _np.zeros_like(sh.C[0])
                    S_sum = _np.zeros_like(sh.S[0])
                C_sum += sh.C[0]
                S_sum += sh.S[0]
                n += 1
            except Exception:
                continue
        if n > 0:
            self._raw_mean_C = C_sum / n
            self._raw_mean_S = S_sum / n
            print(f"  Static field estimated from {n} months "
                  f"(C[2,0]={self._raw_mean_C[2,0]:.4e})")
        else:
            self._raw_mean_C = None
            self._raw_mean_S = None

    def __len__(self) -> int:
        return len(self.items) * self.patches_per_grid

    def __getitem__(self, idx: int) -> _Tuple[torch.Tensor, torch.Tensor]:
        grid_idx = idx // self.patches_per_grid
        entry = self.items[grid_idx]

        raw_grid, filt_grid = self._load_grids(entry)

        # Random patch extraction
        h_max = raw_grid.shape[0] - self.patch_size
        w_max = raw_grid.shape[1] - self.patch_size
        h0 = _np.random.randint(0, max(h_max, 1))
        w0 = _np.random.randint(0, max(w_max, 1))

        raw_patch  = raw_grid[h0:h0 + self.patch_size, w0:w0 + self.patch_size]
        filt_patch = filt_grid[h0:h0 + self.patch_size, w0:w0 + self.patch_size]
        noise_patch = raw_patch - filt_patch

        # To tensor: (1, H, W)
        raw_t   = torch.from_numpy(raw_patch.astype(_np.float32)).unsqueeze(0)
        noise_t = torch.from_numpy(noise_patch.astype(_np.float32)).unsqueeze(0)

        return raw_t, noise_t

    def _load_grids(self, entry: dict) -> tuple:
        """Load or compute raw and filtered EWH grids."""
        key = entry["yyyymm"]
        if key in self._grids:
            return self._grids[key]

        # Raw — subtract static mean field if available
        raw_sh = _read_gsm(entry["raw"])
        if self._raw_mean_C is not None:
            C_anom = raw_sh.C[0] - self._raw_mean_C
            S_anom = raw_sh.S[0] - self._raw_mean_S
        else:
            C_anom, S_anom = raw_sh.C[0], raw_sh.S[0]
        raw_grid, _, _ = _sh_to_ewh(C_anom, S_anom, lmax=LMAX,
                                    nlat=GRID_SHAPE[0], nlon=GRID_SHAPE[1])

        # Filtered (already time-variable only)
        filt_sh = _read_gsm(entry["filt"])
        filt_grid, _, _ = _sh_to_ewh(filt_sh.C[0], filt_sh.S[0], lmax=LMAX,
                                     nlat=GRID_SHAPE[0], nlon=GRID_SHAPE[1])

        self._grids[key] = (raw_grid, filt_grid)
        return raw_grid, filt_grid

    def _precache(self):
        """Pre-synthesize all grids and optionally save to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(self.items):
            key = entry["yyyymm"]
            cache_path = CACHE_DIR / f"{key}.npz"

            if cache_path.exists():
                data = _np.load(cache_path)
                self._grids[key] = (data["raw"], data["filt"])
            else:
                raw, filt = self._load_grids(entry)
                if len(self.items) <= 50 or i % 10 == 0:
                    _np.savez_compressed(cache_path, raw=raw, filt=filt)
                self._grids[key] = (raw, filt)

        print(f"  [{self.split}] {len(self.items)} grids cached "
              f"(×{self.patches_per_grid} patches = {len(self)} samples)")
