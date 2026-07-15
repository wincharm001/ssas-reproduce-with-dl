"""Spherical harmonic coefficient I/O.

Supports:
  - MATLAB .mat files (cSH class format from the SSAS project)
  - ASCII GSM files (GRACE Level-2: L M C S per line)
  - ZIP archives of ASCII GSM files (SSAS filtered results)
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.io import loadmat


class SHCoeffs(NamedTuple):
    """Spherical harmonic coefficients container.

    Attributes
    ----------
    C : np.ndarray
        Cosine coefficients, shape (lmax+1, lmax+1) or (nmonths, lmax+1, lmax+1).
        Matrix format: only m <= l entries contain valid data.
    S : np.ndarray
        Sine coefficients, same shape as C.
    t : np.ndarray | None
        Decimal-year timestamps, shape (nmonths,) or None if single epoch.
    lmax : int
        Maximum spherical harmonic degree.
    label : str
        Data source label for display.
    """
    C: np.ndarray
    S: np.ndarray
    t: np.ndarray | None
    lmax: int
    label: str = ""


# ---------------------------------------------------------------------------
# GRS80 / Earth constants
# ---------------------------------------------------------------------------
GRS80_GM = 3.986004415e14        # geocentric gravitational constant [m³/s²]
GRS80_AE = 6378136.3             # equatorial radius [m]
GRS80_OMEGA = 7292115e-11        # Earth rotation rate [rad/s]
RHO_E = 5517.0                   # average Earth density [kg/m³]
RHO_W = 1000.0                   # water density [kg/m³]


# ===================================================================
# MATLAB .mat reader
# ===================================================================

def read_mat_csh(filepath: str | Path) -> SHCoeffs:
    """Read spherical harmonic coefficients from a MATLAB cSH .mat file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file (e.g. csr06_gsm_2004-12_2005-02.mat).

    Returns
    -------
    SHCoeffs
    """
    filepath = Path(filepath)
    mat = loadmat(str(filepath))

    # The .mat stores an array of cSH structs: shape (1, nmonths)
    sh_array = mat["SH"]
    nmonths = sh_array.shape[1]

    c_list, s_list, t_list = [], [], []
    lmax = None

    for i in range(nmonths):
        item = sh_array[0, i]
        c_mat = item["C"]  # shape (lmax+1, lmax+1)
        s_mat = item["S"]
        tt = float(item["tt"].ravel()[0])

        if lmax is None:
            lmax = c_mat.shape[0] - 1

        c_list.append(c_mat)
        s_list.append(s_mat)
        t_list.append(tt)

    C = np.array(c_list)  # (nmonths, lmax+1, lmax+1)
    S = np.array(s_list)
    t = np.array(t_list)

    # lmax is guaranteed non-None because nmonths >= 1
    assert lmax is not None
    return SHCoeffs(C=C, S=S, t=t, lmax=lmax, label=filepath.stem)


# ===================================================================
# ASCII GSM reader
# ===================================================================

def read_ascii_gsm(filepath: str | Path) -> SHCoeffs:
    """Read a single ASCII GRACE GSM file (GRCOF2 or simple L M C S format).

    Supports:
      - Simple 4-column: L M C S
      - GRCOF2 format: GRCOF2 L M C S sigma_C sigma_S t_start t_end

    Parameters
    ----------
    filepath : str or Path
        Path to the GSM text file.

    Returns
    -------
    SHCoeffs
    """
    filepath = Path(filepath)

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # Filter data lines: either "GRCOF2" prefix or plain numbers
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("header:") or stripped.startswith("  "):
            continue  # YAML header
        # GRCOF2 format or simple numeric
        if stripped.startswith("GRCOF2"):
            parts = stripped.split()
            # GRCOF2 L M C S sigC sigS t0 t1
            l = int(parts[1])
            m = int(parts[2])
            c = float(parts[3].replace("D", "E"))
            s = float(parts[4].replace("D", "E"))
            data_lines.append((l, m, c, s))
        else:
            # Simple format: L M C S
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    l = int(parts[0])
                    m = int(parts[1])
                    c = float(parts[2])
                    s = float(parts[3])
                    data_lines.append((l, m, c, s))
                except ValueError:
                    continue  # skip non-data lines

    if not data_lines:
        raise ValueError(f"No valid SH data found in {filepath}")

    lmax = max(row[0] for row in data_lines)
    C = np.zeros((lmax + 1, lmax + 1))
    S = np.zeros((lmax + 1, lmax + 1))

    for l, m, c, s in data_lines:
        C[l, m] = c
        S[l, m] = s

    return SHCoeffs(C=C[np.newaxis], S=S[np.newaxis], t=None, lmax=lmax,
                    label=filepath.stem)


def read_zip_gsm(zip_path: str | Path, month_indices: list[int] | None = None,
                 ) -> SHCoeffs:
    """Read multiple ASCII GSM files from a ZIP archive.

    Parameters
    ----------
    zip_path : str or Path
        Path to the ZIP file.
    month_indices : list[int] or None
        Specific month indices to read (0-based). If None, reads all.

    Returns
    -------
    SHCoeffs
    """
    zip_path = Path(zip_path)
    zf = zipfile.ZipFile(str(zip_path))
    names = sorted(zf.namelist())

    if month_indices is not None:
        names = [names[i] for i in month_indices]

    c_list, s_list, t_list = [], [], []
    lmax = None

    for name in names:
        with zf.open(name) as f:
            data = np.loadtxt(f)

        l_vals = data[:, 0].astype(int)
        m_vals = data[:, 1].astype(int)
        c_vals = data[:, 2]
        s_vals = data[:, 3]

        cur_lmax = int(l_vals.max())
        if lmax is None:
            lmax = cur_lmax

        C = np.zeros((lmax + 1, lmax + 1))
        S = np.zeros((lmax + 1, lmax + 1))

        for l, m, c, s in zip(l_vals, m_vals, c_vals, s_vals):
            if l <= lmax:
                C[l, m] = c
                S[l, m] = s

        c_list.append(C)
        s_list.append(S)

        # Parse decimal year from filename like CSR06_200204.txt
        yy, mm = int(name[6:10]), int(name[10:12])
        t_list.append(yy + (mm - 0.5) / 12.0)

    C = np.array(c_list)
    S = np.array(s_list)
    t = np.array(t_list)

    assert lmax is not None, "No files found in ZIP archive"
    return SHCoeffs(C=C, S=S, t=t, lmax=lmax,
                    label=Path(zip_path).stem)


# ===================================================================
# Subset / difference utilities
# ===================================================================

def extract_month(sh: SHCoeffs, idx: int) -> SHCoeffs:
    """Extract a single month from multi-month coefficients."""
    return SHCoeffs(
        C=sh.C[idx:idx+1], S=sh.S[idx:idx+1],
        t=sh.t[idx:idx+1] if sh.t is not None else None,
        lmax=sh.lmax, label=f"{sh.label}[{idx}]"
    )


def diff(sh1: SHCoeffs, sh2: SHCoeffs) -> SHCoeffs:
    """Compute the difference between two coefficient sets."""
    assert sh1.lmax == sh2.lmax, "lmax mismatch"
    return SHCoeffs(
        C=sh1.C - sh2.C, S=sh1.S - sh2.S,
        t=sh1.t, lmax=sh1.lmax,
        label=f"{sh1.label} - {sh2.label}"
    )


def degree_rms(sh: SHCoeffs, month_idx: int = 0) -> np.ndarray:
    """Compute per-degree RMS power (degree variance).

    RMS_l = sqrt( Σ_{m=0}^{l} (C_lm² + S_lm²) / (2l+1) ),
    omitting the m=0 term from S (always zero).

    Returns
    -------
    rms : np.ndarray, shape (lmax+1,)
    """
    C = sh.C[month_idx]
    S = sh.S[month_idx]
    lmax = sh.lmax
    rms = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        c2 = np.sum(C[l, :l+1] ** 2)
        s2 = np.sum(S[l, 1:l+1] ** 2)  # S_l0 is always 0
        rms[l] = np.sqrt((c2 + s2) / (2 * l + 1))
    return rms
