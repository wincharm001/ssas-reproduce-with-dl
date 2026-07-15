"""Download raw CSR RL06 GRACE GSM files from GFZ ISDC.

Matches SSAS-filtered months (CSR06_YYYYMM.txt) to raw GSM files
(GSM-2_YYYYDOY-YYYYDOY_GRAC_UTCSR_BA01_0600.gz) and downloads them.

Usage:
    python download_raw_gsm.py              # download all 196 months
    python download_raw_gsm.py --dry-run    # check mapping only
    python download_raw_gsm.py --months 3   # download only 3 months for test
"""

import argparse
import gzip
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests

# ── config ────────────────────────────────────────────────────────
BASE_URL = "https://isdc-data.gfz.de/grace/Level-2/CSR/RL06/"
PROJ_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJ_ROOT / "SSAS_final_result" / "CSR06_200204-202106.zip"
FILT_DIR = PROJ_ROOT / "SSAS_final_result" / "CSR06_200204-202106"
RAW_DIR = PROJ_ROOT / "data" / "raw_gsm_rl06"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SSAS-filter-research/1.0"})


# ── helpers ───────────────────────────────────────────────────────

def doy_to_date(year: int, doy: int) -> date:
    """Convert (year, day-of-year) to date."""
    return date(year, 1, 1) + timedelta(days=doy - 1)


def month_midpoint(yyyymm: str) -> date:
    """Return the ~15th of the month for a YYYYMM string."""
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return date(y, m, 15)


def parse_gsm_filename(fname: str) -> tuple[date, date] | None:
    """Parse 'GSM-2_2004336-2004366_...' → (start_date, end_date)."""
    m = re.match(r"GSM-2_(\d{4})(\d{3})-(\d{4})(\d{3})_", fname)
    if not m:
        return None
    y1, d1, y2, d2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return doy_to_date(y1, d1), doy_to_date(y2, d2)


def get_filtered_months() -> list[str]:
    """Get sorted list of YYYYMM from the filtered ZIP/directory."""
    if FILT_DIR.exists():
        months = []
        for fp in sorted(FILT_DIR.glob("CSR06_*.txt")):
            m = re.match(r"CSR06_(\d{6})\.txt", fp.name)
            if m:
                months.append(m[1])
        return months
    elif ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            months = []
            for name in sorted(zf.namelist()):
                m = re.match(r"CSR06_(\d{6})\.txt", name)
                if m:
                    months.append(m[1])
            return months
    else:
        raise FileNotFoundError(f"Neither {FILT_DIR} nor {ZIP_PATH} found")


def fetch_file_list() -> list[str]:
    """Fetch the directory listing and extract GSM BA01 filenames."""
    print(f"Fetching file list from {BASE_URL} ...")
    resp = SESSION.get(BASE_URL, timeout=30)
    resp.raise_for_status()

    # Parse HTML directory listing for .gz hrefs
    filenames = re.findall(r'href="([^"]+\.gz)"', resp.text)

    # Filter to GSM-2 BA01 only (not BB01, GAC, GAD)
    gsm_files = [f for f in filenames if "GSM-2_" in f and "BA01" in f]
    print(f"  Found {len(filenames)} total .gz files, {len(gsm_files)} GSM BA01 files")
    return gsm_files


def match_months_to_files(
    months: list[str], gsm_files: list[str]
) -> dict[str, str]:
    """Map each YYYYMM to the best matching GSM filename.

    Strategy: for each month, find the GSM file whose date range
    contains the 15th of that month.
    """
    # Parse all GSM files into (start, end) ranges
    gsm_ranges = {}
    for fname in gsm_files:
        parsed = parse_gsm_filename(fname)
        if parsed:
            gsm_ranges[fname] = parsed

    mapping = {}
    unmatched = []

    for yyyymm in months:
        mid = month_midpoint(yyyymm)
        best = None

        for fname, (start, end) in gsm_ranges.items():
            if start <= mid <= end:
                # Found exact match
                best = fname
                break

        if best is None:
            # Try closest match (nearest mid-date to file mid-date)
            best_dist = 9999
            for fname, (start, end) in gsm_ranges.items():
                fmid = start + (end - start) / 2
                dist = abs((mid - fmid).days)
                if dist < best_dist:
                    best_dist = dist
                    best = fname

        if best:
            mapping[yyyymm] = best
        else:
            unmatched.append(yyyymm)

    if unmatched:
        print(f"  WARNING: {len(unmatched)} months could not be matched: {unmatched[:5]}...")

    return mapping


def download_file(fname: str, dest_dir: Path) -> bool:
    """Download a single .gz file and decompress it."""
    dest_gz = dest_dir / fname
    dest_txt = dest_dir / fname.replace(".gz", "")

    if dest_txt.exists():
        return True  # already downloaded

    url = urljoin(BASE_URL, fname)
    try:
        print(f"  Downloading {fname} ... ", end="", flush=True)
        resp = SESSION.get(url, timeout=120)
        resp.raise_for_status()

        # Save .gz
        with open(dest_gz, "wb") as f:
            f.write(resp.content)

        # Decompress
        with gzip.open(dest_gz, "rb") as f_in:
            with open(dest_txt, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove .gz after decompression
        dest_gz.unlink()

        size_kb = dest_txt.stat().st_size / 1024
        print(f"OK ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        # Clean up partial downloads
        for p in [dest_gz, dest_txt]:
            if p.exists():
                p.unlink()
        return False


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download raw CSR RL06 GSM data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show mapping without downloading")
    parser.add_argument("--months", type=int, default=0,
                        help="Download only first N months (0 = all)")
    parser.add_argument("--start", type=str, default="",
                        help="Start from YYYYMM (e.g., 200412)")
    args = parser.parse_args()

    # Get filtered months
    months = get_filtered_months()
    print(f"Filtered months available: {len(months)}")
    print(f"  Range: {months[0]} to {months[-1]}")

    # Apply --start filter
    if args.start:
        months = [m for m in months if m >= args.start]
        print(f"  After --start {args.start}: {len(months)} months")

    # Apply --months limit
    if args.months > 0:
        months = months[:args.months]
        print(f"  Limited to {len(months)} months")

    # Fetch file list
    gsm_files = fetch_file_list()

    # Match
    print("Matching months to GSM files ...")
    mapping = match_months_to_files(months, gsm_files)
    print(f"  Matched: {len(mapping)} / {len(months)} months")

    if args.dry_run:
        print("\n--- Dry Run: First 10 mappings ---")
        for i, (yyyymm, fname) in enumerate(mapping.items()):
            if i >= 10:
                break
            start, end = parse_gsm_filename(fname)
            print(f"  {yyyymm} → {fname}  ({start} to {end})")
        return

    # Download
    print(f"\nDownloading to {RAW_DIR} ...")
    success = 0
    fail = 0
    t0 = time.time()

    for i, (yyyymm, fname) in enumerate(mapping.items()):
        print(f"[{i+1:3d}/{len(mapping)}] {yyyymm}:", end=" ")
        if download_file(fname, RAW_DIR):
            success += 1
        else:
            fail += 1
        time.sleep(0.3)  # polite delay

    elapsed = time.time() - t0
    print(f"\nDone. {success} OK, {fail} failed in {elapsed:.0f}s")
    print(f"Files in {RAW_DIR}: {len(list(RAW_DIR.glob('GSM-2_*')))}")


if __name__ == "__main__":
    main()
