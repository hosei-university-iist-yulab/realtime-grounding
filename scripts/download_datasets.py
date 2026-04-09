#!/usr/bin/env python3
"""
Download datasets for TGP experiments using kagglehub.

Datasets:
1. BDG2 (Building Data Genome Project 2)
2. REDD (Reference Energy Disaggregation Dataset)
3. UK-DALE (UK Domestic Appliance-Level Electricity)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_bdg2():
    """Download BDG2 dataset."""
    print("\n" + "=" * 50)
    print("Downloading BDG2 (Building Data Genome Project 2)...")
    print("=" * 50)

    import kagglehub

    path = kagglehub.dataset_download("claytonmiller/buildingdatagenomeproject2")
    print(f"Downloaded to: {path}")

    # Create symlink to data/raw/bdg2
    target = DATA_DIR / "bdg2_kaggle"
    if not target.exists():
        os.symlink(path, target)
        print(f"Symlinked to: {target}")

    return path


def download_redd():
    """Download REDD dataset."""
    print("\n" + "=" * 50)
    print("Downloading REDD (Reference Energy Disaggregation Dataset)...")
    print("=" * 50)

    import kagglehub

    path = kagglehub.dataset_download("pawelkauf/redd-part")
    print(f"Downloaded to: {path}")

    # Create symlink to data/raw/redd
    target = DATA_DIR / "redd"
    if not target.exists():
        os.symlink(path, target)
        print(f"Symlinked to: {target}")

    return path


def download_ukdale():
    """Download UK-DALE dataset."""
    print("\n" + "=" * 50)
    print("Downloading UK-DALE...")
    print("=" * 50)

    import kagglehub

    path = kagglehub.dataset_download("abdelmdz/uk-dale")
    print(f"Downloaded to: {path}")

    # Create symlink to data/raw/ukdale
    target = DATA_DIR / "ukdale"
    if not target.exists():
        os.symlink(path, target)
        print(f"Symlinked to: {target}")

    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download TGP datasets")
    parser.add_argument("--bdg2", action="store_true", help="Download BDG2")
    parser.add_argument("--redd", action="store_true", help="Download REDD")
    parser.add_argument("--ukdale", action="store_true", help="Download UK-DALE")
    parser.add_argument("--all", action="store_true", help="Download all datasets")

    args = parser.parse_args()

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If no specific dataset selected, download all
    if args.all or not (args.bdg2 or args.redd or args.ukdale):
        args.bdg2 = args.redd = args.ukdale = True

    paths = {}

    if args.bdg2:
        try:
            paths["bdg2"] = download_bdg2()
        except Exception as e:
            print(f"Error downloading BDG2: {e}")

    if args.redd:
        try:
            paths["redd"] = download_redd()
        except Exception as e:
            print(f"Error downloading REDD: {e}")

    if args.ukdale:
        try:
            paths["ukdale"] = download_ukdale()
        except Exception as e:
            print(f"Error downloading UK-DALE: {e}")

    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nDatasets ready in:", DATA_DIR)


if __name__ == "__main__":
    main()
