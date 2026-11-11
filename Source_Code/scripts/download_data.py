#!/usr/bin/env python3
"""Download all raw corpora into data/raw (PhiUSIIL, Kaggle, Hannousse, PhishTank)."""
import argparse

from drift_phish.data import download_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    download_all(ap.parse_args().force)
