"""Dataset acquisition, URL de-duplication and feature caching.

Every dataset is normalised to a DataFrame with columns ``url`` and ``label``
where ``label == 1`` means PHISHING (the positive class used by all metrics).

Datasets used by the reproduction
---------------------------------
phiusiil    PhiUSIIL (UCI 967) - one of the three corpora the paper trains on.
kaggle      "Phishing Site URLs" (Kaggle) 21k balanced subset - the second
            corpus the paper names (mirrored on Hugging Face).
iscx        ISCX-URL-2016 (benign vs phishing), distributed by the dataset
            authors as 79 pre-computed lexical features per URL (the URL
            strings are not included). Used for the paper's Table 5 only.
hannousse   Hannousse & Yahiouhmed (2021) web-phishing URLs, 11,430 balanced
            URLs. Raw URL strings, so it joins the combined corpus and the
            streams in place of the (feature-only) ISCX release.
phishtank   Live PhishTank verified-online feed (the "New URLs" source the
            paper streams). Phishing-only; snapshot date is stored in the file
            name. Used for the "old vs new" drift analysis and as the final
            stream stage (legitimate side comes from Hannousse).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

UCI_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
HF = "https://huggingface.co/datasets"
KAGGLE_FILES = [
    f"{HF}/KaushiGihan/phishing-site-url/resolve/main/data/{s}-00000-of-00001.parquet"
    for s in ("train", "test", "validation")
]
HANNOUSSE_FILES = [
    f"{HF}/pirocheto/phishing-url/resolve/main/data/{s}.parquet" for s in ("train", "test")
]
ISCX_FILE = f"{HF}/bencorn/ISCX-URL-2016/resolve/main/Phishing.csv"
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
UA = "Mozilla/5.0 (cs552-research-reproduction)"


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_all(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    out = RAW / "phiusiil.csv"
    if force or not out.exists():
        print("downloading PhiUSIIL (UCI 967) ...")
        z = zipfile.ZipFile(io.BytesIO(_get(UCI_URL)))
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=["URL", "label"])
        # PhiUSIIL: label 1 = legitimate, 0 = phishing  ->  flip to 1 = phishing.
        pd.DataFrame(
            {"url": df["URL"], "label": 1 - df["label"].astype(int)}
        ).to_csv(out, index=False)

    out = RAW / "kaggle.csv"
    if force or not out.exists():
        print("downloading Kaggle phishing-site-urls subset ...")
        df = pd.concat(
            [pd.read_parquet(io.BytesIO(_get(u))) for u in KAGGLE_FILES], ignore_index=True
        )
        # In this corpus labels: 1 = phishing ("bad"), 0 = legitimate ("good").
        pd.DataFrame({"url": df["text"], "label": df["labels"].astype(int)}).to_csv(
            out, index=False
        )

    out = RAW / "hannousse.csv"
    if force or not out.exists():
        print("downloading Hannousse & Yahiouhmed 2021 ...")
        df = pd.concat(
            [pd.read_parquet(io.BytesIO(_get(u))) for u in HANNOUSSE_FILES], ignore_index=True
        )
        pd.DataFrame(
            {"url": df["url"], "label": (df["status"] == "phishing").astype(int)}
        ).to_csv(out, index=False)

    out = RAW / "iscx_features.csv"
    if force or not out.exists():
        print("downloading ISCX-URL-2016 (benign vs phishing) ...")
        out.write_bytes(_get(ISCX_FILE))

    if force or not list(RAW.glob("phishtank_*.csv")):
        print("downloading live PhishTank feed ...")
        raw = pd.read_csv(io.BytesIO(_get(PHISHTANK_URL)))
        raw = raw[(raw["verified"] == "yes") & (raw["online"] == "yes")]
        snap = pd.Timestamp.now().strftime("%Y%m%d")
        pd.DataFrame(
            {"url": raw["url"], "label": 1, "submission_time": raw["submission_time"]}
        ).to_csv(RAW / f"phishtank_{snap}.csv", index=False)
    print("raw data ready in", RAW)


def load_raw(name: str) -> pd.DataFrame:
    if name == "phishtank":
        files = sorted(RAW.glob("phishtank_*.csv"))
        if not files:
            raise FileNotFoundError("run scripts/download_data.py first")
        df = pd.read_csv(files[-1])
        df["submission_time"] = pd.to_datetime(df["submission_time"], utc=True, errors="coerce")
        df = df.sort_values("submission_time").reset_index(drop=True)
    else:
        df = pd.read_csv(RAW / f"{name}.csv")
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 3]
    # Drop duplicate URLs; drop URLs that appear with conflicting labels.
    conflict = df.groupby("url")["label"].nunique()
    conflict = set(conflict[conflict > 1].index)
    df = df[~df["url"].isin(conflict)].drop_duplicates("url", keep="first").reset_index(drop=True)
    return df


def load_iscx():
    """ISCX-URL-2016 official feature vectors: (X float32, y int, names).

    label 1 = phishing. Infinite values (ratio features) become 0.
    """
    df = pd.read_csv(RAW / "iscx_features.csv").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (df.iloc[:, -1].astype(str).str.lower() == "phishing").astype(int).to_numpy()
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    return X, y, list(df.columns[:-1])


def load_features(name: str, force: bool = False):
    """Return (X float32, y int, urls) for dataset ``name`` with on-disk cache."""
    from .features.extractor import PAPER_FEATURE_NAMES, extract_matrix

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / f"features_{name}.npz"
    df = load_raw(name)
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        if list(z["names"]) == PAPER_FEATURE_NAMES and len(z["y"]) == len(df):
            return z["X"], z["y"], df["url"].to_numpy()
    X = extract_matrix(df["url"].tolist())
    y = df["label"].to_numpy(dtype=int)
    np.savez_compressed(cache, X=X, y=y, names=np.asarray(PAPER_FEATURE_NAMES))
    return X, y, df["url"].to_numpy()
