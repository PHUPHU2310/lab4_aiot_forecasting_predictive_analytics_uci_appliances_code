
from __future__ import annotations

from pathlib import Path
import shutil
import urllib.request
import zipfile
import tempfile

from utils import DATA_DIR

OFFICIAL_PATH = DATA_DIR / "energydata_complete.csv"
SAMPLE_PATH = DATA_DIR / "sample_energydata_complete.csv"

# UCI's static file path can appear with + or %2B depending on client encoding.
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
    "https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip",
]


def download_official_dataset() -> bool:
    if OFFICIAL_PATH.exists():
        print(f"Official UCI dataset already exists: {OFFICIAL_PATH}")
        return True

    DATA_DIR.mkdir(exist_ok=True)
    for url in UCI_ZIP_URLS:
        try:
            print(f"Trying to download UCI Appliances Energy Prediction dataset: {url}")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                zip_path = tmpdir / "appliances_energy_prediction.zip"
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    members = zf.namelist()
                    target = [m for m in members if m.endswith("energydata_complete.csv")]
                    if not target:
                        raise RuntimeError("energydata_complete.csv not found inside UCI zip")
                    zf.extract(target[0], tmpdir)
                    extracted = tmpdir / target[0]
                    shutil.copy2(extracted, OFFICIAL_PATH)
            print(f"Downloaded official UCI dataset to: {OFFICIAL_PATH}")
            return True
        except Exception as exc:
            print(f"Download failed from {url}: {exc}")

    return False


if __name__ == "__main__":
    ok = download_official_dataset()
    if ok:
        print("Dataset ready: official UCI energydata_complete.csv")
    elif SAMPLE_PATH.exists():
        print("No Internet or UCI download failed. Using offline fallback sample for classroom testing.")
        print(f"Fallback sample: {SAMPLE_PATH}")
        print("Note: fallback sample is not the official UCI data. Put energydata_complete.csv in data/ to use the official dataset.")
    else:
        raise FileNotFoundError("No official dataset and no fallback sample found.")
