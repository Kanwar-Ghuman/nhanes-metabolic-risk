"""
Step 1: download the NHANES 2017–March 2020 pre-pandemic files from CDC and
merge them into one participant-level table keyed on SEQN.

    python scripts/01_download.py [--force]
"""
import argparse
import time

import pandas as pd
import requests

from nhanes import COMPONENTS, MERGED_CSV, RAW

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"


def fetch(name, force=False):
    dest = RAW / f"{name}.xpt"
    if dest.exists() and not force:
        return dest

    url = BASE_URL.format(name)
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException as err:
            if attempt == 3:
                raise
            print(f"  {name}: {err} -- retrying")
            time.sleep(5 * attempt)

    # Bad paths on the CDC site return an HTML page with status 200
    if not resp.content.startswith(XPT_MAGIC):
        raise RuntimeError(f"{url} did not return a SAS transport file")

    dest.write_bytes(resp.content)
    return dest


def read_xpt(path, columns):
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing expected columns: {sorted(missing)}")
    df["SEQN"] = df["SEQN"].astype(int)
    return df[["SEQN", *columns]]


def main(force=False):
    tables = {}
    for name, cols in COMPONENTS.items():
        path = fetch(name, force)
        tables[name] = read_xpt(path, cols)
        print(f"{name:<9} {len(tables[name]):>6,} rows  ({path.stat().st_size / 1e6:.1f} MB)")

    merged = tables.pop("P_DEMO")
    for name, df in tables.items():
        merged = merged.merge(df, on="SEQN", how="left", validate="one_to_one")

    merged.to_csv(MERGED_CSV, index=False)
    print(f"\nMerged: {merged.shape[0]:,} participants x {merged.shape[1]} columns -> {MERGED_CSV.relative_to(MERGED_CSV.parents[2])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download files already in data/raw")
    main(parser.parse_args().force)
