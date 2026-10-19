"""Prepare data/creditcard.csv, the Kaggle 'Credit Card Fraud Detection' dataset.

Dataset: transactions by European cardholders over two days in September 2013, collected by
Worldline and the Machine Learning Group of ULB (Universite Libre de Bruxelles). 284,807
rows, 492 frauds (about 0.17%), columns Time, V1..V28 (PCA-anonymised), Amount, Class.
Size: roughly 150 MB uncompressed.
Licence: not stated here. Check the licence and citation requirements on the dataset page
on Kaggle before redistributing or publishing results.

Kaggle blocks anonymous downloads, so this script does not fetch anything by itself.
Two ways to get the file:

1. Local copy. Download the CSV in your browser from the dataset page (search Kaggle for
   'Credit Card Fraud Detection' by the Machine Learning Group - ULB) and run
       python scripts/download_data.py --csv path/to/creditcard.csv
   The file is schema-checked and copied to data/creditcard.csv.

2. Kaggle API. Put your API credentials in ~/.kaggle/kaggle.json (Kaggle account settings,
   'Create New Token') and run
       pip install kaggle
       kaggle datasets download -d <owner>/<dataset-slug> -p data --unzip
   Copy <owner>/<dataset-slug> from the dataset page URL; it is not hard-coded here because
   it was not verified from the build environment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fraud_anomaly.data import SchemaError, read_transactions

DEFAULT_OUT = "data/creditcard.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", help="path to a local creditcard.csv to validate and copy")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    out = Path(args.out)
    if not args.csv:
        if out.is_file():
            print(f"{out} already exists. Nothing to do.")
            return 0
        print(
            "No --csv given and no data found.\n\n" + __doc__.split("Two ways to get the file:")[1],
            file=sys.stderr,
        )
        return 1

    src = Path(args.csv)
    try:
        frame = read_transactions(src, require_label=True)
    except (FileNotFoundError, SchemaError) as exc:
        print(f"Cannot use {src}: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out.resolve():
        shutil.copyfile(src, out)
    print(
        f"{out}: {len(frame):,} rows, {int(frame['Class'].sum())} frauds "
        f"({frame['Class'].mean():.4%}). Run `make train` next."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
