"""Check (or explain how to get) the BAF dataset, then create the splits.

Base.csv is 213 MB, which is too big for GitHub, so it is not in the repository.
Download it from Kaggle ("Bank Account Fraud Dataset Suite (NeurIPS 2022)",
file Base.csv), e.g. with the Kaggle command line tool:

    kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -f Base.csv -p data/raw --unzip

Then run:

    python get_data.py

It checks that the file is exactly the one I used (SHA-256 from
data/splits/split_metadata.json of my run) and creates the splits with
prepare_data.py if they do not exist yet.
"""

import hashlib
import sys
from pathlib import Path

import prepare_data


EXPECTED_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"


def main():
    path = prepare_data.DATA_PATH
    if not path.exists():
        print(__doc__)
        sys.exit(f"{path} does not exist yet.")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"Base.csv has a different SHA-256 ({digest}). The results will not match the report.")
    print("Base.csv is the expected version.")

    if (prepare_data.OUTPUT_DIR / "split_metadata.json").exists():
        print(f"Splits already exist in {prepare_data.OUTPUT_DIR}.")
    else:
        prepare_data.main()


if __name__ == "__main__":
    main()
