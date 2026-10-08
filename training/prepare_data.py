"""Create the initial datasets and a separate pool for the monthly simulation.

"""

import hashlib
import json
from pathlib import Path

import pandas as pd


# Paths and variables
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Base.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "splits"
SIMULATION_MONTHS = 12
ROWS_PER_MONTH = 30_000

SPLIT_SIZES = {
    "train": 400_000,
    "val": 100_000,
    "test": 100_000,
    "simulation_pool": SIMULATION_MONTHS * ROWS_PER_MONTH,
    "reserve": 40_000,
}

def create_splits(data):
    """Take consecutive blocks of rows in the order they appear in Base.csv - for this experiment we assume the order in which applications come is equal to the order of the dataset."""
    splits = {}
    start = 0

    # Each split starts where the previous one ended. The month and fraud label
    # do not affect which rows are selected, and the original order stays intact.
    for name, n_rows in SPLIT_SIZES.items():
        stop = start + n_rows
        splits[name] = data.iloc[start:stop].copy()
        start = stop

    return splits


def check_splits(data, splits):
    """Check the sizes and make sure the splits reproduce the original row order. This shoudlnt be neccesarry but is a precaution"""
    for name, split in splits.items():
        if len(split) != SPLIT_SIZES[name]:
            raise ValueError(f"Unexpected number of rows in {name}: {len(split):,}.")

    source_ids = pd.concat(
        [split["source_id"] for split in splits.values()], ignore_index=True
    )
    if not source_ids.equals(data["source_id"].reset_index(drop=True)):
        raise ValueError("The splits contain repeated, missing or reordered source rows.")


def describe_split(data):
    """Collect the counts and fraud rates for the metadata file. This is useful as the original dataset doesnt follow a linear fraud distribution but has temporal differences"""
    # I record the months for reference, even though they are not used to split.
    by_month = data.groupby("month")["fraud_bool"].agg(
        rows="size", fraud_cases="sum", fraud_rate="mean"
    ).reset_index()
    return {
        "rows": len(data),
        "first_source_id": int(data["source_id"].iloc[0]),
        "last_source_id": int(data["source_id"].iloc[-1]),
        "fraud_cases": int(data["fraud_bool"].sum()),
        "fraud_rate": float(data["fraud_bool"].mean()),
        "by_month": by_month.to_dict(orient="records"),
    }


def main():
    # This records which version of Base.csv was used without copying it again.
    with DATA_PATH.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()

    data = pd.read_csv(DATA_PATH, float_precision="round_trip")
    if not {"month", "fraud_bool"}.issubset(data.columns):
        raise ValueError("The source must contain month and fraud_bool columns.")
    if len(data) != sum(SPLIT_SIZES.values()):
        raise ValueError("These split sizes expect the original dataset with 1,000,000 rows.")
    if not data["fraud_bool"].isin([0, 1]).all():
        raise ValueError("Every fraud_bool value must be either 0 or 1.")
    if "source_id" in data.columns:
        raise ValueError("Use the original Base.csv, which has no source_id column.")

    # The zero-based row number lets me trace an application back to Base.csv.
    # It is metadata and should be left out of the model's input features.
    data.insert(0, "source_id", data.index)
    splits = create_splits(data)
    check_splits(data, splits)

    metadata = {
        "source_file": DATA_PATH.relative_to(PROJECT_DIR).as_posix(),
        "source_sha256": source_hash,
        "source_rows": len(data),
        "library_versions": {"pandas": pd.__version__},
        "source_id_definition": "Zero-based data row in the source CSV, excluding the header.",
        "split_method": "Consecutive row blocks in the original CSV order, without shuffling or stratification.",
        "planned_simulation": {
            "months": SIMULATION_MONTHS,
            "rows_per_month": ROWS_PER_MONTH,
            "label_delay_months": 1,
            "note": "The pool is for artificial monthly batches. Drift and label release are added later.",
        },
        "splits": {},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        filename = f"{name}.csv"
        # Feature values and labels stay as they are. Preprocessing comes later.
        split.to_csv(OUTPUT_DIR / filename, index=False, lineterminator="\n")
        summary = describe_split(split)
        metadata["splits"][name] = {
            "file": filename,
            "selection_rule": (
                f"Source IDs {summary['first_source_id']} to "
                f"{summary['last_source_id']}, inclusive."
            ),
            **summary,
        }
        print(f"{name:16} {len(split):>9,} rows | fraud rate: {summary['fraud_rate']:.4%}")

    metadata_path = OUTPUT_DIR / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved the datasets and metadata to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
