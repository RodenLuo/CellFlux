"""
Create an example input for CellFlux inference.
Picks a treated cell as ground truth and a batch-matched control cell from the BBBC021 dataset.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

METADATA_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/metadata/bbbc021_df_all.csv"
EMBEDDING_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/embeddings/csv/emb_fp.csv"
IMAGE_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/"
OUTPUT_DIR = Path(__file__).parent / "example_input"


def sample_key_to_path(sample_key):
    """Convert SAMPLE_KEY to .npy file path."""
    parts = sample_key.split("_")
    return Path(IMAGE_PATH) / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load drug embeddings to get available drug list
    emb = pd.read_csv(EMBEDDING_PATH, index_col=0)
    available_drugs = set(emb.index.tolist())

    # Read metadata
    df = pd.read_csv(METADATA_PATH, index_col=0)
    treated = df[df["STATE"] == 1]
    ctrl = df[df["STATE"] == 0]

    # Filter treated cells to only those with available drug embeddings
    treated_with_emb = treated[treated["CPD_NAME"].isin(available_drugs)]

    # Pick a random treated cell as ground truth
    trt_row = treated_with_emb.sample(1, random_state=123).iloc[0]
    drug_name = trt_row["CPD_NAME"]
    trt_batch = trt_row["BATCH"]

    # Find a batch-matched control cell
    ctrl_same_batch = ctrl[ctrl["BATCH"] == trt_batch]
    if len(ctrl_same_batch) == 0:
        # Fallback: pick any control cell
        ctrl_row = ctrl.sample(1, random_state=123).iloc[0]
        print(f"WARNING: No control in batch {trt_batch}, using control from batch {ctrl_row['BATCH']}")
    else:
        ctrl_row = ctrl_same_batch.sample(1, random_state=123).iloc[0]

    # Load images
    trt_path = sample_key_to_path(trt_row["SAMPLE_KEY"])
    ctrl_path = sample_key_to_path(ctrl_row["SAMPLE_KEY"])
    trt_img = np.load(trt_path)
    ctrl_img = np.load(ctrl_path)

    # Save control cell
    ctrl_out = OUTPUT_DIR / "control_cell.npy"
    np.save(ctrl_out, ctrl_img)

    # Save ground truth (treated cell)
    gt_out = OUTPUT_DIR / "ground_truth.npy"
    np.save(gt_out, trt_img)

    # Save drug info
    drug_info = {
        "drug_name": drug_name,
        "dose": float(trt_row["DOSE"]),
        "annotation": trt_row["ANNOT"],
        "treated_batch": trt_batch,
        "control_batch": ctrl_row["BATCH"],
        "treated_sample_key": trt_row["SAMPLE_KEY"],
        "control_sample_key": ctrl_row["SAMPLE_KEY"],
    }
    info_out = OUTPUT_DIR / "drug_info.json"
    with open(info_out, "w") as f:
        json.dump(drug_info, f, indent=2)

    print(f"Saved example control cell to: {ctrl_out}")
    print(f"  Source: {ctrl_path}")
    print(f"  Batch: {ctrl_row['BATCH']}")
    print(f"  Shape: {ctrl_img.shape}, dtype: {ctrl_img.dtype}")
    print(f"  Value range: [{ctrl_img.min()}, {ctrl_img.max()}]")
    print()
    print(f"Saved ground truth (treated cell) to: {gt_out}")
    print(f"  Source: {trt_path}")
    print(f"  Drug: {drug_name} (dose={trt_row['DOSE']})")
    print(f"  Annotation: {trt_row['ANNOT']}")
    print(f"  Batch: {trt_batch}")
    print(f"  Shape: {trt_img.shape}, dtype: {trt_img.dtype}")
    print(f"  Value range: [{trt_img.min()}, {trt_img.max()}]")
    print()
    print(f"Saved drug info to: {info_out}")
    print()

    # List available drugs
    drugs = sorted(emb.index.tolist())
    print("Available drugs for prediction:")
    for i, drug in enumerate(drugs):
        marker = " <-- selected" if drug == drug_name else ""
        print(f"  {i+1:2d}. {drug}{marker}")
    print()

    # Print example commands
    print("=" * 60)
    print("Example inference commands:")
    print("=" * 60)
    print()
    print(f"# 1. Predict with ground truth comparison:")
    print(f"python inference.py \\")
    print(f"    --cell_image {ctrl_out} \\")
    print(f"    --drug_name '{drug_name}' \\")
    print(f"    --ground_truth {gt_out} \\")
    print(f"    --output output/output_perturbed.png")
    print()
    print(f"# 2. Predict without ground truth:")
    print(f"python inference.py \\")
    print(f"    --cell_image {ctrl_out} \\")
    print(f"    --drug_name '{drug_name}' \\")
    print(f"    --output output/output_perturbed.png")
    print()
    print("# 3. Start as API server:")
    print(f"python inference.py --api --port 5000")


if __name__ == "__main__":
    main()
