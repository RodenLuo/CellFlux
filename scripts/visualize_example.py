"""Visualize examples with REAL control-treatment pairings from trt2ctrl_idx.json."""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all")
OUTPUT_DIR = Path("/lustre/scratch/users/deng.luo/cellflux_outputs/bbbc021_eval/fid_samples/epoch-100")
SAVE_DIR = Path("/lustre/scratch/users/deng.luo/cellflux_outputs/bbbc021_eval/comparisons")
METADATA = DATA_DIR / "metadata" / "bbbc021_df_all.csv"
PAIR_FILE = Path("/lustre/scratch/users/deng.luo/cellflux_outputs/bbbc021_eval/fid_samples/trt2ctrl_idx.json")

CHANNEL_NAMES = ["Actin (Cytoskeleton)", "Tubulin (Cytoskeleton)", "DAPI (Nucleus)"]
CHANNEL_CMAPS = ["Reds", "Greens", "Blues"]

df = pd.read_csv(METADATA)
with open(PAIR_FILE) as f:
    trt2ctrl = json.load(f)

# Group pairs by drug
drug2pairs = {}
for trt_key, ctrl_key in trt2ctrl.items():
    # Find drug name from generated image directories
    for drug_dir in OUTPUT_DIR.iterdir():
        if not drug_dir.is_dir():
            continue
        gen_file = drug_dir / f"{trt_key}.png"
        if gen_file.exists():
            drug2pairs.setdefault(drug_dir.name, []).append((trt_key, ctrl_key, gen_file))
            break

# Pick drugs from command line or default
if len(sys.argv) > 1:
    drugs = sys.argv[1:]
else:
    available = sorted(drug2pairs.keys())
    drugs = available[:5]

print(f"Visualizing drugs: {drugs}")


def sample_key_to_npy(key):
    parts = key.split("_")
    return DATA_DIR / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")


def load_npy_image(path):
    return np.load(path)


SAVE_DIR.mkdir(parents=True, exist_ok=True)

for drug in drugs:
    if drug not in drug2pairs:
        print(f"  Skipping {drug}: no pairs found")
        continue

    pairs = drug2pairs[drug][:3]  # up to 3 examples per drug

    for trt_key, ctrl_key, gen_file in pairs:
        gt_npy = sample_key_to_npy(trt_key)
        ctrl_npy = sample_key_to_npy(ctrl_key)

        if not gt_npy.exists() or not ctrl_npy.exists():
            print(f"  Skipping {trt_key}: npy not found")
            continue

        ctrl_img = load_npy_image(ctrl_npy)
        gen_img = np.array(Image.open(gen_file))
        gt_img = load_npy_image(gt_npy)

        # 4 rows (composite, ch0, ch1, ch2) x 3 cols (ctrl, output, gt)
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))

        row_labels = ["Composite", CHANNEL_NAMES[0], CHANNEL_NAMES[1], CHANNEL_NAMES[2]]
        col_labels = [
            f"Input (Control)\n{ctrl_key}",
            f"Model Output\nDrug: {drug}",
            f"Ground Truth\n{trt_key}",
        ]

        images = [ctrl_img, gen_img, gt_img]

        for col in range(3):
            img = images[col]
            axes[0, col].imshow(img)
            axes[0, col].axis("off")
            for ch in range(3):
                axes[ch + 1, col].imshow(img[:, :, ch], cmap=CHANNEL_CMAPS[ch], vmin=0, vmax=255)
                axes[ch + 1, col].axis("off")

        for col in range(3):
            axes[0, col].set_title(col_labels[col], fontsize=11, fontweight="bold")

        for row in range(4):
            axes[row, 0].annotate(
                row_labels[row], xy=(-0.15, 0.5), xycoords="axes fraction",
                fontsize=11, fontweight="bold", ha="right", va="center", rotation=90,
            )

        fig.suptitle(f"Drug: {drug}  |  {ctrl_key} → {trt_key}", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0.03, 0, 1, 0.96])

        save_path = SAVE_DIR / f"comparison_{drug}_{trt_key}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")

print("Done.")
