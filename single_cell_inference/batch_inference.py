"""
Batch inference for CellFlux: predict 100 samples and filter for single clear cells.

Workflow:
  1. Sample 100 matched control-treated pairs from the dataset
  2. Run CellFlux inference on each pair
  3. Score control, predicted, and ground truth images for single-cell clarity
  4. Save candidates where ALL three images pass the single-cell filter

Usage:
    # On a GPU node (interactive or via SLURM):
    python batch_inference.py --num_samples 100 --output_dir output/batch

    # With custom score threshold:
    python batch_inference.py --num_samples 100 --score_threshold 80

    # Via SLURM:
    sbatch scripts/run_batch_inference.sh
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CELLFLUX_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, CELLFLUX_ROOT)

from inference import CellFluxPredictor, save_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "/lustre/scratch/users/deng.luo/cellflux_data/hf_repo/checkpoints/cellflux/bbbc021/checkpoint.pth"
DEFAULT_EMBEDDING_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/embeddings/csv/emb_fp.csv"
DEFAULT_IMAGE_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/"
DEFAULT_METADATA_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/metadata/bbbc021_df_all.csv"


def sample_key_to_path(sample_key, image_path=DEFAULT_IMAGE_PATH):
    parts = sample_key.split("_")
    return Path(image_path) / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")


def score_single_cell(img):
    """
    Score how likely the image contains a single clear cell.
    Higher score = more likely a single isolated cell.

    Returns -1 for rejected images, positive float for accepted.
    """
    gray = img.mean(axis=2).astype(float)  # (96, 96)

    mean_brightness = gray.mean()
    if mean_brightness < 15 or mean_brightness > 150:
        return -1.0

    border_width = 12
    top = gray[:border_width, :].mean()
    bottom = gray[-border_width:, :].mean()
    left = gray[:, :border_width].mean()
    right = gray[:, -border_width:].mean()
    border_mean = (top + bottom + left + right) / 4.0

    center = gray[30:66, 30:66].mean()

    if border_mean > 50:
        return -1.0

    contrast = center - border_mean
    if contrast < 20:
        return -1.0

    corner_size = 15
    corners = [
        gray[:corner_size, :corner_size].mean(),
        gray[:corner_size, -corner_size:].mean(),
        gray[-corner_size:, :corner_size].mean(),
        gray[-corner_size:, -corner_size:].mean(),
    ]
    max_corner = max(corners)
    if max_corner > 40:
        return -1.0

    threshold = gray.mean() + gray.std() * 1.5
    bright_mask = gray > threshold
    bright_fraction = bright_mask.sum() / (96 * 96)
    if bright_fraction > 0.4 or bright_fraction < 0.03:
        return -1.0

    score = contrast * 2 - border_mean - max_corner + center * 0.5
    return score


def sample_pairs(metadata_path, embedding_path, image_path, num_samples, seed=42):
    """
    Sample matched control-treated pairs where both control and ground truth
    are single clear cells.
    """
    emb = pd.read_csv(embedding_path, index_col=0)
    available_drugs = set(emb.index.tolist())
    df = pd.read_csv(metadata_path, index_col=0)

    treated = df[(df["STATE"] == 1) & (df["CPD_NAME"].isin(available_drugs))]
    ctrl = df[df["STATE"] == 0]

    # Group controls by batch for fast lookup
    ctrl_by_batch = {batch: group for batch, group in ctrl.groupby("BATCH")}

    rng = np.random.RandomState(seed)
    # Shuffle treated cells
    treated_shuffled = treated.sample(frac=1, random_state=rng).reset_index(drop=True)

    pairs = []
    checked = 0
    for _, trt_row in treated_shuffled.iterrows():
        if len(pairs) >= num_samples:
            break

        checked += 1
        if checked % 500 == 0:
            logger.info(f"  Checked {checked} treated cells, found {len(pairs)} valid pairs so far...")

        batch = trt_row["BATCH"]
        if batch not in ctrl_by_batch:
            continue

        # Check treated image
        trt_path = sample_key_to_path(trt_row["SAMPLE_KEY"], image_path)
        if not trt_path.exists():
            continue
        trt_img = np.load(trt_path)
        trt_score = score_single_cell(trt_img)
        if trt_score < 50:
            continue

        # Find a good control in the same batch
        batch_ctrls = ctrl_by_batch[batch].sample(frac=1, random_state=rng)
        best_ctrl = None
        for _, ctrl_row in batch_ctrls.iterrows():
            ctrl_path = sample_key_to_path(ctrl_row["SAMPLE_KEY"], image_path)
            if not ctrl_path.exists():
                continue
            ctrl_img = np.load(ctrl_path)
            ctrl_score = score_single_cell(ctrl_img)
            if ctrl_score > 50:
                best_ctrl = {
                    "row": ctrl_row,
                    "img": ctrl_img,
                    "score": ctrl_score,
                    "path": str(ctrl_path),
                }
                break  # take first good one

        if best_ctrl is None:
            continue

        pairs.append({
            "drug": trt_row["CPD_NAME"],
            "batch": batch,
            "trt_key": trt_row["SAMPLE_KEY"],
            "trt_img": trt_img,
            "trt_score": trt_score,
            "ctrl_key": best_ctrl["row"]["SAMPLE_KEY"],
            "ctrl_img": best_ctrl["img"],
            "ctrl_score": best_ctrl["score"],
        })

    logger.info(f"Found {len(pairs)} valid pairs after checking {checked} treated cells")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="CellFlux Batch Inference")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to predict")
    parser.add_argument("--output_dir", type=str, default="output/batch",
                        help="Output directory for batch results")
    parser.add_argument("--score_threshold", type=float, default=50.0,
                        help="Minimum single-cell score for filtering (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--cfg_scale", type=float, default=0.2)
    parser.add_argument("--noise_level", type=float, default=1.0)
    parser.add_argument("--use_initial", type=int, default=2, choices=[0, 1, 2])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Step 1: Sample pairs (CPU only, no model needed)
    logger.info(f"Step 1: Sampling {args.num_samples} matched control-treated pairs...")
    pairs = sample_pairs(
        args.metadata, args.embedding, args.image_path,
        args.num_samples, args.seed,
    )

    if not pairs:
        logger.error("No valid pairs found!")
        sys.exit(1)

    # Step 2: Load model and run inference
    logger.info(f"Step 2: Running inference on {len(pairs)} pairs...")
    predictor = CellFluxPredictor(
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding,
        cfg_scale=args.cfg_scale,
        noise_level=args.noise_level,
        use_initial=args.use_initial,
    )

    # Step 3: Predict and filter
    candidates = []
    threshold = args.score_threshold

    for i, pair in enumerate(pairs):
        logger.info(f"  [{i+1}/{len(pairs)}] drug={pair['drug']}, batch={pair['batch']}")

        result = predictor.predict(pair["ctrl_img"], pair["drug"])
        pred_img = result["perturbed_image"]
        pred_score = score_single_cell(pred_img)

        all_pass = (pair["ctrl_score"] >= threshold
                    and pair["trt_score"] >= threshold
                    and pred_score >= threshold)

        entry = {
            "index": i,
            "drug": pair["drug"],
            "batch": pair["batch"],
            "ctrl_key": pair["ctrl_key"],
            "trt_key": pair["trt_key"],
            "ctrl_score": round(pair["ctrl_score"], 1),
            "trt_score": round(pair["trt_score"], 1),
            "pred_score": round(pred_score, 1),
            "all_pass": bool(all_pass),
        }

        # Save images for all samples
        tag = f"{i:03d}_{pair['drug'].replace('/', '_')}"
        img_dir = output_dir / "images" / tag
        img_dir.mkdir(parents=True, exist_ok=True)

        np.save(img_dir / "control.npy", pair["ctrl_img"])
        np.save(img_dir / "predicted.npy", pred_img)
        np.save(img_dir / "ground_truth.npy", pair["trt_img"])

        # Save comparison PNG
        save_comparison(
            pair["ctrl_img"], pred_img, pair["drug"],
            str(img_dir / "comparison_gt.png"),
            ground_truth_img=pair["trt_img"],
        )

        candidates.append(entry)

    # Step 4: Summary
    passed = [c for c in candidates if c["all_pass"]]
    failed = [c for c in candidates if not c["all_pass"]]

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH INFERENCE COMPLETE")
    logger.info(f"  Total predicted: {len(candidates)}")
    logger.info(f"  All three pass (ctrl + pred + gt): {len(passed)}")
    logger.info(f"  Rejected (at least one fails): {len(failed)}")
    logger.info(f"  Score threshold: {threshold}")
    logger.info(f"{'='*60}")

    # Save full results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "total": len(candidates),
            "passed": len(passed),
            "threshold": threshold,
            "samples": candidates,
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save candidates list (passed only)
    candidates_path = output_dir / "candidates.json"
    with open(candidates_path, "w") as f:
        json.dump(passed, f, indent=2)
    logger.info(f"Saved {len(passed)} candidates to {candidates_path}")

    # Print passed candidates
    if passed:
        print(f"\n{'='*60}")
        print(f"CANDIDATES ({len(passed)} passed all filters):")
        print(f"{'='*60}")
        for c in passed:
            tag = f"{c['index']:03d}_{c['drug'].replace('/', '_')}"
            print(f"  [{tag}] ctrl={c['ctrl_score']:.0f} pred={c['pred_score']:.0f} gt={c['trt_score']:.0f}")
        print(f"\nImages saved in: {output_dir / 'images'}/")
    else:
        print("\nNo candidates passed all filters. Try lowering --score_threshold.")


if __name__ == "__main__":
    main()
