"""
CellFlux Single-Cell Inference Script

Usage:
    # Basic: given a control cell image (.npy) and a drug name
    python inference.py \
        --cell_image /path/to/control_cell.npy \
        --drug_name "taxol" \
        --output output_perturbed.png

    # Use a random control from the dataset for a specific drug
    python inference.py \
        --drug_name "taxol" \
        --output output_perturbed.png

    # API mode: starts a Flask server
    python inference.py --api --port 5000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Add CellFlux project root to path (for models, training modules)
CELLFLUX_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, CELLFLUX_ROOT)

from models.model_configs import instantiate_model
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from training.edm_time_discretization import get_time_discretization

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ── Default paths (BBBC021 dataset) ──────────────────────────────────────────
DEFAULT_CHECKPOINT = "/lustre/scratch/users/deng.luo/cellflux_data/hf_repo/checkpoints/cellflux/bbbc021/checkpoint.pth"
DEFAULT_EMBEDDING_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/embeddings/csv/emb_fp.csv"
DEFAULT_IMAGE_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/"
DEFAULT_METADATA_PATH = "/lustre/scratch/users/deng.luo/cellflux_data/bbbc021_all/metadata/bbbc021_df_all.csv"


class CFGScaledModel(ModelWrapper):
    """Classifier-free guidance wrapper for the UNet model."""

    def __init__(self, model):
        super().__init__(model)

    def forward(self, x, t, cfg_scale, extra):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra=extra)
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra=extra)
        return result.to(dtype=torch.float32)


class CellFluxPredictor:
    """
    Encapsulates CellFlux model loading and inference.

    Input requirements:
        - cell_image: numpy array, shape (96, 96, 3), dtype uint8, range [0, 255]
                      This is the control (unperturbed) cell image.
        - drug_name:  string, one of the 35 drug names in the embedding table.
                      e.g. "taxol", "cytochalasin B", "MG-132", etc.

    Output:
        - perturbed cell image: numpy array, shape (96, 96, 3), dtype uint8, range [0, 255]
    """

    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT,
        embedding_path=DEFAULT_EMBEDDING_PATH,
        device=None,
        cfg_scale=0.2,
        noise_level=1.0,
        use_initial=2,
        ode_method="heun2",
        ode_nfe=50,
        edm_schedule=True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg_scale = cfg_scale
        self.noise_level = noise_level
        self.use_initial = use_initial
        self.ode_method = ode_method
        self.ode_nfe = ode_nfe
        self.edm_schedule = edm_schedule

        # Load drug embeddings
        logger.info(f"Loading drug embeddings from {embedding_path}")
        emb_df = pd.read_csv(embedding_path, index_col=0)
        self.drug_names = sorted(emb_df.index.tolist())
        self.drug2emb = {}
        for name in emb_df.index:
            self.drug2emb[name] = torch.tensor(
                emb_df.loc[name].values, dtype=torch.float32
            ).to(self.device)
        self.emb_dim = emb_df.shape[1]  # 1024 for BBBC021
        logger.info(f"Loaded {len(self.drug_names)} drug embeddings (dim={self.emb_dim})")
        logger.info(f"Available drugs: {self.drug_names}")

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        model = instantiate_model(
            architechture="bbbc021",
            is_discrete=False,
            use_ema=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()

        # Wrap for CFG
        self.cfg_model = CFGScaledModel(model=model)
        self.cfg_model.train(False)
        self.solver = ODESolver(velocity_model=self.cfg_model)
        logger.info("Model loaded and ready for inference.")

    def preprocess_image(self, img_np):
        """
        Preprocess a cell image for model input.

        Args:
            img_np: numpy array, shape (96, 96, 3), dtype uint8, range [0, 255]

        Returns:
            torch.Tensor, shape (1, 3, 96, 96), range [-1, 1]
        """
        img = torch.from_numpy(img_np).float()  # (H, W, C)
        img = img.permute(2, 0, 1)  # (C, H, W)
        # Normalize to [-1, 1]: (x + random_noise) / 255 then * 2 - 1
        # For inference we skip random noise dithering
        img = img / 255.0
        img = img * 2.0 - 1.0  # to [-1, 1]
        return img.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def postprocess_image(self, tensor):
        """
        Convert model output tensor back to numpy image.

        Args:
            tensor: torch.Tensor, shape (1, 3, 96, 96), range [-1, 1]

        Returns:
            numpy array, shape (96, 96, 3), dtype uint8, range [0, 255]
        """
        img = torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)  # to [0, 1]
        img = (img * 255.0).to(torch.uint8)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        return img

    def predict(self, cell_image, drug_name):
        """
        Given a control cell image and a drug name, predict the perturbed cell image.

        Args:
            cell_image: numpy array, shape (96, 96, 3), dtype uint8, range [0, 255]
            drug_name: string, drug name (must be in embedding table)

        Returns:
            dict with keys:
                - "perturbed_image": numpy array (96, 96, 3), uint8
                - "drug_name": the drug name used
                - "drug_embedding": the 1024-dim embedding vector (list)
        """
        # Validate drug name
        if drug_name not in self.drug2emb:
            raise ValueError(
                f"Unknown drug '{drug_name}'. Available drugs: {self.drug_names}"
            )

        # Validate image
        assert cell_image.shape == (96, 96, 3), \
            f"Expected image shape (96, 96, 3), got {cell_image.shape}"
        assert cell_image.dtype == np.uint8, \
            f"Expected uint8 dtype, got {cell_image.dtype}"

        # Preprocess
        x_ctrl = self.preprocess_image(cell_image)  # (1, 3, 96, 96)
        z_emb = self.drug2emb[drug_name].unsqueeze(0)  # (1, 1024)

        # Prepare initial state
        if self.use_initial == 1:
            x_0 = x_ctrl
        elif self.use_initial == 2:
            noise = torch.randn_like(x_ctrl) * self.noise_level
            x_0 = x_ctrl + noise
        else:
            x_0 = torch.randn_like(x_ctrl)

        # Time grid
        if self.edm_schedule:
            time_grid = get_time_discretization(nfes=self.ode_nfe)
        else:
            time_grid = torch.tensor([0.0, 1.0], device=self.device)

        # Run ODE solver
        with torch.no_grad():
            synthetic = self.solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=self.ode_method,
                return_intermediates=False,
                step_size=None,
                cfg_scale=self.cfg_scale,
                extra={"concat_conditioning": z_emb},
            )

        # Postprocess
        perturbed_img = self.postprocess_image(synthetic)

        return {
            "perturbed_image": perturbed_img,
            "drug_name": drug_name,
            "drug_embedding": self.drug2emb[drug_name].cpu().tolist(),
        }


def get_random_control_image(metadata_path=DEFAULT_METADATA_PATH, image_path=DEFAULT_IMAGE_PATH):
    """Pick a random control cell image from the dataset."""
    df = pd.read_csv(metadata_path, index_col=0)
    ctrl = df[df["STATE"] == 0]
    row = ctrl.sample(1).iloc[0]

    sample_key = row["SAMPLE_KEY"]
    parts = sample_key.split("_")
    # BBBC021 path: image_path / Week_Plate / field / rest.npy
    npy_path = Path(image_path) / parts[0] / parts[1] / ("_".join(parts[2:]) + ".npy")
    img = np.load(npy_path)
    logger.info(f"Loaded control image: {npy_path} (batch={row['BATCH']})")
    return img, str(npy_path)


def save_comparison(ctrl_img, perturbed_img, drug_name, output_path, ground_truth_img=None):
    """
    Save a side-by-side comparison image.

    If ground_truth_img is None:  control | predicted  (2 panels)
    If ground_truth_img is given: control | predicted | ground truth  (3 panels)
    """
    h, w, c = ctrl_img.shape
    gap = 10
    n_panels = 3 if ground_truth_img is not None else 2
    canvas_w = w * n_panels + gap * (n_panels - 1)
    canvas = np.ones((h + 30, canvas_w, c), dtype=np.uint8) * 255

    canvas[30:30 + h, :w, :] = ctrl_img
    canvas[30:30 + h, w + gap:w * 2 + gap, :] = perturbed_img
    if ground_truth_img is not None:
        canvas[30:30 + h, w * 2 + gap * 2:, :] = ground_truth_img

    img = Image.fromarray(canvas)
    img.save(output_path)
    logger.info(f"Saved comparison to {output_path}")


def run_cli(args):
    """Run single prediction from command line."""
    predictor = CellFluxPredictor(
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding,
        cfg_scale=args.cfg_scale,
        noise_level=args.noise_level,
        use_initial=args.use_initial,
    )

    # Load cell image
    if args.cell_image:
        cell_img = np.load(args.cell_image)
        logger.info(f"Loaded cell image: {args.cell_image}, shape={cell_img.shape}")
    else:
        cell_img, path = get_random_control_image(args.metadata, args.image_path)
        logger.info(f"Using random control cell: {path}")

    # Run prediction
    result = predictor.predict(cell_img, args.drug_name)

    # Load ground truth if provided
    gt_img = None
    if args.ground_truth:
        gt_img = np.load(args.ground_truth)
        logger.info(f"Loaded ground truth image: {args.ground_truth}, shape={gt_img.shape}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    perturbed = result["perturbed_image"]
    Image.fromarray(perturbed).save(output_path)
    logger.info(f"Saved perturbed image to {output_path}")

    # Save comparison without ground truth (control | predicted)
    comparison_path = output_path.with_name(output_path.stem + "_comparison.png")
    save_comparison(cell_img, perturbed, args.drug_name, str(comparison_path))

    # Save comparison with ground truth (control | predicted | ground truth)
    if gt_img is not None:
        comparison_gt_path = output_path.with_name(output_path.stem + "_comparison_gt.png")
        save_comparison(cell_img, perturbed, args.drug_name, str(comparison_gt_path), ground_truth_img=gt_img)

    # Save raw npy
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, perturbed)
    logger.info(f"Saved raw npy to {npy_path}")

    print(f"\nDone! Drug: {args.drug_name}")
    print(f"  Output image:  {output_path}")
    print(f"  Comparison:    {comparison_path}")
    if gt_img is not None:
        print(f"  Comparison GT: {comparison_gt_path}")
    print(f"  Raw npy:       {npy_path}")


def run_api(args):
    """Run as a Flask API server."""
    try:
        from flask import Flask, request, jsonify, send_file
    except ImportError:
        print("Flask is required for API mode. Install with: pip install flask")
        sys.exit(1)

    import io
    import base64

    app = Flask(__name__)

    logger.info("Loading model for API mode...")
    predictor = CellFluxPredictor(
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding,
        cfg_scale=args.cfg_scale,
        noise_level=args.noise_level,
        use_initial=args.use_initial,
    )

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        POST /predict

        Request JSON body:
        {
            "drug_name": "taxol",
            "cell_image_base64": "<base64-encoded .npy bytes>",  // optional
            "cell_image_path": "/path/to/cell.npy",              // optional
            "use_random_control": true                            // optional, default false
        }

        Response JSON:
        {
            "drug_name": "taxol",
            "perturbed_image_base64": "<base64-encoded PNG>",
            "image_shape": [96, 96, 3]
        }
        """
        data = request.get_json()
        if not data or "drug_name" not in data:
            return jsonify({"error": "drug_name is required"}), 400

        drug_name = data["drug_name"]

        # Get cell image
        try:
            if "cell_image_base64" in data:
                npy_bytes = base64.b64decode(data["cell_image_base64"])
                cell_img = np.load(io.BytesIO(npy_bytes))
            elif "cell_image_path" in data:
                cell_img = np.load(data["cell_image_path"])
            elif data.get("use_random_control", False):
                cell_img, _ = get_random_control_image(args.metadata, args.image_path)
            else:
                return jsonify({
                    "error": "Provide cell_image_base64, cell_image_path, or set use_random_control=true"
                }), 400

            result = predictor.predict(cell_img, drug_name)

            # Encode output as base64 PNG
            img = Image.fromarray(result["perturbed_image"])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Also encode control image
            ctrl_img = Image.fromarray(cell_img)
            ctrl_buf = io.BytesIO()
            ctrl_img.save(ctrl_buf, format="PNG")
            ctrl_b64 = base64.b64encode(ctrl_buf.getvalue()).decode("utf-8")

            return jsonify({
                "drug_name": drug_name,
                "perturbed_image_base64": img_b64,
                "control_image_base64": ctrl_b64,
                "image_shape": list(result["perturbed_image"].shape),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Internal error: {str(e)}"}), 500

    @app.route("/drugs", methods=["GET"])
    def list_drugs():
        """GET /drugs - List all available drug names."""
        return jsonify({"drugs": predictor.drug_names})

    @app.route("/health", methods=["GET"])
    def health():
        """GET /health - Health check."""
        return jsonify({"status": "ok", "device": str(predictor.device)})

    logger.info(f"Starting API server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(description="CellFlux Single-Cell Inference")

    # Mode
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--port", type=int, default=5000, help="API server port")

    # Input
    parser.add_argument("--cell_image", type=str, default=None,
                        help="Path to control cell image (.npy, shape 96x96x3, uint8)")
    parser.add_argument("--drug_name", type=str, default=None,
                        help="Drug/compound name (e.g. 'taxol', 'MG-132')")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to ground truth treated cell image (.npy) for comparison")

    # Output
    parser.add_argument("--output", type=str, default="output/output_perturbed.png",
                        help="Output image path")

    # Model
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING_PATH,
                        help="Path to drug embedding CSV")

    # Data (for random control)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_PATH,
                        help="Path to metadata CSV")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to image directory")

    # Generation params
    parser.add_argument("--cfg_scale", type=float, default=0.2,
                        help="Classifier-free guidance scale")
    parser.add_argument("--noise_level", type=float, default=1.0,
                        help="Noise level for control image")
    parser.add_argument("--use_initial", type=int, default=2, choices=[0, 1, 2],
                        help="0=random noise, 1=control image, 2=control+noise")

    args = parser.parse_args()

    if args.api:
        run_api(args)
    else:
        if args.drug_name is None:
            parser.error("--drug_name is required in CLI mode")
        run_cli(args)


if __name__ == "__main__":
    main()
