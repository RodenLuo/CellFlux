# CellFlux Single-Cell Inference

Predict drug-perturbed cell images from control (unperturbed) cell images using CellFlux.

**Input:** a 96x96x3 uint8 `.npy` control cell image + a drug name
**Output:** a 96x96x3 uint8 predicted perturbed cell image

## Quick Start (SLURM CLI)

```bash
cd /home/deng.luo/projects/CellFlux/single_cell_inference

# Run with example input (lactacystin, with ground truth comparison)
sbatch scripts/run_inference.sh \
    --drug_name lactacystin \
    --cell_image example_input/control_cell.npy \
    --ground_truth example_input/ground_truth.npy \
    --output output/output_perturbed.png
```

This produces 4 output files:

| File | Description |
|---|---|
| `output/output_perturbed.png` | Predicted perturbed cell image |
| `output/output_perturbed.npy` | Raw numpy array of the prediction |
| `output/output_perturbed_comparison.png` | Side-by-side: control \| predicted |
| `output/output_perturbed_comparison_gt.png` | Side-by-side: control \| predicted \| ground truth |

Check job status and output:

```bash
squeue -u $USER
cat logs/inference_<JOBID>.out
cat logs/inference_<JOBID>.err
```

More examples:

```bash
# Without ground truth (produces 3 files, no _comparison_gt.png)
sbatch scripts/run_inference.sh \
    --drug_name taxol \
    --cell_image example_input/control_cell.npy \
    --output output/output_perturbed.png

# Use a random control cell from the dataset
sbatch scripts/run_inference.sh \
    --drug_name taxol \
    --output output/output_perturbed.png
```

## Batch Inference

Sample N matched control-treated pairs, run predictions, and filter for single clear cells:

```bash
sbatch scripts/run_batch_inference.sh --num_samples 100 --output_dir output/batch

# With custom score threshold
sbatch scripts/run_batch_inference.sh --num_samples 200 --score_threshold 80
```

Results are saved to `output/batch/results.json` and images under `output/batch/images/`.

## API Server

### 1. Start the server on a compute node

```bash
cd /home/deng.luo/projects/CellFlux/single_cell_inference

# Start API server (allocates 4 hours by default)
sbatch --time=04:00:00 scripts/run_inference.sh --api --port 5000
```

### 2. Find which compute node it landed on

```bash
# Check job status
squeue -u $USER

# Example output:
#   JOBID PARTITION  NAME      USER  ST  TIME  NODES  NODELIST
#   12345 gpumid     cellflux  deng  R   0:30  1      gpumid-07
```

Note the node name (e.g. `gpumid-07`). Confirm the server is ready:

```bash
cat logs/inference_<JOBID>.out
# Look for: "Starting API server on port 5000"
```

### 3. SSH tunnel from your local machine

Set up a two-hop tunnel: **local** -> **login node** -> **compute node**.

```bash
# Run this on your LOCAL machine (replace gpumid-07 with the actual node):
ssh -N -L 5000:gpumid-07:5000 deng.luo@cscc-login-2
```

Now `http://localhost:5000` on your local machine forwards to the API server.

### 4. Use the API

#### Health check

```bash
curl http://localhost:5000/health
# {"device":"cuda","status":"ok"}
```

#### List available drugs

```bash
curl http://localhost:5000/drugs
# {"drugs":["ALLN","AZ138",...,"vincristine"]}
```

#### Predict (with a local .npy file)

```bash
# Encode a .npy file as base64 and send it
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"drug_name\": \"taxol\",
    \"cell_image_base64\": \"$(base64 -w0 example_input/control_cell.npy)\"
  }"
```

#### Predict (with a server-side file path)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "taxol",
    "cell_image_path": "/home/deng.luo/projects/CellFlux/single_cell_inference/example_input/control_cell.npy"
  }'
```

#### Predict (random control from dataset)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "taxol",
    "use_random_control": true
  }'
```

#### Response format

```json
{
  "drug_name": "taxol",
  "perturbed_image_base64": "<base64-encoded PNG>",
  "control_image_base64": "<base64-encoded PNG>",
  "image_shape": [96, 96, 3]
}
```

To decode and save the output image (Python):

```python
import base64
import json

resp = json.loads(response_text)
with open("perturbed.png", "wb") as f:
    f.write(base64.b64decode(resp["perturbed_image_base64"]))
```

### 5. Stop the server

```bash
scancel <JOBID>
```

## Python API (In-Process)

Use `CellFluxPredictor` directly in your own scripts (must run on a GPU node):

```python
import numpy as np
from single_cell_inference.inference import CellFluxPredictor

predictor = CellFluxPredictor()

cell_img = np.load("example_input/control_cell.npy")  # (96, 96, 3) uint8
result = predictor.predict(cell_img, "taxol")

perturbed = result["perturbed_image"]  # (96, 96, 3) uint8
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--cell_image` | None | Path to control cell `.npy` (96x96x3 uint8). If omitted, picks a random control from the dataset. |
| `--drug_name` | (required in CLI) | Drug name, see list below |
| `--ground_truth` | None | Optional ground truth `.npy` for comparison |
| `--output` | `output/output_perturbed.png` | Output image path |
| `--checkpoint` | `...checkpoint.pth` | Model checkpoint path |
| `--embedding` | `...emb_fp.csv` | Drug embedding CSV path |
| `--cfg_scale` | 0.2 | Classifier-free guidance scale |
| `--noise_level` | 1.0 | Noise level added to control image |
| `--use_initial` | 2 | Initial state: 0=random noise, 1=control only, 2=control+noise |
| `--api` | false | Run as Flask API server |
| `--port` | 5000 | API server port |

## Available Drugs (35)

```
ALLN              AZ138             AZ258             AZ841
DMSO              MG-132            PD-169316         PP-2
alsterpaullone    anisomycin        bryostatin        camptothecin
chlorambucil      cisplatin         colchicine        cyclohexamide
cytochalasin B    cytochalasin D    demecolcine       docetaxel
emetine           epothilone B      etoposide         floxuridine
lactacystin       latrunculin B     methotrexate      mevinolin/lovastatin
mitomycin C       mitoxantrone      nocodazole        proteasome inhibitor I
simvastatin       taxol             vincristine
```

## Example Input

The `example_input/` folder contains a pre-selected pair:

- `control_cell.npy` — control cell from batch Week6_31661
- `ground_truth.npy` — same cell treated with **lactacystin**
- `drug_info.json` — metadata for this pair

To regenerate example input: `python create_example_input.py`
