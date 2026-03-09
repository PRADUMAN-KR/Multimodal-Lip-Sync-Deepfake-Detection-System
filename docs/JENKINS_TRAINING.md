# Running training from Jenkins

Training can be run from Jenkins using the pipeline and script below. Paths and hyperparameters are driven by **environment variables** so the same command works in the container without hardcoding paths.

## 1. Run script directly (e.g. inside your container)

From the **repository root** (e.g. `$WORKSPACE` in Jenkins):

```bash
chmod +x scripts/run_finetune_jenkins.sh
./scripts/run_finetune_jenkins.sh
```

Optional overrides via environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data/AVLips1 2` | Path to data (must contain `0_real/`, `1_fake/` or equivalent) |
| `PRETRAINED` | `weights/best_model.pth` | Pretrained checkpoint |
| `OUTPUT_DIR` | `weights` | Where to save checkpoints |
| `EPOCHS` | `36` | Total epochs |
| `FREEZE_EPOCHS` | `8` | Epochs with frozen encoders |
| `BATCH_SIZE` | `8` | Batch size |
| `LR` | `2e-4` | Learning rate (new layers) |
| `LR_ENCODER` | `2e-5` | Learning rate (encoders when unfrozen) |
| `LOG_EVERY` | `5` | Log every N batches (0 = epoch summaries only) |
| `DEVICE` | *(auto)* | `cuda`, `mps`, or `cpu` (empty = auto-detect) |

Example with custom paths in the container:

```bash
export DATA_DIR=/mnt/training-data/avlips
export PRETRAINED=/mnt/weights/backbone.pth
export OUTPUT_DIR=/mnt/weights/run_001
./scripts/run_finetune_jenkins.sh
```

## 2. Jenkins pipeline (`Jenkinsfile`)

The repo includes a `Jenkinsfile` that:

- Runs the same script from the Jenkins workspace
- Exposes parameters (data dir, pretrained path, output dir, epochs, batch size, etc.)
- Archives `weights/*.pth` on success

**Setup:**

1. Create a **Pipeline** job in Jenkins.
2. Set "Pipeline script from SCM" and point to this repo; branch as needed.
3. Ensure the agent (or container) has:
   - Python 3 with PyTorch, mediapipe, and project deps
   - Data and pretrained weights available at the paths you pass (e.g. mounted volumes or copied in a prior step)
4. If you use a **Docker agent**, build an image that installs the app (e.g. `pip install -e .` or `pip install -r requirements.txt`) and set the workspace to the repo root.

**Parameters** in the job match the env vars above; adjust default values in the Jenkinsfile to match your container layout (e.g. `/data/train`, `/models/pretrained.pth`).

## 3. Single-line command (no script)

If you prefer to call Python from Jenkins without the shell script:

```bash
cd "$WORKSPACE"   # or your repo root inside the container

python3 -m app.training.finetune \
  --data-dir "${DATA_DIR:-data/AVLips1 2}" \
  --pretrained "${PRETRAINED:-weights/best_model.pth}" \
  --output-dir "${OUTPUT_DIR:-weights}" \
  --epochs "${EPOCHS:-36}" \
  --freeze-epochs "${FREEZE_EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --lr 2e-4 \
  --lr-encoder 2e-5 \
  --contrastive-weight 0.1 \
  --use-augmentation \
  --early-stopping-patience 8 \
  --log-every "${LOG_EVERY:-5}"
```

Set `DATA_DIR`, `PRETRAINED`, `OUTPUT_DIR`, etc. in the Jenkins job (environment or parameters) so the container uses the correct paths.

---

## 4. ExecuteTask-style job (workdir + command + user)

If your Jenkins job uses parameters **workdir**, **command**, and **user** (e.g. code is rsync’d to the agent at `/home/ubuntu/lip_sync_service/`):

| Parameter | Value |
|-----------|--------|
| **workdir** | `/home/ubuntu/lip_sync_service` (or `$MY_WORKSPACE` if it points to this path on the agent) |
| **user** | `ubuntu` |

**command** — use the venv’s Python so you don’t depend on `source activate` in the job:

```bash
/home/ubuntu/lip_sync_service/venv/bin/python -m app.training.finetune --data-dir "data/AVLips1 2" --pretrained "weights/best_model.pth" --epochs 36 --freeze-epochs 8 --batch-size 8 --lr 2e-4 --lr-encoder 2e-5 --contrastive-weight 0.1 --use-augmentation --early-stopping-patience 8 --log-every 5
```

Or activate venv then run (if the job runs in a login shell):

```bash
source /home/ubuntu/lip_sync_service/venv/bin/activate && python -m app.training.finetune --data-dir "data/AVLips1 2" --pretrained "weights/best_model.pth" --epochs 36 --freeze-epochs 8 --batch-size 8 --lr 2e-4 --lr-encoder 2e-5 --contrastive-weight 0.1 --use-augmentation --early-stopping-patience 8 --log-every 5
```

Or run the Jenkins script (same venv):

```bash
/home/ubuntu/lip_sync_service/venv/bin/python /home/ubuntu/lip_sync_service/scripts/set_resource_limits.py /home/ubuntu/lip_sync_service/venv/bin/python -m app.training.finetune --data-dir "data/AVLips1 2" --pretrained "weights/best_model.pth" --epochs 36 --freeze-epochs 8 --batch-size 8 --lr 2e-4 --lr-encoder 2e-5 --contrastive-weight 0.1 --use-augmentation --early-stopping-patience 8 --log-every 5
```

**Note:** If you rsync without `venv/`, the agent must have a venv already at `/home/ubuntu/lip_sync_service/venv` (created once on the agent), or use system/python from the container and omit the `venv/bin/python` path.
