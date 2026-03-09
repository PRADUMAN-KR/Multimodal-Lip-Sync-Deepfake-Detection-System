#!/bin/bash
# Jenkins/container-friendly finetune runner.
# Uses environment variables so Jenkins can override paths and hyperparameters.
# Run from repo root (e.g. Jenkins workspace) or from scripts/ with: ./scripts/run_finetune_jenkins.sh

set -e

# Repo root: allow running from scripts/ or from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv first if present (VENV_DIR can override location)
VENV_PATH="${VENV_DIR:-$REPO_ROOT/venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  echo "Activating venv: $VENV_PATH"
  set +e
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
  set -e
  PYTHON_BIN="python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Config from environment (defaults for local/dev)
DATA_DIR="${DATA_DIR:-data/AVLips1 2}"
PRETRAINED="${PRETRAINED:-weights/best_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-weights}"
EPOCHS="${EPOCHS:-36}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-4}"
LR_ENCODER="${LR_ENCODER:-2e-5}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
LOG_EVERY="${LOG_EVERY:-5}"
# PYTHON_BIN set above after venv activation (or default python3 if no venv)
# DEVICE: leave unset to auto-detect (CUDA > MPS > CPU)

echo "=============================================="
echo "Lip-sync finetune (Jenkins/container)"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  DATA_DIR=$DATA_DIR"
echo "  PRETRAINED=$PRETRAINED"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  EPOCHS=$EPOCHS FREEZE_EPOCHS=$FREEZE_EPOCHS BATCH_SIZE=$BATCH_SIZE"
echo "  LR=$LR LR_ENCODER=$LR_ENCODER"
echo "  PYTHON=$PYTHON_BIN"
echo "=============================================="

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR not found: $DATA_DIR"
  exit 1
fi
if [ ! -f "$PRETRAINED" ]; then
  echo "ERROR: PRETRAINED checkpoint not found: $PRETRAINED"
  exit 1
fi

EXTRA_ARGS=()
if [ -n "$DEVICE" ]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi

# Optional: resource limits (skip if set_resource_limits.py not available)
if [ -f "scripts/set_resource_limits.py" ]; then
  RUN_CMD=("$PYTHON_BIN" scripts/set_resource_limits.py "$PYTHON_BIN")
else
  RUN_CMD=("$PYTHON_BIN")
fi

"${RUN_CMD[@]}" -m app.training.finetune \
  --data-dir "$DATA_DIR" \
  --pretrained "$PRETRAINED" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --freeze-epochs "$FREEZE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-encoder "$LR_ENCODER" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" \
  --use-augmentation \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --log-every "$LOG_EVERY" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Checkpoints: $OUTPUT_DIR/best_model_accuracy.pth, $OUTPUT_DIR/best_model_f1.pth"
