#!/bin/bash
# Fine-tune from best_model.pth (e.g. after initial training for a few epochs).
# Saves to weights_finetune/ so you keep the original weights/best_model.pth.

set -e
cd "$(dirname "$0")/.."

PRETRAINED="${1:-weights/best_model.pth}"
OUTPUT_DIR="${2:-weights_finetune}"

if [ ! -f "$PRETRAINED" ]; then
  echo "Pretrained weights not found: $PRETRAINED"
  echo "Usage: $0 [path_to_best_model.pth] [output_dir]"
  exit 1
fi

echo "Pretrained: $PRETRAINED"
echo "Output dir: $OUTPUT_DIR"
echo ""

python scripts/set_resource_limits.py python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained "$PRETRAINED" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --lr 1e-4 \
  --lr-encoder 1e-5

echo ""
echo "Done. Best checkpoints: $OUTPUT_DIR/best_model_accuracy.pth (most accurate), $OUTPUT_DIR/best_model_loss.pth (lowest loss)"
