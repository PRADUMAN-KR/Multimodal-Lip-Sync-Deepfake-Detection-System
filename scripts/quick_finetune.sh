#!/bin/bash
# Quick Fine-Tuning Script for Beginners
# Just run this script and it will handle everything!

set -e  # Exit on error

echo "🚀 Starting Fine-Tuning Process..."
echo ""

# Check if MediaPipe is installed
echo "📦 Checking dependencies..."
python3 -c "import mediapipe" 2>/dev/null || {
    echo "⚠️  MediaPipe not found. Installing..."
    pip install mediapipe>=0.10
}

# Check if data directory exists
if [ ! -d "data/AVLips1 2" ]; then
    echo "❌ Error: data/AVLips1 2 directory not found!"
    echo "   Please make sure your training data is in: data/AVLips1 2/"
    exit 1
fi

# Check if weights directory exists
mkdir -p weights

# Check if we have pre-trained weights
if [ -f "weights/best_model.pth" ]; then
    echo "✅ Found existing weights: weights/best_model.pth"
    echo "   Starting fine-tuning..."
    echo ""
    
    python3 -m app.training.finetune \
        --data-dir "data/AVLips1 2" \
        --pretrained weights/best_model.pth \
        --epochs 30 \
        --freeze-epochs 10 \
        --batch-size 4 \
        --use-augmentation \
        --device mps
    
    echo ""
    echo "✅ Fine-tuning complete!"
    echo "   Updated model saved to: weights/best_model.pth"
    
else
    echo "⚠️  No pre-trained weights found."
    echo "   Starting initial training first..."
    echo ""
    
    echo "📚 Step 1: Initial Training (this may take a while)..."
    python3 -m app.training.train \
        --data-dir "data/AVLips1 2" \
        --epochs 50 \
        --batch-size 4 \
        --device mps
    
    echo ""
    echo "✅ Initial training complete!"
    echo ""
    echo "📚 Step 2: Fine-Tuning..."
    
    python3 -m app.training.finetune \
        --data-dir "data/AVLips1 2" \
        --pretrained weights/best_model.pth \
        --epochs 30 \
        --freeze-epochs 10 \
        --batch-size 4 \
        --use-augmentation \
        --device mps
    
    echo ""
    echo "✅ Fine-tuning complete!"
    echo "   Final model saved to: weights/best_model.pth"
fi

echo ""
echo "🎉 All done! Your model is ready to use."
echo "   Restart your service with: python3 -m app.main"
