# Setting Up Python 3.11/3.12 for MediaPipe

## Option 1: Using Homebrew (Recommended for Mac M1)

### Step 1: Install Python 3.11 via Homebrew

```bash
# Install Python 3.11 (keeps your Python 3.14 intact)
brew install python@3.11
```

### Step 2: Create a Virtual Environment with Python 3.11

```bash
cd /Users/macsolution/Desktop/lip_sync_service

# Create venv using Python 3.11
/opt/homebrew/bin/python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python version
python --version
# Should show: Python 3.11.x
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies (including MediaPipe)
pip install -r requirements.txt
```

### Step 4: Verify MediaPipe Works

```bash
python -c "import mediapipe; print('MediaPipe OK!')"
```

---

## Option 2: Using pyenv (If you have it)

```bash
# Install Python 3.11
pyenv install 3.11.9

# Set local Python version for this project
cd /Users/macsolution/Desktop/lip_sync_service
pyenv local 3.11.9

# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Option 3: Using Conda/Mamba (If you have it)

```bash
# Create environment with Python 3.11
conda create -n lip_sync python=3.11 -y
conda activate lip_sync

# Install dependencies
pip install -r requirements.txt
```

---

## After Setup: Always Activate Your Environment

**Before running training or the service:**

```bash
cd /Users/macsolution/Desktop/lip_sync_service
source venv/bin/activate  # or: conda activate lip_sync
```

You'll see `(venv)` in your terminal prompt when it's active.

---

## Why This Works

- ✅ **Python 3.14 stays installed** - you can use it for other projects
- ✅ **Python 3.11 in venv** - MediaPipe works perfectly
- ✅ **Project isolated** - dependencies don't conflict
- ✅ **Easy to switch** - just activate/deactivate venv

---

## Troubleshooting

### "python3.11: command not found"

If Homebrew didn't add it to PATH:

```bash
# Find where Homebrew installed it
ls -la /opt/homebrew/bin/python3.11

# Use full path to create venv
/opt/homebrew/bin/python3.11 -m venv venv
```

### "MediaPipe still fails after install"

Try reinstalling MediaPipe specifically:

```bash
pip uninstall mediapipe
pip install --upgrade --force-reinstall mediapipe
```

### Check Your Current Python

```bash
which python
python --version
```

Should show Python 3.11.x when venv is activated.

---

## Quick Test

After setup, test everything:

```bash
# Activate venv
source venv/bin/activate

# Test MediaPipe
python -c "import mediapipe; print('✅ MediaPipe works!')"

# Test your imports
python -c "from app.preprocessing.face_detection import FaceDetector; print('✅ Face detection imports OK!')"
```

If both work, you're ready to train! 🚀
