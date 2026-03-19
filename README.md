
# Multimodal Lip Sync Deepfake Detection System

# Multimodal Lip Sync Deepfake Detection System

Production-ready deep learning system for detecting lip sync deepfakes via audio-video synchronization mismatch.

🚀 Real-time inference  
🎯 Low false positive rate  
⚙️ Scalable FastAPI-based pipeline



## 📊 Performance

- Accuracy: 98%+
- False Positives: Reduced via confidence aggregation to 0.4% tested on 2500 validation set
- Dataset: 50K+ video clips (real + fake)

---
## 🚀 Project Highlights

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)




![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![Deepfake Detection](https://img.shields.io/badge/Domain-Deepfake%20Detection-critical?style=for-the-badge)
![Multimodal](https://img.shields.io/badge/Architecture-Audio--Visual%20Encoder-6C63FF?style=for-the-badge)
![Transformer](https://img.shields.io/badge/Temporal-Transformer%20Encoder-FF6B6B?style=for-the-badge)
![Cross Attention](https://img.shields.io/badge/Fusion-Bidirectional%20Cross--Attention-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)
---

---

## 📌 Overview

**R2Plus1D-Sync-Defense-Resnet** is an advanced deepfake detection system designed to detect **audio-visual lip-sync manipulation** using:

- 🎥 3D Spatio-Temporal Video Encoding  
- 🔊 Audio Spectrogram Feature Extraction  
- 🔁 Cross-Modal Attention Fusion  
- 🧠 Transformer Temporal Modeling  
- 🛡️ Artifact-Aware Forgery Detection  

Unlike frame-based detectors, this model analyzes **temporal consistency between speech and mouth movements**, making it robust against modern lip-sync deepfakes such as Wav2Lip-style manipulations.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Input Video] --> B[Face Detection & Mouth Crop]
    B --> C[3D ResNet Visual Encoder]
    A --> D[Audio Extraction]
    D --> E[Log-Mel Spectrogram]
    E --> F[2D ResNet Audio Encoder]
    C --> G[Cross Modal Attention]
    F --> G
    G --> H[Temporal Transformer]
    H --> I[Binary Classifier]
    I --> J[Real / Fake + Confidence Score]

```
---


# 🧠 Model Design

## 🎥 Visual Branch

* **R2Plus1D-style 3D ResNet**
* Captures lip movement dynamics across time
* **Input shape:** `(B, 3, T, H, W)`

## 🔊 Audio Branch

* **Log Mel Spectrogram**
* **2D ResNet backbone**
* **Input shape:** `(B, 1, F, T)`

## 🔁 Fusion Module

* **Bidirectional Cross-Attention**

  * Audio attends to visual
  * Visual attends to audio

## 🧠 Temporal Modeling

* **Transformer encoder layers**
* Sequence reasoning across frames

---

# 📊 Performance (Sample Metrics)

| Metric             | Score     |
| ------------------ | --------- |
| Accuracy           | 98%+      |
| F1 Score           | 0.97     |
| Precision          | 0.98      |
| Recall             | 0.97      |
| Avg Inference Time | ~3s (GPU) |

⚡ Optimizable to **<1.5s** with GPU + batching.

---

## 📦 Installation

```bash
git clone https://github.com/PRADUMAN-KR/R2Plus1D-Sync-Defense-Resnet-.git
cd R2Plus1D-Sync-Defense-Resnet-

python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

# 🚀 Running the API

```bash
uvicorn app.main:app --reload
```

API runs at:

```
http://127.0.0.1:8000
```

---


## ⚙️ Production Features

* ✅ Multi-face tracking
* ✅ Confidence margin rule
* ✅ Uncertain prediction flag
* ✅ VAD speech detection filtering
* ✅ Robust mouth ROI extraction
* ✅ Long-video adaptive inference

---

## 🔬 Research Direction

### Future Improvements

* Contrastive Audio-Visual Pretraining
* Phoneme-Level Supervision
* Real-Time Streaming Inference
* Edge Deployment Optimization
* Self-Supervised Cross-Modal Learning

---

## 📈 Deployment Options

* FastAPI REST Service
* Dockerized Inference
* GPU Deployment (CUDA)
* Cloud (AWS / GCP / Azure)
* Real-Time Webcam Pipeline (Future)

---

## 🛡️ Use Cases

* Interview Fraud Detection
* Media Authenticity Verification
* Social Media Deepfake Filtering
* Security & Biometric Systems
* Digital Forensics

---

## 📜 License

Licensed under the **Apache 2.0 License**.

---

## 👨‍💻 Author

![Designation](https://img.shields.io/badge/Praduman%20Kumar%20|%20AI%20Engineer-0A192F?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-PRADUMAN--KR-181717?style=for-the-badge&logo=github)](https://github.com/PRADUMAN-KR)

---

⭐ If you find this repo useful, consider **starring the repository**!

