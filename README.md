
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
    V["Video Frames (B,3,T,H,W)"] --> VE["Visual Encoder (3D ResNet)"]
    A["Audio Spectrogram (B,1,F,T_a)"] --> AE["Audio Encoder (2D ResNet)"]

    VE --> VP["Visual Projection -> v_emb (B,T,256)"]
    AE --> AP["Audio Projection -> a_emb (B,T,256)"]

    VP --> CMA["Cross-Modal Attention + Gated Fusion"]
    AP --> CMA
    CMA --> F["Fused Sequence (B,T,256)"]

    F --> TT["Temporal Transformer (Multi-scale + CLS)"]
    TT --> CLS["Sync Semantic Feature (B,256)"]

    VE --> AD["Artifact Detector (Raw + Delta + High-Freq)"]
    CLS --> AD
    AD --> AF["Artifact Feature (B,128)"]

    CLS --> M["Merge Features (B,384)"]
    AF --> M

    M --> H["Classification Head (MLP)"]
    H --> O["Logit / Lip-Sync Authenticity Score"]

    style VE fill:#2a2f3a,stroke:#7aa2ff,color:#fff
    style AE fill:#2a2f3a,stroke:#7aa2ff,color:#fff
    style CMA fill:#3a2f4a,stroke:#b38cff,color:#fff
    style TT fill:#2f3a2f,stroke:#6adf91,color:#fff
    style AD fill:#4a2f2f,stroke:#ff8a8a,color:#fff

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

