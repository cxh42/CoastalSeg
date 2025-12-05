# CoastalSeg — Installation & User Guide

**Overview**  
CoastalSeg is a project for coastal image segmentation. This guide explains how to set up a Windows + NVIDIA GPU environment, fetch pretrained assets, (optionally) retrain, and launch the app for interactive use.

---

## 1) Prerequisites

- **Hardware:** NVIDIA GPU (recent driver recommended for CUDA 12.x runtime wheels)
- **Disk space:** ~1.31 GB
- **Recommended tools:** Miniconda (or Anaconda), VS Code, Git

> Tip: You can verify your GPU/driver with `nvidia-smi` in a Command Prompt (or PowerShell).

---

## 2) Clone the repository

```bash
git clone https://github.com/cxh42/CoastalSeg.git
cd CoastalSeg
```

---

## 3) Create and activate a Conda environment (Python 3.12)

```bash
conda create -n CoastalSeg python=3.12
conda activate CoastalSeg
```

---

## 4) Install PyTorch (CUDA 12.6 build) and project dependencies

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**(Optional) Quick GPU check in Python:**
```python
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5) Hugging Face login (recommended to avoid rate limits)

Run once on the machine; authenticated downloads are throttled less.

```bash
hf auth login  # paste your HF token
```


---

## 6) Download pretrained models

```bash
python scripts/fetch_models.py
```
This fetches the trained weights and reference vectors needed for inference.

---

## 7) (Optional) Download training datasets

```bash
python scripts/fetch_datasets.py
```
Only needed if you plan to retrain locally. Inference does not require these datasets.
You can also see https://huggingface.co/datasets/AveMujica/CostalSeg-SJ and https://huggingface.co/datasets/AveMujica/CostalSeg-MM directly.
---

## 8) (Optional) Retrain the models locally

If you want to reproduce training locally:
```bash
# Train the "Metal Marcy" model
python SegmentModelTraining/MetalMarcy/train.py

# Train the "Silhouette Jaenette" model
python SegmentModelTraining/SilhouetteJaenette/train.py
```
These scripts will run end-to-end and place the resulting model weights where the app can find them automatically. If you only want to use pretrained models, you can skip this section.

---

## 9) Launch the interactive app (GUI)

```bash
python app.py
```
Running this command opens a browser-based interface. You can upload your own images for processing, and sample images are included in the repository. If a browser tab does not open automatically, copy the local URL printed in the terminal into your browser.

---

## 10) Troubleshooting

- **GPU not detected (`torch.cuda.is_available() == False`)**  
  - Ensure the correct PyTorch wheel is installed (CUDA 12.6 build as shown above).  
  - Update to a recent NVIDIA driver compatible with CUDA 12.x.  
  - Close and reopen your terminal, then re-activate the Conda environment.

- **Browser doesn’t open automatically**  
  - Press Enter again


---

## 11) Quick command summary

```bash
# Clone
git clone https://github.com/cxh42/CoastalSeg.git
cd CoastalSeg

# Conda env
conda create -n CoastalSeg python=3.12
conda activate CoastalSeg

# Install
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Fetch models (required)
python scripts/fetch_models.py

# (Optional) Fetch datasets
python scripts/fetch_datasets.py

# (Optional) Train
python SegmentModelTraining/MetalMarcy/train.py
python SegmentModelTraining/SilhouetteJaenette/train.py

# Launch app
python app.py
```
