# CoastalSeg: Image segmentation for coastal erosion monitoring

## About
University of Washington Capstone project: Machine Learning for Community-Driven Coastal Erosion Monitoring and Management.  
We present a system for multi-class segmentation and analysis of community user-uploaded images, while also integrating outlier detection and multi-image perspective correction.  This is an image processing system developed for coastal research at the University of Washington Applied Physics Laboratory.  
Segmentation Model: DeepLabV3Plus with EfficientNet-B6, achieved 0.93 IoU score.

[Xinghao Chen](https://cxh42.github.io/) <sup>1,</sup><sup>2</sup>, [Zheheng Li](https://github.com/Martyr12333) <sup>1,</sup><sup>2</sup>, [Dylan Scott](https://github.com/dwilsons) <sup>1,</sup><sup>2</sup>, Aaryan Shah <sup>1,</sup><sup>2</sup>, Bauka Zhandulla <sup>1,</sup><sup>2</sup>, Sarah Li <sup>1,</sup><sup>2</sup>

<sup>1 </sup>University of Washington&emsp; <sup>2 </sup>University of Washington Applied Physics Laboratory &emsp;

<div style="display: flex; justify-content: center;">
    <img src="assets/originalshow.jpg" style="width: 49%;" />
    <img src="assets/overlayshow.webp" style="width: 49%;" />
</div>

Try image segmentation demo at  

https://huggingface.co/spaces/AveMujica/MetalMarcy  
https://huggingface.co/spaces/AveMujica/SilhouetteJaenette  
https://huggingface.co/spaces/AveMujica/CoastalSegment (slower, integrates outlier detection and spatial alignment, mainly used for the [MyCoast](https://mycoast.org/wa).)  

## Environmental Setups
```bash
git clone https://github.com/cxh42/CoastalSeg.git
cd CoastalSeg
conda create -n CoastalSeg python=3.12
conda activate CoastalSeg
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Hugging Face login (recommended to avoid rate limits)
Run once on the machine; authenticated downloads are throttled less.
```bash
pip install -U huggingface_hub hf_transfer
huggingface-cli login  # paste your HF token
```
On PowerShell without interactivity:
```powershell
setx HF_TOKEN "YOUR_HF_TOKEN"
```

## Preparation

**1. Download pretrained models:**

```bash
python scripts/fetch_models.py
```

**2. (Optional) Download training datasets:**

Needed only if you plan to retrain; inference does not require datasets.

```bash
python scripts/fetch_datasets.py
```

**3. (Optional) Training segmentation model from scratch:**

Start training (two sites):

```bash
# Metal Marcy
python SegmentModelTraining/MetalMarcy/train.py

# Silhouette Jaenette
python SegmentModelTraining/SilhouetteJaenette/train.py
```

## Run
```bash
python app.py
```
By running app.py, a graphical interactive interface will automatically open in the browser.

## Batch Processing

For bulk processing without GUI, use the script `batch_infer.py`. It supports two workflows:

- Segmentation on a folder of images
- Spatial alignment (one reference) + segmentation for a folder of targets

Examples

```bash
# 1) Batch segmentation (save segmentation map + overlay)
python batch_infer.py segment \
  --location "Metal Marcy" \
  --input path/to/images \
  --output outputs/mm_segment \
  --overlay

# 2) Batch segmentation with outlier detection (adds outlier column to summary.csv)
python batch_infer.py segment \
  --location "Silhouette Jaenette" \
  --input path/to/images \
  --output outputs/sj_segment \
  --outlier

# 3) Batch spatial alignment + segmentation
python batch_infer.py align \
  --location "Metal Marcy" \
  --reference reference_images/MM/2025-01-26_16-36-00_MM.jpg \
  --input path/to/targets \
  --output outputs/mm_aligned \
  --overlay
```
