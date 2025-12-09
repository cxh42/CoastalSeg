import os
import re
import io
import cv2
import base64
import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from glob import glob
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pipeline.ImgOutlier import detect_outliers
from pipeline.normalization import align_images

# Global Configuration
MODEL_PATHS = {
    "Metal Marcy": "models/MM_best_model.pth",
    "Silhouette Jaenette": "models/SJ_best_model.pth"
}

REFERENCE_VECTOR_PATHS = {
    "Metal Marcy": "models/MM_mean.npy",
    "Silhouette Jaenette": "models/SJ_mean.npy"
}

REFERENCE_IMAGE_DIRS = {
    "Metal Marcy": "reference_images/MM",
    "Silhouette Jaenette": "reference_images/SJ"
}

def extract_image_datetime(image_path):
    """
    Extract date/time string from image filename formatted like YYYY-MM-DD_HH-MM-SS_*.*
    """
    if not image_path:
        return "", ""
    filename = os.path.basename(image_path)
    match = re.search(r"(\d{4}-\d{2}-\d{2})[_-](\d{2}-\d{2}-\d{2})", filename)
    if match:
        date_part, time_part = match.groups()
        time_part = time_part.replace("-", ":")
        return filename, f"{date_part} {time_part}"
    return filename, ""

# Category names and color mapping
CLASSES = ['Background', 'Cobbles', 'Dry sand', 'Plant', 'Sky', 'Water', 'Wet sand']
COLORS = [
    [0, 0, 0],        # background - black
    [139, 137, 137],  # cobbles - dark gray
    [255, 228, 181],  # drysand - light yellow
    [0, 128, 0],      # plant - green
    [135, 206, 235],  # sky - sky blue
    [0, 0, 255],      # water - blue
    [194, 178, 128]   # wetsand - sand brown
]

# Load model function
def load_model(model_path, device="cuda"):
    """
    Load the segmentation model from the specified path
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded PyTorch model or None if loading failed
    """
    try:
        model = smp.create_model(
            "DeepLabV3Plus",
            encoder_name="efficientnet-b6",
            in_channels=3,
            classes=len(CLASSES),
            encoder_weights=None
        )
        state_dict = torch.load(model_path, map_location=device)
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

# Load reference vector
def load_reference_vector(vector_path):
    """
    Load the reference vector used for outlier detection
    
    Args:
        vector_path (str): Path to the reference vector file
        
    Returns:
        np.array: Reference vector or empty list if loading failed
    """
    try:
        ref_vector = np.load(vector_path)
        print(f"Reference vector loaded successfully: {vector_path}")
        return ref_vector
    except Exception as e:
        print(f"Reference vector loading failed {vector_path}: {e}")
        return []

# Load reference images
def load_reference_images(ref_dir):
    """
    Load reference images from the specified directory
    
    Args:
        ref_dir (str): Directory containing reference images
        
    Returns:
        list: List of loaded reference images or empty list if loading failed
    """
    try:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(ref_dir, ext)))
        image_files.sort()
        reference_images = []
        for file in image_files[:4]:
            img = cv2.imread(file)
            if img is not None:
                reference_images.append(img)
        print(f"Loaded {len(reference_images)} images from {ref_dir}")
        return reference_images
    except Exception as e:
        print(f"Image loading failed {ref_dir}: {e}")
        return []

# Preprocess the image
def preprocess_image(image):
    """
    Preprocess an image for model inference
    
    Args:
        image (np.array): Input image in RGB format
        
    Returns:
        tuple: (preprocessed image tensor, original height, original width)
    """
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_tensor, orig_h, orig_w

# Generate segmentation map and visualization
def generate_segmentation_map(prediction, orig_h, orig_w):
    """
    Generate a segmentation map from model prediction
    
    Args:
        prediction (torch.Tensor): Model prediction
        orig_h (int): Original image height
        orig_w (int): Original image width
        
    Returns:
        np.array: Segmentation map as a colored image
    """
    mask = prediction.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = mask_resized.copy()
    for idx in range(1, len(CLASSES)):
        class_mask = (mask_resized == idx).astype(np.uint8)
        dilated_mask = cv2.dilate(class_mask, kernel, iterations=2)
        dilated_effect = dilated_mask & (mask_resized == 0)
        processed_mask[dilated_effect > 0] = idx
    segmentation_map = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        segmentation_map[processed_mask == idx] = color
    return segmentation_map

# Analysis result with Pie Chart (including background)
def create_analysis_result(mask, location="Unknown", image_path=None):
    """
    Create a pie chart visualization of the terrain distribution
    
    Args:
        mask (np.array): Segmentation mask
        location (str): Location name for display
        image_path (str): Path of the input image for date extraction
        
    Returns:
        tuple: (HTML content, pie bytes, percentages dict, text summary, metadata dict)
    """
    total_pixels = mask.size if mask.size else 1
    filename, date_text = extract_image_datetime(image_path)
    date_label = date_text if date_text else "Date Unknown"
    location_label = location if location else "Unknown"

    # Calculate percentages for every class
    percentages = {cls: round((np.sum(mask == i) / total_pixels) * 100, 1)
                   for i, cls in enumerate(CLASSES)}

    # Use non-zero classes for the pie wedges to avoid clutter
    non_zero_items = [(cls, pct) for cls, pct in percentages.items() if pct > 0]
    if not non_zero_items:
        non_zero_items = [("Background", 100.0)]
    sorted_items = sorted(non_zero_items, key=lambda x: x[1], reverse=True)

    fig = Figure(figsize=(8, 5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [np.array(COLORS[CLASSES.index(cls)]) / 255 for cls in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )

    for idx, autotext in enumerate(autotexts):
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
        autotext.set_color('white' if np.mean(colors[idx]) < 0.5 else 'black')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f"{labels[i]} ({values[i]}%)")
        for i in range(len(labels))
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_title(f"Analysis Results - {date_label}")
    ax.axis('equal')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pie_bytes = buf.read()
    img_str = "data:image/png;base64," + base64.b64encode(pie_bytes).decode('utf-8')

    sorted_classes = sorted(percentages, key=percentages.get, reverse=True)
    percentage_lines = [f"{cls}: {percentages[cls]}%" for cls in sorted_classes]
    text_block = "<br>".join([
        f"Image date: {date_text or 'Unknown'}",
        f"Location: {location_label}",
        f"Filename: {filename or 'N/A'}",
        ""
    ] + percentage_lines)

    result_html = f"""
    <div style='display:flex; flex-direction:column; align-items:center; text-align:center; gap:10px;'>
        <img src='{img_str}' alt='Terrain Distribution Pie Chart' style='max-width:100%; height:auto;'>
        <div style='font-size:14px; line-height:1.5;'>{text_block}</div>
    </div>
    """

    text_summary = "\n".join([
        f"Filename: {filename or 'N/A'}",
        f"Image date: {date_text or 'Unknown'}",
        f"Location: {location_label}",
        "Percentages:"
    ] + [f"- {cls}: {percentages[cls]}%" for cls in sorted_classes])

    metadata = {
        "filename": filename,
        "image_date": date_text or "Unknown",
        "location": location_label
    }

    return result_html, pie_bytes, percentages, text_summary, metadata

# Merge and overlay
def create_overlay(image, segmentation_map, alpha=0.5):
    """
    Create an overlay of the original image and segmentation map
    
    Args:
        image (np.array): Original image in RGB format
        segmentation_map (np.array): Segmentation map
        alpha (float): Transparency value for the overlay
        
    Returns:
        np.array: Overlay image
    """
    if image.shape[:2] != segmentation_map.shape[:2]:
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1-alpha, segmentation_map, alpha, 0)

# Perform segmentation
def perform_segmentation(model, image_bgr, location, image_path=None):
    """
    Perform segmentation on an image
    
    Args:
        model: Loaded PyTorch model
        image_bgr (np.array): Input image in BGR format
        location (str): Location name for display
        image_path (str): Original image path for metadata extraction
        
    Returns:
        tuple: (segmentation map, overlay image, analysis HTML, save payload)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(image_rgb)
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
    seg_map = generate_segmentation_map(prediction, orig_h, orig_w)  # RGB
    overlay = create_overlay(image_rgb, seg_map)
    mask = prediction.argmax(1).squeeze().cpu().numpy()
    analysis_html, pie_bytes, percentages, text_summary, metadata = create_analysis_result(
        mask, location=location, image_path=image_path
    )
    save_payload = {
        "seg_map": seg_map,
        "overlay": overlay,
        "pie_bytes": pie_bytes,
        "text_summary": text_summary,
        "percentages": percentages,
        "metadata": metadata
    }
    return seg_map, overlay, analysis_html, save_payload

def predict_mask_and_analysis(model, image_bgr, location, image_path=None):
    """
    Run model on the original (pre-alignment) image and return
    the raw class-index mask (model resolution) and analysis HTML.

    Args:
        model: Loaded PyTorch model
        image_bgr (np.array): Input image in BGR format
        location (str): Location name for display
        image_path (str): Original image path for metadata extraction

    Returns:
        tuple: (mask_1024, orig_h, orig_w, analysis_html, analysis payload)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor, orig_h, orig_w = preprocess_image(image_rgb)
    with torch.no_grad():
        prediction = model(image_tensor.to(device))
    mask_1024 = prediction.argmax(1).squeeze().cpu().numpy()
    analysis_html, pie_bytes, percentages, text_summary, metadata = create_analysis_result(
        mask_1024, location=location, image_path=image_path
    )
    analysis_payload = {
        "pie_bytes": pie_bytes,
        "text_summary": text_summary,
        "percentages": percentages,
        "metadata": metadata
    }
    return mask_1024, orig_h, orig_w, analysis_html, analysis_payload

# Split the processing into separate functions for progressive display

def run_segmentation(location, input_image_path, progress=gr.Progress()):
    """
    Run image segmentation task independently
    
    Args:
        location (str): Location name for model selection
        input_image_path (str): Input image path
        progress: Gradio progress indicator
        
    Returns:
        tuple: (segmentation map, overlay image, analysis HTML, save payload)
    """
    if not input_image_path or (isinstance(input_image_path, str) and not os.path.exists(input_image_path)):
        return None, None, "Please upload an image to analyze", {}
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        return None, None, "Error: Unable to load model", {}
    
    # Process the image
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        return None, None, "Error: Unable to read the image file", {}
    
    progress(0.3, desc="Performing segmentation (GPU)...")
    seg_map, overlay, analysis, save_payload = perform_segmentation(
        model, image_bgr, location, image_path=input_image_path
    )
    
    progress(1.0, desc="Segmentation complete")
    # Include image path for saving later
    save_payload["image_path"] = input_image_path
    save_payload["mode"] = "single"
    return seg_map, overlay, analysis, save_payload

def run_outlier_detection(location, input_image_path, progress=gr.Progress()):
    """
    Run outlier detection task independently
    
    Args:
        location (str): Location name for model selection
        input_image_path (str): Input image path
        progress: Gradio progress indicator
        
    Returns:
        tuple: (status HTML, warning HTML)
    """
    if not input_image_path or (isinstance(input_image_path, str) and not os.path.exists(input_image_path)):
        return "No image detected", ""
    
    # Choose device for outlier detection (prefer GPU if available)
    det_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading reference data...")
    
    # Load reference data
    ref_vector = load_reference_vector(REFERENCE_VECTOR_PATHS[location]) if os.path.exists(REFERENCE_VECTOR_PATHS[location]) else []
    ref_images = load_reference_images(REFERENCE_IMAGE_DIRS[location])
    
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        return "Error: Unable to read image for outlier detection", ""
    
    # Perform outlier detection
    progress(0.3, desc=f"Performing outlier detection ({det_device.upper()})...")
    is_outlier = False
    
    # Run detection using selected device
    if len(ref_vector) > 0:
        filtered, _ = detect_outliers(ref_images, [image_bgr], ref_vector, device=det_device)
        is_outlier = len(filtered) == 0
    else:
        filtered, _ = detect_outliers(ref_images, [image_bgr], device=det_device)
        is_outlier = len(filtered) == 0
    
    progress(1.0, desc="Outlier detection complete")
    outlier_status = "Outlier Detection: <span style='color:red;font-weight:bold'>Failed</span>" if is_outlier else "Outlier Detection: <span style='color:green;font-weight:bold'>Passed</span>"
    
    # Add warning to analysis if outlier
    if is_outlier:
        outlier_warning = "<div style='color:red;font-weight:bold;margin-bottom:10px'>Warning: Image did not pass outlier detection. Results may be less accurate!</div>"
        return outlier_status, outlier_warning
    
    return outlier_status, ""

def update_analysis_with_warning(analysis, warning):
    """
    Update analysis HTML with warning message if needed
    
    Args:
        analysis (str): Original analysis HTML
        warning (str): Warning message to prepend
        
    Returns:
        str: Updated analysis HTML
    """
    if warning and analysis:
        return warning + analysis
    return analysis

# Spatial Alignment with progressive display
def run_alignment_and_segmentation(location, reference_image_path, input_image_path, progress=gr.Progress()):
    """
    Run spatial alignment and segmentation with progressive display
    
    Args:
        location (str): Location name for model selection
        reference_image_path (str): Reference image path
        input_image_path (str): Input image path to analyze
        progress: Gradio progress indicator
        
    Returns:
        tuple: (
            reference image,
            aligned image,
            segmentation map (aligned),
            overlay image (aligned),
            pre-alignment analysis HTML,
            aligned analysis HTML,
            status HTML,
            save payload
        )
    """
    if (not reference_image_path or not os.path.exists(reference_image_path)) or \
       (not input_image_path or not os.path.exists(input_image_path)):
        message = "Please upload both reference and target images for analysis"
        return None, None, None, None, message, message, "Not processed", {}
    
    # Set up GPU device
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show loading status
    progress(0, desc="Loading segmentation model...")
    
    model = load_model(MODEL_PATHS[location], gpu_device)
    
    if model is None:
        return None, None, None, None, "Error: Unable to load model", "Error: Unable to load model", "Analysis failed", {}
    
    ref_bgr = cv2.imread(reference_image_path)
    tgt_bgr = cv2.imread(input_image_path)
    if ref_bgr is None or tgt_bgr is None:
        message = "Error: Unable to read provided images"
        return None, None, None, None, message, message, "Analysis failed", {}
    
    # 1) Perform segmentation on pre-aligned target image
    progress(0.3, desc="Performing segmentation (pre-alignment)...")
    mask_1024, orig_h, orig_w, analysis_pre, _ = predict_mask_and_analysis(
        model, tgt_bgr, location, image_path=input_image_path
    )

    # Resize mask back to original target size and lightly postprocess (same as visualization path)
    mask_resized = cv2.resize(mask_1024, (tgt_bgr.shape[1], tgt_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = mask_resized.copy()
    for idx in range(1, len(CLASSES)):
        class_mask = (mask_resized == idx).astype(np.uint8)
        dilated_mask = cv2.dilate(class_mask, kernel, iterations=2)
        dilated_effect = dilated_mask & (mask_resized == 0)
        processed_mask[dilated_effect > 0] = idx

    # 2) Perform spatial alignment on images and warp the segmentation mask using the same transform
    progress(0.6, desc="Performing spatial alignment...")
    ref_seg_dummy = np.zeros(ref_bgr.shape[:2], dtype=np.uint8)
    aligned_imgs, aligned_segs = align_images([ref_bgr, tgt_bgr], [ref_seg_dummy, processed_mask.astype(np.uint8)])
    aligned_tgt_bgr = aligned_imgs[1]
    aligned_mask = aligned_segs[1]

    # 3) Prepare display images
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    aligned_tgt_rgb = cv2.cvtColor(aligned_tgt_bgr, cv2.COLOR_BGR2RGB)

    # Colorize aligned mask for visualization
    seg_map_aligned = np.zeros((aligned_mask.shape[0], aligned_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        seg_map_aligned[aligned_mask == idx] = color

    # Overlay on aligned image
    overlay_aligned = create_overlay(aligned_tgt_rgb, seg_map_aligned)

    status = "Spatial Alignment: <span style='color:green;font-weight:bold'>Successfully Completed</span>"

    # 4) Compute analysis from aligned mask (includes black borders)
    analysis_aligned_html, pie_bytes, percentages, text_summary, metadata = create_analysis_result(
        aligned_mask.astype(np.uint8), location=location, image_path=input_image_path
    )
    save_payload = {
        "seg_map": seg_map_aligned,
        "overlay": overlay_aligned,
        "pie_bytes": pie_bytes,
        "text_summary": text_summary,
        "percentages": percentages,
        "metadata": metadata,
        "image_path": input_image_path,
        "mode": "alignment"
    }

    # 5) Return both analyses: pre-alignment and aligned
    progress(1.0, desc="Analysis complete")
    return ref_rgb, aligned_tgt_rgb, seg_map_aligned, overlay_aligned, analysis_pre, analysis_aligned_html, status, save_payload

def save_results(save_payload):
    """
    Save segmentation outputs (segmentation map, overlay, pie chart) and a text summary.
    """
    if not isinstance(save_payload, dict) or not save_payload:
        return "No results to save. Please run an analysis first."

    seg_map = save_payload.get("seg_map")
    overlay = save_payload.get("overlay")
    pie_bytes = save_payload.get("pie_bytes")
    percentages = save_payload.get("percentages", {})
    metadata = save_payload.get("metadata", {})

    if seg_map is None or overlay is None or pie_bytes is None:
        return "Missing data to save. Please rerun the analysis."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    seg_path = os.path.join(save_dir, "segmentation_map.png")
    ovl_path = os.path.join(save_dir, "overlay.png")
    pie_path = os.path.join(save_dir, "analysis_pie.png")
    txt_path = os.path.join(save_dir, "summary.txt")

    cv2.imwrite(seg_path, cv2.cvtColor(seg_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(ovl_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    with open(pie_path, "wb") as f:
        f.write(pie_bytes)

    sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    text_lines = [
        f"Mode: {save_payload.get('mode', 'N/A')}",
        f"Location: {metadata.get('location', 'Unknown')}",
        f"Filename: {metadata.get('filename') or 'N/A'}",
        f"Image date: {metadata.get('image_date', 'Unknown')}"
    ]
    if save_payload.get("image_path"):
        text_lines.append(f"Source path: {save_payload['image_path']}")
    text_lines.append("")
    text_lines.append("Percentages (descending):")
    text_lines.extend([f"- {cls}: {pct}%" for cls, pct in sorted_percentages])
    if save_payload.get("text_summary"):
        text_lines.append("")
        text_lines.append("Detail summary:")
        text_lines.append(save_payload["text_summary"])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    return f"Saved files to {save_dir}:<br>- {seg_path}<br>- {ovl_path}<br>- {pie_path}<br>- {txt_path}"

# Create the Gradio interface with progressive display
def create_interface():
    """
    Create the Gradio web interface with progressive result display
    
    Returns:
        gradio.Blocks: Gradio interface
    """
    with gr.Blocks(title="Coastal Erosion Analysis System") as demo:
        gr.Markdown("""# Coastal Erosion Analysis System

Upload coastal photographs for segmentation analysis and spatial alignment. The system identifies terrain types including background, cobbles, sand, plants, sky, and water.""")
        
        # Store analysis content for updating with warnings
        current_analysis = gr.State("")
        outlier_warning = gr.State("")
        single_save_state = gr.State({})
        align_save_state = gr.State({})
        
        with gr.Tabs():
            with gr.TabItem("Single Image Segmentation"):
                with gr.Row():
                    loc1 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])

                with gr.Row():
                    inp = gr.Image(label="Input Image", type="filepath")
                    seg = gr.Image(label="Segmentation Map", type="numpy")
                    ovl = gr.Image(label="Overlay Visualization", type="numpy")

                with gr.Row():
                    btn1 = gr.Button("Run Analysis", variant="primary")

                # Built-in example images
                gr.Examples(
                    label="Examples",
                    examples=[
                        ["Metal Marcy", "reference_images/MM/2025-01-26_16-36-00_MM.jpg"],
                        ["Metal Marcy", "reference_images/MM/2025-01-25_13-55-00_MM.jpg"],
                        ["Silhouette Jaenette", "reference_images/SJ/2025-01-26_14-43-00_SJ.jpg"],
                        ["Silhouette Jaenette", "reference_images/SJ/2025-01-23_11-22-00_SJ.jpg"],
                    ],
                    inputs=[loc1, inp],
                    examples_per_page=4,
                )

                status1 = gr.HTML(label="Outlier Detection Status")
                res1 = gr.HTML(label="Terrain Analysis")
                save_status1 = gr.HTML(label="Save Status")
                save_btn1 = gr.Button("Save Results", variant="secondary")
                
                # When the button is clicked, run both functions in parallel
                btn1.click(
                    fn=run_segmentation,
                    inputs=[loc1, inp],
                    outputs=[seg, ovl, res1, single_save_state]
                ).then(
                    fn=lambda analysis: analysis,
                    inputs=[res1],
                    outputs=[current_analysis]
                )
                
                # Also start outlier detection
                btn1.click(
                    fn=run_outlier_detection,
                    inputs=[loc1, inp],
                    outputs=[status1, outlier_warning]
                ).then(
                    # Update analysis with warning if needed
                    fn=update_analysis_with_warning,
                    inputs=[current_analysis, outlier_warning],
                    outputs=[res1]
                )
                
                save_btn1.click(
                    fn=save_results,
                    inputs=[single_save_state],
                    outputs=[save_status1]
                )
            
            with gr.TabItem("Spatial Alignment Segmentation"):
                with gr.Row():
                    loc2 = gr.Radio(list(MODEL_PATHS.keys()), label="Select Location", value=list(MODEL_PATHS.keys())[0])

                with gr.Row():
                    ref_img = gr.Image(label="Reference Image", type="filepath")
                    tgt_img = gr.Image(label="Target Image for Analysis", type="filepath")

                with gr.Row():
                    btn2 = gr.Button("Run Spatial Alignment Analysis", variant="primary")

                # Built-in paired examples (reference, target)
                gr.Examples(
                    label="Examples",
                    examples=[
                        [
                            "Metal Marcy",
                            "reference_images/MM/2025-01-26_16-36-00_MM.jpg",
                            "reference_images/MM/2025-01-25_13-55-00_MM.jpg",
                        ],
                        [
                            "Silhouette Jaenette",
                            "reference_images/SJ/2025-01-26_14-43-00_SJ.jpg",
                            "reference_images/SJ/2025-01-23_11-22-00_SJ.jpg",
                        ],
                    ],
                    inputs=[loc2, ref_img, tgt_img],
                    examples_per_page=2,
                )

                with gr.Row():
                    orig = gr.Image(label="Original Reference", type="numpy")
                    aligned = gr.Image(label="Aligned Image", type="numpy")
                
                with gr.Row():
                    seg2 = gr.Image(label="Segmentation Map", type="numpy")
                    ovl2 = gr.Image(label="Overlay Visualization", type="numpy")
                
                status2 = gr.HTML(label="Spatial Alignment Status")
                res2_pre = gr.HTML(label="Pre-alignment Analysis")
                res2_aligned = gr.HTML(label="Aligned Analysis (with borders)")
                save_status2 = gr.HTML(label="Save Status")
                save_btn2 = gr.Button("Save Results", variant="secondary")
                
                # For alignment, we use the progressive display function
                btn2.click(
                    fn=run_alignment_and_segmentation,
                    inputs=[loc2, ref_img, tgt_img],
                    outputs=[orig, aligned, seg2, ovl2, res2_pre, res2_aligned, status2, align_save_state]
                )

                save_btn2.click(
                    fn=save_results,
                    inputs=[align_save_state],
                    outputs=[save_status2]
                )
    
    return demo

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    for path in ["models", "reference_images/MM", "reference_images/SJ"]:
        os.makedirs(path, exist_ok=True)
    
    # Check if model files exist
    for p in MODEL_PATHS.values():
        if not os.path.exists(p):
            print(f"Error: Model file {p} does not exist!")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(inbrowser=True)
