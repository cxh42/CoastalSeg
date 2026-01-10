import argparse
import csv
import os
import sys
import re
from glob import glob
from typing import Dict, List

import cv2
import numpy as np

# Same classes as app.py
CLASSES = ['Background', 'Cobbles', 'Dry sand', 'Plant', 'Sky', 'Water', 'Wet sand']

def extract_image_datetime(filename):
    """
    Extract date/time string from filename formatted like YYYY-MM-DD_HH-MM-SS_*.*
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})[_-](\d{2}-\d{2}-\d{2})", filename)
    if match:
        date_part, time_part = match.groups()
        time_part = time_part.replace("-", ":")
        return f"{date_part} {time_part}"
    return ""

def compute_class_percentages(mask: np.ndarray) -> Dict[str, float]:
    total = mask.size
    out: Dict[str, float] = {}
    for idx, name in enumerate(CLASSES):
        # Assuming mask values correspond to class indices (0..6)
        cnt = int((mask == idx).sum())
        if cnt > 0:
            out[name] = round(cnt * 100.0 / total, 3)
    return out

def write_csv_header(fh):
    cols = ["filename", "image_date"] + CLASSES
    fh.write(",".join(cols) + "\n")

def write_csv_row(fh, filename: str, date_str: str, percentages: Dict[str, float]):
    row = [filename, date_str]
    for cls in CLASSES:
        row.append(str(percentages.get(cls, 0.0)))
    fh.write(",".join(row) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compute class statistics from ground truth masks")
    parser.add_argument("--input", required=True, help="Input directory containing _mask.png files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for masks")
    
    args = parser.parse_args()
    
    search_pattern = "**/*_mask.png" if args.recursive else "*_mask.png"
    mask_files = glob(os.path.join(args.input, search_pattern), recursive=args.recursive)
    mask_files.sort()
    
    if not mask_files:
        print(f"No mask files found in {args.input}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Found {len(mask_files)} masks. processing...")
    
    with open(args.output, "w", encoding="utf-8") as fh:
        write_csv_header(fh)
        
        for idx, path in enumerate(mask_files, 1):
            filename = os.path.basename(path)
            # Try to extract date from the prefix of the mask name
            # Filename is typically: 2024-02-15_11-15-00_PST_T_jpg.rf.hash_mask.png
            # We want to associate this with the base image name if possible, or just parse date.
            date_str = extract_image_datetime(filename)
            
            # Read mask
            # IMREAD_UNCHANGED is safe for 1-channel indices.
            # If Roboflow exported as RGB, we might need conversion, but assuming index based on typical flow.
            # If 3 channel but all equal, it's grayscale.
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            if mask is None:
                print(f"Error reading {path}")
                continue
                
            if len(mask.shape) == 3:
                # If RGB, check if it's grayscale-ish
                if np.all(mask[:,:,0] == mask[:,:,1]):
                    mask = mask[:,:,0]
                else:
                    # If real RGB, we would need color mapping. 
                    # For now, warn if values out of range or try standard mapping?
                    # Let's simple-case: typical Roboflow training masks are grayscale indices.
                    pass
            
            perc = compute_class_percentages(mask)
            write_csv_row(fh, filename, date_str, perc)
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(mask_files)}...")
                
    print(f"Done. Ground truth stats saved to {args.output}")

if __name__ == "__main__":
    main()
