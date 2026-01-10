import cv2
import numpy as np
import sys
import os

def inspect(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    # Read as is (unchanged) to detect if it's grayscale or color
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read image")
        return
        
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    
    unique_vals = np.unique(img)
    print(f"Unique values: {unique_vals}")
    
    if len(img.shape) == 3:
        # Check if it's actually grayscale saved as RGB (all channels identical)
        if np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,1] == img[:,:,2]):
             print("It is a grayscale image saved as 3-channel.")
        else:
             print("It is a color image.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        # Fallback to a hardcoded known path for testing if no arg provided
        # (User will provide one or I will run it with one)
        print("Please provide a mask path.")
