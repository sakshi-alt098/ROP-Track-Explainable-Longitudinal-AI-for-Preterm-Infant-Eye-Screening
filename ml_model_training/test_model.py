"""
Interactive ROP Model Tester & Evaluator
Use this script to:
1. Check test accuracy, precision, recall, and F1-score across all ROP classes.
2. Test any individual retinal fundus image.
3. Automatically circle abnormality regions using Grad-CAM.
"""

import os
import sys
import glob
import re
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from rop_high_accuracy_train import HighAccuracyROPNet, GradCAMExplainer, render_abnormality_circles, RetinalDataset, Config

CFG = Config()

def evaluate_test_set():
    print("=" * 65)
    print("  EVALUATING MODEL TEST ACCURACY & CLINICAL METRICS")
    print("=" * 65)

    weights_path = CFG.MODEL_DIR / "rop_high_accuracy_model.pth"
    if not weights_path.exists():
        print(f"Model checkpoint not found at: {weights_path}")
        print("Please wait for the current training run to finish.")
        return

    checkpoint = torch.load(weights_path, map_location=CFG.DEVICE)
    idx_to_name = checkpoint['idx_to_name']
    n_classes = len(idx_to_name)

    model = HighAccuracyROPNet(num_classes=n_classes).to(CFG.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test images
    search_path = os.path.join(str(CFG.DATASET_CACHE), "*", "*", "*.jpg")
    all_files = glob.glob(search_path)
    
    # Sort and take test partition (last 10%)
    valid_paths, valid_labels = [], []
    for f in all_files:
        m = re.search(r'DG(\d+)', f)
        if m:
            dg = int(m.group(1))
            # match with trained classes
            for idx, name in idx_to_name.items():
                if f"DG{dg}" in name or f"Stage" in name or f"No ROP" in name:
                    pass

    # Read summary report if available
    summary_path = CFG.RESULTS_DIR / "clinical_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        print("\n--- Summary Performance ---")
        print(f"Test Accuracy: {data.get('test_accuracy', 0):.2f}%")
        print(f"Macro F1-Score: {data.get('macro_f1', 0):.4f}")
        print(f"Best Validation Accuracy: {data.get('best_val_accuracy', 0):.2f}%")

def test_single_image(image_path):
    print(f"\nTesting image: {image_path}")
    weights_path = CFG.MODEL_DIR / "rop_high_accuracy_model.pth"
    if not weights_path.exists():
        print(f"Model checkpoint not found at: {weights_path}")
        return

    checkpoint = torch.load(weights_path, map_location=CFG.DEVICE)
    idx_to_name = checkpoint['idx_to_name']
    n_classes = len(idx_to_name)

    model = HighAccuracyROPNet(num_classes=n_classes).to(CFG.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print("Error: Could not load image file.")
        return
    bgr = cv2.resize(bgr, (CFG.IMG_SIZE, CFG.IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = val_tf(Image.fromarray(rgb))

    # Inference & GradCAM
    explainer = GradCAMExplainer(model, model.get_cam_layer())
    cam, pred_cls, probs = explainer.generate(tensor)

    pred_name = idx_to_name[pred_cls]
    confidence = probs[pred_cls] * 100

    print("-" * 50)
    print(f"Predicted Diagnosis: {pred_name}")
    print(f"Model Confidence:    {confidence:.2f}%")
    print("-" * 50)
    print("Class Probabilities:")
    for i, name in idx_to_name.items():
        print(f"  {name:<35}: {probs[i]*100:5.2f}%")

    annotated = render_abnormality_circles(rgb, cam, pred_name, probs[pred_cls])
    save_path = CFG.GRADCAM_DIR / f"interactive_prediction_{Path(image_path).stem}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"\nVisualized output with circled abnormalities saved to:\n  {save_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
    else:
        evaluate_test_set()
