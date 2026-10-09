"""
Module 6: Single-Image Inference CLI Tool with Automated Abnormality Circle Visualizer
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .config import cfg
from .model import ROPCustomCNN
from .explainability import GradCAMExplainer, render_abnormality_circles

def test_image(image_path):
    print("=" * 60)
    print(f"  ROP EYE SCREENING INFERENCE & ABNORMALITY LOCALIZATION")
    print("=" * 60)
    print(f"Processing image: {image_path}")

    weights_path = cfg.MODEL_DIR / "rop_custom_cnn_best.pth"
    if not weights_path.exists():
        # Fallback to high accuracy checkpoint
        weights_path = Path(r"C:\Users\LOQ\OneDrive\Desktop\synapse model\rop_outputs\models\rop_high_accuracy_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    idx_to_name = checkpoint['idx_to_name']

    model = ROPCustomCNN(num_classes=len(idx_to_name)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print("Error: Could not read image.")
        return

    bgr = cv2.resize(bgr, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = val_tf(Image.fromarray(rgb))

    explainer = GradCAMExplainer(model, model.get_cam_target_layer(), device=device)
    cam, pred_cls, probs = explainer.generate(tensor)

    pred_name = idx_to_name[pred_cls]
    conf = probs[pred_cls] * 100

    print("-" * 50)
    print(f"Predicted Diagnosis: {pred_name}")
    print(f"Model Confidence:    {conf:.2f}%")
    print("-" * 50)

    annotated = render_abnormality_circles(rgb, cam, pred_name, probs[pred_cls])
    save_path = cfg.GRADCAM_DIR / f"prediction_{Path(image_path).stem}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"Result with circled abnormalities saved to:\n  {save_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        print("Usage: python -m ml_model_training.inference <image_path>")
