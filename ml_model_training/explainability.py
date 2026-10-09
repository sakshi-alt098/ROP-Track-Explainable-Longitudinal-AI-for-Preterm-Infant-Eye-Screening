"""
Module 4: Explainability & Abnormality Detection with Grad-CAM and Circle Highlighting
"""

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class GradCAMExplainer:
    def __init__(self, model, target_layer, device="cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._fhook)
        target_layer.register_full_backward_hook(self._bhook)

    def _fhook(self, m, i, o): self.activations = o.detach()
    def _bhook(self, m, gi, go): self.gradients = go[0].detach()

    def generate(self, img_tensor, target_class=None):
        self.model.eval()
        inp = img_tensor.unsqueeze(0).to(self.device)
        inp.requires_grad = True

        out = self.model(inp)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        self.model.zero_grad()
        out[0, target_class].backward()

        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam, target_class, F.softmax(out, dim=1)[0].detach().cpu().numpy()

def render_abnormality_circles(orig_rgb, cam_map, class_name, confidence):
    """
    Renders Grad-CAM heatmaps and draws bounding circles around detected abnormalities.
    """
    h, w = orig_rgb.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig_rgb, 0.6, heatmap, 0.4, 0)

    # Threshold highest activation regions to locate clinical pathology
    thresh = (cam_resized > 0.62).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_count = 0
    circle_color = (255, 60, 60) if "No ROP" not in class_name else (60, 255, 60)

    for c in contours:
        if cv2.contourArea(c) > 70:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            r = max(int(radius), 12)
            cv2.circle(blended, center, r, circle_color, 2)
            cv2.circle(blended, center, 3, circle_color, -1)
            cv2.putText(blended, f"Abn-{circle_count+1}", (center[0] + r + 4, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, circle_color, 1)
            circle_count += 1

    banner = np.zeros((50, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    cv2.putText(banner, f"{class_name} ({confidence*100:.1f}%)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(banner, f"Abnormalities Circled: {circle_count}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, circle_color, 1)

    return np.vstack([banner, blended])
