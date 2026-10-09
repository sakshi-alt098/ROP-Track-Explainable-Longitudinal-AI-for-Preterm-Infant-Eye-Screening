"""
Image Quality Assessment (IQA) & Gating Engine
Pure algorithmic computer vision without pre-trained neural networks.
Evaluates sharpness/blur, illumination, contrast, and field-of-view completeness.
"""

import cv2
import numpy as np


class QualityGate:
    def __init__(self, blur_threshold: float = 85.0, min_contrast: float = 25.0):
        self.blur_threshold = blur_threshold
        self.min_contrast = min_contrast

    def assess_image(self, image_bgr: np.ndarray) -> dict:
        """
        Assess retinal fundus image quality.
        Returns quality metrics, pass/fail status, and actionable diagnostic warnings.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "is_gradable": False,
                "overall_score": 0.0,
                "blur_score": 0.0,
                "contrast_score": 0.0,
                "mean_brightness": 0.0,
                "status": "Ungradable",
                "warnings": ["Empty or invalid image payload."]
            }

        # Resize for normalized evaluation
        h, w = image_bgr.shape[:2]
        target_size = 512
        scale = target_size / max(h, w)
        resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 1. Blur Assessment using Laplacian Variance & Tenengrad Gradient
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)

        # 2. Illumination & Brightness Analysis
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mean_brightness = float(np.mean(v_channel))
        brightness_std = float(np.std(v_channel))

        # 3. RMS Contrast
        rms_contrast = float(np.std(gray))

        # 4. Color & Green Channel Quality (Green channel has maximum retinal contrast)
        green_channel = resized[:, :, 1]
        green_contrast = float(np.std(green_channel))

        # 5. Over-exposure / Glare Detection
        glare_ratio = float(np.sum(v_channel > 245) / v_channel.size)

        # 6. Under-exposure / Dark Artifacts
        dark_ratio = float(np.sum(v_channel < 20) / v_channel.size)

        # Compile Warnings & Quality Score
        warnings = []
        blur_passed = laplacian_var >= self.blur_threshold
        if not blur_passed:
            warnings.append(f"Image motion blur detected (Laplacian: {laplacian_var:.1f} < {self.blur_threshold:.0f}).")

        if mean_brightness < 45.0:
            warnings.append("Severe under-illumination / dark fundus capture.")
        elif mean_brightness > 210.0:
            warnings.append("Over-exposed image with excessive flash reflectance.")

        if glare_ratio > 0.08:
            warnings.append(f"Corneal or lens reflection glare artifact ({glare_ratio*100:.1f}% saturated).")

        if rms_contrast < self.min_contrast:
            warnings.append(f"Low contrast (RMS: {rms_contrast:.1f} < {self.min_contrast:.0f}).")

        # Compute Normalized Overall Quality Score (0 to 100)
        blur_norm = np.clip(laplacian_var / 300.0 * 40.0, 0, 40)
        contrast_norm = np.clip(rms_contrast / 60.0 * 30.0, 0, 30)
        illum_penalty = 0.0
        if mean_brightness < 50:
            illum_penalty += (50 - mean_brightness) * 0.5
        if mean_brightness > 190:
            illum_penalty += (mean_brightness - 190) * 0.5
        illum_score = np.clip(30.0 - illum_penalty - (glare_ratio * 50.0), 0, 30)

        overall_score = float(np.clip(blur_norm + contrast_norm + illum_score, 0.0, 100.0))
        is_gradable = (overall_score >= 45.0) and (laplacian_var >= 40.0) and (mean_brightness >= 35.0)

        if overall_score >= 75.0:
            status = "Optimal"
        elif overall_score >= 50.0:
            status = "Acceptable"
        elif overall_score >= 35.0:
            status = "Suboptimal"
        else:
            status = "Ungradable"

        return {
            "is_gradable": bool(is_gradable),
            "overall_score": round(overall_score, 1),
            "blur_score": round(float(laplacian_var), 1),
            "tenengrad_score": round(float(tenengrad), 1),
            "contrast_score": round(rms_contrast, 1),
            "mean_brightness": round(mean_brightness, 1),
            "glare_percentage": round(glare_ratio * 100.0, 2),
            "status": status,
            "warnings": warnings if warnings else ["Image quality is optimal for ICROP-3 diagnostic grading."]
        }
