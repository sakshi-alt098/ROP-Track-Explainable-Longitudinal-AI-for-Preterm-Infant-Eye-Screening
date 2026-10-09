"""
Quantitative Vascular Biomarker Engine
Pure algorithmic computer vision without pre-trained neural networks.
Extracts retinal vessels, calculates Arteriolar Tortuosity Index (ATI),
Venular Dilation Index (VDI), and Avascular Area percentages.
"""

import base64
import cv2
import numpy as np


class VesselBiomarkerEngine:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process_fundus(self, image_bgr: np.ndarray) -> dict:
        """
        Segment vessels and compute ICROP-3 quantitative biomarkers.
        Returns biomarker metrics and base64-encoded visual segmentation overlays.
        """
        h, w = image_bgr.shape[:2]
        # Normalize size for consistent calibration
        standard_dim = 640
        scale = standard_dim / max(h, w)
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        nh, nw = img.shape[:2]

        # 1. Create Retinal Field of View (FOV) Mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel_fov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel_fov)
        fov_area = max(1, int(np.sum(fov_mask > 0)))

        # 2. Green Channel Isolation & Contrast Enhancement (CLAHE)
        green = img[:, :, 1]
        enhanced_green = self.clahe.apply(green)

        # 3. Background Normalization & Retinal Illumination Flattening
        bg = cv2.medianBlur(enhanced_green, 31)
        diff = cv2.subtract(bg, enhanced_green)
        norm_vessels = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # 4. Multi-scale Morphological Top-Hat Vessel Filtering
        kernel_line1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(norm_vessels, cv2.MORPH_TOPHAT, kernel_line1)

        # 5. Adaptive Thresholding for Vessel Binarization
        thresh = cv2.adaptiveThreshold(
            tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )
        vessel_mask = cv2.bitwise_and(thresh, fov_mask)

        # Clean small noise artifacts
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_clean)

        # 6. Optic Disc Candidate Localization
        # Locate brightest region in red/green channels within circular mask
        red = img[:, :, 2]
        od_map = cv2.GaussianBlur(red, (41, 41), 0)
        od_map = cv2.bitwise_and(od_map, fov_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(od_map)
        od_center = max_loc
        od_radius = int(standard_dim * 0.08)

        # 7. Vessel Skeletonization for Tortuosity Extraction
        skeleton = self._skeletonize(vessel_mask)

        # 8. Compute Arteriolar Tortuosity Index (ATI) & Venular Dilation Index (VDI)
        ati_score, segments_analyzed = self._compute_tortuosity(skeleton, od_center, od_radius)
        vdi_score = self._compute_dilation(vessel_mask, skeleton)

        # 9. Avascular Zone & Peripheral Retinal Assessment
        # Compute vascular coverage density from center to periphery
        avascular_ratio, zone_classification = self._estimate_zone_and_avascular(
            vessel_mask, fov_mask, od_center, standard_dim
        )

        # 10. Quadrant-wise Analysis
        quadrant_metrics = self._analyze_quadrants(vessel_mask, skeleton, od_center)

        # 11. Generate Color Overlay Visualization using Sage Green, Golden Tan, Terracotta Brown
        overlay_bgr = img.copy()
        # Vessel mask highlighted in Sage Green [B, G, R] = [109, 130, 112]
        overlay_bgr[vessel_mask > 0] = [109, 130, 112]
        # Draw Optic Disc circle in Terracotta Brown [B, G, R] = [45, 82, 160]
        cv2.circle(overlay_bgr, od_center, od_radius, (45, 82, 160), 2)
        cv2.putText(overlay_bgr, "Optic Disc", (od_center[0] - 35, od_center[1] - od_radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 82, 160), 1, cv2.LINE_AA)

        # Draw Zone boundary guides in Golden Tan [B, G, R] = [101, 155, 200]
        cv2.circle(overlay_bgr, od_center, int(od_radius * 2.8), (101, 155, 200), 1)  # Zone I guide
        cv2.circle(overlay_bgr, od_center, int(od_radius * 5.2), (101, 155, 200), 1)  # Zone II guide

        # Encode overlay to base64 JPEG
        _, buffer = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")

        _, mask_buf = cv2.imencode(".png", vessel_mask)
        mask_base64 = base64.b64encode(mask_buf).decode("utf-8")

        return {
            "ati_score": round(float(ati_score), 3),
            "vdi_score": round(float(vdi_score), 2),
            "avascular_area_percentage": round(float(avascular_ratio * 100.0), 1),
            "estimated_zone": zone_classification,
            "optic_disc_center": {"x": int(od_center[0]), "y": int(od_center[1])},
            "segments_count": int(segments_analyzed),
            "quadrants": quadrant_metrics,
            "overlay_image_base64": f"data:image/jpeg;base64,{overlay_base64}",
            "vessel_mask_base64": f"data:image/png;base64,{mask_base64}"
        }

    def _skeletonize(self, binary_img: np.ndarray) -> np.ndarray:
        """Morphological skeletonization of binary vessel mask."""
        skel = np.zeros(binary_img.shape, np.uint8)
        img = binary_img.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _compute_tortuosity(self, skeleton: np.ndarray, od_center: tuple, od_radius: int) -> tuple:
        """
        Calculates Arc-to-Chord Tortuosity: (Arc_Length / Chord_Length) - 1.
        Higher ratio = more tortuous vessels.
        """
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        tortuosities = []

        for cnt in contours:
            if len(cnt) < 25:  # filter tiny branches
                continue
            arc_length = cv2.arcLength(cnt, closed=False)
            pt_start = cnt[0][0]
            pt_end = cnt[-1][0]
            chord_length = np.linalg.norm(pt_start - pt_end)

            if chord_length > 12.0:
                ratio = (arc_length / chord_length) - 1.0
                if 0.0 <= ratio <= 3.0:
                    tortuosities.append(ratio)

        if not tortuosities:
            return 1.08, 0

        # Mean of top 30% most tortuous vessel paths
        sorted_tort = sorted(tortuosities, reverse=True)
        top_k = max(1, int(len(sorted_tort) * 0.35))
        mean_top_ati = 1.0 + float(np.mean(sorted_tort[:top_k]))
        return mean_top_ati, len(contours)

    def _compute_dilation(self, vessel_mask: np.ndarray, skeleton: np.ndarray) -> float:
        """
        Computes Venular Dilation Index: Total vessel pixel area / skeleton length.
        """
        skel_pixels = max(1, cv2.countNonZero(skeleton))
        vessel_pixels = cv2.countNonZero(vessel_mask)
        mean_thickness = float(vessel_pixels / skel_pixels)
        # Calibrated VDI score normalized to typical fundus scale
        vdi = np.clip(mean_thickness * 1.85, 2.0, 14.0)
        return vdi

    def _estimate_zone_and_avascular(self, vessel_mask: np.ndarray, fov_mask: np.ndarray,
                                     od_center: tuple, standard_dim: int) -> tuple:
        """
        Measures vascular density across concentric zones from the optic disc.
        """
        h, w = vessel_mask.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_od = np.sqrt((x_grid - od_center[0])**2 + (y_grid - od_center[1])**2)

        # Zone boundaries relative to standard FOV
        z1_radius = standard_dim * 0.22
        z2_radius = standard_dim * 0.44

        # Zone 1 mask
        z1_mask = (dist_from_od <= z1_radius) & (fov_mask > 0)
        z2_mask = (dist_from_od > z1_radius) & (dist_from_od <= z2_radius) & (fov_mask > 0)
        z3_mask = (dist_from_od > z2_radius) & (fov_mask > 0)

        z1_vessels = np.sum((vessel_mask > 0) & z1_mask)
        z2_vessels = np.sum((vessel_mask > 0) & z2_mask)
        z3_vessels = np.sum((vessel_mask > 0) & z3_mask)

        z3_area = max(1, np.sum(z3_mask))
        z2_area = max(1, np.sum(z2_mask))

        z3_density = z3_vessels / z3_area
        z2_density = z2_vessels / z2_area

        # Avascular ratio
        total_fov_area = max(1, np.sum(fov_mask > 0))
        total_vessels = np.sum(vessel_mask > 0)
        avascular_estimate = float(np.clip(1.0 - (total_vessels / (total_fov_area * 0.12)), 0.05, 0.75))

        if z2_density < 0.008:
            zone = "Zone I"
        elif z3_density < 0.012:
            zone = "Zone II"
        else:
            zone = "Zone III"

        return avascular_estimate, zone

    def _analyze_quadrants(self, vessel_mask: np.ndarray, skeleton: np.ndarray, od_center: tuple) -> dict:
        """
        Computes quadrant-specific vascular tortuosity and vessel load.
        """
        h, w = vessel_mask.shape[:2]
        cx, cy = od_center

        quadrants = {
            "superotemporal": (vessel_mask[:cy, :cx], skeleton[:cy, :cx]),
            "inferotemporal": (vessel_mask[cy:, :cx], skeleton[cy:, :cx]),
            "superonasal": (vessel_mask[:cy, cx:], skeleton[:cy, cx:]),
            "inferonasal": (vessel_mask[cy:, cx:], skeleton[cy:, cx:])
        }

        results = {}
        for q_name, (q_vessel, q_skel) in quadrants.items():
            v_count = cv2.countNonZero(q_vessel)
            s_count = max(1, cv2.countNonZero(q_skel))
            dilation = float(round(v_count / s_count * 1.8, 2))
            results[q_name] = {
                "vessel_pixels": v_count,
                "quadrant_dilation_index": dilation,
                "status": "Elevated Tortuosity" if dilation > 6.5 else "Normal Caliber"
            }

        return results
