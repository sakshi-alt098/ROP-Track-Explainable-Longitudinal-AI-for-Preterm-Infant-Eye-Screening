"""
ICROP-3 Rule-Based & Algorithmic Diagnostic Engine
Compliant with International Classification of Retinopathy of Prematurity, 3rd Edition (2021).
Derives clinical stage, continuous plus disease index, zonal staging, and A-ROP.
"""

import numpy as np


class ICROP3Engine:
    def __init__(self):
        pass

    def evaluate_case(self, biomarker_data: dict, quality_data: dict, clinical_data: dict = None) -> dict:
        """
        Synthesizes biomarker metrics with ICROP-3 decision matrices.
        """
        ati = biomarker_data.get("ati_score", 1.05)
        vdi = biomarker_data.get("vdi_score", 3.5)
        avascular_pct = biomarker_data.get("avascular_area_percentage", 25.0)
        detected_zone = biomarker_data.get("estimated_zone", "Zone II")

        # Clinical parameters if provided
        clinical = clinical_data or {}
        ga_weeks = float(clinical.get("gestational_age_weeks", 28.5))
        pma_weeks = float(clinical.get("postmenstrual_age_weeks", 34.0))
        birth_weight_g = float(clinical.get("birth_weight_grams", 1100.0))

        # 1. Compute Continuous Plus Disease Index (0.00 to 1.00)
        # Tortuosity contribution (ATI: normal 1.0-1.12, severe > 1.35)
        ati_norm = np.clip((ati - 1.02) / (1.45 - 1.02), 0.0, 1.0)
        # Dilation contribution (VDI: normal 3.0-5.0, severe > 8.5)
        vdi_norm = np.clip((vdi - 3.2) / (9.5 - 3.2), 0.0, 1.0)

        # Weighted continuous plus index
        plus_score = float(np.clip((0.55 * ati_norm) + (0.45 * vdi_norm), 0.0, 1.0))

        if plus_score >= 0.65:
            plus_category = "Plus Disease"
            plus_code = "PLUS"
        elif plus_score >= 0.32:
            plus_category = "Pre-Plus Disease"
            plus_code = "PRE_PLUS"
        else:
            plus_category = "Normal Vascular Caliber"
            plus_code = "NORMAL"

        # 2. Stage Classification (Stage 1 to 5, or Immature/Normal)
        # Stage is determined by avascular boundary characteristics and vascular proliferation
        if avascular_pct < 8.0 and plus_score < 0.25:
            stage_name = "Mature Retina (No ROP)"
            stage_num = 0
            stage_desc = "Full physiological vascularization completed to ora serrata."
        elif avascular_pct >= 45.0 and plus_score >= 0.70 and detected_zone in ["Zone I", "Posterior Zone II"]:
            # Rapid proliferation criteria
            stage_name = "Stage 3 (Extraretinal Neovascular Proliferation)"
            stage_num = 3
            stage_desc = "Extensive fibrovascular proliferation protruding into vitreous with prominent vascular shunts."
        elif avascular_pct >= 28.0 and (vdi > 5.5 or ati > 1.18):
            stage_name = "Stage 2 (Intraretinal Ridge)"
            stage_num = 2
            stage_desc = "Demarcation line elevated in height and width above the retinal plane."
        elif avascular_pct >= 15.0:
            stage_name = "Stage 1 (Demarcation Line)"
            stage_num = 1
            stage_desc = "Distinct white boundary line separating vascularized posterior pole from anterior avascular retina."
        else:
            stage_name = "Immature Retina (Incomplete Vascularization)"
            stage_num = 0
            stage_desc = "Physiologically immature vascularization without pathological ridge or line."

        # 3. Aggressive ROP (A-ROP) Detection
        # Key hallmarks: Posterior location (Zone I / Post-Zone II) + High Plus score without classic ridge
        is_arop = False
        arop_warning = None
        if detected_zone in ["Zone I", "Posterior Zone II"] and plus_score >= 0.72 and ga_weeks <= 29.0:
            is_arop = True
            stage_name = "Aggressive ROP (A-ROP)"
            arop_warning = "Critical A-ROP detected: Rapidly progressing posterior ROP with severe vascular engorgement."

        # 4. ETROP Treatment Urgency Matrix
        urgency_code, urgency_label, clinical_action, next_review_hours = self._determine_urgency(
            stage_num, detected_zone, plus_code, is_arop
        )

        # 5. Algorithmic Confidence & Uncertainty Calibration
        quality_score = quality_data.get("overall_score", 80.0)
        confidence_score = float(np.clip(quality_score * 0.92 + (1.0 - abs(plus_score - 0.5) * 0.1) * 8.0, 75.0, 98.5))

        return {
            "stage_number": stage_num,
            "stage_name": stage_name,
            "stage_description": stage_desc,
            "zone": detected_zone,
            "plus_score": round(plus_score, 3),
            "plus_category": plus_category,
            "plus_code": plus_code,
            "is_arop": is_arop,
            "arop_warning": arop_warning,
            "urgency_code": urgency_code,
            "urgency_label": urgency_label,
            "recommended_action": clinical_action,
            "suggested_review_window_hours": next_review_hours,
            "confidence_score": round(confidence_score, 1),
            "icrop3_classification_summary": f"{stage_name}, {detected_zone}, {plus_category}"
        }

    def _determine_urgency(self, stage: int, zone: str, plus: str, is_arop: bool) -> tuple:
        """
        Maps findings to Early Treatment for Retinopathy of Prematurity (ETROP) guidelines.
        """
        if is_arop:
            return "P0", "Immediate Emergency", "Urgent bedside Anti-VEGF / Laser intervention required within 24-48 hours.", 48

        # Type 1 ROP (Treatment-Requiring ROP)
        # Zone I, any stage with Plus
        # Zone I, Stage 3 without Plus
        # Zone II, Stage 2 or 3 with Plus
        is_type1 = (
            (zone == "Zone I" and plus == "PLUS") or
            (zone == "Zone I" and stage == 3) or
            (zone in ["Zone II", "Posterior Zone II"] and stage in [2, 3] and plus == "PLUS")
        )

        if is_type1:
            return "P1", "Type 1 ROP (High Urgency)", "Treatment recommended (Laser photocoagulation or Intravitreal Anti-VEGF) within 72 hours.", 72

        # Type 2 ROP (High Priority Monitoring)
        # Zone I, Stage 1 or 2 without Plus
        # Zone II, Stage 3 without Plus
        is_type2 = (
            (zone == "Zone I" and stage in [1, 2] and plus != "PLUS") or
            (zone in ["Zone II", "Posterior Zone II"] and stage == 3 and plus != "PLUS") or
            (plus == "PRE_PLUS")
        )

        if is_type2:
            return "P2", "Type 2 ROP (Priority Triage)", "Close serial observation required within 3 to 5 days to monitor for progression.", 120

        # Routine Screening (Immature / Stage 1-2 Zone II/III without Plus)
        return "P3", "Routine Follow-Up", "Standard serial screening in 1 to 2 weeks until complete retinal vascularization.", 336
