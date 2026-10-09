"""
Multimodal Longitudinal Progression & Systemic Risk Engine
Combines image biomarkers with neonatal clinical risk factors (WINROP / CHOP-ROP principles).
Forecasts progression trajectory and risk of treatment-requiring ROP (TR-ROP).
"""

import numpy as np


class LongitudinalProgressionEngine:
    def __init__(self):
        pass

    def compute_progression(self, current_eval: dict, clinical_data: dict, visit_history: list = None) -> dict:
        """
        Synthesizes current visit findings, neonatal clinical risk factors, and prior visit history.
        """
        # Extract clinical parameters
        ga = float(clinical_data.get("gestational_age_weeks", 28.0))
        bw = float(clinical_data.get("birth_weight_grams", 1050.0))
        pma = float(clinical_data.get("postmenstrual_age_weeks", 33.5))
        weight_gain_rate = float(clinical_data.get("weight_gain_g_per_day", 12.0))
        o2_days = float(clinical_data.get("supplemental_o2_days", 14.0))

        # 1. Baseline Systemic Neonatal Risk Score (WINROP / CHOP-ROP inspired)
        # Low birth weight & lower GA increase systemic baseline risk
        ga_risk = np.clip((32.0 - ga) / 8.0, 0.0, 1.0) * 0.35
        bw_risk = np.clip((1500.0 - bw) / 1000.0, 0.0, 1.0) * 0.30
        
        # Suboptimal weight gain (<15g/day is a major signal in WINROP)
        wg_risk = np.clip((18.0 - weight_gain_rate) / 15.0, 0.0, 1.0) * 0.20
        
        # Prolonged supplemental oxygen duration
        o2_risk = np.clip(o2_days / 30.0, 0.0, 1.0) * 0.15

        systemic_risk_score = float(np.clip(ga_risk + bw_risk + wg_risk + o2_risk, 0.05, 0.98))

        # 2. Visit-to-Visit Delta Tracking (if prior visits exist)
        current_plus = current_eval.get("plus_score", 0.3)
        current_stage = current_eval.get("stage_number", 1)
        history = visit_history or []

        delta_plus = 0.0
        delta_stage = 0
        progression_velocity = "Stable"
        prev_pma = None

        if len(history) > 0:
            last_visit = history[-1]
            last_plus = float(last_visit.get("plus_score", current_plus))
            last_stage = int(last_visit.get("stage_number", current_stage))
            prev_pma = float(last_visit.get("pma_weeks", pma - 1.0))
            weeks_elapsed = max(0.5, pma - prev_pma)

            delta_plus = round((current_plus - last_plus) / weeks_elapsed, 3)
            delta_stage = current_stage - last_stage

            if delta_plus > 0.18 or delta_stage >= 1:
                progression_velocity = "Rapid Progression"
            elif delta_plus > 0.05:
                progression_velocity = "Slow Progression"
            elif delta_plus < -0.05 or delta_stage < 0:
                progression_velocity = "Regressing / Maturing"
            else:
                progression_velocity = "Stable"

        # 3. Overall 14-Day Treatment-Requiring ROP (TR-ROP) Risk Index
        # Combined weighted formula: 50% Image Plus/Stage + 35% Systemic Risk + 15% Progression Delta
        delta_risk_component = 0.0
        if progression_velocity == "Rapid Progression":
            delta_risk_component = 0.20
        elif progression_velocity == "Regressing / Maturing":
            delta_risk_component = -0.15

        tr_rop_risk = float(np.clip(
            (0.40 * current_plus) +
            (0.15 * (current_stage / 4.0)) +
            (0.35 * systemic_risk_score) +
            delta_risk_component,
            0.02, 0.99
        ))

        # 4. Generate 14-Day Trajectory Curve (Days 0, 2, 4, 7, 10, 14)
        days = [0, 2, 4, 7, 10, 14]
        risk_curve = []
        base_val = tr_rop_risk * 100.0

        growth_rate = 0.05
        if progression_velocity == "Rapid Progression":
            growth_rate = 0.12
        elif progression_velocity == "Regressing / Maturing":
            growth_rate = -0.08

        for d in days:
            projected = np.clip(base_val * np.exp(growth_rate * (d / 7.0)), 1.0, 99.0)
            risk_curve.append({
                "day": d,
                "projected_risk_percentage": round(float(projected), 1)
            })

        return {
            "systemic_risk_score": round(systemic_risk_score * 100.0, 1),
            "tr_rop_risk_percentage": round(tr_rop_risk * 100.0, 1),
            "progression_velocity": progression_velocity,
            "delta_plus_per_week": delta_plus,
            "delta_stage": delta_stage,
            "trajectory_forecast_14d": risk_curve,
            "risk_strata": "Critical" if tr_rop_risk >= 0.70 else ("Moderate" if tr_rop_risk >= 0.35 else "Low")
        }
