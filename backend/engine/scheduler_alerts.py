"""
Automated Scheduler & Loss-to-Follow-Up (LTFU) Alert Engine
Generates risk-calibrated screening return dates, clinician summaries,
parent-friendly explanations, and simulated SMS/WhatsApp message alerts.
"""

from datetime import datetime, timedelta


class SchedulerAlertsEngine:
    def __init__(self):
        pass

    def generate_schedule_and_alerts(self, patient_id: str, baby_name: str, parent_phone: str,
                                     eval_data: dict, longitudinal_data: dict) -> dict:
        """
        Creates next appointment schedule, notification messages, and bilingual/friendly summaries.
        """
        urgency = eval_data.get("urgency_code", "P3")
        review_hours = eval_data.get("suggested_review_window_hours", 336)
        stage_name = eval_data.get("stage_name", "Stage 1")
        plus_cat = eval_data.get("plus_category", "Normal")
        zone = eval_data.get("zone", "Zone II")

        now = datetime.now()
        next_visit_date = now + timedelta(hours=review_hours)
        formatted_date = next_visit_date.strftime("%A, %d %B %Y (%I:%M %p)")

        # Generate WhatsApp & SMS Alert Payloads
        if urgency in ["P0", "P1"]:
            urgency_banner = "URGENT ROP CLINICAL ALERT"
            sms_text = (
                f"URGENT: NICU Eye Screening Alert for Baby {baby_name} (ID: {patient_id}). "
                f"Retinal exam indicates {eval_data.get('urgency_label')}. "
                f"Immediate follow-up required by {next_visit_date.strftime('%d-%b-%Y')}. "
                f"Please reply YES to confirm attendance or call NICU triage immediately."
            )
        elif urgency == "P2":
            urgency_banner = "PRIORITY ROP RE-EXAM SCHEDULED"
            sms_text = (
                f"NICU Follow-up for Baby {baby_name}: Priority eye check scheduled for "
                f"{next_visit_date.strftime('%d-%b-%Y')}. "
                f"Vascular monitoring required. Reply YES to confirm."
            )
        else:
            urgency_banner = "ROUTINE ROP SCREENING SCHEDULED"
            sms_text = (
                f"Reminder for Baby {baby_name}: Next routine eye screening scheduled for "
                f"{next_visit_date.strftime('%d-%b-%Y')}. Thank you."
            )

        whatsapp_text = (
            f"*{urgency_banner}*\n\n"
            f"• *Patient:* Baby of {baby_name} (ID: {patient_id})\n"
            f"• *Current Finding:* {stage_name} ({zone}, {plus_cat})\n"
            f"• *Urgency:* {eval_data.get('urgency_label')}\n"
            f"• *Next Exam Window:* {formatted_date}\n"
            f"• *Doctor's Instructions:* {eval_data.get('recommended_action')}\n\n"
            f"👉 *Action:* Please confirm receipt and attendance by tapping: https://rop-track.med/confirm/{patient_id}"
        )

        # Parent-Friendly Plain Language Summary
        parent_explanation = self._generate_parent_summary(stage_name, plus_cat, urgency)

        return {
            "scheduled_datetime": next_visit_date.isoformat(),
            "formatted_schedule": formatted_date,
            "window_hours": review_hours,
            "urgency_code": urgency,
            "sms_payload": {
                "recipient": parent_phone,
                "message": sms_text,
                "dispatch_status": "Queued (Simulated)"
            },
            "whatsapp_payload": {
                "recipient": parent_phone,
                "message": whatsapp_text,
                "dispatch_status": "Queued (Simulated)"
            },
            "parent_friendly_explanation": parent_explanation
        }

    def _generate_parent_summary(self, stage_name: str, plus_cat: str, urgency: str) -> str:
        """Translates technical ICROP-3 findings into compassionate, clear language for parents."""
        if urgency in ["P0", "P1"]:
            return (
                "The blood vessels in your baby's eyes are growing unusually and need active medical attention. "
                "The eye doctor needs to examine your baby within the next 48 to 72 hours to protect their vision. "
                "Timely care is very effective when done quickly."
            )
        elif urgency == "P2":
            return (
                "The blood vessels in your baby's retina are still developing and showing mild changes. "
                "No immediate laser treatment is needed right now, but a close re-check is required in a few days "
                "to make sure the vessels continue to heal and grow normally."
            )
        else:
            return (
                "Your baby's retinal blood vessels are still growing normally as expected for their age. "
                "A routine check-up in 1 to 2 weeks is recommended to monitor continuous healthy eye development."
            )
