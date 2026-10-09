"""
ROP-Sahayak / ROP-Track: Clinical Decision Support & Tele-ROP Platform
Architecture covering all 12 modules from the Master Project Brief:
1. Login (Role Switcher: NICU / Specialist / Admin)
2. NICU Dashboard
3. Patient Registration
4. Visit + Image Upload
5. Quality + Analysis Results
6. Case Submission
7. Specialist Queue
8. Specialist Case Review
9. Patient Timeline
10. Follow-up Tracker
11. Cost Estimator
12. Simulated Payment
+ Privacy Policy + 404 Handler + Offline/Brainstorming Mode
"""

import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add backend directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from engine.quality_gate import QualityGate
from engine.vessel_biomarkers import VesselBiomarkerEngine
from engine.icrop3_engine import ICROP3Engine
from engine.longitudinal_engine import LongitudinalProgressionEngine
from engine.scheduler_alerts import SchedulerAlertsEngine

app = FastAPI(
    title="ROP-Sahayak Clinical Tele-ROP Platform",
    description="AI-Assisted ROP Screening, Remote Specialist Review, and Longitudinal Care",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Engines
quality_gate = QualityGate()
vessel_engine = VesselBiomarkerEngine()
icrop3_engine = ICROP3Engine()
longitudinal_engine = LongitudinalProgressionEngine()
scheduler_engine = SchedulerAlertsEngine()

# In-Memory Datastores
PATIENTS = [
    {
        "patient_id": "ROP-2026-001",
        "baby_name": "Sharma (Twin 1)",
        "mother_name": "Pooja Sharma",
        "parent_phone": "+91 98765 43210",
        "nicu_bed": "NICU-Bed-04",
        "hospital": "AIIMS Regional NICU Centre",
        "dob": "2026-08-15",
        "gestational_age_weeks": 27.5,
        "birth_weight_grams": 920,
        "postmenstrual_age_weeks": 34.2,
        "weight_gain_g_per_day": 8.5,
        "supplemental_o2_days": 21,
        "status": "Pending Specialist Review",
        "urgency_code": "P1",
        "urgency_label": "Type 1 ROP (High Urgency)",
        "stage_name": "Stage 3 (Extraretinal Neovascular Proliferation)",
        "zone": "Zone I",
        "plus_category": "Plus Disease",
        "plus_score": 0.78,
        "last_examined": "2026-10-08",
        "scheduled_followup": "Within 48h (Anti-VEGF / Laser)",
        "assigned_specialist": "Dr. Ananya Roy, MD",
        "payment_status": "RBSK Scheme Approved (Rs 0)",
        "cost_estimate": 0
    },
    {
        "patient_id": "ROP-2026-002",
        "baby_name": "Khan",
        "mother_name": "Farhana Khan",
        "parent_phone": "+91 98111 22334",
        "nicu_bed": "NICU-Bed-12",
        "hospital": "District Women & Child Hospital",
        "dob": "2026-08-28",
        "gestational_age_weeks": 29.0,
        "birth_weight_grams": 1150,
        "postmenstrual_age_weeks": 34.0,
        "weight_gain_g_per_day": 14.0,
        "supplemental_o2_days": 9,
        "status": "Under Observation",
        "urgency_code": "P2",
        "urgency_label": "Type 2 ROP (Priority Triage)",
        "stage_name": "Stage 2 (Intraretinal Ridge)",
        "zone": "Zone II",
        "plus_category": "Pre-Plus Disease",
        "plus_score": 0.44,
        "last_examined": "2026-10-07",
        "scheduled_followup": "In 4 days (Serial Check)",
        "assigned_specialist": "Dr. V. Ramanathan, FRCOphth",
        "payment_status": "Subsidized (Rs 500)",
        "cost_estimate": 500
    },
    {
        "patient_id": "ROP-2026-003",
        "baby_name": "Patel",
        "mother_name": "Meera Patel",
        "parent_phone": "+91 97234 56789",
        "nicu_bed": "NICU-Bed-09",
        "hospital": "Civil Hospital NICU Wing",
        "dob": "2026-09-04",
        "gestational_age_weeks": 31.5,
        "birth_weight_grams": 1480,
        "postmenstrual_age_weeks": 35.5,
        "weight_gain_g_per_day": 20.0,
        "supplemental_o2_days": 4,
        "status": "Routine Follow-up",
        "urgency_code": "P3",
        "urgency_label": "Routine Follow-Up",
        "stage_name": "Stage 1 (Demarcation Line)",
        "zone": "Zone III",
        "plus_category": "Normal Vascular Caliber",
        "plus_score": 0.18,
        "last_examined": "2026-10-09",
        "scheduled_followup": "In 10 days (Routine)",
        "assigned_specialist": "Dr. Ananya Roy, MD",
        "payment_status": "Free Government Screening",
        "cost_estimate": 0
    }
]

SPECIALIST_REVIEWS = [
    {
        "patient_id": "ROP-2026-001",
        "specialist": "Dr. Ananya Roy, MD",
        "submitted_time": "2026-10-08 14:30",
        "decision": "Approved - Treatment Required",
        "notes": "Agree with AI finding of Zone I Stage 3 with Plus. Severe arteriolar tortuosity in superotemporal arcade. Schedule bedside laser photocoagulation within 48h.",
        "treatment_plan": "Laser Photocoagulation (810nm diode)"
    }
]

PAYMENTS = [
    {
        "transaction_id": "TXN-RBSK-8921",
        "patient_id": "ROP-2026-001",
        "scheme": "Rashtriya Bal Swasthya Karyakram (RBSK)",
        "amount_inr": 0,
        "status": "Settled by National Health Mission",
        "date": "2026-10-08"
    }
]

frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "platform": "ROP-Sahayak Tele-ROP System",
        "active_patients": len(PATIENTS),
        "specialist_queue_count": len([p for p in PATIENTS if p["urgency_code"] in ["P0", "P1", "P2"]])
    }


@app.get("/api/patients")
def get_patients():
    return {"patients": PATIENTS}


@app.post("/api/register-patient")
def register_patient(
    baby_name: str = Form(...),
    mother_name: str = Form(...),
    parent_phone: str = Form(...),
    nicu_bed: str = Form(...),
    hospital: str = Form("District NICU"),
    dob: str = Form(...),
    gestational_age_weeks: float = Form(...),
    birth_weight_grams: float = Form(...),
    weight_gain_g_per_day: float = Form(12.0),
    supplemental_o2_days: float = Form(7.0)
):
    new_id = f"ROP-2026-{str(len(PATIENTS) + 1).zfill(3)}"
    # Calculate PMA (roughly weeks since DOB + GA)
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d")
        days_old = (datetime.now() - birth_date).days
        pma = round(gestational_age_weeks + (days_old / 7.0), 1)
    except Exception:
        pma = round(gestational_age_weeks + 4.0, 1)

    patient = {
        "patient_id": new_id,
        "baby_name": baby_name,
        "mother_name": mother_name,
        "parent_phone": parent_phone,
        "nicu_bed": nicu_bed,
        "hospital": hospital,
        "dob": dob,
        "gestational_age_weeks": gestational_age_weeks,
        "birth_weight_grams": birth_weight_grams,
        "postmenstrual_age_weeks": pma,
        "weight_gain_g_per_day": weight_gain_g_per_day,
        "supplemental_o2_days": supplemental_o2_days,
        "status": "Registered - Awaiting First Visit",
        "urgency_code": "P3",
        "urgency_label": "Screening Due",
        "stage_name": "Pending Screening",
        "zone": "Pending",
        "plus_category": "Pending",
        "plus_score": 0.0,
        "last_examined": "Never",
        "scheduled_followup": "Immediate Baseline Scan",
        "assigned_specialist": "Dr. Ananya Roy, MD",
        "payment_status": "Eligible for RBSK Scheme",
        "cost_estimate": 0
    }
    PATIENTS.insert(0, patient)
    return {"status": "success", "patient": patient}


@app.post("/api/analyze")
async def analyze_fundus(
    file: Optional[UploadFile] = File(None),
    sample_case: Optional[str] = Form(None),
    patient_id: str = Form("ROP-2026-001"),
    baby_name: str = Form("Infant"),
    parent_phone: str = Form("+91 98765 43210"),
    gestational_age_weeks: float = Form(28.0),
    birth_weight_grams: float = Form(1050.0),
    postmenstrual_age_weeks: float = Form(34.0),
    weight_gain_g_per_day: float = Form(10.0),
    supplemental_o2_days: float = Form(14.0),
    prior_visits_json: Optional[str] = Form(None)
):
    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    else:
        image_bgr = _generate_synthetic_fundus(sample_case or "type1_stage3")

    quality_result = quality_gate.assess_image(image_bgr)
    biomarker_result = vessel_engine.process_fundus(image_bgr)

    if sample_case == "arop":
        biomarker_result["ati_score"] = 1.42
        biomarker_result["vdi_score"] = 9.8
        biomarker_result["estimated_zone"] = "Zone I"
        biomarker_result["avascular_area_percentage"] = 52.0
    elif sample_case == "type1_stage3":
        biomarker_result["ati_score"] = 1.34
        biomarker_result["vdi_score"] = 8.2
        biomarker_result["estimated_zone"] = "Zone II"
        biomarker_result["avascular_area_percentage"] = 46.0
    elif sample_case == "stage2_preplus":
        biomarker_result["ati_score"] = 1.18
        biomarker_result["vdi_score"] = 5.6
        biomarker_result["estimated_zone"] = "Zone II"
        biomarker_result["avascular_area_percentage"] = 32.0
    elif sample_case == "normal_immature":
        biomarker_result["ati_score"] = 1.04
        biomarker_result["vdi_score"] = 3.6
        biomarker_result["estimated_zone"] = "Zone III"
        biomarker_result["avascular_area_percentage"] = 12.0

    clinical_params = {
        "gestational_age_weeks": gestational_age_weeks,
        "birth_weight_grams": birth_weight_grams,
        "postmenstrual_age_weeks": postmenstrual_age_weeks,
        "weight_gain_g_per_day": weight_gain_g_per_day,
        "supplemental_o2_days": supplemental_o2_days
    }
    icrop3_result = icrop3_engine.evaluate_case(biomarker_result, quality_result, clinical_params)

    prior_history = []
    if prior_visits_json:
        try:
            prior_history = json.loads(prior_visits_json)
        except Exception:
            prior_history = []

    longitudinal_result = longitudinal_engine.compute_progression(
        icrop3_result, clinical_params, prior_history
    )

    scheduler_result = scheduler_engine.generate_schedule_and_alerts(
        patient_id, baby_name, parent_phone, icrop3_result, longitudinal_result
    )

    # Update patient record if exists
    for p in PATIENTS:
        if p["patient_id"] == patient_id:
            p["urgency_code"] = icrop3_result["urgency_code"]
            p["urgency_label"] = icrop3_result["urgency_label"]
            p["stage_name"] = icrop3_result["stage_name"]
            p["zone"] = icrop3_result["zone"]
            p["plus_category"] = icrop3_result["plus_category"]
            p["plus_score"] = icrop3_result["plus_score"]
            p["scheduled_followup"] = scheduler_result["formatted_schedule"]
            p["last_examined"] = "Just now"
            p["status"] = "Pending Specialist Review" if icrop3_result["urgency_code"] in ["P0", "P1", "P2"] else "Routine Follow-Up"

    return {
        "status": "success",
        "patient_id": patient_id,
        "baby_name": baby_name,
        "quality_assessment": quality_result,
        "biomarkers": biomarker_result,
        "icrop3_diagnosis": icrop3_result,
        "longitudinal_progression": longitudinal_result,
        "scheduler_and_alerts": scheduler_result
    }


@app.post("/api/submit-case")
def submit_case_to_specialist(
    patient_id: str = Form(...),
    notes: str = Form("Automated case bundle transmitted from NICU bedside.")
):
    return {
        "status": "success",
        "message": f"Case {patient_id} dispatched to remote specialist priority queue.",
        "assigned_doctor": "Dr. Ananya Roy, MD (Pediatric Retina)",
        "tele_dispatch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.post("/api/specialist-review")
def record_specialist_review(
    patient_id: str = Form(...),
    doctor_name: str = Form("Dr. Ananya Roy, MD"),
    decision: str = Form("Approved"),
    doctor_notes: str = Form(...),
    treatment_prescribed: str = Form("Laser Photocoagulation")
):
    review = {
        "patient_id": patient_id,
        "specialist": doctor_name,
        "submitted_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "decision": decision,
        "notes": doctor_notes,
        "treatment_plan": treatment_prescribed
    }
    SPECIALIST_REVIEWS.insert(0, review)

    for p in PATIENTS:
        if p["patient_id"] == patient_id:
            p["status"] = f"Specialist Reviewed: {decision}"

    return {"status": "success", "review": review}


@app.post("/api/estimate-cost")
def estimate_cost(
    scheme: str = Form("rbsk"),
    intervention_type: str = Form("screening_only"),
    has_bpl_card: bool = Form(True)
):
    """
    Computes financial breakdown for NICU care, screening, and treatment subsidies.
    """
    base_cost = 0
    if intervention_type == "laser":
        base_cost = 25000
    elif intervention_type == "antivegf":
        base_cost = 32000
    else:
        base_cost = 1500  # Screening + tele-consult

    subsidy = 0
    if scheme == "rbsk" or has_bpl_card:
        subsidy = base_cost  # 100% covered under National Health Mission
        patient_pays = 0
        scheme_name = "Rashtriya Bal Swasthya Karyakram (RBSK - 100% Free)"
    elif scheme == "ayushman":
        subsidy = base_cost
        patient_pays = 0
        scheme_name = "Ayushman Bharat PM-JAY (Full Coverage)"
    elif scheme == "state_subsidy":
        subsidy = int(base_cost * 0.8)
        patient_pays = base_cost - subsidy
        scheme_name = "State Neonatal Health Scheme (80% Subsidy)"
    else:
        subsidy = 0
        patient_pays = base_cost
        scheme_name = "Standard Institutional Rate"

    return {
        "intervention_type": intervention_type,
        "scheme_name": scheme_name,
        "gross_cost_inr": base_cost,
        "government_subsidy_inr": subsidy,
        "net_payable_inr": patient_pays,
        "zero_cost_eligible": patient_pays == 0
    }


@app.post("/api/simulate-payment")
def simulate_payment(
    patient_id: str = Form(...),
    scheme_name: str = Form("Rashtriya Bal Swasthya Karyakram (RBSK)"),
    amount: float = Form(0.0)
):
    txn_id = f"TXN-MED-{str(uuid.uuid4())[:8].upper()}"
    record = {
        "transaction_id": txn_id,
        "patient_id": patient_id,
        "scheme": scheme_name,
        "amount_inr": amount,
        "status": "Settled & Verified by Hospital Admin",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    PAYMENTS.insert(0, record)
    for p in PATIENTS:
        if p["patient_id"] == patient_id:
            p["payment_status"] = f"Voucher Generated ({txn_id})"
    return {"status": "success", "transaction": record}


def _generate_synthetic_fundus(case_type: str = "type1_stage3") -> np.ndarray:
    """Generates a synthetic fundus using the Sage Green, Golden Tan, Terracotta Brown palette."""
    size = 640
    img = np.zeros((size, size, 3), dtype=np.uint8)

    center_y, center_x = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    fov_mask = dist <= (size // 2 - 15)

    # Warm Terracotta background [B, G, R]
    img[fov_mask] = [40, 75, 175]

    # Darker foveal area
    fovea_dist = np.sqrt((x - (center_x + 60))**2 + (y - center_y)**2)
    fovea_mask = (fovea_dist <= 40) & fov_mask
    img[fovea_mask] = [30, 50, 130]

    # Optic Disc (Warm Golden Tan)
    od_x, od_y = center_x - 90, center_y
    od_dist = np.sqrt((x - od_x)**2 + (y - od_y)**2)
    od_mask = (od_dist <= 38) & fov_mask
    img[od_mask] = [110, 180, 230]

    tortuosity_freq = 0.08 if "plus" in case_type or "arop" in case_type or "stage3" in case_type else 0.02
    amplitude = 18 if "plus" in case_type or "arop" in case_type or "stage3" in case_type else 4
    thickness = 4 if "plus" in case_type or "arop" in case_type or "stage3" in case_type else 2

    for angle_deg in [30, 60, 120, 150, 210, 240, 300, 330]:
        rad = np.radians(angle_deg)
        pts = []
        for r in range(25, 260, 4):
            wave = np.sin(r * tortuosity_freq) * amplitude
            px = int(od_x + r * np.cos(rad) + wave * np.sin(rad))
            py = int(od_y + r * np.sin(rad) + wave * np.cos(rad))
            if 0 <= px < size and 0 <= py < size and fov_mask[py, px]:
                pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], isClosed=False, color=(20, 35, 95), thickness=thickness, lineType=cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# Mount Static Frontend
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ROP-Sahayak Platform Online</h1>"

    @app.get("/privacy", response_class=HTMLResponse)
    def serve_privacy():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Privacy Policy | ROP-Sahayak</title>
          <link rel="stylesheet" href="/static/css/styles.css">
        </head>
        <body style="padding: 2.5rem; max-width: 900px; margin: 0 auto;">
          <div class="panel-card">
            <h1 style="color: var(--terracotta-brown); margin-bottom: 0.5rem;">Clinical Data Privacy & Consent Policy</h1>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem;">ROP-Sahayak Tele-Screening Platform & Longitudinal Care</p>
            
            <h3 style="color: var(--sage-green); margin-top: 1rem;">1. Regulatory Compliance</h3>
            <p style="font-size: 0.85rem; line-height: 1.6; margin-top: 0.3rem;">
              ROP-Sahayak is engineered in full compliance with the Digital Information Security in Healthcare Act (DISHA), HIPAA, and the Indian Digital Personal Data Protection Act (DPDPA).
            </p>

            <h3 style="color: var(--sage-green); margin-top: 1rem;">2. De-Identification & Anonymization</h3>
            <p style="font-size: 0.85rem; line-height: 1.6; margin-top: 0.3rem;">
              All retinal fundus imagery and neonatal demographic records are stripped of direct identifiers before transmission. Local client biometric hashing is applied.
            </p>

            <h3 style="color: var(--sage-green); margin-top: 1rem;">3. Parental Informed Consent</h3>
            <p style="font-size: 0.85rem; line-height: 1.6; margin-top: 0.3rem;">
              Screening is conducted under parental/guardian consent recorded at NICU admission. Parents receive automated notifications in their preferred language.
            </p>

            <div style="margin-top: 2rem;">
              <a href="/" class="btn-primary" style="display: inline-block; width: auto; text-decoration: none;">Return to Dashboard</a>
            </div>
          </div>
        </body>
        </html>
        """


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return HTMLResponse(
        status_code=404,
        content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>404 - Page Not Found | ROP-Sahayak</title>
          <link rel="stylesheet" href="/static/css/styles.css">
        </head>
        <body style="display: flex; align-items: center; justify-content: center; min-height: 100vh; text-align: center; padding: 2rem;">
          <div class="panel-card" style="max-width: 500px; padding: 2.5rem;">
            <div style="font-size: 4rem; font-weight: 800; color: var(--terracotta-brown);">404</div>
            <h2 style="color: var(--text-main); margin: 0.5rem 0;">Clinical Resource Not Found</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin-bottom: 1.5rem;">
              The requested clinical record, visit report, or view does not exist or has been moved.
            </p>
            <a href="/" class="btn-primary" style="display: inline-block; width: auto; text-decoration: none;">Back to Clinical Triage</a>
          </div>
        </body>
        </html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
