# ROP-Track: Explainable Multi-Visit AI for Retinopathy of Prematurity Screening and Follow-Up

> Explainable multi-visit AI system for retinopathy of prematurity (ROP) screening, progression risk prediction, and automated follow-up scheduling.

![status](https://img.shields.io/badge/status-in%20development-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> ⚠️ **Research prototype. Not a medical device.** ROP-Track is a clinical *decision-support* tool intended for research and education. It must not be used as a substitute for examination and judgement by a qualified ophthalmologist.

---

## Why this project?

Retinopathy of Prematurity (ROP) is a leading cause of preventable childhood blindness. Screening depends on repeated retinal exams over several weeks, and in many regions there are far fewer pediatric retina specialists than preterm babies who need screening.

Most existing ROP AI work:

- classifies **one image at a time**, ignoring that ROP is a *progression* disease,
- offers explainability limited to **heatmaps** (e.g., Grad-CAM) that clinicians find hard to act on,
- stops at the prediction, with no link to **follow-up scheduling** or loss-to-follow-up prevention,
- struggles to **generalize across cameras and hospitals**.

ROP-Track targets these gaps.

---

## Key Features

### Core
- Retinal image upload / capture (multi-view per eye)
- ROP stage, zone, plus disease, and aggressive ROP assessment
- Per-finding explanations
- Visit-wise clinical reports with a timeline
- Automatic notification to the ophthalmologist with a suggested next appointment

### Novel contributions
| # | Feature | What it adds |
|---|---------|--------------|
| 1 | **Longitudinal progression engine** | Uses all past visits (plus clinical data) to estimate the risk of progression to treatment-requiring ROP; produces visit-to-visit **change maps** |
| 2 | **Concept-based explainability** | Predicts clinically meaningful concepts (zone, stage, demarcation line/ridge, vessel tortuosity, vessel width) and derives the diagnosis from them, instead of relying only on heatmaps |
| 3 | **Follow-up scheduling & loss-to-follow-up alerts** | Risk-based next-visit suggestion (doctor confirms), reminders to parents, and missed-appointment alerts |
| 4 | **Image quality gate** *(planned)* | Flags blurry or incomplete captures (e.g., optic disc not visible) so they can be retaken |
| 5 | **Uncertainty-aware triage** *(planned)* | Model abstains when unsure and ranks cases in a doctor's priority queue |
| 6 | **Parent-friendly report** *(planned)* | Simple, local-language summary of findings and next steps |
| 7 | **Cross-camera robustness / offline mode** *(planned)* | Domain generalization and lightweight on-device inference for low-connectivity NICUs |

---

## System Architecture

```
Capture / Upload
      │
      ▼
Quality Gate ──► Preprocessing
      │
      ▼
Vessel Segmentation + Concept Model (stage / zone / plus)
      │
      ▼
Temporal Progression Model (past visits + clinical data)
      │
      ▼
Uncertainty + Explanation Layer
      │
      ├──► Report Generator (clinical PDF + parent version)
      │
      ▼
Doctor Dashboard (priority queue, approve / override)
      │
      ▼
Scheduler + Notification Service (WhatsApp / SMS / email)
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Models | PyTorch, torchvision, OpenCV |
| Explainability | Captum / pytorch-grad-cam, custom concept & vessel metrics |
| Backend | FastAPI, PostgreSQL, SQLAlchemy |
| Async jobs | Celery, Redis |
| Frontend | React *(or Flutter)* |
| Reports | WeasyPrint / ReportLab |
| Deployment | Docker, ONNX / TFLite (on-device inference) |

---

## Repository Structure

```
rop-track/
├── data/            # (gitignored) raw + processed images
├── models/          # quality gate, segmentation, concept model, temporal model
├── explain/         # concept explanations, vessel metrics, change maps
├── backend/         # FastAPI app, scheduler, notifications
├── frontend/        # doctor dashboard + parent report
├── reports/         # PDF templates
├── notebooks/       # experiments and EDA
├── tests/
├── docs/
├── README.md
└── .gitignore
```

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/rop-track.git
cd rop-track

# 2. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend (dev)
uvicorn backend.main:app --reload
```

> Setup steps will be finalized as modules are added.

---

## Data

- Public ROP datasets are small and limited, so this project plans to combine **public data**, **transfer learning**, **augmentation**, and (where possible) **hospital-partnered data** under proper ethics approval.
- **Never commit patient data.** `data/` is gitignored. Only fully anonymized or public data may be used, and any clinical data requires institutional approval and consent.

---

## Evaluation Plan

- **Sensitivity for treatment-requiring ROP** (primary priority)
- AUROC, per-class sensitivity / specificity
- Agreement with expert grading (Cohen's kappa)
- Calibration of predicted probabilities
- Progression model: C-index / time-to-event metrics
- Explainability: clinician usefulness ratings and agreement with expert annotations
- Robustness: train on one camera/source, test on another

Results will be added here as experiments are completed.

---

## Roadmap

- [ ] Literature review and dataset collection
- [ ] Baseline stage / plus classification model
- [ ] Image quality gate
- [ ] Vessel segmentation + vessel metrics
- [ ] Concept-based explainability layer
- [ ] Multi-visit timeline and progression risk model
- [ ] Report generation (clinical + parent)
- [ ] Doctor dashboard
- [ ] Scheduling and notification service
- [ ] Cross-camera evaluation
- [ ] Paper / demo

---

## Ethics and Limitations

- Decision support only; a clinician makes every final decision.
- Model performance depends heavily on data quality, camera type, and population; results may not transfer across settings.
- Predictions include uncertainty estimates and should be reviewed, not blindly trusted.
- Any real-world use would require clinical validation and regulatory approval.

---

## Contributing

Contributions, issues, and ideas are welcome. Please open an issue to discuss major changes first.

## License

Released under the MIT License. See `LICENSE` for details.

## Acknowledgements

Thanks to the clinicians and researchers whose work on ROP screening and explainable medical AI informs this project.
