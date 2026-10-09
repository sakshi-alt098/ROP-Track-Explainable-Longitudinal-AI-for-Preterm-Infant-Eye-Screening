// ROP-Sahayak Full-Stack Interactive Controller
let currentRole = "nicu";
let selectedPreset = "type1_stage3";
let latestAnalysisData = null;
let introDismissed = false;

document.addEventListener("DOMContentLoaded", () => {
  initIntroSequence();
  initScrollAnimations();
  initLoginForm();
  loadPatients();
  runAnalysis();

  // Handle Patient Registration Form Submit
  const regForm = document.getElementById("patientRegForm");
  if (regForm) {
    regForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData();
      formData.append("baby_name", document.getElementById("regBabyName").value);
      formData.append("mother_name", document.getElementById("regMotherName").value);
      formData.append("parent_phone", document.getElementById("regParentPhone").value);
      formData.append("nicu_bed", document.getElementById("regNicuBed").value);
      formData.append("dob", document.getElementById("regDob").value);
      formData.append("gestational_age_weeks", document.getElementById("regGa").value);
      formData.append("birth_weight_grams", document.getElementById("regBw").value);
      formData.append("weight_gain_g_per_day", document.getElementById("regWg").value);
      formData.append("supplemental_o2_days", document.getElementById("regO2").value);

      try {
        const res = await fetch("/api/register-patient", { method: "POST", body: formData });
        const data = await res.json();
        alert(`Infant ${data.patient.baby_name} registered successfully with ID: ${data.patient.patient_id}`);
        await loadPatients();
        switchView("visitUploadView");
      } catch (err) {
        alert("Registration failed. Please check network connection.");
      }
    });
  }

  // Handle Analysis Run
  const runBtn = document.getElementById("runAnalysisBtn");
  if (runBtn) {
    runBtn.addEventListener("click", () => runAnalysis());
  }

  // File input change
  const fileIn = document.getElementById("fundusFileInput");
  if (fileIn) {
    fileIn.addEventListener("change", () => {
      if (fileIn.files.length > 0) {
        selectedPreset = null;
        runAnalysis();
      }
    });
  }
});

// Fullscreen Intro Video Sequence
function initIntroSequence() {
  const introOverlay = document.getElementById("introSplash");
  const introVideo = document.getElementById("introEyeVideo");
  const progressFill = document.getElementById("introProgressFill");

  if (!introOverlay || !introVideo) return;

  // Set slow-motion playback on the intro video
  introVideo.playbackRate = 0.55;
  introVideo.addEventListener("play", () => {
    introVideo.playbackRate = 0.55;
  });

  // Animate progress bar
  setTimeout(() => {
    if (progressFill) progressFill.style.width = "100%";
  }, 100);

  // Auto-dismiss after slow-motion presentation
  setTimeout(() => {
    dismissIntro();
  }, 3200);
}

window.dismissIntro = function () {
  if (introDismissed) return;
  introDismissed = true;

  const introOverlay = document.getElementById("introSplash");
  const introVideo = document.getElementById("introEyeVideo");

  if (introOverlay) {
    introOverlay.classList.add("fade-out");
    setTimeout(() => {
      if (introVideo) introVideo.pause();
      introOverlay.style.display = "none";
    }, 850);
  }

  // Trigger login card scale-in and laser scan animation (bottom -> top -> bottom -> disappear)
  const loginCard = document.querySelector(".login-card");
  const scanLine = document.querySelector(".login-card .hologram-scan-line");
  if (loginCard) {
    loginCard.classList.remove("scale-in");
    setTimeout(() => loginCard.classList.add("scale-in"), 50);
  }
  if (scanLine) {
    scanLine.style.animation = "none";
    scanLine.offsetHeight; /* trigger reflow */
    scanLine.style.animation = "scanBottomTopBottom 3.2s cubic-bezier(0.4, 0, 0.2, 1) forwards";
  }
};

// Scroll & View Animation Initializer
function initScrollAnimations() {
  const elements = document.querySelectorAll(".ss-scroll");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("scale-in");
        }
      });
    },
    { threshold: 0.1 }
  );

  elements.forEach((el) => observer.observe(el));
}

// Neumorphic Hospital Login System
function initLoginForm() {
  const loginForm = document.getElementById("loginForm");
  const passwordToggle = document.getElementById("passwordToggle");
  const passwordInput = document.getElementById("password");
  const emailInput = document.getElementById("email");
  const emailError = document.getElementById("emailError");
  const passwordError = document.getElementById("passwordError");
  const loginSubmitBtn = document.getElementById("loginSubmitBtn");
  const successMessage = document.getElementById("successMessage");

  // Password Visibility Toggle
  if (passwordToggle && passwordInput) {
    passwordToggle.addEventListener("click", () => {
      const isPassword = passwordInput.type === "password";
      passwordInput.type = isPassword ? "text" : "password";
      passwordToggle.classList.toggle("show-pass", isPassword);
    });
  }

  // Form Submit Handler with Validation
  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
      let isValid = true;

      emailError.innerText = "";
      passwordError.innerText = "";

      const emailVal = emailInput.value.trim();
      const passVal = passwordInput.value.trim();

      if (!emailVal) {
        emailError.innerText = "Institutional hospital email is required.";
        isValid = false;
      } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailVal)) {
        emailError.innerText = "Please enter a valid institutional email.";
        isValid = false;
      }

      if (!passVal) {
        passwordError.innerText = "Hospital security password/PIN is required.";
        isValid = false;
      } else if (passVal.length < 6) {
        passwordError.innerText = "PIN must be at least 6 characters.";
        isValid = false;
      }

      if (!isValid) return;

      // Show Neumorphic Loading State
      loginSubmitBtn.classList.add("loading");

      setTimeout(() => {
        loginSubmitBtn.classList.remove("loading");
        loginForm.style.display = "none";
        document.querySelector(".divider").style.display = "none";
        document.querySelector(".social-login").style.display = "none";
        document.querySelector(".signup-link").style.display = "none";
        successMessage.classList.add("show");

        setTimeout(() => {
          setRole("nicu");
          switchView("nicuDashboardView");
          // Reset login view for future use
          loginForm.style.display = "block";
          document.querySelector(".divider").style.display = "flex";
          document.querySelector(".social-login").style.display = "flex";
          document.querySelector(".signup-link").style.display = "block";
          successMessage.classList.remove("show");
        }, 1200);
      }, 900);
    });
  }
}

window.quickLogin = function (hospitalName) {
  const emailInput = document.getElementById("email");
  if (emailInput) emailInput.value = `duty.doctor@${hospitalName.toLowerCase().replace(/[^a-z0-9]/g, "")}.in`;
  const form = document.getElementById("loginForm");
  if (form) form.dispatchEvent(new Event("submit"));
};

// View Navigation Switcher
window.switchView = function (viewId) {
  document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active"));
  const target = document.getElementById(viewId);
  if (target) {
    target.classList.add("active");
    // Trigger scroll scale-in animation on new view elements
    target.querySelectorAll(".ss-scroll").forEach((el) => {
      el.classList.remove("scale-in");
      setTimeout(() => el.classList.add("scale-in"), 50);
    });

    if (viewId === "loginView") {
      const scanLine = target.querySelector(".login-card .hologram-scan-line");
      if (scanLine) {
        scanLine.style.animation = "none";
        scanLine.offsetHeight; /* trigger reflow */
        scanLine.style.animation = "scanBottomTopBottom 3.2s cubic-bezier(0.4, 0, 0.2, 1) forwards";
      }
    }
  }

  document.querySelectorAll(".screen-link").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("onclick")?.includes(viewId));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// Role Switcher
window.setRole = function (role) {
  currentRole = role;
  document.querySelectorAll(".role-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.role === role);
  });

  if (role === "specialist") {
    switchView("specialistQueueView");
  } else if (role === "admin") {
    switchView("costEstimatorView");
  } else if (role === "login") {
    switchView("loginView");
  } else {
    switchView("nicuDashboardView");
  }
};

window.setPresetAndRun = function (preset) {
  selectedPreset = preset;
  const fileIn = document.getElementById("fundusFileInput");
  if (fileIn) fileIn.value = "";
  runAnalysis();
};

async function loadPatients() {
  try {
    const res = await fetch("/api/patients");
    const data = await res.json();
    renderNicuTable(data.patients);
    renderSpecialistQueue(data.patients);
    updateDashboardCounters(data.patients);
  } catch (err) {
    console.error("Failed to load patient registry", err);
  }
}

function updateDashboardCounters(patients) {
  const urgent = patients.filter((p) => p.urgency_code === "P0" || p.urgency_code === "P1").length;
  const priority = patients.filter((p) => p.urgency_code === "P2").length;
  const routine = patients.filter((p) => p.urgency_code === "P3").length;

  document.getElementById("statUrgentCount").innerText = urgent;
  document.getElementById("statPriorityCount").innerText = priority;
  document.getElementById("statRoutineCount").innerText = routine;
}

function renderNicuTable(patients) {
  const tbody = document.getElementById("nicuTableBody");
  if (!tbody) return;

  tbody.innerHTML = patients
    .map(
      (p) => `
    <tr>
      <td><strong>${p.patient_id}</strong></td>
      <td>${p.baby_name}<br><span style="color:var(--text-muted);font-size:0.7rem;">M: ${p.mother_name || "N/A"}</span></td>
      <td><span class="badge badge-tan">${p.nicu_bed || "NICU"}</span></td>
      <td>${p.gestational_age_weeks}w / ${p.birth_weight_grams}g</td>
      <td>${p.postmenstrual_age_weeks}w</td>
      <td>${p.stage_name} (${p.plus_category})</td>
      <td><span class="badge ${getUrgencyBadgeClass(p.urgency_code)}">${p.urgency_code}: ${p.urgency_label}</span></td>
      <td><span style="font-size:0.75rem;color:var(--text-muted);">${p.status}</span></td>
      <td>
        <button class="btn-outline" style="font-size:0.7rem;padding:0.3rem 0.5rem;" onclick="selectPatientForScreening('${p.patient_id}')">
          Screen
        </button>
      </td>
    </tr>
  `
    )
    .join("");
}

function renderSpecialistQueue(patients) {
  const queueBody = document.getElementById("specialistQueueBody");
  if (!queueBody) return;

  const urgentPatients = patients.filter((p) => p.urgency_code === "P0" || p.urgency_code === "P1" || p.urgency_code === "P2");

  queueBody.innerHTML = urgentPatients
    .map(
      (p) => `
    <tr>
      <td><span class="badge ${getUrgencyBadgeClass(p.urgency_code)}">${p.urgency_code}</span></td>
      <td><strong>${p.patient_id}</strong><br><span style="font-size:0.7rem;color:var(--text-muted);">${p.baby_name}</span></td>
      <td>${p.hospital || "District NICU"}</td>
      <td>${p.postmenstrual_age_weeks}w PMA (GA: ${p.gestational_age_weeks}w)</td>
      <td>${p.stage_name} (${p.plus_category})</td>
      <td><span style="font-size:0.75rem;">Plus: ${p.plus_score ? p.plus_score.toFixed(3) : "0.780"}</span></td>
      <td>
        <button class="btn-primary" style="font-size:0.7rem;padding:0.35rem 0.65rem;" onclick="openSpecialistReview('${p.patient_id}')">
          Review Case
        </button>
      </td>
    </tr>
  `
    )
    .join("");
}

window.selectPatientForScreening = function (id) {
  const select = document.getElementById("visitPatientSelect");
  if (select) select.value = id;
  switchView("visitUploadView");
};

window.openSpecialistReview = function (id) {
  switchView("specialistReviewView");
};

async function runAnalysis() {
  const formData = new FormData();
  const fileIn = document.getElementById("fundusFileInput");
  const pma = document.getElementById("visitPma")?.value || 34.0;
  const wg = document.getElementById("visitWg")?.value || 10.0;

  formData.append("patient_id", "ROP-2026-001");
  formData.append("baby_name", "Sharma (Twin 1)");
  formData.append("postmenstrual_age_weeks", pma);
  formData.append("weight_gain_g_per_day", wg);

  if (fileIn && fileIn.files.length > 0) {
    formData.append("file", fileIn.files[0]);
  } else if (selectedPreset) {
    formData.append("sample_case", selectedPreset);
  }

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    latestAnalysisData = data;
    renderAnalysisResults(data);
  } catch (err) {
    console.error("Analysis execution error", err);
  }
}

function renderAnalysisResults(data) {
  const { quality_assessment, biomarkers, icrop3_diagnosis, longitudinal_progression, scheduler_and_alerts } = data;

  // Quality & Urgency Header
  document.getElementById("resUrgencyBadge").innerText = `${icrop3_diagnosis.urgency_code}: ${icrop3_diagnosis.urgency_label}`;
  document.getElementById("resUrgencyBadge").className = `badge ${getUrgencyBadgeClass(icrop3_diagnosis.urgency_code)}`;
  document.getElementById("resActionHeading").innerText = icrop3_diagnosis.recommended_action;
  document.getElementById("resQualityScore").innerText = `${quality_assessment.status} (${quality_assessment.overall_score}%)`;
  document.getElementById("resQualityWarning").innerText = quality_assessment.warnings[0] || "Quality acceptable.";

  // Images
  const overlaySrc = biomarkers.overlay_image_base64;
  document.getElementById("resOverlayImg").src = overlaySrc;
  if (document.getElementById("capturePreviewImg")) document.getElementById("capturePreviewImg").src = overlaySrc;
  if (document.getElementById("specOrigImg")) document.getElementById("specOrigImg").src = overlaySrc;
  if (document.getElementById("specOverlayImg")) document.getElementById("specOverlayImg").src = overlaySrc;

  // Diagnostic fields
  document.getElementById("resStageName").innerText = icrop3_diagnosis.stage_name;
  document.getElementById("resStageDesc").innerText = icrop3_diagnosis.stage_description;
  document.getElementById("resZoneBadge").innerText = icrop3_diagnosis.zone;
  document.getElementById("resPlusText").innerText = `${icrop3_diagnosis.plus_category} (${icrop3_diagnosis.plus_score.toFixed(3)})`;

  const meter = document.getElementById("resPlusMeter");
  const pct = Math.min(100, Math.max(0, icrop3_diagnosis.plus_score * 100));
  meter.style.width = `${pct}%`;
  meter.style.backgroundColor = pct > 65 ? "var(--terracotta-brown)" : pct > 30 ? "var(--golden-tan)" : "var(--sage-green)";

  // Biomarkers
  document.getElementById("resAtiVal").innerText = biomarkers.ati_score;
  document.getElementById("resVdiVal").innerText = biomarkers.vdi_score;

  // Longitudinal Progression
  document.getElementById("timelineSysRisk").innerText = `${longitudinal_progression.systemic_risk_score}%`;
  document.getElementById("timelineTrRopRisk").innerText = `${longitudinal_progression.tr_rop_risk_percentage}%`;
  document.getElementById("timelineVelocity").innerText = longitudinal_progression.progression_velocity;

  renderTrajectorySvg(longitudinal_progression.trajectory_forecast_14d);

  // Scheduler & Follow-up Tracker
  document.getElementById("trackerNextVisit").innerText = scheduler_and_alerts.formatted_schedule;
  document.getElementById("trackerParentSummary").innerText = scheduler_and_alerts.parent_friendly_explanation;
  document.getElementById("trackerMsgPreview").innerText = scheduler_and_alerts.whatsapp_payload.message;
}

function renderTrajectorySvg(points) {
  const svg = document.getElementById("timelineSvg");
  if (!svg || !points) return;

  const w = 700;
  const h = 140;
  const coords = points.map((p, i) => {
    const x = (i / (points.length - 1)) * (w - 80) + 40;
    const y = h - (p.projected_risk_percentage / 100) * (h - 40) - 20;
    return `${x},${y}`;
  });

  const path = `M ${coords.join(" L ")}`;

  svg.innerHTML = `
    <line x1="40" y1="${h - 20}" x2="${w - 40}" y2="${h - 20}" stroke="#E0D3C1" stroke-width="1"/>
    <path d="${path}" fill="none" stroke="#A84B38" stroke-width="2.5"/>
    ${points
      .map((p, i) => {
        const [cx, cy] = coords[i].split(",");
        return `
        <circle cx="${cx}" cy="${cy}" r="4" fill="#FAF6EF" stroke="#A84B38" stroke-width="2"/>
        <text x="${cx}" y="${h - 5}" font-size="10" fill="#7A6F68" text-anchor="middle">Day ${p.day}</text>
        <text x="${cx}" y="${parseFloat(cy) - 8}" font-size="10" font-weight="700" fill="#A84B38" text-anchor="middle">${p.projected_risk_percentage}%</text>
      `;
      })
      .join("")}
  `;
}

window.submitCaseToDoctor = async function () {
  try {
    const formData = new FormData();
    formData.append("patient_id", "ROP-2026-001");
    formData.append("notes", document.getElementById("submitCaseNotes").value);

    const res = await fetch("/api/submit-case", { method: "POST", body: formData });
    const data = await res.json();
    alert(`Case submitted successfully!\n${data.message}`);
    setRole("specialist");
  } catch (err) {
    alert("Submission failed.");
  }
};

window.saveSpecialistReview = async function () {
  const formData = new FormData();
  formData.append("patient_id", "ROP-2026-001");
  formData.append("doctor_name", "Dr. Ananya Roy, MD");
  formData.append("decision", document.getElementById("specDecision").value);
  formData.append("doctor_notes", document.getElementById("specNotes").value);
  formData.append("treatment_prescribed", document.getElementById("specTreatmentPlan").value);

  try {
    const res = await fetch("/api/specialist-review", { method: "POST", body: formData });
    const data = await res.json();
    alert("Review signed & report dispatched to NICU!");
    await loadPatients();
    switchView("nicuDashboardView");
  } catch (err) {
    alert("Failed to save review.");
  }
};

window.updateCostEstimate = async function () {
  const scheme = document.getElementById("costScheme").value;
  const intervention = document.getElementById("costIntervention").value;

  const formData = new FormData();
  formData.append("scheme", scheme);
  formData.append("intervention_type", intervention);
  formData.append("has_bpl_card", "true");

  try {
    const res = await fetch("/api/estimate-cost", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("costGross").innerText = `Rs ${data.gross_cost_inr.toLocaleString()}`;
    document.getElementById("costSubsidy").innerText = `- Rs ${data.government_subsidy_inr.toLocaleString()}`;
    document.getElementById("costNet").innerText = data.zero_cost_eligible ? "Rs 0 (100% Free Scheme)" : `Rs ${data.net_payable_inr.toLocaleString()}`;
  } catch (err) {
    console.error("Cost estimation failed", err);
  }
};

function getUrgencyBadgeClass(code) {
  if (code === "P0" || code === "P1") return "badge-terracotta";
  if (code === "P2") return "badge-tan";
  return "badge-sage";
}
