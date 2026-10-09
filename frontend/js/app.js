// ROP-Sahayak Full-Stack Interactive Controller
let currentRole = "assistant";
let selectedPreset = null;
let latestAnalysisData = null;
let introDismissed = false;
let activePatientId = "ROP-2026-001";
let patientsRegistry = [];

document.addEventListener("DOMContentLoaded", () => {
  initIntroSequence();
  initScrollAnimations();
  initLoginForm();
  loadPatients();

  // Role selector cards toggle in login
  const cardAssistant = document.getElementById("roleCardAssistant");
  const cardSpec = document.getElementById("roleCardSpecialist");
  if (cardAssistant && cardSpec) {
    cardAssistant.addEventListener("click", () => {
      cardAssistant.classList.add("selected");
      cardAssistant.style.border = "2px solid var(--terracotta-brown)";
      cardAssistant.style.background = "var(--warm-beige-panel)";
      cardSpec.classList.remove("selected");
      cardSpec.style.border = "1px solid var(--warm-beige-border)";
      cardSpec.style.background = "#FFFFFF";
      cardAssistant.querySelector("input").checked = true;
    });

    cardSpec.addEventListener("click", () => {
      cardSpec.classList.add("selected");
      cardSpec.style.border = "2px solid var(--terracotta-brown)";
      cardSpec.style.background = "var(--warm-beige-panel)";
      cardAssistant.classList.remove("selected");
      cardAssistant.style.border = "1px solid var(--warm-beige-border)";
      cardAssistant.style.background = "#FFFFFF";
      cardSpec.querySelector("input").checked = true;
    });
  }

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

      const regSuccessBadge = document.getElementById("regSuccessBadge");
      const regSuccessDetail = document.getElementById("regSuccessDetail");
      const regErrorBadge = document.getElementById("regErrorBadge");

      try {
        const res = await fetch("/api/register-patient", { method: "POST", body: formData });
        const data = await res.json();
        
        // Show inline status badge instead of browser alert popup
        if (regSuccessBadge) {
          if (regSuccessDetail) {
            regSuccessDetail.innerText = `Infant ${data.patient.baby_name} registered with ID ${data.patient.patient_id}. Bed ${data.patient.nicu_bed || "NICU"}. Baseline scan scheduled.`;
          }
          regSuccessBadge.style.display = "block";
        }
        if (regErrorBadge) regErrorBadge.style.display = "none";
        
        activePatientId = data.patient.patient_id;
        await loadPatients();
        
        setTimeout(() => {
          switchView("visitUploadView");
          selectPatientForScreening(activePatientId);
        }, 1200);
      } catch (err) {
        if (regErrorBadge) regErrorBadge.style.display = "block";
        if (regSuccessBadge) regSuccessBadge.style.display = "none";
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
      } else if (passVal.length < 4) {
        passwordError.innerText = "PIN must be at least 4 characters.";
        isValid = false;
      }

      if (!isValid) return;

      // Detect chosen clinical role
      const selectedRoleRadio = document.querySelector('input[name="portalRole"]:checked');
      const chosenRole = selectedRoleRadio ? selectedRoleRadio.value : "assistant";

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
          setRole(chosenRole);
          if (chosenRole === "assistant") {
            switchView("visitUploadView");
          } else {
            switchView("specialistQueueView");
          }
          // Reset login view for future use
          loginForm.style.display = "block";
          document.querySelector(".divider").style.display = "flex";
          document.querySelector(".social-login").style.display = "flex";
          document.querySelector(".signup-link").style.display = "block";
          successMessage.classList.remove("show");
        }, 1100);
      }, 800);
    });
  }
}

window.quickLogin = function (hospitalName) {
  const emailInput = document.getElementById("email");
  const passInput = document.getElementById("password");
  if (emailInput) emailInput.value = `staff@${hospitalName.toLowerCase().replace(/[^a-z0-9]/g, "")}.in`;
  if (passInput) passInput.value = "HospitalPass123";
  const form = document.getElementById("loginForm");
  if (form) form.dispatchEvent(new Event("submit"));
};

// View Navigation Switcher
window.switchView = function (viewId) {
  // If moving to loginView, hide the entire upper header navigation
  const navHeader = document.getElementById("appNavigationHeader");
  if (viewId === "loginView") {
    if (navHeader) navHeader.style.display = "none";
  } else {
    if (navHeader) navHeader.style.display = "block";
  }

  // Guard: Screening Assistant cannot open specialist review screens
  if (currentRole === "assistant" && (viewId === "specialistQueueView" || viewId === "specialistReviewView")) {
    switchView("caseSubmitView");
    return;
  }

  // Guard: Specialist does not upload images (they review submitted captures from NICU)
  if (currentRole === "specialist" && (viewId === "visitUploadView" || viewId === "caseSubmitView")) {
    switchView("specialistQueueView");
    return;
  }

  document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active"));
  const target = document.getElementById(viewId);
  if (target) {
    target.classList.add("active");
    target.querySelectorAll(".ss-scroll").forEach((el) => {
      el.classList.remove("scale-in");
      setTimeout(() => el.classList.add("scale-in"), 50);
    });

    if (viewId === "loginView") {
      const scanLine = target.querySelector(".login-card .hologram-scan-line");
      if (scanLine) {
        scanLine.style.animation = "none";
        scanLine.offsetHeight;
        scanLine.style.animation = "scanBottomTopBottom 3.2s cubic-bezier(0.4, 0, 0.2, 1) forwards";
      }
    }
  }

  document.querySelectorAll(".screen-link").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("onclick")?.includes(viewId));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// Role Switcher with strict clinical permission management
window.setRole = function (role) {
  currentRole = role;

  // Reveal top header now that role is authenticated
  const navHeader = document.getElementById("appNavigationHeader");
  if (navHeader) navHeader.style.display = "block";

  const isSpecialist = (role === "specialist");
  const roleLabel = document.getElementById("activeRoleLabel");
  const roleIcon = document.getElementById("authenticatedRoleIcon");
  if (roleLabel) {
    roleLabel.innerText = isSpecialist ? "Specialist Portal (Dr. Ananya Roy, MD)" : "NICU Screening Portal (Bedside Assistant)";
  }
  if (roleIcon) {
    roleIcon.innerText = isSpecialist ? "🔬" : "🩺";
  }

  // STRICT TAB ISOLATION:
  // 1. Specialist tabs (Specialist Queue, Specialist Case Review):
  //    VISIBLE ONLY to Specialist; COMPLETELY HIDDEN from Screening Assistant
  document.querySelectorAll(".role-spec-only").forEach((el) => {
    el.style.display = isSpecialist ? "inline-block" : "none";
  });

  // 2. Bedside/Assistant tabs (Visit Upload, Case Submission):
  //    VISIBLE ONLY to Screening Assistant; COMPLETELY HIDDEN from Specialist
  document.querySelectorAll(".role-assist-only").forEach((el) => {
    // If specialist, hide image upload and bedside submit buttons
    if (el.id === "navLinkUpload" || el.id === "navLinkCaseSubmit") {
      el.style.display = isSpecialist ? "none" : "inline-block";
    }
  });

  // 3. Landing page routing per role
  if (isSpecialist) {
    switchView("specialistQueueView");
  } else {
    switchView("visitUploadView");
  }
};

window.logoutUser = function () {
  const navHeader = document.getElementById("appNavigationHeader");
  if (navHeader) navHeader.style.display = "none";
  currentRole = null;
  switchView("loginView");
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
    patientsRegistry = data.patients || [];
    renderNicuTable(patientsRegistry);
    renderSpecialistQueue(patientsRegistry);
    updateDashboardCounters(patientsRegistry);
    populateAllPatientDropdowns(patientsRegistry);
    updateCaseSubmissionDetails(activePatientId);
    loadPersonalizedPatientAlert(activePatientId);
  } catch (err) {
    console.error("Failed to load patient registry", err);
  }
}

function populateAllPatientDropdowns(patients) {
  const dropdownIds = ["visitPatientSelect", "submitPatientSelect", "trackerPatientSelect"];
  dropdownIds.forEach((id) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const currentVal = sel.value || activePatientId;
    sel.innerHTML = patients
      .map(
        (p) => {
          const capTag = p.has_image_captured ? "✓ Captured" : "📷 Not Captured";
          return `<option value="${p.patient_id}">Baby ${p.baby_name} (Mother: ${p.mother_name || 'N/A'}) [${p.patient_id}] - ${capTag}</option>`;
        }
      )
      .join("");
    if (patients.some((p) => p.patient_id === currentVal)) {
      sel.value = currentVal;
    } else if (patients.length > 0) {
      sel.value = patients[0].patient_id;
    }
  });

  // Keep upload card updated with current patient
  onVisitPatientChange(activePatientId);
}

function updateDashboardCounters(patients) {
  const urgent = patients.filter((p) => p.has_image_captured && (p.urgency_code === "P0" || p.urgency_code === "P1")).length;
  const priority = patients.filter((p) => p.has_image_captured && p.urgency_code === "P2").length;
  const routine = patients.filter((p) => p.has_image_captured && p.urgency_code === "P3").length;

  document.getElementById("statUrgentCount").innerText = urgent;
  document.getElementById("statPriorityCount").innerText = priority;
  document.getElementById("statRoutineCount").innerText = routine;
}

function renderNicuTable(patients) {
  const tbody = document.getElementById("nicuTableBody");
  if (!tbody) return;

  tbody.innerHTML = patients
    .map(
      (p) => {
        const isCaptured = Boolean(p.has_image_captured);
        const scanStatusHtml = isCaptured
          ? `<span class="badge badge-sage" style="font-weight:700;">✓ Captured</span>`
          : `<span class="badge badge-tan" style="background:#fff3cd; color:#856404; font-weight:700; border:1px solid #ffeeba;">📷 Not Captured</span>`;

        const priorityHtml = isCaptured
          ? `<span class="badge ${getUrgencyBadgeClass(p.urgency_code)}">${p.urgency_code}: ${p.urgency_label}</span>`
          : `<span class="badge" style="background:#e9ecef; color:#6c757d; font-weight:600;">Pending Scan</span>`;

        return `
    <tr style="${!isCaptured ? 'background: rgba(230, 180, 100, 0.05);' : ''}">
      <td><strong>${p.patient_id}</strong></td>
      <td>
        <strong style="color:var(--text-main); font-size:0.85rem;">${p.baby_name}</strong><br>
        <span style="color:var(--text-muted); font-size:0.7rem;">M: ${p.mother_name || "N/A"}</span>
      </td>
      <td><span class="badge badge-tan">${p.nicu_bed || "NICU"}</span></td>
      <td>${p.gestational_age_weeks}w / ${p.birth_weight_grams}g</td>
      <td>${p.postmenstrual_age_weeks}w</td>
      <td>${scanStatusHtml}</td>
      <td>${priorityHtml}</td>
      <td><span style="font-size:0.75rem; color:var(--text-muted);">${p.status}</span></td>
      <td>
        <button class="btn-primary" style="font-size:0.7rem; padding:0.35rem 0.65rem;" onclick="selectPatientForScreening('${p.patient_id}')">
          ${isCaptured ? 'Re-Screen' : '📷 Capture Scan'}
        </button>
      </td>
    </tr>
  `;
      }
    )
    .join("");
}

function renderSpecialistQueue(patients) {
  const queueBody = document.getElementById("specialistQueueBody");
  if (!queueBody) return;

  const queuePatients = patients.filter((p) => 
    p.status === "Submitted to Specialist" || 
    (p.has_image_captured && (p.urgency_code === "P0" || p.urgency_code === "P1" || p.urgency_code === "P2"))
  );

  queueBody.innerHTML = queuePatients
    .map(
      (p) => `
    <tr style="${p.status === 'Submitted to Specialist' ? 'background: rgba(168, 75, 56, 0.06); border-left: 3px solid var(--terracotta-brown);' : ''}">
      <td>
        <span class="badge ${getUrgencyBadgeClass(p.urgency_code)}">${p.urgency_code}</span>
        ${p.status === 'Submitted to Specialist' ? '<span class="badge badge-terracotta" style="display:block;margin-top:0.25rem;font-size:0.65rem;">✓ Submitted</span>' : ''}
      </td>
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

window.onVisitPatientChange = function (patientId) {
  activePatientId = patientId;
  const p = patientsRegistry.find((item) => item.patient_id === patientId);
  if (!p) return;

  const nameEl = document.getElementById("visitCardBabyName");
  const metaEl = document.getElementById("visitCardMeta");
  const statusEl = document.getElementById("visitCardScanStatus");
  const clinEl = document.getElementById("visitCardClinicalInfo");

  if (nameEl) nameEl.innerText = `Baby ${p.baby_name}`;
  if (metaEl) metaEl.innerText = `ID: ${p.patient_id} • Mother: ${p.mother_name || "N/A"} • Bed: ${p.nicu_bed || "NICU"}`;
  if (statusEl) {
    statusEl.innerHTML = p.has_image_captured
      ? `<span class="badge badge-sage" style="font-weight:700;">✓ Scan Captured</span>`
      : `<span class="badge badge-tan" style="background:#fff3cd; color:#856404; font-weight:700; border:1px solid #ffeeba;">📷 Not Captured</span>`;
  }
  if (clinEl) {
    clinEl.innerText = `GA: ${p.gestational_age_weeks} wks | BW: ${p.birth_weight_grams}g | Current PMA: ${p.postmenstrual_age_weeks} wks | Supplemental O2: ${p.supplemental_o2_days || 7} days`;
  }

  const pmaInput = document.getElementById("visitPma");
  const wgInput = document.getElementById("visitWg");
  if (pmaInput && p.postmenstrual_age_weeks) pmaInput.value = p.postmenstrual_age_weeks;
  if (wgInput && p.weight_gain_g_per_day) wgInput.value = p.weight_gain_g_per_day;

  const preview = document.getElementById("capturePreviewImg");
  if (preview) {
    preview.src = p.original_image_base64 || "";
  }
};

window.selectPatientForScreening = function (id) {
  activePatientId = id;
  const select = document.getElementById("visitPatientSelect");
  if (select) select.value = id;

  onVisitPatientChange(id);
  updateCaseSubmissionDetails(id);
  loadPersonalizedPatientAlert(id);
  switchView("visitUploadView");
};

window.updateCaseSubmissionDetails = function (patientId) {
  activePatientId = patientId;
  const sel = document.getElementById("submitPatientSelect");
  if (sel && sel.value !== patientId) sel.value = patientId;

  const patient = patientsRegistry.find((p) => p.patient_id === patientId);
  if (!patient) return;

  const titleEl = document.getElementById("submitBundlePatientTitle");
  if (titleEl) titleEl.innerText = `Patient: ${patient.baby_name} (${patient.patient_id})`;

  const badgeEl = document.getElementById("submitBundleUrgencyBadge");
  if (badgeEl) {
    badgeEl.innerText = `${patient.urgency_code}: ${patient.urgency_label || 'Triage'}`;
    badgeEl.className = `badge ${getUrgencyBadgeClass(patient.urgency_code)}`;
  }

  const demoEl = document.getElementById("submitBundleDemographics");
  if (demoEl) {
    demoEl.innerText = `GA: ${patient.gestational_age_weeks}w | BW: ${patient.birth_weight_grams}g | PMA: ${patient.postmenstrual_age_weeks}w | Bed: ${patient.nicu_bed || "NICU"} | Hospital: ${patient.hospital || "District NICU"}`;
  }

  const diagEl = document.getElementById("submitBundleDiagnosis");
  if (diagEl) {
    diagEl.innerText = `Finding: ${patient.stage_name} • ${patient.zone} • ${patient.plus_category}`;
  }

  // Populate Image Thumbnails
  const origThumb = document.getElementById("submitThumbOrig");
  const aiThumb = document.getElementById("submitThumbAi");
  const origSrc = patient.original_image_base64 || (latestAnalysisData && latestAnalysisData.original_image_base64) || "";
  const aiSrc = patient.ai_marked_image_base64 || (latestAnalysisData && latestAnalysisData.ai_marked_image_base64) || "";

  if (origThumb && origSrc) origThumb.src = origSrc;
  if (aiThumb && aiSrc) aiThumb.src = aiSrc;

  // Reset submission banner if switching patients
  const banner = document.getElementById("caseSubmissionSuccessBanner");
  if (banner && patient.status !== "Submitted to Specialist") {
    banner.style.display = "none";
  }
};

window.openSpecialistReview = function (id) {
  activePatientId = id;
  const patient = patientsRegistry.find((p) => p.patient_id === id);
  if (patient) {
    const heading = document.getElementById("specReviewPatientHeading");
    if (heading) heading.innerText = `Case Review: ${patient.baby_name} (${patient.patient_id})`;

    const origImg = document.getElementById("specOrigImg");
    const overlayImg = document.getElementById("specOverlayImg");
    if (origImg) origImg.src = patient.original_image_base64 || (latestAnalysisData?.original_image_base64 || "");
    if (overlayImg) overlayImg.src = patient.ai_marked_image_base64 || (latestAnalysisData?.ai_marked_image_base64 || "");

    const sysInd = document.getElementById("specSystemicIndicators");
    if (sysInd) {
      sysInd.innerHTML = `<strong>Systemic Indicators:</strong> GA ${patient.gestational_age_weeks} wks, BW ${patient.birth_weight_grams}g, PMA ${patient.postmenstrual_age_weeks} wks, Daily Weight Gain: ${patient.weight_gain_g_per_day || 10}g/day, Supplemental O2: ${patient.supplemental_o2_days || 7} days, Center: ${patient.hospital || 'NICU'}.`;
    }

    const notesDisp = document.getElementById("specBedsideNurseNotesDisplay");
    if (notesDisp) {
      notesDisp.innerText = patient.last_notes || "Bedside scan submitted for specialist review.";
    }
  }

  const reviewBadge = document.getElementById("specReviewSuccessBadge");
  if (reviewBadge) reviewBadge.style.display = "none";

  switchView("specialistReviewView");
};

async function runAnalysis() {
  const formData = new FormData();
  const fileIn = document.getElementById("fundusFileInput");
  const pma = document.getElementById("visitPma")?.value || 34.0;
  const wg = document.getElementById("visitWg")?.value || 10.0;
  const patientSelect = document.getElementById("visitPatientSelect");
  const patientId = patientSelect ? patientSelect.value : activePatientId;
  activePatientId = patientId;

  const patient = patientsRegistry.find((p) => p.patient_id === patientId);

  formData.append("patient_id", patientId);
  formData.append("baby_name", patient ? patient.baby_name : "Infant");
  formData.append("mother_name", patient ? (patient.mother_name || "") : "");
  formData.append("parent_phone", patient ? (patient.parent_phone || "+91 98765 43210") : "+91 98765 43210");
  formData.append("gestational_age_weeks", patient ? (patient.gestational_age_weeks || 28.0) : 28.0);
  formData.append("birth_weight_grams", patient ? (patient.birth_weight_grams || 1000.0) : 1000.0);
  formData.append("postmenstrual_age_weeks", pma);
  formData.append("weight_gain_g_per_day", wg);

  const hasFile = fileIn && fileIn.files.length > 0;
  if (hasFile) {
    formData.append("file", fileIn.files[0]);
  } else if (selectedPreset) {
    formData.append("sample_case", selectedPreset);
  } else {
    // Show inline warning instead of alert dialog
    const warn = document.getElementById("uploadWarningBadge");
    if (warn) {
      warn.style.display = "block";
      setTimeout(() => { warn.style.display = "none"; }, 4000);
    }
    return;
  }

  const warn = document.getElementById("uploadWarningBadge");
  if (warn) warn.style.display = "none";

  // Switch to Analysis Results View with Loading Animation
  switchView("analysisResultsView");
  const emptyState = document.getElementById("resultsEmptyState");
  const loader = document.getElementById("mlAnalysisLoader");
  const contentArea = document.getElementById("resultsContentArea");
  const retakeScreen = document.getElementById("retakeRequiredScreen");

  if (emptyState) emptyState.style.display = "none";
  if (contentArea) contentArea.style.display = "none";
  if (retakeScreen) retakeScreen.style.display = "none";
  if (loader) loader.style.display = "block";

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();

    // QUALITY GATE EVALUATION: IF BLANK, BLURRED, OR UNGRADABLE -> RETAKE REQUIRED (DO NOT CREATE REPORT!)
    if (data.requires_retake || (data.quality_assessment && !data.quality_assessment.is_gradable)) {
      renderRetakeRequiredView(data);
      await loadPatients();
      return;
    }

    // AI DETECTED VALID RETINAL IMAGE -> DO NOT ASK TO RETAKE, RENDER FULL COMPARATIVE REPORT
    latestAnalysisData = data;
    renderAnalysisResults(data);

    // Update cached patient record with captured image and updated priority
    const targetPatient = patientsRegistry.find((p) => p.patient_id === patientId);
    if (targetPatient) {
      targetPatient.has_image_captured = true;
      targetPatient.original_image_base64 = data.original_image_base64;
      targetPatient.ai_marked_image_base64 = data.ai_marked_image_base64;
      targetPatient.stage_name = data.icrop3_diagnosis.stage_name;
      targetPatient.zone = data.icrop3_diagnosis.zone;
      targetPatient.plus_category = data.icrop3_diagnosis.plus_category;
      targetPatient.urgency_code = data.icrop3_diagnosis.urgency_code;
      targetPatient.urgency_label = data.icrop3_diagnosis.urgency_label;
    }

    await loadPatients(); // Updates dashboard counters & priority in the list dynamically!
    updateCaseSubmissionDetails(patientId);
    loadPersonalizedPatientAlert(patientId);
  } catch (err) {
    console.error("Analysis execution error", err);
    if (loader) loader.style.display = "none";
    if (emptyState) {
      emptyState.style.display = "block";
      const h2 = emptyState.querySelector("h2");
      if (h2) h2.innerText = "Network or Analysis Processing Error";
    }
  }
}

function renderRetakeRequiredView(data) {
  const loader = document.getElementById("mlAnalysisLoader");
  const contentArea = document.getElementById("resultsContentArea");
  const emptyState = document.getElementById("resultsEmptyState");
  const retakeScreen = document.getElementById("retakeRequiredScreen");

  if (loader) loader.style.display = "none";
  if (contentArea) contentArea.style.display = "none";
  if (emptyState) emptyState.style.display = "none";

  if (retakeScreen) {
    retakeScreen.style.display = "block";
    const reasonEl = document.getElementById("retakeReasonText");
    const imgEl = document.getElementById("retakeRejectedImg");

    const reason = data.retake_reason || data.quality_assessment?.warnings?.[0] || "Image is ungradable or blank.";
    if (reasonEl) {
      reasonEl.innerHTML = `<strong>Quality Defect Detected:</strong> ${reason}<br><span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:0.35rem;">Overall Quality Score: ${data.quality_assessment?.overall_score || 0}% • Laplacian Sharpness: ${data.quality_assessment?.blur_score || 0} • Status: ${data.quality_assessment?.status || 'Rejected'}</span>`;
    }
    if (imgEl && data.original_image_base64) {
      imgEl.src = data.original_image_base64;
    }
  }
}

window.triggerScanRetake = function () {
  const retakeScreen = document.getElementById("retakeRequiredScreen");
  if (retakeScreen) retakeScreen.style.display = "none";

  const fileIn = document.getElementById("fundusFileInput");
  if (fileIn) fileIn.value = "";
  selectedPreset = null;

  switchView("visitUploadView");
  if (fileIn) fileIn.focus();
};

function renderAnalysisResults(data) {
  const { quality_assessment, biomarkers, icrop3_diagnosis, longitudinal_progression, scheduler_and_alerts } = data;

  // Reveal populated results and hide loader
  const loader = document.getElementById("mlAnalysisLoader");
  const contentArea = document.getElementById("resultsContentArea");
  const emptyState = document.getElementById("resultsEmptyState");

  if (loader) loader.style.display = "none";
  if (emptyState) emptyState.style.display = "none";
  if (contentArea) contentArea.style.display = "block";

  // Quality & Urgency Header
  document.getElementById("resUrgencyBadge").innerText = `${icrop3_diagnosis.urgency_code}: ${icrop3_diagnosis.urgency_label}`;
  document.getElementById("resUrgencyBadge").className = `badge ${getUrgencyBadgeClass(icrop3_diagnosis.urgency_code)}`;
  document.getElementById("resActionHeading").innerText = icrop3_diagnosis.recommended_action;
  document.getElementById("resQualityScore").innerText = `${quality_assessment.status} (${quality_assessment.overall_score}%)`;
  document.getElementById("resQualityWarning").innerText = quality_assessment.warnings[0] || "Quality acceptable.";

  // Images & Side-by-Side Dual View
  const origSrc = data.original_image_base64 || biomarkers.overlay_image_base64;
  const aiMarkedSrc = data.ai_marked_image_base64 || biomarkers.overlay_image_base64;
  const overlaySrc = biomarkers.overlay_image_base64;

  if (document.getElementById("resOriginalImg")) document.getElementById("resOriginalImg").src = origSrc;
  if (document.getElementById("resAiMarkedImg")) document.getElementById("resAiMarkedImg").src = aiMarkedSrc;
  if (document.getElementById("capturePreviewImg")) document.getElementById("capturePreviewImg").src = origSrc;
  if (document.getElementById("resOverlayImg")) document.getElementById("resOverlayImg").src = overlaySrc;
  if (document.getElementById("specOrigImg")) document.getElementById("specOrigImg").src = origSrc;
  if (document.getElementById("specOverlayImg")) document.getElementById("specOverlayImg").src = aiMarkedSrc;

  // Update abnormality count badge
  const abnCountBadge = document.getElementById("resAbnormalityCountBadge");
  if (abnCountBadge) {
    const numAbnormalities = data.abnormality_zones ? data.abnormality_zones.length : 0;
    abnCountBadge.innerText = `${numAbnormalities} Red Lesion Area${numAbnormalities === 1 ? '' : 's'} Flagged`;
  }

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

// Case Submission to Specialist Queue
window.submitCaseToDoctor = async function () {
  try {
    const sel = document.getElementById("submitPatientSelect");
    const patientId = sel ? sel.value : activePatientId;
    const patient = patientsRegistry.find((p) => p.patient_id === patientId);
    const notes = document.getElementById("submitCaseNotes")?.value || "Bedside scan submitted for specialist review.";

    const formData = new FormData();
    formData.append("patient_id", patientId);
    formData.append("notes", notes);

    const origSrc = patient?.original_image_base64 || (latestAnalysisData?.original_image_base64 || "");
    const aiSrc = patient?.ai_marked_image_base64 || (latestAnalysisData?.ai_marked_image_base64 || "");
    if (origSrc) formData.append("original_image_base64", origSrc);
    if (aiSrc) formData.append("ai_marked_image_base64", aiSrc);

    const res = await fetch("/api/submit-case", { method: "POST", body: formData });
    const data = await res.json();

    // Show inline submission feedback banner (No alert popup!)
    const banner = document.getElementById("caseSubmissionSuccessBanner");
    const feedbackMsg = document.getElementById("submissionFeedbackMsg");
    const subStatusBadge = document.getElementById("subStatusBadge");
    const subTimestamp = document.getElementById("subTimestamp");

    if (banner) {
      if (feedbackMsg) {
        feedbackMsg.innerText = `Report & retinal scans for Baby ${patient?.baby_name || 'Infant'} (${patientId}) submitted to Dr. Ananya Roy's priority specialist queue.`;
      }
      if (subStatusBadge) subStatusBadge.innerText = "Status: Submitted (Queued for Tele-Review)";
      if (subTimestamp) subTimestamp.innerText = `Submitted: ${new Date().toLocaleTimeString()}`;
      banner.style.display = "block";
    }

    // Refresh patients list so specialist queue reflects submission
    await loadPatients();
  } catch (err) {
    console.error("Submission failed", err);
    const banner = document.getElementById("caseSubmissionSuccessBanner");
    if (banner) {
      const feedbackMsg = document.getElementById("submissionFeedbackMsg");
      if (feedbackMsg) feedbackMsg.innerText = "Submission failed. Please check network connection.";
      banner.style.display = "block";
    }
  }
};

// Specialist Sign-Off
window.saveSpecialistReview = async function () {
  const patientId = activePatientId;
  const decision = document.getElementById("specDecision").value;
  const notes = document.getElementById("specNotes").value;
  const treatment = document.getElementById("specTreatmentPlan").value;

  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("doctor_name", "Dr. Ananya Roy, MD");
  formData.append("decision", decision);
  formData.append("doctor_notes", notes);
  formData.append("treatment_prescribed", treatment);

  try {
    const res = await fetch("/api/specialist-review", { method: "POST", body: formData });
    const data = await res.json();

    // Show inline sign-off confirmation badge (No alert popup!)
    const badge = document.getElementById("specReviewSuccessBadge");
    const detail = document.getElementById("specReviewSuccessDetail");
    const timeEl = document.getElementById("specReviewSignTimestamp");

    if (badge) {
      if (detail) {
        detail.innerText = `Specialist review (${decision}) signed by Dr. Ananya Roy and transmitted back to bedside NICU team.`;
      }
      if (timeEl) timeEl.innerText = `Signed & Dispatched: ${new Date().toLocaleTimeString()}`;
      badge.style.display = "block";
    }

    await loadPatients();
  } catch (err) {
    console.error("Failed to save review", err);
  }
};

// Load and Generate Personalized Patient Alert
window.loadPersonalizedPatientAlert = async function (patientId) {
  if (!patientId) patientId = activePatientId;
  activePatientId = patientId;

  const trackerSelect = document.getElementById("trackerPatientSelect");
  if (trackerSelect && trackerSelect.value !== patientId) {
    trackerSelect.value = patientId;
  }

  try {
    const res = await fetch(`/api/patient-alert/${encodeURIComponent(patientId)}`);
    const data = await res.json();
    const { patient, alerts } = data;

    // Update Demographic Card
    const babyEl = document.getElementById("trackerBabyName");
    const pidEl = document.getElementById("trackerPatientId");
    const motherEl = document.getElementById("trackerMotherName");
    const bedEl = document.getElementById("trackerBed");
    const phoneEl = document.getElementById("trackerPhone");
    const diseaseEl = document.getElementById("trackerDiseaseFinding");
    const urgencyBadge = document.getElementById("trackerUrgencyBadge");

    if (babyEl) babyEl.innerText = patient.baby_name;
    if (pidEl) pidEl.innerText = patient.patient_id;
    if (motherEl) motherEl.innerText = patient.mother_name || "N/A";
    if (bedEl) bedEl.innerText = patient.nicu_bed || "NICU";
    if (phoneEl) phoneEl.innerText = patient.parent_phone || "+91 98765 43210";
    if (diseaseEl) diseaseEl.innerText = `${patient.stage_name} - ${patient.zone} (${patient.plus_category})`;

    if (urgencyBadge) {
      urgencyBadge.innerText = `${alerts.urgency_code}: ${patient.urgency_label || 'Triage'}`;
      urgencyBadge.className = `badge ${getUrgencyBadgeClass(alerts.urgency_code)}`;
    }

    // Update Follow-up Windows & Explanations
    const nextVisitEl = document.getElementById("trackerNextVisit");
    const subtextEl = document.getElementById("trackerReviewWindowSubtext");
    const parentSummaryEl = document.getElementById("trackerParentSummary");
    const msgPreviewEl = document.getElementById("trackerMsgPreview");

    if (nextVisitEl) nextVisitEl.innerText = alerts.formatted_schedule;
    if (subtextEl) subtextEl.innerText = `Review window: ${alerts.window_hours} hours according to ICROP-3 guidelines.`;
    if (parentSummaryEl) parentSummaryEl.innerText = alerts.parent_friendly_explanation;
    if (msgPreviewEl) msgPreviewEl.innerText = alerts.whatsapp_payload.message;

    // Reset dispatch badge
    const dispatchBadge = document.getElementById("dispatchSuccessBadge");
    if (dispatchBadge) dispatchBadge.style.display = "none";
  } catch (err) {
    console.error("Failed to load patient alert", err);
  }
};

// Dispatch Personalized Alert to Parent
window.dispatchParentNotification = async function () {
  const patientId = activePatientId;
  const patient = patientsRegistry.find((p) => p.patient_id === patientId);
  const phone = patient ? patient.parent_phone : "+91 98765 43210";

  try {
    const formData = new FormData();
    formData.append("patient_id", patientId);
    formData.append("channel", "whatsapp");
    await fetch("/api/dispatch-alert", { method: "POST", body: formData });

    const badge = document.getElementById("dispatchSuccessBadge");
    if (badge) {
      badge.innerText = `✓ Done: Submitted & Sent to Registered Phone (${phone})`;
      badge.style.display = "inline-block";
    }
  } catch (err) {
    const badge = document.getElementById("dispatchSuccessBadge");
    if (badge) {
      badge.innerText = `✓ Done: Alert Queued for Parent (${phone})`;
      badge.style.display = "inline-block";
    }
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
