// ===== CONSTANTS =====
const POSES = {
  front: "Look straight at the camera",
  left: "Turn your face to the left",
  right: "Turn your face to the right",
  down: "Tilt your head down",
};

const QUALITY_WARNINGS = {
  blur: "Image is blurry! Please maintain your position.",
  dark: "Too dark! Please increase lighting.",
  bright: "Too bright! Please reduce lighting.",
  multiple_faces:
    "Multiple faces detected! Please keep only one face in the frame.",
  no_face:
    "No face detected! Please position your face in the center of the frame.",
  outside_roi: "Please place your face in the center capture area.",
  wrong_angle: "Incorrect face angle! Please follow the instructions.",
};

const COUNTDOWN_START = 3;
const VALIDATION_INTERVAL_MS = 500;
const CAMERA_SETTINGS = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: "user",
  },
};

// ===== APPLICATION STATE =====
const state = {
  sessionId: null,
  personName: "",
  currentPose: "front",
  completedPoses: {},
  processingCapture: false,
  stream: null,
  validationInterval: null,
  countdownTimer: null,
  captureInProgress: false,
  availableCameras: [],
};

// ===== DOM ELEMENTS =====
// Cache all DOM elements for better performance
const elements = {
  step1: document.getElementById("step1"),
  step2: document.getElementById("step2"),
  step3: document.getElementById("step3"),
  step4: document.getElementById("step4"),
  personName: document.getElementById("personName"),
  startBtn: document.getElementById("startBtn"),
  nameError: document.getElementById("nameError"),
  videoElement: document.getElementById("videoElement"),
  faceOverlay: document.getElementById("faceOverlay"),
  roiOverlay: document.getElementById("roiOverlay"),
  poseOverlay: document.getElementById("poseOverlay"),
  issuesOverlay: document.getElementById("issuesOverlay"),
  countdownOverlay: document.getElementById("countdownOverlay"),
  flashOverlay: document.getElementById("flashOverlay"),
  poseInstructions: document.getElementById("poseInstructions"),
  manualCaptureBtn: document.getElementById("manualCaptureBtn"),
  cancelBtn: document.getElementById("cancelBtn"),
  previewContainer: document.getElementById("previewContainer"),
  successMessage: document.getElementById("successMessage"),
  errorMessage: document.getElementById("errorMessage"),
  successText: document.getElementById("successText"),
  errorText: document.getElementById("errorText"),
  doneBtn: document.getElementById("doneBtn"),
  cameraSelect: document.getElementById("cameraSelect"),
};

// ===== EVENT LISTENERS =====
elements.startBtn.addEventListener("click", startCapture);
elements.manualCaptureBtn.addEventListener("click", captureManually);
elements.cancelBtn.addEventListener("click", cancelProcess);
elements.doneBtn.addEventListener("click", resetProcess);
elements.personName.addEventListener("keypress", (e) => {
  if (e.key === "Enter") startCapture();
});
elements.cameraSelect.addEventListener("change", switchCamera);

// ===== CAMERA MANAGEMENT =====

/**
 * Lists available cameras and updates dropdown
 */
function enumerateWebcams() {
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      state.availableCameras = videoDevices;

      // Create camera options HTML
      const optionsHtml = videoDevices.length
        ? videoDevices
            .map(
              (device, idx) =>
                `<option value="${device.deviceId}">${
                  device.label || `Camera ${idx + 1}`
                }</option>`
            )
            .join("")
        : '<option value="">No cameras found</option>';

      elements.cameraSelect.innerHTML = optionsHtml;
    })
    .catch((err) => {
      console.error("Error listing cameras:", err);
      elements.cameraSelect.innerHTML =
        '<option value="">Error loading cameras</option>';
    });
}

/**
 * Switch to the selected camera
 */
function switchCamera() {
  if (!state.stream) return;

  // Stop current camera stream
  state.stream.getTracks().forEach((track) => track.stop());

  // Start new camera with selected device
  initCamera()
    .then(() => {
      // Resume validation if active
      if (state.validationInterval) {
        startValidation();
      }
    })
    .catch((error) => {
      console.error("Camera switch error:", error);
      showIssue(`Failed to switch camera: ${error.message}`);
    });
}

/**
 * Initialize camera with current settings
 */
async function initCamera() {
  try {
    // Use selected camera if available
    const selectedDeviceId = elements.cameraSelect.value;
    const constraints = {
      video: selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId } }
        : CAMERA_SETTINGS.video,
    };

    // Stop existing stream if any
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
    }

    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.videoElement.srcObject = state.stream;

    // Wait for video to load
    await new Promise((resolve) => {
      elements.videoElement.onloadedmetadata = resolve;
    });

    // After camera permission, update labels
    if (!state.availableCameras.some((cam) => cam.label)) {
      setTimeout(enumerateWebcams, 500);
    }

    setupROI();
    initializeFaceOverlay();
    return true;
  } catch (error) {
    throw new Error(`Camera access error: ${error.message}`);
  }
}

/**
 * Setup the region of interest indicator
 */
function setupROI() {
  const video = elements.videoElement;
  const roi = elements.roiOverlay;

  // Make ROI 1/3 of the minimum video dimension
  const size = Math.min(video.videoWidth, video.videoHeight) / 3;
  const centerX = video.videoWidth / 2;
  const centerY = video.videoHeight / 2;

  // Set ROI position as percentage of video dimensions
  roi.style.left = `${((centerX - size) / video.videoWidth) * 100}%`;
  roi.style.top = `${((centerY - size) / video.videoHeight) * 100}%`;
  roi.style.width = `${((size * 2) / video.videoWidth) * 100}%`;
  roi.style.height = `${((size * 2) / video.videoHeight) * 100}%`;
}

/**
 * Initialize the face detection overlay canvas
 */
function initializeFaceOverlay() {
  elements.faceOverlay.width = elements.videoElement.videoWidth;
  elements.faceOverlay.height = elements.videoElement.videoHeight;
}

/**
 * Stop camera and clear all timers
 */
function stopCamera() {
  // Clear all timers
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }

  cancelCountdown();

  // Stop camera
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    elements.videoElement.srcObject = null;
  }

  clearFaceOverlay();
}

/**
 * Clear the face detection overlay
 */
function clearFaceOverlay() {
  const canvas = elements.faceOverlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ===== SESSION MANAGEMENT =====

/**
 * Start the capture process
 */
async function startCapture() {
  const name = elements.personName.value.trim();
  if (!name) {
    elements.nameError.classList.remove("hidden");
    return;
  }

  elements.nameError.classList.add("hidden");
  elements.startBtn.disabled = true;

  try {
    // Show loading indicator
    showButtonLoading(elements.startBtn, "Loading...");

    // Start session with backend
    const session = await startCaptureSession(name);

    // Save session data
    state.sessionId = session.sessionId;
    state.personName = session.personName;
    state.currentPose = session.currentPose;

    // Setup UI and camera
    showStep(2);
    enumerateWebcams();
    await initCamera();
    updatePoseUI(state.currentPose);
    startValidation();
  } catch (error) {
    console.error("Capture initialization error:", error);
    showError(`Failed to initialize: ${error.message}`);
    resetButtonState(elements.startBtn, "Start Capture");
  }
}

/**
 * Show loading state on button
 */
function showButtonLoading(button, text) {
  button.innerHTML = `
    <div class="inline-block animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
    ${text}`;
}

/**
 * Reset button to normal state
 */
function resetButtonState(button, text) {
  button.disabled = false;
  button.innerHTML = text;
}

/**
 * Start a capture session with the server
 */
async function startCaptureSession(name) {
  try {
    const response = await fetch("/api/person/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });

    // Check for error responses
    if (!response.ok) {
      const errorData = await getErrorData(response);
      throw new Error(errorData);
    }

    const data = await response.json();

    // Handle existing person case
    if (
      data.existing_person &&
      !(await confirmPersonOverwrite(data.person_name))
    ) {
      await cancelSession(data.session_id);
      throw new Error("Session cancelled by user");
    }

    // Preload face detector
    await preloadDetector(data.session_id, data.current_pose);

    return {
      sessionId: data.session_id,
      personName: data.person_name,
      currentPose: data.current_pose,
      existingPerson: data.existing_person,
    };
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

/**
 * Extract error data from response
 */
async function getErrorData(response) {
  try {
    const errorData = await response.json();
    return errorData.detail || `Server error: ${response.status}`;
  } catch {
    const errorText = await response.text();
    return errorText || `Server error: ${response.status}`;
  }
}

/**
 * Cancel a session on the server
 */
async function cancelSession(sessionId) {
  await fetch(`/api/person/cancel/${sessionId}`, {
    method: "DELETE",
  });
}

/**
 * Confirm if user wants to overwrite existing person
 */
async function confirmPersonOverwrite(name) {
  return confirm(
    `Person "${name}" already exists. Do you want to replace the existing data?`
  );
}

/**
 * Preload face detector with empty image
 */
async function preloadDetector(sessionId, pose) {
  // Create blank canvas for initial detector loading
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const blob = await new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
  });

  const formData = new FormData();
  formData.append("file", blob);
  formData.append("session_id", sessionId);
  formData.append("pose", pose);

  await fetch("/api/person/validate_image", {
    method: "POST",
    body: formData,
  });
}

// ===== UI MANAGEMENT =====

/**
 * Show a specific step and hide others
 */
function showStep(stepNumber) {
  for (let i = 1; i <= 4; i++) {
    elements[`step${i}`].classList.toggle("hidden", i !== stepNumber);
  }
}

/**
 * Update UI for the current pose
 */
function updatePoseUI(pose) {
  elements.poseOverlay.textContent = `Pose: ${pose.toUpperCase()}`;
  elements.poseInstructions.textContent = POSES[pose] || "";
  elements.manualCaptureBtn.disabled = false;
  updatePreviewItems();
}

/**
 * Update preview items for all poses
 */
function updatePreviewItems() {
  elements.previewContainer.innerHTML = "";

  // Create preview for each pose type
  Object.keys(POSES).forEach((pose) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    const isCaptured = state.completedPoses[pose];
    const isCurrentPose = pose === state.currentPose;

    // Create pose name element
    const nameEl = document.createElement("p");
    nameEl.className = "font-medium " + (isCurrentPose ? "text-blue-400" : "");
    nameEl.textContent = pose.charAt(0).toUpperCase() + pose.slice(1);

    // Create appropriate preview based on pose status
    isCaptured
      ? createCapturedPosePreview(item, nameEl, pose)
      : createPendingPosePreview(item, nameEl, isCurrentPose);

    elements.previewContainer.appendChild(item);
  });
}

/**
 * Create preview for a captured pose
 */
function createCapturedPosePreview(item, nameEl, pose) {
  // Create image element
  const img = document.createElement("img");
  img.src = state.completedPoses[pose];
  img.alt = `${pose} pose`;
  img.className = "mt-2";

  // Create status indicator
  const statusEl = document.createElement("span");
  statusEl.className =
    "inline-block mt-2 px-2 py-1 text-xs font-medium bg-green-900/30 text-green-400 rounded";
  statusEl.innerHTML = '<i class="fas fa-check mr-1"></i>Captured';

  // Combine elements
  const headerEl = document.createElement("div");
  headerEl.className = "flex justify-between items-center";
  headerEl.appendChild(nameEl);
  headerEl.appendChild(statusEl);

  item.appendChild(headerEl);
  item.appendChild(img);
}

/**
 * Create preview for a pending pose
 */
function createPendingPosePreview(item, nameEl, isCurrentPose) {
  item.appendChild(nameEl);

  // Create placeholder
  const placeholder = document.createElement("div");
  placeholder.className =
    "h-32 bg-slate-800/50 rounded flex items-center justify-center text-slate-400 mt-2";

  if (isCurrentPose) {
    placeholder.innerHTML = '<i class="fas fa-camera mr-2"></i>Current';
    placeholder.className += " border border-blue-500/30";
  } else {
    placeholder.innerHTML = '<i class="fas fa-clock mr-2"></i>Pending';
  }

  item.appendChild(placeholder);
}

/**
 * Show success message and finish process
 */
function showSuccessMessage(message) {
  stopAllProcesses();
  elements.successMessage.classList.remove("hidden");
  elements.errorMessage.classList.add("hidden");
  elements.successText.textContent = message;
  showStep(4);
}

/**
 * Show error message
 */
function showError(message) {
  stopAllProcesses();
  elements.errorMessage.classList.remove("hidden");
  elements.successMessage.classList.add("hidden");
  elements.errorText.textContent = message;
  showStep(4);
}

/**
 * Stop all active processes
 */
function stopAllProcesses() {
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }
  stopCamera();
}

// ===== FACE VALIDATION =====

/**
 * Start face validation interval
 */
function startValidation() {
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
  }
  state.validationInterval = setInterval(validateFrame, VALIDATION_INTERVAL_MS);
}

/**
 * Validate current video frame
 */
async function validateFrame(silentMode = false) {
  if (!state.stream || !state.sessionId) return false;

  try {
    // Capture and validate current frame
    const blob = await captureFrameAsBlob();
    const formData = createValidationFormData(blob);

    try {
      const response = await fetch("/api/person/validate_image", {
        method: "POST",
        body: formData,
      });

      // Handle 404 (session ended)
      if (response.status === 404) {
        if (!silentMode) {
          clearInterval(state.validationInterval);
          state.validationInterval = null;
          console.log("Session ended, stopping validation");
        }
        return false;
      }

      if (!response.ok) {
        throw new Error(`Validation failed: ${response.status}`);
      }

      const result = await response.json();

      // Update UI if not in silent mode
      if (!silentMode) {
        updateValidationUI(result);
      }

      return result.valid;
    } catch (fetchError) {
      // Handle network errors
      if (!silentMode && state.validationInterval) {
        clearInterval(state.validationInterval);
        state.validationInterval = null;
        console.error("Validation stopped:", fetchError);
      }
      return false;
    }
  } catch (error) {
    console.error("Frame validation error:", error);
    if (!silentMode) {
      clearInterval(state.validationInterval);
      showIssue("Validation error: " + error.message);
    }
    return false;
  }
}

/**
 * Create form data for validation request
 */
function createValidationFormData(blob) {
  const formData = new FormData();
  formData.append("file", blob);
  formData.append("session_id", state.sessionId);
  formData.append("pose", state.currentPose);
  return formData;
}

/**
 * Capture current video frame as blob
 */
async function captureFrameAsBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = elements.videoElement.videoWidth;
  canvas.height = elements.videoElement.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(elements.videoElement, 0, 0);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
  });
}

/**
 * Update UI based on validation results
 */
function updateValidationUI(validation) {
  // Handle validation issues
  if (validation.issues && validation.issues.length > 0) {
    handleValidationIssues(validation.issues);
  } else {
    handleValidationSuccess(validation);
  }

  // Update face visualization
  updateFaceVisualization(validation.pose_data?.faces);
}

/**
 * Handle validation issues
 */
function handleValidationIssues(issues) {
  // Show all issues as combined message
  const messages = issues.map((issue) => QUALITY_WARNINGS[issue] || issue);
  showIssue(messages.join(" "));

  // Cancel any countdown and disable manual capture
  cancelCountdown();
  elements.manualCaptureBtn.disabled = true;
}

/**
 * Handle successful validation
 */
function handleValidationSuccess(validation) {
  // Hide issues and enable manual capture
  hideIssue();
  elements.manualCaptureBtn.disabled = false;

  // Check if we can start auto-capture
  const hasFace = validation.pose_data?.faces?.length > 0;

  if (!state.captureInProgress && hasFace) {
    startAutoCaptureCountdown();
  }
}

/**
 * Show issue message
 */
function showIssue(message) {
  elements.issuesOverlay.classList.remove("hidden");
  elements.issuesOverlay.textContent = message;
}

/**
 * Hide issue message
 */
function hideIssue() {
  elements.issuesOverlay.classList.add("hidden");
}

/**
 * Update face detection visualization
 */
function updateFaceVisualization(faces) {
  if (faces && faces.length > 0) {
    drawFaceBoxes(faces);
  } else {
    clearFaceOverlay();
  }
}

/**
 * Draw face bounding boxes and landmarks
 */
function drawFaceBoxes(faces) {
  const canvas = elements.faceOverlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  faces.forEach((face) => {
    const [x1, y1, x2, y2] = face.bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    // Draw bounding box
    ctx.strokeStyle = "#10B981"; // Green
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, width, height);

    // Draw confidence
    ctx.fillStyle = "#10B981";
    ctx.font = "14px Arial";
    ctx.fillText(
      `${Math.round(face.confidence * 100)}%`,
      x1,
      y1 > 20 ? y1 - 5 : y1 + 15
    );

    // Draw landmarks
    if (face.landmarks && face.landmarks.length > 0) {
      ctx.fillStyle = "#ef4444"; // Red
      face.landmarks.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  });
}

// ===== CAPTURE MANAGEMENT =====

/**
 * Start automatic capture countdown
 */
function startAutoCaptureCountdown() {
  cancelCountdown(); // Cancel any existing countdown

  // Start new countdown
  state.captureInProgress = true;
  elements.countdownOverlay.classList.remove("hidden");

  let countdown = COUNTDOWN_START;
  elements.countdownOverlay.textContent = countdown;

  // Define countdown update function
  const updateCountdown = async () => {
    // Re-validate before continuing
    const isValid = await validateFrame(true);

    // Cancel if validation fails
    if (!isValid) {
      cancelCountdown();
      return;
    }

    countdown--;

    if (countdown <= 0) {
      elements.countdownOverlay.classList.add("hidden");
      captureCurrentPose();
      return;
    }

    elements.countdownOverlay.textContent = countdown;
    state.countdownTimer = setTimeout(updateCountdown, 1000);
  };

  // Start countdown
  state.countdownTimer = setTimeout(updateCountdown, 1000);
}

/**
 * Cancel active countdown
 */
function cancelCountdown() {
  if (state.countdownTimer) {
    clearTimeout(state.countdownTimer);
    state.countdownTimer = null;
    elements.countdownOverlay.classList.add("hidden");
  }
  state.captureInProgress = false;
}

/**
 * Manual capture trigger
 */
async function captureManually() {
  if (state.processingCapture) return;

  cancelCountdown();
  state.captureInProgress = true;
  await captureCurrentPose();
}

/**
 * Capture current pose and send to server
 */
async function captureCurrentPose() {
  state.processingCapture = true;
  elements.manualCaptureBtn.disabled = true;

  try {
    // Capture frame with flash effect
    const blob = await captureFrameWithFlash();

    // Send to server
    const formData = new FormData();
    formData.append("file", blob);
    formData.append("session_id", state.sessionId);
    formData.append("pose", state.currentPose);

    const response = await fetch("/api/person/capture", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Capture request failed");
    }

    const result = await response.json();

    // Handle result
    if (result.success) {
      handleSuccessfulCapture(result);
    } else {
      showIssue("Capture failed: " + result.message);
    }
  } catch (error) {
    console.error("Capture error:", error);
    showIssue("Capture error: " + error.message);
  } finally {
    state.captureInProgress = false;
    state.processingCapture = false;
    elements.manualCaptureBtn.disabled = false;
  }
}

/**
 * Capture frame with flash effect
 */
async function captureFrameWithFlash() {
  // Capture frame
  const canvas = document.createElement("canvas");
  canvas.width = elements.videoElement.videoWidth;
  canvas.height = elements.videoElement.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(elements.videoElement, 0, 0);

  // Show flash effect
  elements.flashOverlay.classList.add("flash-active");
  setTimeout(() => {
    elements.flashOverlay.classList.remove("flash-active");
  }, 500);

  // Return as blob
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
  });
}

/**
 * Handle successful capture
 */
async function handleSuccessfulCapture(result) {
  // Update captured pose in state
  state.completedPoses[result.pose] = result.image_path;

  // Check if process is complete
  if (result.completed) {
    await completeCapture();
  } else {
    // Move to next pose
    state.currentPose = result.next_pose;
    updatePoseUI(state.currentPose);
  }
}

/**
 * Complete capture process and finalize
 */
async function completeCapture() {
  // Stop validation immediately
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }

  stopCamera();
  showStep(3);

  try {
    // Finalize on server
    const response = await fetch(`/api/person/finalize/${state.sessionId}`, {
      method: "POST",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to finalize person");
    }

    // Clear session to prevent further API calls
    state.sessionId = null;

    // Update local storage flags
    localStorage.setItem("datasetUpdated", "true");
    localStorage.setItem("lastAddedPerson", state.personName);

    // Show success message
    showSuccessMessage(
      data.message || "Person successfully added to the recognition system"
    );
  } catch (error) {
    console.error("Finalization error:", error);
    showError("Failed to complete: " + error.message);
  }
}

// ===== PROCESS MANAGEMENT =====

/**
 * Cancel current capture process
 */
async function cancelProcess() {
  if (!confirm("Are you sure you want to cancel? All progress will be lost.")) {
    return;
  }

  try {
    // Stop processes before canceling on server
    stopCamera();

    if (state.sessionId) {
      await fetch(`/api/person/cancel/${state.sessionId}`, {
        method: "DELETE",
      });
    }
  } catch (error) {
    console.error("Cancel error:", error);
  } finally {
    resetProcess();
  }
}

/**
 * Reset application state
 */
function resetProcess() {
  // Reset state
  Object.assign(state, {
    sessionId: null,
    personName: "",
    currentPose: "front",
    completedPoses: {},
    processingCapture: false,
    stream: null,
    validationInterval: null,
    countdownTimer: null,
    captureInProgress: false,
  });

  // Return to main page if completed
  if (!elements.step4.classList.contains("hidden")) {
    window.location.href = "/";
    return;
  }

  // Reset UI
  elements.personName.value = "";
  elements.startBtn.disabled = false;
  elements.startBtn.innerHTML = "Start Capture";
  showStep(1);
}

// Don't enumerate cameras on page load - wait for user interaction
