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

// ===== CAMERA ENUMERATION =====

/**
 * Enumerate available webcams and populate the dropdown
 */
function enumerateWebcams() {
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      state.availableCameras = videoDevices;

      // Populate the dropdown
      elements.cameraSelect.innerHTML = videoDevices.length
        ? videoDevices
            .map(
              (device, idx) =>
                `<option value="${device.deviceId}">${
                  device.label || `Camera ${idx + 1}`
                }</option>`
            )
            .join("")
        : '<option value="">No cameras found</option>';
    })
    .catch((err) => {
      console.error("Error enumerating devices: ", err);
      elements.cameraSelect.innerHTML =
        '<option value="">Error loading cameras</option>';
    });
}

/**
 * Switch to the selected camera
 */
function switchCamera() {
  if (state.stream) {
    // Stop current camera
    state.stream.getTracks().forEach((track) => track.stop());

    // Start new camera with selected device
    initCamera()
      .then(() => {
        // Resume validation if it was active
        if (state.validationInterval) {
          startValidation();
        }
      })
      .catch((error) => {
        console.error("Error switching camera:", error);
        showIssue("Failed to switch camera: " + error.message);
      });
  }
}

// ===== SESSION MANAGEMENT FUNCTIONS =====

/**
 * Starts the capture process by initiating a session and setting up the camera
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
    elements.startBtn.innerHTML =
      '<div class="inline-block animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2"></div> Loading...';

    // Start a new session with the backend
    const { sessionId, personName, currentPose, existingPerson } =
      await startCaptureSession(name);

    // Save session data
    state.sessionId = sessionId;
    state.personName = personName;
    state.currentPose = currentPose;

    // Setup UI and start camera
    showStep(2);

    // Enumerate webcams before initializing camera
    enumerateWebcams();

    await initCamera();
    updatePoseUI(state.currentPose);
    startValidation();
  } catch (error) {
    console.error("Error starting capture:", error);
    showError("Failed to initialize capture: " + error.message);
    elements.startBtn.disabled = false;
    elements.startBtn.innerHTML = "Start Capture";
  }
}

/**
 * Initiates a capture session with the backend
 * @param {string} name - The person's name
 * @returns {Object} Session information
 */
async function startCaptureSession(name) {
  // Start a new session
  try {
    const response = await fetch("/api/person/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });

    // Kiểm tra nếu response không phải 2xx
    if (!response.ok) {
      // Thử đọc lỗi dưới dạng JSON nếu có
      try {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      } catch (jsonError) {
        // Nếu không phải JSON, sử dụng text hoặc status code
        const errorText = await response.text();
        throw new Error(errorText || `Server error: ${response.status}`);
      }
    }

    const data = await response.json();

    // Check if person exists and confirm overwrite
    if (data.existing_person) {
      const confirmOverwrite = confirm(
        `Person "${data.person_name}" already exists in the dataset. Do you want to replace the existing data?`
      );
      if (!confirmOverwrite) {
        // Cancel session if user doesn't want to overwrite
        await fetch(`/api/person/cancel/${data.session_id}`, {
          method: "DELETE",
        });
        throw new Error("Session cancelled by user");
      }
    }

    // Initialize detector
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
 * Preloads the face detector by sending an initial validation request
 */
async function preloadDetector(sessionId, pose) {
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

// ===== CAMERA MANAGEMENT =====

/**
 * Initializes the camera and sets up video display
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

    // Wait for video to be ready
    await new Promise((resolve) => {
      elements.videoElement.onloadedmetadata = resolve;
    });

    // After getting permission, update camera labels if needed
    if (!state.availableCameras.some((cam) => cam.label)) {
      setTimeout(enumerateWebcams, 500);
    }

    setupROI();
    initializeFaceOverlay();

    return true;
  } catch (error) {
    console.error("Camera initialization error:", error);
    throw new Error("Could not access camera: " + error.message);
  }
}

/**
 * Sets up the region of interest overlay
 */
function setupROI() {
  const video = elements.videoElement;
  const roi = elements.roiOverlay;

  const size = Math.min(video.videoWidth, video.videoHeight) / 3;
  const centerX = video.videoWidth / 2;
  const centerY = video.videoHeight / 2;

  roi.style.left = `${((centerX - size) / video.videoWidth) * 100}%`;
  roi.style.top = `${((centerY - size) / video.videoHeight) * 100}%`;
  roi.style.width = `${((size * 2) / video.videoWidth) * 100}%`;
  roi.style.height = `${((size * 2) / video.videoHeight) * 100}%`;
}

/**
 * Initialize the face overlay canvas
 */
function initializeFaceOverlay() {
  elements.faceOverlay.width = elements.videoElement.videoWidth;
  elements.faceOverlay.height = elements.videoElement.videoHeight;
}

/**
 * Stops all camera and timing operations
 */
function stopCamera() {
  // Clear intervals and timers
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }

  cancelCountdown();

  // Stop camera stream
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    elements.videoElement.srcObject = null;
  }

  // Clear canvas
  clearFaceOverlay();
}

/**
 * Clears any drawings from the face overlay
 */
function clearFaceOverlay() {
  const canvas = elements.faceOverlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ===== UI MANAGEMENT =====

/**
 * Shows a specific step and hides others
 * @param {number} stepNumber - The step to show (1-4)
 */
function showStep(stepNumber) {
  for (let i = 1; i <= 4; i++) {
    elements[`step${i}`].classList.toggle("hidden", i !== stepNumber);
  }
}

/**
 * Updates UI based on the current pose
 * @param {string} pose - The pose to display
 */
function updatePoseUI(pose) {
  elements.poseOverlay.textContent = `Pose: ${pose.toUpperCase()}`;
  elements.poseInstructions.textContent = POSES[pose] || "";
  elements.manualCaptureBtn.disabled = false;
  updatePreviewItems();
}

/**
 * Updates the preview items for all poses
 */
function updatePreviewItems() {
  // Clear existing previews
  elements.previewContainer.innerHTML = "";

  // Create preview items for all poses
  Object.keys(POSES).forEach((pose) => {
    const item = document.createElement("div");
    item.className = "preview-item";

    // Check pose status
    const isCaptured = state.completedPoses[pose];
    const isCurrentPose = pose === state.currentPose;

    // Add pose name
    const nameEl = document.createElement("p");
    nameEl.className = "font-medium " + (isCurrentPose ? "text-blue-400" : "");
    nameEl.textContent = pose.charAt(0).toUpperCase() + pose.slice(1);

    if (isCaptured) {
      // Create captured pose display
      createCapturedPosePreview(item, nameEl, pose);
    } else {
      // Create pending pose display
      createPendingPosePreview(item, nameEl, isCurrentPose);
    }

    elements.previewContainer.appendChild(item);
  });
}

/**
 * Creates a preview item for a captured pose
 */
function createCapturedPosePreview(item, nameEl, pose) {
  const img = document.createElement("img");
  img.src = state.completedPoses[pose];
  img.alt = `${pose} pose`;
  img.className = "mt-2";

  // Add status indicator
  const statusEl = document.createElement("span");
  statusEl.className =
    "inline-block mt-2 px-2 py-1 text-xs font-medium bg-green-900/30 text-green-400 rounded";
  statusEl.innerHTML = '<i class="fas fa-check mr-1"></i>Captured';

  // Container for name and status
  const headerEl = document.createElement("div");
  headerEl.className = "flex justify-between items-center";
  headerEl.appendChild(nameEl);
  headerEl.appendChild(statusEl);

  item.appendChild(headerEl);
  item.appendChild(img);
}

/**
 * Creates a preview item for a pending pose
 */
function createPendingPosePreview(item, nameEl, isCurrentPose) {
  item.appendChild(nameEl);

  // Add placeholder
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
 * Shows a success message and completes the process
 */
function showSuccessMessage(message) {
  // Stop all validation and camera processes first
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }
  stopCamera();

  elements.successMessage.classList.remove("hidden");
  elements.errorMessage.classList.add("hidden");
  elements.successText.textContent = message;
  showStep(4);
}

/**
 * Shows an error message
 */
function showError(message) {
  // Stop all validation and camera processes first
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }
  stopCamera();

  elements.errorMessage.classList.remove("hidden");
  elements.successMessage.classList.add("hidden");
  elements.errorText.textContent = message;
  showStep(4);
}

// ===== FACE DETECTION & VALIDATION =====

/**
 * Starts the validation interval
 */
function startValidation() {
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
  }

  state.validationInterval = setInterval(validateFrame, VALIDATION_INTERVAL_MS);
}

/**
 * Validates the current frame for face detection and quality issues
 * @param {boolean} silentMode - If true, doesn't update the UI with results
 * @returns {boolean} Whether the frame passes validation
 */
async function validateFrame(silentMode = false) {
  if (!state.stream || !state.sessionId) return false;

  try {
    // Capture the current frame
    const blob = await captureFrameAsBlob();

    // Send to validation endpoint
    const formData = new FormData();
    formData.append("file", blob);
    formData.append("session_id", state.sessionId);
    formData.append("pose", state.currentPose);

    // Use try-catch specifically for the fetch operation
    try {
      const response = await fetch("/api/person/validate_image", {
        method: "POST",
        body: formData,
      });

      // Check for 404 (session ended) and stop validation if found
      if (response.status === 404) {
        if (!silentMode) {
          clearInterval(state.validationInterval);
          state.validationInterval = null;
          console.log("Session ended or not found, stopping validation");
        }
        return false;
      }

      if (!response.ok) {
        throw new Error(
          `Validation request failed with status: ${response.status}`
        );
      }

      const result = await response.json();

      // Update UI based on validation (only if not in silent mode)
      if (!silentMode) {
        updateValidationUI(result);
      }

      return result.valid;
    } catch (fetchError) {
      // Handle specifically network errors or server errors
      if (!silentMode && state.validationInterval) {
        clearInterval(state.validationInterval);
        state.validationInterval = null;
        console.error("Stopping validation due to API error:", fetchError);
      }
      return false;
    }
  } catch (error) {
    console.error("Error validating frame:", error);
    if (!silentMode) {
      clearInterval(state.validationInterval);

      // Show error in the issues overlay
      showIssue("Validation error: " + error.message);
    }
    return false;
  }
}

/**
 * Captures the current video frame as an image blob
 * @returns {Promise<Blob>} The captured frame as a JPEG blob
 */
async function captureFrameAsBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = elements.videoElement.videoWidth;
  canvas.height = elements.videoElement.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(elements.videoElement, 0, 0);

  return await new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
  });
}

/**
 * Updates the UI based on validation results
 * @param {Object} validation - The validation result object
 */
function updateValidationUI(validation) {
  // Check for issues
  if (validation.issues && validation.issues.length > 0) {
    // Show issues and handle UI elements
    handleValidationIssues(validation.issues);
  } else {
    // No issues, proceed with validation
    handleValidationSuccess(validation);
  }

  // Update face detection visualization
  updateFaceVisualization(validation.pose_data?.faces);
}

/**
 * Handles validation issues by showing warnings and canceling countdown
 * @param {string[]} issues - Array of issue codes
 */
function handleValidationIssues(issues) {
  // Show issues overlay
  const messages = issues.map((issue) => QUALITY_WARNINGS[issue] || issue);
  showIssue(messages.join(" "));

  // Always fully cancel any countdown in progress
  cancelCountdown();

  // Disable manual capture during issues
  elements.manualCaptureBtn.disabled = true;
}

/**
 * Handles successful validation by starting auto-capture if possible
 * @param {Object} validation - The validation result
 */
function handleValidationSuccess(validation) {
  // Hide any issues
  hideIssue();

  // Enable manual capture
  elements.manualCaptureBtn.disabled = false;

  // Check if we have a valid face for auto-capture
  const hasFace =
    validation.pose_data &&
    validation.pose_data.faces &&
    validation.pose_data.faces.length > 0;

  // Start auto-capture if not already in progress and we have a face
  if (!state.captureInProgress && hasFace) {
    startAutoCaptureCountdown();
  }
}

/**
 * Shows an issue message in the issues overlay
 * @param {string} message - The issue message to display
 */
function showIssue(message) {
  elements.issuesOverlay.classList.remove("hidden");
  elements.issuesOverlay.textContent = message;
}

/**
 * Hides the issues overlay
 */
function hideIssue() {
  elements.issuesOverlay.classList.add("hidden");
}

/**
 * Updates the face visualization based on detection results
 * @param {Array|null} faces - Array of detected faces or null
 */
function updateFaceVisualization(faces) {
  if (faces && faces.length > 0) {
    drawFaceBoxes(faces);
  } else {
    clearFaceOverlay();
  }
}

/**
 * Draws face bounding boxes and landmarks
 * @param {Array} faces - Array of face detection results
 */
function drawFaceBoxes(faces) {
  const canvas = elements.faceOverlay;
  const ctx = canvas.getContext("2d");

  // Clear previous drawings
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw each face
  faces.forEach((face) => {
    const [x1, y1, x2, y2] = face.bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    // Draw bounding box
    ctx.strokeStyle = "#10B981"; // Green
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, width, height);

    // Draw confidence score
    ctx.fillStyle = "#10B981";
    ctx.font = "14px Arial";
    ctx.fillText(
      `${Math.round(face.confidence * 100)}%`,
      x1,
      y1 > 20 ? y1 - 5 : y1 + 15
    );

    // Draw landmarks if available
    if (face.landmarks && face.landmarks.length > 0) {
      ctx.fillStyle = "#ef4444"; // Red dots for landmarks
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
 * Starts the auto-capture countdown
 * Completely new countdown each time - no resuming
 */
function startAutoCaptureCountdown() {
  // Cancel any existing countdown first
  cancelCountdown();

  // Start new countdown
  state.captureInProgress = true;
  elements.countdownOverlay.classList.remove("hidden");

  // Always start from COUNTDOWN_START (typically 3)
  let countdown = COUNTDOWN_START;
  elements.countdownOverlay.textContent = countdown;

  const updateCountdown = async () => {
    // Re-check validation before continuing
    const isValid = await validateFrame(true);

    // If validation fails at any point, cancel the entire countdown
    if (!isValid) {
      cancelCountdown();
      return;
    }

    countdown--;

    if (countdown <= 0) {
      // Capture when countdown reaches 0
      elements.countdownOverlay.classList.add("hidden");
      captureCurrentPose();
      return;
    }

    elements.countdownOverlay.textContent = countdown;
    state.countdownTimer = setTimeout(updateCountdown, 1000);
  };

  // Start the countdown process
  state.countdownTimer = setTimeout(updateCountdown, 1000);
}

/**
 * Cancels and resets any active countdown
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
 * Manually triggers a capture
 */
async function captureManually() {
  if (state.processingCapture) return;

  // Cancel any auto-capture countdown
  cancelCountdown();

  state.captureInProgress = true;
  await captureCurrentPose();
}

/**
 * Captures the current pose and sends it to the server
 */
async function captureCurrentPose() {
  state.processingCapture = true;
  elements.manualCaptureBtn.disabled = true;

  try {
    // Capture current frame
    const blob = await captureFrameWithFlash();

    // Send to capture endpoint
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

    if (result.success) {
      handleSuccessfulCapture(result);
    } else {
      showIssue("Capture failed: " + result.message);
    }
  } catch (error) {
    console.error("Error capturing pose:", error);
    showIssue("Capture error: " + error.message);
  } finally {
    state.captureInProgress = false;
    state.processingCapture = false;
    elements.manualCaptureBtn.disabled = false;
  }
}

/**
 * Captures a frame with a camera flash effect
 * @returns {Promise<Blob>} The captured frame as a JPEG blob
 */
async function captureFrameWithFlash() {
  // Capture the frame
  const canvas = document.createElement("canvas");
  canvas.width = elements.videoElement.videoWidth;
  canvas.height = elements.videoElement.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(elements.videoElement, 0, 0);

  // Flash effect
  elements.flashOverlay.classList.add("flash-active");
  setTimeout(() => {
    elements.flashOverlay.classList.remove("flash-active");
  }, 500);

  // Convert to blob
  return await new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
  });
}

/**
 * Handles a successful pose capture
 * @param {Object} result - The capture result from the server
 */
async function handleSuccessfulCapture(result) {
  // Update state with captured pose
  state.completedPoses[result.pose] = result.image_path;

  // Move to next pose or complete
  if (result.completed) {
    await completeCapture();
  } else {
    state.currentPose = result.next_pose;
    updatePoseUI(state.currentPose);
  }
}

/**
 * Completes the capture process and finalizes the person
 */
async function completeCapture() {
  // Stop validation interval immediately to prevent 404 errors
  if (state.validationInterval) {
    clearInterval(state.validationInterval);
    state.validationInterval = null;
  }

  // Stop camera processes
  stopCamera();

  // Show processing screen
  showStep(3);

  try {
    // Send finalize request
    const response = await fetch(`/api/person/finalize/${state.sessionId}`, {
      method: "POST",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to finalize person");
    }

    // Clear session ID after successful finalization to prevent further API calls
    state.sessionId = null;

    // Show success
    showSuccessMessage(
      data.message || "Person successfully added to the recognition system"
    );
  } catch (error) {
    console.error("Error finalizing capture:", error);
    showError("Failed to complete process: " + error.message);
  }
}

// ===== PROCESS MANAGEMENT =====

/**
 * Cancels the current capture process
 */
async function cancelProcess() {
  if (confirm("Are you sure you want to cancel? All progress will be lost.")) {
    try {
      // Stop all validation and camera processes FIRST before cancelling on server
      // This prevents additional API calls after session deletion
      stopCamera();

      if (state.sessionId) {
        await fetch(`/api/person/cancel/${state.sessionId}`, {
          method: "DELETE",
        });
      }
    } catch (error) {
      console.error("Error cancelling process:", error);
    } finally {
      resetProcess();
    }
  }
}

/**
 * Resets the application to its initial state
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

  // Reset UI
  elements.personName.value = "";
  elements.startBtn.disabled = false;
  elements.startBtn.innerHTML = "Start Capture";

  // Return to first step
  showStep(1);
}

// Initialize camera selection dropdown on page load
document.addEventListener("DOMContentLoaded", () => {
  // Don't enumerate cameras yet - wait until user clicks start
  // This avoids permission prompts on page load
});
