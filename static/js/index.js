/**
 * Face Recognition System Frontend
 * Provides real-time face detection, recognition, and security notifications
 */

// ===== Core State =====
const state = {
  socket: null,
  isRecognizing: false,
  lastFrameSent: 0,
  frameInterval: 100, // Limit to 10 fps to server
  totalFrames: 0,
  framesLastSecond: 0,
  currentFPS: 0,
  activeStream: null,
  availableCameras: [],
  detectedFaces: [],
  datasetUsers: [],
  faceIdentityMap: new Map(), // Maps track_id to recognized identity
  stats: {
    latency: 0,
    recognition_time: 0,
    detection_time: 0,
    spoof_detection_time: 0,
  },
};

// ===== DOM Elements =====
const elements = {
  // Video and display elements
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  faceBoxesContainer: document.getElementById("face-boxes-container"),
  cameraOverlay: document.getElementById("camera-overlay"),
  recordingIndicator: document.getElementById("recording-indicator"),

  // Lists and counters
  detectedFacesList: document.getElementById("detected-faces-list"),
  datasetUsersList: document.getElementById("dataset-users-list"),
  fpsCounter: document.getElementById("fps-counter"),
  detectedCount: document.getElementById("detected-count"),
  datasetCount: document.getElementById("dataset-count"),

  // Control buttons
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  addPersonBtn: document.getElementById("add-person-btn"),
  updateDatasetBtn: document.getElementById("update-dataset-btn"),
  cleanTempBtn: document.getElementById("clean-temp-btn"),
  testNotificationBtn: document.getElementById("test-notification-btn"),
  enableCameraBtn: document.getElementById("enable-camera-btn"),
  cameraSelect: document.getElementById("camera-select"),

  // Status and popups
  connectionStatus: document.getElementById("connection-status"),
  connectionStatusText: document.getElementById("connectionStatus"),
  statsBtn: document.getElementById("statsBtn"),
  statsPopup: document.getElementById("stats-popup"),
  showTechStackBtn: document.getElementById("showTechStackBtn"),
  closeTechStackBtn: document.getElementById("closeTechStackBtn"),
  techStackPopup: document.getElementById("techStackPopup"),

  // Stats display
  latencyStat: document.getElementById("latency-stat"),
  recogTimeStat: document.getElementById("recog-time-stat"),
  spoofTimeStat: document.getElementById("spoof-time-stat"),
  detectTimeStat: document.getElementById("detect-time-stat"),

  // Feature indicators
  featureFaceDetection: document.getElementById("feature-face-detection"),
  featureRecognition: document.getElementById("feature-recognition"),
  featureAntiSpoofing: document.getElementById("feature-anti-spoofing"),
  featureMaskDetection: document.getElementById("feature-mask-detection"),
  featureTracking: document.getElementById("feature-tracking"),

  // Notification settings
  notificationSettings: document.getElementById("notification-settings"),
  closeNotificationSettings: document.getElementById(
    "close-notification-settings"
  ),
  notificationEmail: document.getElementById("notification-email"),
  emailError: document.getElementById("email-error"),
  enableNotifications: document.getElementById("enable-notifications"),
  saveNotificationSettings: document.getElementById(
    "save-notification-settings"
  ),
};

const ctx = elements.canvas.getContext("2d");

// ===== Initialization =====
/**
 * Initialize the application
 */
function init() {
  enumerateWebcams();
  checkForDatasetUpdates();

  if (!localStorage.getItem("datasetUpdated")) {
    loadData();
  }

  loadNotificationSettings();
  setupEventListeners();

  // Calculate FPS every second
  setInterval(calculateFPS, 1000);

  elements.notificationSettings.classList.remove("hidden");
  elements.recordingIndicator.classList.add("hidden");
  requestNotificationPermission();
}

/**
 * Request browser notification permission
 */
function requestNotificationPermission() {
  if (
    Notification.permission !== "granted" &&
    Notification.permission !== "denied"
  ) {
    Notification.requestPermission();
  }
}

// ===== Camera Management =====
/**
 * Get list of available webcams and populate dropdown
 */
function enumerateWebcams() {
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      state.availableCameras = videoDevices;

      // Update camera selection dropdown
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
      console.error("Error enumerating devices:", err);
      elements.cameraSelect.innerHTML =
        '<option value="">Error loading cameras</option>';
    });
}

/**
 * Start camera with selected device
 * @returns {Promise} Stream promise
 */
function startCamera() {
  const selectedDeviceId = elements.cameraSelect.value;

  const constraints = {
    video: selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
  };

  // Stop existing stream if any
  stopCamera();

  return navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      state.activeStream = stream;
      elements.video.srcObject = stream;
      elements.cameraOverlay.classList.add("hidden");

      // Re-enumerate cameras if needed to get labels
      if (!state.availableCameras.some((cam) => cam.label)) {
        setTimeout(enumerateWebcams, 500);
      }

      // Set canvas dimensions to match video
      elements.video.onloadedmetadata = () => {
        elements.canvas.width = elements.video.videoWidth;
        elements.canvas.height = elements.video.videoHeight;
      };

      return stream;
    })
    .catch((err) => {
      console.error("Camera error:", err);
      elements.cameraOverlay.classList.remove("hidden");
      showNotification("Failed to access camera", "error");
      throw err;
    });
}

/**
 * Stop active camera stream
 */
function stopCamera() {
  if (state.activeStream) {
    state.activeStream.getTracks().forEach((track) => track.stop());
    state.activeStream = null;
    elements.video.srcObject = null;
  }
}

/**
 * Start camera and recognition
 */
function startAll() {
  if (state.isRecognizing) return;

  startCamera()
    .then(() => startRecognition())
    .catch((err) => console.error("Failed to start camera:", err));
}

/**
 * Stop recognition but keep camera active with pause overlay
 */
function stopAll() {
  stopRecognition();

  elements.faceBoxesContainer.innerHTML = "";
  showNotification("Face recognition paused", "info");

  // Create pause overlay
  const pauseOverlay = document.createElement("div");
  pauseOverlay.className =
    "absolute inset-0 flex items-center justify-center bg-slate-900/50";
  pauseOverlay.innerHTML = `
    <div class="text-center p-4 bg-slate-800/90 rounded-lg">
      <i class="fas fa-pause-circle text-3xl mb-2 text-blue-400"></i>
      <h3 class="text-lg font-medium">Recognition Paused</h3>
      <p class="text-sm text-slate-300">Click Start to resume</p>
    </div>
  `;

  const cameraContainer = document.querySelector(".camera-container");
  cameraContainer.appendChild(pauseOverlay);

  // Store reference to remove later
  window.pauseOverlay = pauseOverlay;
}

/**
 * Switch to different camera
 */
function switchCamera() {
  if (state.isRecognizing) {
    stopRecognition();
    startCamera()
      .then(() => startRecognition())
      .catch((err) => console.error("Failed to switch camera:", err));
  } else if (state.activeStream) {
    startCamera().catch((err) =>
      console.error("Failed to switch camera:", err)
    );
  }
}

// ===== WebSocket Communication =====
/**
 * Setup WebSocket connection to server
 */
function setupWebSocketConnection() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  state.socket = new WebSocket(wsUrl);

  state.socket.onopen = () => {
    updateConnectionStatus("connected");
    showNotification("WebSocket connected", "success");
  };

  state.socket.onmessage = (event) => handleWebSocketMessage(event.data);

  state.socket.onclose = () => {
    updateConnectionStatus("disconnected");

    // Auto-reconnect if actively recognizing
    if (state.isRecognizing) {
      setTimeout(setupWebSocketConnection, 3000);
      showNotification("Connection lost. Reconnecting...", "warning");
    }
  };

  state.socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    updateConnectionStatus("disconnected");
    showNotification("Connection error", "error");
  };
}

/**
 * Update connection status UI
 * @param {string} status - Connection status: connected, connecting, disconnected
 */
function updateConnectionStatus(status) {
  const statusElement = elements.connectionStatus;
  statusElement.className = "connection-status " + status;

  const statusText = {
    connected: "Connected",
    connecting: "Connecting...",
    disconnected: "Disconnected",
  }[status];

  statusElement.textContent = statusText;
  elements.connectionStatusText.textContent = statusText;
}

/**
 * Process incoming WebSocket messages
 * @param {string} data - Raw message data
 */
function handleWebSocketMessage(data) {
  try {
    const message = JSON.parse(data);

    switch (message.type) {
      case "recognition_result":
        handleRecognitionResults(message.faces, message.performance);
        break;
      case "security_notification":
        handleSecurityNotification(message.notification);
        break;
      case "error":
        showNotification("Server error: " + message.message, "error");
        break;
      case "command_result":
        handleCommandResult(message);
        break;
    }
  } catch (error) {
    console.error("Error handling websocket message:", error);
  }
}

/**
 * Process server command results
 * @param {Object} message - Command result message
 */
function handleCommandResult(message) {
  const { command, success, data, message: errorMessage } = message;

  switch (command) {
    case "cleanup_temp":
      if (success) {
        showNotification(
          `Cleaned up ${data.deleted_count} temporary files`,
          "success"
        );
      } else {
        showNotification("Failed to clean temporary files", "error");
      }
      // Reset button state
      elements.cleanTempBtn.innerHTML =
        '<i class="fas fa-trash-alt text-xl"></i><span class="text-sm">Clean Temp</span>';
      elements.cleanTempBtn.disabled = false;
      break;

    case "reset_caches":
      showNotification(
        success
          ? "Recognition caches reset successfully"
          : "Failed to reset caches",
        success ? "success" : "error"
      );
      break;

    case "embeddings_complete":
      if (success) {
        showNotification("Embeddings regenerated successfully", "success");
        loadData(); // Reload dataset to reflect changes
      } else {
        showNotification(
          `Error generating embeddings: ${errorMessage || "Unknown error"}`,
          "error"
        );
      }

      // Reset button state
      elements.updateDatasetBtn.innerHTML =
        '<i class="fas fa-sync-alt text-xl"></i><span class="text-sm">Update Dataset</span>';
      elements.updateDatasetBtn.disabled = false;
      break;
  }
}

// ===== Face Recognition =====
/**
 * Start face recognition process
 */
function startRecognition() {
  if (state.isRecognizing) return;

  // Remove pause overlay if exists
  if (window.pauseOverlay) {
    window.pauseOverlay.remove();
    window.pauseOverlay = null;
  }

  // Connect WebSocket
  setupWebSocketConnection();
  updateConnectionStatus("connecting");

  state.isRecognizing = true;
  elements.startBtn.disabled = true;
  elements.stopBtn.disabled = false;
  elements.recordingIndicator.classList.remove("hidden");

  // Start sending frames
  sendVideoFrames();
}

/**
 * Stop face recognition process
 */
function stopRecognition() {
  if (!state.isRecognizing) return;

  state.isRecognizing = false;
  elements.startBtn.disabled = false;
  elements.stopBtn.disabled = true;
  elements.recordingIndicator.classList.add("hidden");

  // Close WebSocket if open
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.close();
  }

  // Clear face boxes
  elements.faceBoxesContainer.innerHTML = "";
}

/**
 * Capture and send video frames to server via WebSocket
 */
function sendVideoFrames() {
  if (!state.isRecognizing) return;

  state.totalFrames++;
  state.framesLastSecond++;

  const now = Date.now();
  if (now - state.lastFrameSent >= state.frameInterval) {
    state.lastFrameSent = now;

    try {
      if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        // Capture and send frame
        ctx.drawImage(
          elements.video,
          0,
          0,
          elements.canvas.width,
          elements.canvas.height
        );
        const frameDataUrl = elements.canvas.toDataURL("image/jpeg", 0.7);

        state.socket.send(
          JSON.stringify({
            type: "frame",
            data: frameDataUrl,
          })
        );
      }
    } catch (error) {
      console.error("Error sending frame:", error);
    }
  }

  // Request next frame
  requestAnimationFrame(sendVideoFrames);
}

/**
 * Process recognition results from server
 * @param {Array} faces - Detected faces data
 * @param {Object} performance - Performance metrics
 */
function handleRecognitionResults(faces, performance) {
  if (!state.isRecognizing) return;

  // Update performance stats
  if (performance) {
    state.stats.latency = Date.now() - state.lastFrameSent;
    if (performance.detection_time)
      state.stats.detection_time = performance.detection_time;
    if (performance.recognition_time)
      state.stats.recognition_time = performance.recognition_time;
    if (performance.spoof_detection_time)
      state.stats.spoof_detection_time = performance.spoof_detection_time;

    updateStatsDisplay();
  }

  // Clear existing face boxes
  elements.faceBoxesContainer.innerHTML = "";

  // Process detections
  const updatesNeeded = [];

  if (faces && faces.length > 0) {
    faces.forEach((face) => {
      // Create face box for live display
      createFaceBox(face);

      // Use track ID to maintain identity persistence
      const trackId = face.track_id !== undefined ? face.track_id : -1;

      if (trackId >= 0) {
        const faceId = `track_${trackId}`;
        const currentName = face.name || "Unknown";
        const previousIdentity = state.faceIdentityMap.get(faceId);

        // Add new face or update when identity changes
        if (!previousIdentity || previousIdentity !== currentName) {
          state.faceIdentityMap.set(faceId, currentName);

          updatesNeeded.push({
            ...face,
            faceId: faceId,
            timestamp: new Date(),
            identity_changed: previousIdentity ? true : false,
            previous_identity: previousIdentity,
          });
        }
      } else {
        // Face without track_id gets temporary ID
        const tempId = `face_${Date.now()}_${Math.random()
          .toString(36)
          .substring(7)}`;
        updatesNeeded.push({
          ...face,
          faceId: tempId,
          timestamp: new Date(),
        });
      }
    });
  }

  // Update detected faces list
  if (updatesNeeded.length > 0) {
    // Add new faces to top of list, maintain max 20 items
    state.detectedFaces = [...updatesNeeded, ...state.detectedFaces].slice(
      0,
      20
    );
    updateDetectedFacesList();
  }
}

/**
 * Create visual box around detected face
 * @param {Object} face - Face detection data
 */
function createFaceBox(face) {
  const [x1, y1, x2, y2] = face.bbox;

  // Calculate scale between video and display dimensions
  const displayWidth = elements.video.clientWidth;
  const displayHeight = elements.video.clientHeight;
  const videoWidth = elements.video.videoWidth || elements.canvas.width;
  const videoHeight = elements.video.videoHeight || elements.canvas.height;

  // Scale coordinates to match display size
  const scaleX = displayWidth / videoWidth;
  const scaleY = displayHeight / videoHeight;

  const scaledX1 = x1 * scaleX;
  const scaledY1 = y1 * scaleY;
  const scaledWidth = (x2 - x1) * scaleX;
  const scaledHeight = (y2 - y1) * scaleY;

  // Create face box element
  const faceBox = document.createElement("div");
  faceBox.className = "face-box";

  // Position and size the box
  Object.assign(faceBox.style, {
    width: `${scaledWidth}px`,
    height: `${scaledHeight}px`,
    left: `${scaledX1}px`,
    top: `${scaledY1}px`,
    borderColor: face.is_spoofed
      ? "#ef4444" // Red for spoofed
      : face.is_unknown
      ? "#f97316" // Orange for unknown
      : "#22c55e", // Green for known faces
  });

  // Create face label
  const faceLabel = document.createElement("div");
  faceLabel.className = "face-label";

  // Set label text with status indicators
  let labelText = face.name || "Unknown";
  if (face.is_spoofed) labelText += " [SPOOF]";
  if (face.wearing_mask) labelText += " [MASK]";

  faceLabel.textContent = labelText;

  // Add confidence percentage if available
  if (face.confidence) {
    const confidenceSpan = document.createElement("span");
    confidenceSpan.className = "ml-1 text-xs text-slate-400";
    confidenceSpan.textContent = `${(face.confidence * 100).toFixed(0)}%`;
    faceLabel.appendChild(confidenceSpan);
  }

  faceBox.appendChild(faceLabel);
  elements.faceBoxesContainer.appendChild(faceBox);
}

// ===== UI Updates =====
/**
 * Calculate and display FPS (frames per second)
 */
function calculateFPS() {
  state.currentFPS = state.framesLastSecond;
  elements.fpsCounter.textContent = `${state.currentFPS} FPS`;
  state.framesLastSecond = 0;
}

/**
 * Update performance statistics display
 */
function updateStatsDisplay() {
  elements.latencyStat.textContent = `${state.stats.latency}ms`;
  elements.recogTimeStat.textContent = state.stats.recognition_time
    ? `${state.stats.recognition_time.toFixed(1)}ms`
    : "-";
  elements.spoofTimeStat.textContent = state.stats.spoof_detection_time
    ? `${state.stats.spoof_detection_time.toFixed(1)}ms`
    : "-";
  elements.detectTimeStat.textContent = state.stats.detection_time
    ? `${state.stats.detection_time.toFixed(1)}ms`
    : "-";
}

/**
 * Update feature indicator status based on system capabilities
 * @param {Object} features - Enabled features from server
 */
function updateFeatureStatus(features) {
  updateFeatureElement(elements.featureFaceDetection, true); // Always enabled
  updateFeatureElement(elements.featureRecognition, features.face_recognition);
  updateFeatureElement(elements.featureAntiSpoofing, features.anti_spoofing);
  updateFeatureElement(elements.featureMaskDetection, features.mask_detection);
  updateFeatureElement(elements.featureTracking, features.face_tracking);
}

/**
 * Update single feature indicator element
 * @param {HTMLElement} element - Feature indicator element
 * @param {boolean} enabled - Whether feature is enabled
 */
function updateFeatureElement(element, enabled) {
  if (enabled) {
    element.classList.add("enabled");
    element.classList.remove("disabled");
    element.querySelector("i").className = "fas fa-square-check";
  } else {
    element.classList.add("disabled");
    element.classList.remove("enabled");
    element.querySelector("i").className = "fas fa-square-xmark";
  }
}

/**
 * Update UI list of detected faces
 */
function updateDetectedFacesList() {
  if (state.detectedFaces.length === 0) {
    elements.detectedFacesList.innerHTML = `
      <div class="text-center py-8 text-slate-400">
        <i class="fas fa-user-slash text-3xl mb-2"></i>
        <p>No faces detected yet</p>
      </div>
    `;
    elements.detectedCount.textContent = "0";
    return;
  }

  elements.detectedCount.textContent = state.detectedFaces.length.toString();

  elements.detectedFacesList.innerHTML = state.detectedFaces
    .map((face) => {
      // Determine status classes based on face properties
      const isUnknown = face.is_unknown;
      const isSpoofed = face.is_spoofed;

      const statusClass = isSpoofed
        ? "bg-red-500/10 border border-red-500/20"
        : isUnknown
        ? "bg-amber-500/10 border border-amber-500/20"
        : "bg-green-500/10 border border-green-500/20";

      const statusBadgeClass = isSpoofed
        ? "bg-red-500/20 text-red-500"
        : isUnknown
        ? "bg-amber-500/20 text-amber-500"
        : "bg-green-500/20 text-green-500";

      // Face info
      const name = face.name || "Unknown";
      const statusText = isSpoofed
        ? "SPOOF"
        : face.confidence
        ? `${(face.confidence * 100).toFixed(0)}%`
        : "";

      const imgUrl =
        face.image ||
        `https://ui-avatars.com/api/?name=${encodeURIComponent(
          name
        )}&background=random`;

      const trackInfo = face.track_id >= 0 ? `#${face.track_id}` : "";

      // Identity change badge
      const identityChangedBadge = face.identity_changed
        ? `<div class="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-300 ml-1">ID Changed</div>`
        : "";

      return `
      <div class="flex items-center gap-3 p-3 rounded-lg ${statusClass}">
        <div class="w-12 h-12 rounded-full overflow-hidden bg-slate-700">
          <img src="${imgUrl}" alt="${name}" class="w-full h-full object-cover">
        </div>
        <div class="flex-1 min-w-0">
          <div class="flex items-center">
            <h4 class="font-medium truncate">${name}</h4>
            ${identityChangedBadge}
          </div>
          ${
            face.previous_identity
              ? `<p class="text-xs text-slate-400">Was: ${face.previous_identity}</p>`
              : ""
          }
          <p class="text-xs text-slate-400">${formatTime(face.timestamp)}</p>
        </div>
        <div class="text-xs px-2 py-1 rounded-full ${statusBadgeClass}">
          ${statusText}
        </div>
      </div>
    `;
    })
    .join("");
}

/**
 * Update UI list of dataset users
 */
function updateDatasetUsersList() {
  if (state.datasetUsers.length === 0) {
    elements.datasetUsersList.innerHTML = `
      <div class="text-center py-8 text-slate-400">
        <i class="fas fa-user-plus text-3xl mb-2"></i>
        <p>No users in dataset</p>
      </div>
    `;
    elements.datasetCount.textContent = "0";
    return;
  }

  elements.datasetCount.textContent = state.datasetUsers.length.toString();

  elements.datasetUsersList.innerHTML = state.datasetUsers
    .map(
      (user) => `
    <div class="flex items-center justify-between p-3 rounded-lg hover:bg-slate-700/50">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-full overflow-hidden bg-slate-700">
          <img src="${user.image_path}" alt="${user.name}" class="w-full h-full object-cover">
        </div>
        <span class="font-medium">${user.name}</span>
      </div>
      <button class="text-red-400 hover:text-red-500 p-1 rounded-full hover:bg-red-500/10" data-user-id="${user.id}">
        <i class="fas fa-trash-alt text-sm"></i>
      </button>
    </div>
  `
    )
    .join("");
}

// ===== Data Management =====
/**
 * Load data from server APIs
 * @returns {Promise} Promise that resolves when data is loaded
 */
function loadData() {
  // Load dataset users
  const datasetPromise = fetch("/api/dataset/users")
    .then((response) => response.json())
    .then((data) => {
      state.datasetUsers = data;
      elements.datasetCount.textContent = data.length;
      updateDatasetUsersList();
      return data;
    })
    .catch((error) => {
      console.error("Failed to load dataset users:", error);
      state.datasetUsers = [];
      elements.datasetCount.textContent = "0";
      updateDatasetUsersList();
      return [];
    });

  // Get system status
  fetch("/api/status")
    .then((response) => response.json())
    .then((data) => updateFeatureStatus(data.features))
    .catch((error) => console.error("Failed to load system status:", error));

  return datasetPromise;
}

/**
 * Check for dataset updates from other pages
 */
function checkForDatasetUpdates() {
  const datasetUpdated = localStorage.getItem("datasetUpdated");

  if (datasetUpdated === "true") {
    // Clear flags
    localStorage.removeItem("datasetUpdated");
    const lastAddedPerson = localStorage.getItem("lastAddedPerson") || "";
    localStorage.removeItem("lastAddedPerson");

    // Reload dataset users
    loadData().then(() => {
      if (lastAddedPerson) {
        showNotification(
          `${lastAddedPerson} was successfully added to the dataset`,
          "success"
        );
      } else {
        showNotification("Dataset updated successfully", "success");
      }
    });
  }
}

/**
 * Load notification settings from server
 */
function loadNotificationSettings() {
  fetch("/api/settings/email")
    .then((response) => response.json())
    .then((data) => {
      elements.notificationEmail.value = data.email || "";
      elements.enableNotifications.checked = data.enable_notifications;
    })
    .catch((error) =>
      console.error("Error loading notification settings:", error)
    );
}

/**
 * Update and regenerate embedding dataset
 */
function updateDataset() {
  if (
    !confirm(
      "This will regenerate all embeddings with augmentation. This may take some time. Continue?"
    )
  ) {
    return;
  }

  // Show loading state
  elements.updateDatasetBtn.innerHTML =
    '<i class="fas fa-spinner fa-spin text-xl"></i><span class="text-sm">Processing...</span>';
  elements.updateDatasetBtn.disabled = true;

  fetch("/api/regenerate_embeddings", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showNotification(
          "Embedding regeneration started in background",
          "info"
        );

        setTimeout(() => {
          showNotification(
            "This process may take a few minutes. The system will update automatically when complete.",
            "info"
          );
        }, 3000);
      } else {
        showNotification("Failed to start embedding regeneration", "error");
        resetUpdateButton();
      }
    })
    .catch((error) => {
      console.error("Error regenerating embeddings:", error);
      showNotification("Error generating embeddings", "error");
      resetUpdateButton();
    });

  function resetUpdateButton() {
    elements.updateDatasetBtn.innerHTML =
      '<i class="fas fa-sync-alt text-xl"></i><span class="text-sm">Update Dataset</span>';
    elements.updateDatasetBtn.disabled = false;
  }
}

/**
 * Clean temporary image files
 */
function cleanTempFiles() {
  if (
    !confirm("Are you sure you want to delete all temporary capture files?")
  ) {
    return;
  }

  // Show loading
  elements.cleanTempBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
  elements.cleanTempBtn.disabled = true;

  // Send command via WebSocket if connected
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(
      JSON.stringify({
        type: "command",
        command: "cleanup_temp",
        days: 0,
      })
    );
    // Button will be reset when we get response
  } else {
    // Fallback to REST API
    fetch("/api/cleanup/temp_captures?days=0", { method: "DELETE" })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          showNotification(
            `Cleaned up ${data.deleted_count} temporary files`,
            "success"
          );
        } else {
          showNotification("Failed to clean temporary files", "error");
        }
      })
      .catch((error) => {
        console.error("Error cleaning temp files:", error);
        showNotification("Error cleaning temporary files", "error");
      })
      .finally(() => {
        elements.cleanTempBtn.innerHTML =
          '<i class="fas fa-trash-alt text-xl"></i><span class="text-sm">Clean Temp</span>';
        elements.cleanTempBtn.disabled = false;
      });
  }
}

/**
 * Delete user from recognition dataset
 * @param {string} userId - User ID to delete
 */
function deleteUser(userId) {
  if (!confirm("Are you sure you want to delete this user from the dataset?")) {
    return;
  }

  const deleteButton = document.querySelector(
    `button[data-user-id="${userId}"]`
  );

  // Show loading state
  if (deleteButton) {
    deleteButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    deleteButton.disabled = true;
  }

  // Call API to delete user
  fetch(`/api/dataset/users/${userId}`, { method: "DELETE" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (data.success) {
        // Update local data and UI
        state.datasetUsers = state.datasetUsers.filter(
          (user) => user.id !== userId
        );
        updateDatasetUsersList();
        showNotification(`User ${userId} deleted from dataset`, "success");

        // Restart recognition if active
        if (state.isRecognizing) {
          stopRecognition();
          setTimeout(startRecognition, 500);
        }
      } else {
        showNotification("Failed to delete user", "error");
        resetDeleteButton();
      }
    })
    .catch((error) => {
      console.error("Error deleting user:", error);
      showNotification(`Error deleting user: ${error.message}`, "error");
      resetDeleteButton();
    });

  function resetDeleteButton() {
    if (deleteButton) {
      deleteButton.innerHTML = '<i class="fas fa-trash-alt text-sm"></i>';
      deleteButton.disabled = false;
    }
  }
}

// ===== Notifications =====
/**
 * Handle security notifications from backend
 * @param {Object} data - Notification data
 */
function handleSecurityNotification(data) {
  const { type, camera_id, new_count, total_count } = data;

  // Create appropriate message based on notification type
  let message = "";
  if (type === "new_face") {
    message =
      new_count > 1
        ? `Phát hiện ${new_count} người lạ mới tại camera ${camera_id}`
        : `Phát hiện người lạ mới tại camera ${camera_id}`;
  } else if (type === "periodic") {
    message = `Cảnh báo: ${total_count} người lạ đang xuất hiện tại camera ${camera_id}`;
  }

  // Show in-app notification
  showNotification(message, "warning");

  // Show browser notification if permitted
  if (Notification.permission === "granted") {
    const notification = createSecurityNotification(data);
    new Notification(notification.title, notification);
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission().then((permission) => {
      if (permission === "granted") {
        const notification = createSecurityNotification(data);
        new Notification(notification.title, notification);
      }
    });
  }
}

/**
 * Create notification content that matches email format
 * @param {Object} data - Notification data
 * @returns {Object} Notification options
 */
function createSecurityNotification(data) {
  const { type, camera_id, new_count, total_count, timestamp } = data;

  // Format time
  const time = new Date(timestamp || Date.now()).toLocaleString("vi-VN");

  // Default icon
  const defaultIcon = "https://cdn-icons-png.flaticon.com/512/1680/1680012.png";

  // Create notification content based on type
  if (type === "new_face") {
    const title = "CẢNH BÁO AN NINH";
    let body =
      new_count > 1
        ? `Hệ thống đã phát hiện ${new_count} người lạ mới tại camera ${camera_id}.\n`
        : `Hệ thống đã phát hiện người lạ mới tại camera ${camera_id}.\n`;

    body += `Tổng cộng có ${total_count} người lạ đang xuất hiện.\n`;
    body += `Thời gian: ${time}\n`;
    body += `Vui lòng kiểm tra!`;

    return {
      title,
      body,
      icon: "/static/images/alert-icon.png" || defaultIcon,
    };
  } else if (type === "periodic") {
    return {
      title: "CẬP NHẬT AN NINH",
      body:
        `Hiện có ${total_count} người lạ đang xuất hiện tại camera ${camera_id}.\n` +
        `Thời gian: ${time}\n` +
        `Đây là thông báo định kỳ.`,
      icon: "/static/images/info-icon.png" || defaultIcon,
    };
  }

  // Default fallback
  return {
    title: "THÔNG BÁO HỆ THỐNG",
    body: "Có cảnh báo mới từ hệ thống an ninh",
    icon: defaultIcon,
  };
}

/**
 * Test notification system with mock data
 */
function testNotification() {
  // Create sample security notification
  const mockData = {
    type: "new_face",
    camera_id: "Camera Test",
    new_count: 1,
    total_count: 3,
    timestamp: new Date().toISOString(),
  };

  handleSecurityNotification(mockData);
}

/**
 * Save notification email settings
 */
function saveNotificationSettingsHandler() {
  const email = elements.notificationEmail.value.trim();
  const enableNotifs = elements.enableNotifications.checked;

  // Validate email
  if (email && !/^\S+@\S+\.\S+$/.test(email)) {
    elements.emailError.classList.remove("hidden");
    return;
  }

  elements.emailError.classList.add("hidden");

  // Save to server
  fetch("/api/settings/email", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email: email,
      enable_notifications: enableNotifs,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showNotification("Notification settings saved", "success");
        elements.notificationSettings.classList.add("hidden");
      } else {
        showNotification(
          "Failed to save settings: " + (data.message || "Unknown error"),
          "error"
        );
      }
    })
    .catch((error) => {
      console.error("Error saving notification settings:", error);
      showNotification("Error saving settings", "error");
    });
}

// ===== Utility Functions =====
/**
 * Format timestamp for display
 * @param {Date} date - Date to format
 * @returns {string} Formatted time string
 */
function formatTime(date) {
  if (!date) return "";

  const now = new Date();
  const diff = now - date;

  if (diff < 60000) {
    return "Just now";
  } else if (diff < 3600000) {
    const mins = Math.floor(diff / 60000);
    return `${mins} min${mins !== 1 ? "s" : ""} ago`;
  } else if (diff < 86400000) {
    const hours = Math.floor(diff / 3600000);
    return `${hours} hour${hours !== 1 ? "s" : ""} ago`;
  } else {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
}

/**
 * Display notification toast
 * @param {string} message - Notification message
 * @param {string} type - Notification type: success, error, warning, info
 */
function showNotification(message, type = "info") {
  // Map type to color class
  const typeClasses = {
    success: "bg-green-600",
    error: "bg-red-600",
    warning: "bg-amber-600",
    info: "bg-blue-600",
  };

  // Map type to icon
  const typeIcons = {
    success: "fa-check-circle",
    error: "fa-exclamation-circle",
    warning: "fa-exclamation-triangle",
    info: "fa-info-circle",
  };

  const toast = document.createElement("div");
  toast.className = `fixed bottom-4 left-4 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 ${
    typeClasses[type] || typeClasses.info
  }`;

  toast.innerHTML = `
    <i class="fas ${typeIcons[type] || typeIcons.info}"></i>
    <span>${message}</span>
  `;

  document.body.appendChild(toast);

  // Remove after 3 seconds
  setTimeout(() => toast.remove(), 3000);
}

// ===== Event Listeners =====
/**
 * Set up all event handlers
 */
function setupEventListeners() {
  // Camera controls
  elements.enableCameraBtn.addEventListener("click", startAll);
  elements.startBtn.addEventListener("click", startAll);
  elements.stopBtn.addEventListener("click", stopAll);
  elements.cameraSelect.addEventListener("change", switchCamera);

  // Control panel
  elements.addPersonBtn.addEventListener(
    "click",
    () => (window.location.href = "/add_person")
  );
  elements.updateDatasetBtn.addEventListener("click", updateDataset);
  elements.cleanTempBtn.addEventListener("click", cleanTempFiles);
  elements.testNotificationBtn.addEventListener("click", testNotification);

  // Stats and popups
  elements.statsBtn.addEventListener("click", () =>
    elements.statsPopup.classList.toggle("hidden")
  );
  elements.showTechStackBtn.addEventListener("click", () =>
    elements.techStackPopup.classList.toggle("hidden")
  );
  elements.closeTechStackBtn.addEventListener("click", () =>
    elements.techStackPopup.classList.add("hidden")
  );

  // Notification settings
  elements.closeNotificationSettings.addEventListener("click", () =>
    elements.notificationSettings.classList.add("hidden")
  );
  elements.saveNotificationSettings.addEventListener(
    "click",
    saveNotificationSettingsHandler
  );

  // Dataset user management - delegation pattern for better performance
  elements.datasetUsersList.addEventListener("click", (event) => {
    const deleteBtn = event.target.closest("button");
    if (deleteBtn) {
      const userId = deleteBtn.dataset.userId;
      if (userId) deleteUser(userId);
    }
  });

  // Check for updates when page becomes visible
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkForDatasetUpdates();
    }
  });

  // Check for updates when window gets focus
  window.addEventListener("focus", checkForDatasetUpdates);
}

// Initialize on page load
window.onload = init;
