// ===== WebSocket Connection =====
let socket = null;
let isRecognizing = false;
let lastFrameSent = 0;
let frameInterval = 100; // Max 10 fps to server
let totalFrames = 0;
let framesLastSecond = 0;
let lastFpsUpdate = Date.now();
let currentFPS = 0;
let stats = {
  latency: 0,
  recognition_time: 0,
  detection_time: 0,
  spoof_detection_time: 0,
};

// New variables for camera control
let activeStream = null;
let availableCameras = [];

// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const faceBoxesContainer = document.getElementById("face-boxes-container");
const detectedFacesList = document.getElementById("detected-faces-list");
const datasetUsersList = document.getElementById("dataset-users-list");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const addPersonBtn = document.getElementById("add-person-btn");
const updateDatasetBtn = document.getElementById("update-dataset-btn");
const cleanTempBtn = document.getElementById("clean-temp-btn");
const testNotificationBtn = document.getElementById("test-notification-btn");
const cameraOverlay = document.getElementById("camera-overlay");
const enableCameraBtn = document.getElementById("enable-camera-btn");
const recordingIndicator = document.getElementById("recording-indicator");
const fpsCounter = document.getElementById("fps-counter");
const detectedCount = document.getElementById("detected-count");
const datasetCount = document.getElementById("dataset-count");
const connectionStatus = document.getElementById("connection-status");
const connectionStatusText = document.getElementById("connectionStatus");
const statsBtn = document.getElementById("statsBtn");
const statsPopup = document.getElementById("stats-popup");
const latencyStat = document.getElementById("latency-stat");
const recogTimeStat = document.getElementById("recog-time-stat");
const spoofTimeStat = document.getElementById("spoof-time-stat");
const detectTimeStat = document.getElementById("detect-time-stat");
const showTechStackBtn = document.getElementById("showTechStackBtn");
const closeTechStackBtn = document.getElementById("closeTechStackBtn");
const techStackPopup = document.getElementById("techStackPopup");
const cameraSelect = document.getElementById("camera-select"); // New camera select element

// Feature indicators
const featureFaceDetection = document.getElementById("feature-face-detection");
const featureRecognition = document.getElementById("feature-recognition");
const featureAntiSpoofing = document.getElementById("feature-anti-spoofing");
const featureMaskDetection = document.getElementById("feature-mask-detection");
const featureTracking = document.getElementById("feature-tracking");

// Notification Settings
const notificationSettings = document.getElementById("notification-settings");
const closeNotificationSettings = document.getElementById(
  "close-notification-settings"
);
const notificationEmail = document.getElementById("notification-email");
const emailError = document.getElementById("email-error");
const enableNotifications = document.getElementById("enable-notifications");
const saveNotificationSettings = document.getElementById(
  "save-notification-settings"
);

// App State
let detectedFaces = [];
let datasetUsers = [];
let faceIdentityMap = new Map(); // Map để theo dõi danh tính khuôn mặt theo track_id

// Initialize the app
function init() {
  // Enumerate available webcams
  enumerateWebcams();

  // Don't immediately access camera, wait for user to click Start
  cameraOverlay.classList.remove("hidden");

  // Load initial data
  loadData();

  // Load notification settings
  loadNotificationSettings();

  // Set up event listeners
  setupEventListeners();

  // Calculate FPS periodically
  setInterval(calculateFPS, 1000);

  // Show notification settings by default
  notificationSettings.classList.remove("hidden");

  // Hide recording indicator initially
  recordingIndicator.classList.add("hidden");

  // Yêu cầu quyền thông báo khi khởi động
  requestNotificationPermission();
}

// Yêu cầu quyền thông báo trình duyệt
function requestNotificationPermission() {
  if (
    Notification.permission !== "granted" &&
    Notification.permission !== "denied"
  ) {
    Notification.requestPermission();
  }
}

// Load notification settings from server
function loadNotificationSettings() {
  fetch("/api/settings/email")
    .then((response) => response.json())
    .then((data) => {
      // Update UI with current settings
      notificationEmail.value = data.email || "";
      enableNotifications.checked = data.enable_notifications;
    })
    .catch((error) => {
      console.error("Error loading notification settings:", error);
    });
}

// Enumerate available webcams
function enumerateWebcams() {
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      availableCameras = videoDevices;

      // Populate the dropdown
      cameraSelect.innerHTML = videoDevices.length
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
      cameraSelect.innerHTML =
        '<option value="">Error loading cameras</option>';
    });
}

// Check camera access - modified to not auto-start
function checkCameraAccess() {
  navigator.permissions
    .query({ name: "camera" })
    .then((permissionStatus) => {
      if (permissionStatus.state === "granted") {
        // Just show camera overlay, don't auto-start
        cameraOverlay.classList.remove("hidden");
      } else {
        cameraOverlay.classList.remove("hidden");
      }

      permissionStatus.onchange = () => {
        if (permissionStatus.state === "granted") {
          cameraOverlay.classList.remove("hidden");
        }
      };
    })
    .catch(() => {
      // Fallback for browsers that don't support permissions API
      cameraOverlay.classList.remove("hidden");
    });
}

// Start camera stream - updated for device selection
function startCamera() {
  const selectedDeviceId = cameraSelect.value;

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
      activeStream = stream;
      video.srcObject = stream;
      cameraOverlay.classList.add("hidden");

      // Update camera selection with labels if they were empty before
      if (!availableCameras.some((cam) => cam.label)) {
        setTimeout(enumerateWebcams, 500); // Re-enumerate after permission granted
      }

      // Set canvas dimensions to match video
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      };

      return stream;
    })
    .catch((err) => {
      console.error("Camera error: ", err);
      cameraOverlay.classList.remove("hidden");
      showNotification("Failed to access camera", "error");
      throw err;
    });
}

// Stop camera stream - new function
function stopCamera() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => {
      track.stop();
    });
    activeStream = null;
    video.srcObject = null;
  }
}

// New function to start both camera and recognition
function startAll() {
  if (isRecognizing) return;

  // Start camera with current selection
  startCamera()
    .then(() => {
      startRecognition();
    })
    .catch((err) => {
      console.error("Failed to start camera:", err);
      // Error handling is in startCamera
    });
}

// New function to stop both camera and recognition
function stopAll() {
  stopRecognition();

  // Không tắt camera
  // stopCamera();

  // Không hiển thị overlay yêu cầu quyền camera
  // cameraOverlay.classList.remove("hidden");

  // Thay vào đó, hiển thị overlay thông báo trạng thái tạm dừng
  const faceBoxesContainer = document.getElementById("face-boxes-container");
  faceBoxesContainer.innerHTML = "";

  // Hiển thị thông báo nhỏ tạm dừng
  showNotification("Face recognition paused", "info");

  // Hiển thị overlay với thông báo tạm dừng thay vì "Camera Access Required"
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

  // Lưu để có thể xóa sau này
  window.pauseOverlay = pauseOverlay;
}

// Function to switch camera
function switchCamera() {
  // Only restart if already active
  if (isRecognizing) {
    stopRecognition();
    startCamera()
      .then(() => {
        startRecognition();
      })
      .catch((err) => {
        console.error("Failed to switch camera:", err);
      });
  } else if (activeStream) {
    startCamera().catch((err) => {
      console.error("Failed to switch camera:", err);
    });
  }
}

// Update feature indicators based on system status
function updateFeatureStatus(features) {
  updateFeatureElement(featureFaceDetection, true); // Always enabled
  updateFeatureElement(featureRecognition, features.face_recognition);
  updateFeatureElement(featureAntiSpoofing, features.anti_spoofing);
  updateFeatureElement(featureMaskDetection, features.mask_detection);
  updateFeatureElement(featureTracking, features.face_tracking);
}

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

// Set up event listeners - updated for new functions
function setupEventListeners() {
  // Camera controls - updated for new functions
  enableCameraBtn.addEventListener("click", startAll);
  startBtn.addEventListener("click", startAll);
  stopBtn.addEventListener("click", stopAll);
  cameraSelect.addEventListener("change", switchCamera);

  // Control panel buttons
  addPersonBtn.addEventListener("click", () => {
    window.location.href = "/add_person";
  });
  updateDatasetBtn.addEventListener("click", updateDataset);
  cleanTempBtn.addEventListener("click", cleanTempFiles);
  testNotificationBtn.addEventListener("click", testNotification);

  // Stats button
  statsBtn.addEventListener("click", () => {
    statsPopup.classList.toggle("hidden");
  });

  // Tech stack popup
  showTechStackBtn.addEventListener("click", () => {
    techStackPopup.classList.toggle("hidden");
  });
  closeTechStackBtn.addEventListener("click", () => {
    techStackPopup.classList.add("hidden");
  });

  // Notification settings
  closeNotificationSettings.addEventListener("click", () =>
    notificationSettings.classList.add("hidden")
  );
  saveNotificationSettings.addEventListener(
    "click",
    saveNotificationSettingsHandler
  );

  // Dataset users list - improved click handler
  datasetUsersList.addEventListener("click", (event) => {
    // Check if delete button was clicked
    const deleteBtn = event.target.closest("button");
    if (deleteBtn) {
      const userId = deleteBtn.dataset.userId;
      if (userId) {
        deleteUser(userId);
      }
    }
  });
}

// Load initial data (for datasets)
function loadData() {
  // Get real dataset users
  fetch("/api/dataset/users")
    .then((response) => response.json())
    .then((data) => {
      datasetUsers = data;
      datasetCount.textContent = datasetUsers.length;
      updateDatasetUsersList();
    })
    .catch((error) => {
      console.error("Failed to load dataset users:", error);
      // Fallback to empty array if API fails
      datasetUsers = [];
      datasetCount.textContent = "0";
      updateDatasetUsersList();
    });

  // Get system status from API
  fetch("/api/status")
    .then((response) => response.json())
    .then((data) => {
      updateFeatureStatus(data.features);
    })
    .catch((error) => {
      console.error("Failed to load system status:", error);
    });
}

// WebSocket Connection
function setupWebSocketConnection() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  socket = new WebSocket(wsUrl);

  socket.onopen = (event) => {
    updateConnectionStatus("connected");
    showNotification("WebSocket connected", "success");
  };

  socket.onmessage = (event) => {
    handleWebSocketMessage(event.data);
  };

  socket.onclose = (event) => {
    updateConnectionStatus("disconnected");

    // Attempt to reconnect if actively recognizing
    if (isRecognizing) {
      setTimeout(setupWebSocketConnection, 3000);
      showNotification("Connection lost. Reconnecting...", "warning");
    }
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    updateConnectionStatus("disconnected");
    showNotification("Connection error", "error");
  };
}

// Update connection status UI
function updateConnectionStatus(status) {
  const statusElement = document.getElementById("connection-status");
  statusElement.className = "connection-status " + status;

  switch (status) {
    case "connected":
      statusElement.textContent = "Connected";
      connectionStatusText.textContent = "Connected";
      break;
    case "connecting":
      statusElement.textContent = "Connecting...";
      connectionStatusText.textContent = "Connecting...";
      break;
    case "disconnected":
      statusElement.textContent = "Disconnected";
      connectionStatusText.textContent = "Disconnected";
      break;
  }
}

// Hàm xử lý thông báo bảo mật với định dạng giống email
function handleSecurityNotification(data) {
  // Hiển thị thông báo trong ứng dụng
  const { type, camera_id, new_count, total_count } = data;

  let message = "";
  if (type === "new_face") {
    if (new_count > 1) {
      message = `Phát hiện ${new_count} người lạ mới tại camera ${camera_id}`;
    } else {
      message = `Phát hiện người lạ mới tại camera ${camera_id}`;
    }
  } else if (type === "periodic") {
    message = `Cảnh báo: ${total_count} người lạ đang xuất hiện tại camera ${camera_id}`;
  }

  // Hiển thị thông báo trong ứng dụng
  showNotification(message, "warning");

  // Hiển thị thông báo trình duyệt nếu được cho phép
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

// Tạo nội dung thông báo đồng bộ với email
function createSecurityNotification(data) {
  const { type, camera_id, new_count, total_count, timestamp } = data;

  // Định dạng thời gian
  const time = new Date(timestamp || Date.now()).toLocaleString("vi-VN");

  // Tạo nội dung thông báo dựa trên loại
  if (type === "new_face") {
    const title = "CẢNH BÁO AN NINH";
    let body;

    if (new_count > 1) {
      body = `Hệ thống đã phát hiện ${new_count} người lạ mới tại camera ${camera_id}.\n`;
    } else {
      body = `Hệ thống đã phát hiện người lạ mới tại camera ${camera_id}.\n`;
    }

    body += `Tổng cộng có ${total_count} người lạ đang xuất hiện.\n`;
    body += `Thời gian: ${time}\n`;
    body += `Vui lòng kiểm tra!`;

    return {
      title: title,
      body: body,
      icon:
        "/static/images/alert-icon.png" ||
        "https://cdn-icons-png.flaticon.com/512/1680/1680012.png",
    };
  } else if (type === "periodic") {
    return {
      title: "CẬP NHẬT AN NINH",
      body:
        `Hiện có ${total_count} người lạ đang xuất hiện tại camera ${camera_id}.\n` +
        `Thời gian: ${time}\n` +
        `Đây là thông báo định kỳ.`,
      icon:
        "/static/images/info-icon.png" ||
        "https://cdn-icons-png.flaticon.com/512/1680/1680012.png",
    };
  }

  // Mặc định nếu không có loại phù hợp
  return {
    title: "THÔNG BÁO HỆ THỐNG",
    body: "Có cảnh báo mới từ hệ thống an ninh",
    icon: "https://cdn-icons-png.flaticon.com/512/1680/1680012.png",
  };
}

// Handle messages from WebSocket
function handleWebSocketMessage(data) {
  try {
    const message = JSON.parse(data);

    if (message.type === "recognition_result") {
      // Process recognition results
      handleRecognitionResults(message.faces, message.performance);
    } else if (message.type === "security_notification") {
      // Xử lý thông báo bảo mật từ backend
      handleSecurityNotification(message.notification);
    } else if (message.type === "error") {
      showNotification("Server error: " + message.message, "error");
    } else if (message.type === "command_result") {
      handleCommandResult(message);
    }
  } catch (error) {
    console.error("Error handling websocket message:", error);
  }
}

// Handle command results
function handleCommandResult(message) {
  if (message.command === "cleanup_temp") {
    if (message.success) {
      showNotification(
        `Cleaned up ${message.data.deleted_count} temporary files`,
        "success"
      );
    } else {
      showNotification("Failed to clean temporary files", "error");
    }
  } else if (message.command === "reset_caches") {
    if (message.success) {
      showNotification("Recognition caches reset successfully", "success");
    } else {
      showNotification("Failed to reset caches", "error");
    }
  } else if (message.command === "embeddings_complete") {
    // Xử lý thông báo hoàn thành việc tạo embeddings
    if (message.success) {
      showNotification("Embeddings regenerated successfully", "success");

      // Reload dataset users to reflect changes
      loadData();
    } else {
      showNotification(
        `Error generating embeddings: ${message.message || "Unknown error"}`,
        "error"
      );
    }

    // Đảm bảo nút được reset về trạng thái ban đầu
    updateDatasetBtn.innerHTML =
      '<i class="fas fa-sync-alt text-xl"></i><span class="text-sm">Update Dataset</span>';
    updateDatasetBtn.disabled = false;
  }
}

// Handle recognition results - cập nhật logic theo dõi danh tính
function handleRecognitionResults(faces, performance) {
  if (!isRecognizing) return;

  // Update stats
  if (performance) {
    stats.latency = Date.now() - lastFrameSent;
    if (performance.detection_time)
      stats.detection_time = performance.detection_time;
    if (performance.recognition_time)
      stats.recognition_time = performance.recognition_time;
    if (performance.spoof_detection_time)
      stats.spoof_detection_time = performance.spoof_detection_time;

    updateStatsDisplay();
  }

  // Clear existing face boxes
  faceBoxesContainer.innerHTML = "";

  // Process detections
  const updatesNeeded = [];

  if (faces && faces.length > 0) {
    faces.forEach((face) => {
      // Create face box for live display
      createFaceBox(face);

      // Lấy track_id để theo dõi danh tính
      const trackId = face.track_id !== undefined ? face.track_id : -1;

      if (trackId >= 0) {
        // Nếu có track_id hợp lệ
        const faceId = `track_${trackId}`;
        const currentName = face.name || "Unknown";
        const previousIdentity = faceIdentityMap.get(faceId);

        // Thêm khuôn mặt mới hoặc cập nhật khi danh tính thay đổi
        if (!previousIdentity || previousIdentity !== currentName) {
          faceIdentityMap.set(faceId, currentName);

          // Đánh dấu cần cập nhật danh sách
          updatesNeeded.push({
            ...face,
            faceId: faceId,
            timestamp: new Date(),
            identity_changed: previousIdentity ? true : false,
            previous_identity: previousIdentity,
          });
        }
      } else {
        // Khuôn mặt không có track_id vẫn được thêm vào để hiển thị
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

  // Thêm khuôn mặt mới/cập nhật vào danh sách
  if (updatesNeeded.length > 0) {
    // Thêm khuôn mặt mới lên đầu danh sách
    detectedFaces = [...updatesNeeded, ...detectedFaces].slice(0, 20); // Giữ tối đa 20 khuôn mặt
    updateDetectedFacesList();
  }
}

// Update stats display
function updateStatsDisplay() {
  latencyStat.textContent = `${stats.latency}ms`;
  recogTimeStat.textContent = `${
    stats.recognition_time ? stats.recognition_time.toFixed(1) + "ms" : "-"
  }`;
  spoofTimeStat.textContent = `${
    stats.spoof_detection_time
      ? stats.spoof_detection_time.toFixed(1) + "ms"
      : "-"
  }`;
  detectTimeStat.textContent = `${
    stats.detection_time ? stats.detection_time.toFixed(1) + "ms" : "-"
  }`;
}

// Create face box element
function createFaceBox(face) {
  const [x1, y1, x2, y2] = face.bbox;

  // Tính toán tỷ lệ giữa kích thước hiển thị và kích thước xử lý
  const videoElement = document.getElementById("video");
  const displayWidth = videoElement.clientWidth;
  const displayHeight = videoElement.clientHeight;
  const videoWidth = videoElement.videoWidth || canvas.width;
  const videoHeight = videoElement.videoHeight || canvas.height;

  // Tính toán tọa độ đã điều chỉnh tỷ lệ
  const scaleX = displayWidth / videoWidth;
  const scaleY = displayHeight / videoHeight;

  const scaledX1 = x1 * scaleX;
  const scaledY1 = y1 * scaleY;
  const scaledWidth = (x2 - x1) * scaleX;
  const scaledHeight = (y2 - y1) * scaleY;

  // Tạo và định vị face box
  const faceBox = document.createElement("div");
  faceBox.className = "face-box";
  faceBox.style.width = `${scaledWidth}px`;
  faceBox.style.height = `${scaledHeight}px`;
  faceBox.style.left = `${scaledX1}px`;
  faceBox.style.top = `${scaledY1}px`;
  // Set border color based on face status
  if (face.is_spoofed) {
    faceBox.style.borderColor = "#ef4444"; // Red for spoofed faces
  } else if (face.is_unknown) {
    faceBox.style.borderColor = "#f97316"; // Orange for unknown faces
  } else {
    faceBox.style.borderColor = "#22c55e"; // Green for known faces
  }

  // Create label
  const faceLabel = document.createElement("div");
  faceLabel.className = "face-label";

  // Set label text
  let labelText = face.name || "Unknown";

  if (face.is_spoofed) {
    labelText += " [SPOOF]";
  }

  if (face.wearing_mask) {
    labelText += " [MASK]";
  }

  faceLabel.textContent = labelText;

  // Add confidence if available
  if (face.confidence) {
    const confidenceSpan = document.createElement("span");
    confidenceSpan.className = "ml-1 text-xs text-slate-400";
    confidenceSpan.textContent = `${(face.confidence * 100).toFixed(0)}%`;
    faceLabel.appendChild(confidenceSpan);
  }

  faceBox.appendChild(faceLabel);
  faceBoxesContainer.appendChild(faceBox);
}

// Start WebRTC recognition - modified to focus on recognition only
function startRecognition() {
  if (isRecognizing) return;

  // Xóa overlay tạm dừng nếu có
  if (window.pauseOverlay) {
    window.pauseOverlay.remove();
    window.pauseOverlay = null;
  }

  // Connect WebSocket
  setupWebSocketConnection();
  updateConnectionStatus("connecting");

  isRecognizing = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  recordingIndicator.classList.remove("hidden");

  // Start sending frames
  sendVideoFrames();
}

// Stop recognition - đã cập nhật để reset danh sách đã phát hiện
function stopRecognition() {
  if (!isRecognizing) return;

  isRecognizing = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  recordingIndicator.classList.add("hidden");

  // Reset theo dõi khuôn mặt đã phát hiện
  // detectedFaceIds.clear(); - Không cần thiết nữa vì chúng ta sử dụng faceIdentityMap

  // Close WebSocket if open
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  // Clear face boxes
  faceBoxesContainer.innerHTML = "";
}

// Send video frames via WebSocket
function sendVideoFrames() {
  if (!isRecognizing) return;

  totalFrames++;
  framesLastSecond++;

  const now = Date.now();
  if (now - lastFrameSent >= frameInterval) {
    lastFrameSent = now;

    try {
      if (socket && socket.readyState === WebSocket.OPEN) {
        // Capture frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to data URL
        const frameDataUrl = canvas.toDataURL("image/jpeg", 0.7);

        // Send via WebSocket
        socket.send(
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

// Calculate and update FPS
function calculateFPS() {
  currentFPS = framesLastSecond;
  fpsCounter.textContent = `${currentFPS} FPS`;
  framesLastSecond = 0;
}

// Update detected faces list UI - cập nhật để hiển thị thông tin thay đổi danh tính
function updateDetectedFacesList() {
  if (detectedFaces.length === 0) {
    detectedFacesList.innerHTML = `
            <div class="text-center py-8 text-slate-400">
                <i class="fas fa-user-slash text-3xl mb-2"></i>
                <p>No faces detected yet</p>
            </div>
        `;
    detectedCount.textContent = "0";
    return;
  }

  detectedCount.textContent = detectedFaces.length.toString();

  detectedFacesList.innerHTML = detectedFaces
    .map((face) => {
      // Determine status class based on face properties
      let statusClass = face.is_unknown
        ? "bg-amber-500/10 border border-amber-500/20"
        : "bg-green-500/10 border border-green-500/20";
      let statusBadgeClass = face.is_unknown
        ? "bg-amber-500/20 text-amber-500"
        : "bg-green-500/20 text-green-500";

      if (face.is_spoofed) {
        statusClass = "bg-red-500/10 border border-red-500/20";
        statusBadgeClass = "bg-red-500/20 text-red-500";
      }

      // Determine face name and status text
      const name = face.name || "Unknown";
      let statusText = face.confidence
        ? `${(face.confidence * 100).toFixed(0)}%`
        : "";

      if (face.is_spoofed) {
        statusText = "SPOOF";
      }

      // Sử dụng ảnh từ khuôn mặt được capture
      const imgUrl =
        face.image ||
        `https://ui-avatars.com/api/?name=${encodeURIComponent(
          name
        )}&background=random`;

      // Hiển thị ID khuôn mặt nếu có
      const trackInfo = face.track_id >= 0 ? `#${face.track_id}` : "";

      // Thêm thông báo nếu danh tính thay đổi
      const identityChangedBadge = face.identity_changed
        ? `<div class="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-300 ml-1">ID Changed</div>`
        : "";

      return `
        <div class="flex items-center gap-3 p-3 rounded-lg ${statusClass}">
          <div class="w-12 h-12 rounded-full overflow-hidden bg-slate-700">
              <img src="${imgUrl}" alt="${name}" class="w-full h-full object-cover" style="transition: all 0.2s ease;">
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
              <p class="text-xs text-slate-400">${formatTime(
                face.timestamp
              )}</p>
          </div>
          <div class="text-xs px-2 py-1 rounded-full ${statusBadgeClass}">
              ${statusText}
          </div>
      </div>
        `;
    })
    .join("");
}

// Update dataset users list UI - fixed to use data-user-id
function updateDatasetUsersList() {
  if (datasetUsers.length === 0) {
    datasetUsersList.innerHTML = `
        <div class="text-center py-8 text-slate-400">
          <i class="fas fa-user-plus text-3xl mb-2"></i>
          <p>No users in dataset</p>
        </div>
      `;
    datasetCount.textContent = "0";
    return;
  }

  datasetCount.textContent = datasetUsers.length.toString();

  datasetUsersList.innerHTML = datasetUsers
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

// Format time for display
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

// Update dataset
function updateDataset() {
  // Show loading and confirmation dialog
  if (
    !confirm(
      "This will regenerate all embeddings with augmentation. This may take some time. Continue?"
    )
  ) {
    return;
  }

  // Show loading state
  updateDatasetBtn.innerHTML =
    '<i class="fas fa-spinner fa-spin text-xl"></i><span class="text-sm">Processing...</span>';
  updateDatasetBtn.disabled = true;

  // Call the new API endpoint to regenerate embeddings
  fetch("/api/regenerate_embeddings", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showNotification(
          "Embedding regeneration started in background",
          "info"
        );

        // Show additional message about waiting
        setTimeout(() => {
          showNotification(
            "This process may take a few minutes. The system will update automatically when complete.",
            "info"
          );
        }, 3000);
      } else {
        showNotification("Failed to start embedding regeneration", "error");

        // Reset button
        updateDatasetBtn.innerHTML =
          '<i class="fas fa-sync-alt text-xl"></i><span class="text-sm">Update Dataset</span>';
        updateDatasetBtn.disabled = false;
      }
    })
    .catch((error) => {
      console.error("Error regenerating embeddings:", error);
      showNotification("Error generating embeddings", "error");

      // Reset button
      updateDatasetBtn.innerHTML =
        '<i class="fas fa-sync-alt text-xl"></i><span class="text-sm">Update Dataset</span>';
      updateDatasetBtn.disabled = false;
    });
}

// Clean temp files
function cleanTempFiles() {
  // Show confirmation dialog
  if (confirm("Are you sure you want to delete all temporary capture files?")) {
    // Show loading
    cleanTempBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    cleanTempBtn.disabled = true;

    // Send command via WebSocket if connected
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "command",
          command: "cleanup_temp",
          days: 0,
        })
      );

      // Will reset button when we get a response
    } else {
      // Fallback to REST API
      fetch("/api/cleanup/temp_captures?days=0", {
        method: "DELETE",
      })
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
          cleanTempBtn.innerHTML =
            '<i class="fas fa-trash-alt text-xl"></i><span class="text-sm">Clean Temp</span>';
          cleanTempBtn.disabled = false;
        });
    }
  }
}

// Test notification - updated to use same format as security notifications
function testNotification() {
  // Tạo dữ liệu mẫu cho thông báo bảo mật
  const mockData = {
    type: "new_face",
    camera_id: "Camera Test",
    new_count: 1,
    total_count: 3,
    timestamp: new Date().toISOString(),
  };

  // Gọi hàm xử lý thông báo bảo mật
  handleSecurityNotification(mockData);
}

// Save notification settings
function saveNotificationSettingsHandler() {
  const email = notificationEmail.value.trim();
  const enableNotifs = enableNotifications.checked;

  if (email && !/^\S+@\S+\.\S+$/.test(email)) {
    emailError.classList.remove("hidden");
    return;
  }

  emailError.classList.add("hidden");

  // Save to server via API
  fetch("/api/settings/email", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email: email,
      enable_notifications: enableNotifs,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showNotification("Notification settings saved", "success");
        notificationSettings.classList.add("hidden");
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

// Delete user from dataset - updated to work with data-user-id
function deleteUser(userId) {
  if (confirm("Are you sure you want to delete this user from the dataset?")) {
    // Find the button by data attribute instead of onclick
    const deleteButton = document.querySelector(
      `button[data-user-id="${userId}"]`
    );

    // Show loading state
    if (deleteButton) {
      deleteButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
      deleteButton.disabled = true;
    }

    // Call API to delete the user
    fetch(`/api/dataset/users/${userId}`, {
      method: "DELETE",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.success) {
          // Remove from local data and update UI
          datasetUsers = datasetUsers.filter((user) => user.id !== userId);
          updateDatasetUsersList();
          showNotification(`User ${userId} deleted from dataset`, "success");

          // Restart recognition if active to apply changes
          if (isRecognizing) {
            stopRecognition();
            setTimeout(startRecognition, 500);
          }
        } else {
          showNotification("Failed to delete user", "error");

          // Reset button state
          if (deleteButton) {
            deleteButton.innerHTML = '<i class="fas fa-trash-alt text-sm"></i>';
            deleteButton.disabled = false;
          }
        }
      })
      .catch((error) => {
        console.error("Error deleting user:", error);
        showNotification(`Error deleting user: ${error.message}`, "error");

        // Reset button state
        if (deleteButton) {
          deleteButton.innerHTML = '<i class="fas fa-trash-alt text-sm"></i>';
          deleteButton.disabled = false;
        }
      });
  }
}

// Show notification toast
function showNotification(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `fixed bottom-4 left-4 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 ${
    type === "success"
      ? "bg-green-600"
      : type === "error"
      ? "bg-red-600"
      : type === "warning"
      ? "bg-amber-600"
      : "bg-blue-600"
  }`;

  toast.innerHTML = `
        <i class="fas ${
          type === "success"
            ? "fa-check-circle"
            : type === "error"
            ? "fa-exclamation-circle"
            : type === "warning"
            ? "fa-exclamation-triangle"
            : "fa-info-circle"
        }"></i>
        <span>${message}</span>
    `;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}

// Initialize the app
window.onload = init;
