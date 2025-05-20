/**
 * WebGL Bounding Box Renderer
 * Optimized for rendering face detection boxes efficiently using WebGL
 */

class WebGLBoxRenderer {
  /**
   * Create a new WebGL Box Renderer
   * @param {HTMLElement} container - Container element to append the canvas to
   * @param {Object} options - Configuration options
   */
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign(
      {
        lineWidth: 2,
        defaultColor: [0, 1, 0, 1], // Green
        unknownColor: [1, 0.45, 0.1, 1], // Orange
        spoofColor: [1, 0, 0, 1], // Red
        textColor: [1, 1, 1, 1], // White
        textBackgroundColor: [0, 0, 0, 0.7], // Semi-transparent black
        useCanvas2DFallback: true,
        showLabels: true, // New option to control label visibility
        showConfidence: true, // Option to show/hide confidence percentages
        debugMode: false,
      },
      options
    );

    this.boxes = new Map(); // Map of box id -> box data
    this.canvas = null;
    this.gl = null;
    this.program = null;
    this.canvas2d = null;
    this.ctx2d = null;
    this.isWebGLSupported = false;
    this.frameId = null;
    this.videoElement = null;
    this.pixelRatio = window.devicePixelRatio || 1;

    // Initialize the renderer
    this.init();
  }

  /**
   * Initialize the WebGL renderer
   */
  init() {
    // Create canvas element
    this.canvas = document.createElement("canvas");
    this.canvas.className = "webgl-overlay";
    this.canvas.style.position = "absolute";
    this.canvas.style.top = "0";
    this.canvas.style.left = "0";
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.pointerEvents = "none";
    this.container.appendChild(this.canvas);

    // Try to initialize WebGL
    try {
      this.gl =
        this.canvas.getContext("webgl") ||
        this.canvas.getContext("experimental-webgl");

      if (!this.gl) {
        throw new Error("WebGL not supported");
      }

      this.isWebGLSupported = true;
      this.initWebGL();

      if (this.options.debugMode) {
        console.log("WebGL initialized successfully");
      }
    } catch (error) {
      console.warn("WebGL initialization failed:", error);

      if (this.options.useCanvas2DFallback) {
        this.initCanvas2DFallback();
      } else {
        throw error;
      }
    }

    // Set up resize observer to handle container size changes
    this.resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    this.resizeObserver.observe(this.container);

    // Initial size update
    this.updateCanvasSize();
  }

  /**
   * Initialize WebGL context and shaders
   */
  initWebGL() {
    const gl = this.gl;

    // Create shader program
    const vertexShaderSource = `
      attribute vec2 a_position;
      attribute vec4 a_color;

      uniform vec2 u_resolution;
      
      varying vec4 v_color;
      
      void main() {
        // Convert pixel coordinates to clip space
        vec2 clipSpace = (a_position / u_resolution) * 2.0 - 1.0;
        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        
        // Pass color to fragment shader
        v_color = a_color;
      }
    `;

    const fragmentShaderSource = `
      precision mediump float;
      
      varying vec4 v_color;
      
      void main() {
        gl_FragColor = v_color;
      }
    `;

    // Create and compile shaders
    const vertexShader = this.compileShader(
      gl,
      gl.VERTEX_SHADER,
      vertexShaderSource
    );
    const fragmentShader = this.compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      fragmentShaderSource
    );

    // Create program and link shaders
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(this.program);
      gl.deleteProgram(this.program);
      throw new Error(`Failed to link program: ${error}`);
    }

    // Get attribute and uniform locations
    this.positionAttributeLocation = gl.getAttribLocation(
      this.program,
      "a_position"
    );
    this.colorAttributeLocation = gl.getAttribLocation(this.program, "a_color");
    this.resolutionUniformLocation = gl.getUniformLocation(
      this.program,
      "u_resolution"
    );

    // Create buffers
    this.positionBuffer = gl.createBuffer();
    this.colorBuffer = gl.createBuffer();
  }

  /**
   * Initialize Canvas 2D fallback if WebGL is not supported
   */
  initCanvas2DFallback() {
    this.canvas2d = document.createElement("canvas");
    this.canvas2d.className = "canvas2d-overlay";
    this.canvas2d.style.position = "absolute";
    this.canvas2d.style.top = "0";
    this.canvas2d.style.left = "0";
    this.canvas2d.style.width = "100%";
    this.canvas2d.style.height = "100%";
    this.canvas2d.style.pointerEvents = "none";
    this.container.appendChild(this.canvas2d);

    this.ctx2d = this.canvas2d.getContext("2d");

    console.log("Using Canvas 2D fallback for rendering");
  }

  /**
   * Compile a WebGL shader
   * @param {WebGLRenderingContext} gl - WebGL context
   * @param {number} type - Shader type (VERTEX_SHADER or FRAGMENT_SHADER)
   * @param {string} source - Shader source code
   * @returns {WebGLShader} Compiled shader
   */
  compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Failed to compile shader: ${error}`);
    }

    return shader;
  }

  /**
   * Update canvas size to match container dimensions
   */
  updateCanvasSize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // Update both canvases if they exist
    if (this.canvas) {
      this.canvas.width = width * this.pixelRatio;
      this.canvas.height = height * this.pixelRatio;
    }

    if (this.canvas2d) {
      this.canvas2d.width = width * this.pixelRatio;
      this.canvas2d.height = height * this.pixelRatio;
    }

    // Update WebGL viewport
    if (this.isWebGLSupported && this.gl) {
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    // Re-render with new size
    this.render();
  }

  /**
   * Set the video element to use for scaling coordinates
   * @param {HTMLVideoElement} videoElement - Video element
   */
  setVideoElement(videoElement) {
    this.videoElement = videoElement;
  }

  /**
   * Calculate scaling factors for converting between video and display coordinates
   * @returns {Object} Scaling factors {scaleX, scaleY, videoWidth, videoHeight}
   */
  getScalingFactors() {
    const containerWidth = this.container.clientWidth;
    const containerHeight = this.container.clientHeight;

    let videoWidth = containerWidth;
    let videoHeight = containerHeight;

    // If we have a video element, use its dimensions
    if (this.videoElement) {
      videoWidth = this.videoElement.videoWidth || containerWidth;
      videoHeight = this.videoElement.videoHeight || containerHeight;
    }

    return {
      scaleX: containerWidth / videoWidth,
      scaleY: containerHeight / videoHeight,
      videoWidth: videoWidth,
      videoHeight: videoHeight,
    };
  }

  /**
   * Add or update a bounding box
   * @param {string|number} id - Unique identifier for the box
   * @param {Object} box - Box data {bbox: [x1, y1, x2, y2], name, is_unknown, is_spoofed, confidence}
   */
  updateBox(id, box) {
    this.boxes.set(id, box);

    // Render on next animation frame
    if (!this.frameId) {
      this.frameId = requestAnimationFrame(() => {
        this.render();
        this.frameId = null;
      });
    }
  }

  /**
   * Update multiple boxes at once
   * @param {Array} boxes - Array of boxes with IDs
   */
  updateBoxes(boxes) {
    // Clear existing boxes
    this.boxes.clear();

    // Add new boxes
    for (const box of boxes) {
      const id =
        box.track_id !== undefined ? box.track_id : Math.random().toString(36);
      this.boxes.set(id, box);
    }

    // Render on next animation frame
    if (!this.frameId) {
      this.frameId = requestAnimationFrame(() => {
        this.render();
        this.frameId = null;
      });
    }
  }

  /**
   * Remove a box by ID
   * @param {string|number} id - Box ID to remove
   */
  removeBox(id) {
    if (this.boxes.has(id)) {
      this.boxes.delete(id);
      this.render();
    }
  }

  /**
   * Clear all boxes
   */
  clearBoxes() {
    this.boxes.clear();
    this.render();
  }

  /**
   * Render all boxes using WebGL or Canvas 2D fallback
   */
  render() {
    if (this.boxes.size === 0) {
      this.clear();
      return;
    }

    if (this.isWebGLSupported) {
      this.renderWebGL();
    } else if (this.ctx2d) {
      this.renderCanvas2D();
    }
  }

  /**
   * Clear the canvas
   */
  clear() {
    if (this.isWebGLSupported && this.gl) {
      this.gl.clearColor(0, 0, 0, 0);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    } else if (this.ctx2d) {
      this.ctx2d.clearRect(0, 0, this.canvas2d.width, this.canvas2d.height);
    }
  }

  /**
   * Render boxes using WebGL
   */
  renderWebGL() {
    const gl = this.gl;
    const { scaleX, scaleY } = this.getScalingFactors();

    // Clear the canvas
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use our shader program
    gl.useProgram(this.program);

    // Set the resolution uniform
    gl.uniform2f(
      this.resolutionUniformLocation,
      this.canvas.width,
      this.canvas.height
    );

    // Calculate positions and colors for all boxes
    const positions = [];
    const colors = [];

    const lineWidth = this.options.lineWidth * this.pixelRatio;

    // Process each box
    for (const box of this.boxes.values()) {
      const [x1, y1, x2, y2] = box.bbox;

      // Scale box coordinates
      const scaledX1 = x1 * scaleX * this.pixelRatio;
      const scaledY1 = y1 * scaleY * this.pixelRatio;
      const scaledX2 = x2 * scaleX * this.pixelRatio;
      const scaledY2 = y2 * scaleY * this.pixelRatio;

      // Determine box color based on properties
      let color;
      if (box.is_spoofed) {
        color = this.options.spoofColor;
      } else if (box.is_unknown) {
        color = this.options.unknownColor;
      } else {
        color = this.options.defaultColor;
      }

      // Add vertices for the box (6 vertices for 2 triangles)
      // Top line
      this.addRect(
        positions,
        colors,
        scaledX1,
        scaledY1,
        scaledX2,
        scaledY1 + lineWidth,
        color
      );

      // Bottom line
      this.addRect(
        positions,
        colors,
        scaledX1,
        scaledY2 - lineWidth,
        scaledX2,
        scaledY2,
        color
      );

      // Left line
      this.addRect(
        positions,
        colors,
        scaledX1,
        scaledY1 + lineWidth,
        scaledX1 + lineWidth,
        scaledY2 - lineWidth,
        color
      );

      // Right line
      this.addRect(
        positions,
        colors,
        scaledX2 - lineWidth,
        scaledY1 + lineWidth,
        scaledX2,
        scaledY2 - lineWidth,
        color
      );
    }

    // Upload position data
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.positionAttributeLocation);
    gl.vertexAttribPointer(
      this.positionAttributeLocation,
      2,
      gl.FLOAT,
      false,
      0,
      0
    );

    // Upload color data
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.colorAttributeLocation);
    gl.vertexAttribPointer(
      this.colorAttributeLocation,
      4,
      gl.FLOAT,
      false,
      0,
      0
    );

    // Enable blending for transparency
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Draw the boxes
    gl.drawArrays(gl.TRIANGLES, 0, positions.length / 2);

    // Clean up
    gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Draw text with Canvas 2D API (more appropriate for text)
    if (this.options.showLabels) {
      this.drawLabelsCanvas2D();
    }
  }

  /**
   * Add rectangle vertices to arrays
   */
  addRect(positions, colors, x1, y1, x2, y2, color) {
    // First triangle
    positions.push(x1, y1, x1, y2, x2, y1);
    // Second triangle
    positions.push(x1, y2, x2, y2, x2, y1);

    // Add colors for each vertex (6 vertices)
    for (let i = 0; i < 6; i++) {
      colors.push(...color);
    }
  }

  /**
   * Draw text labels using Canvas 2D API
   * WebGL is not well-suited for text rendering, so we use Canvas 2D for labels
   */
  drawLabelsCanvas2D() {
    // Skip label rendering if showLabels is false
    if (!this.options.showLabels) return;

    if (!this.canvas2d) {
      this.canvas2d = document.createElement("canvas");
      this.canvas2d.className = "canvas2d-overlay";
      this.canvas2d.style.position = "absolute";
      this.canvas2d.style.top = "0";
      this.canvas2d.style.left = "0";
      this.canvas2d.style.width = "100%";
      this.canvas2d.style.height = "100%";
      this.canvas2d.style.pointerEvents = "none";
      this.container.appendChild(this.canvas2d);
      this.ctx2d = this.canvas2d.getContext("2d");

      // Match size with WebGL canvas
      this.canvas2d.width = this.canvas.width;
      this.canvas2d.height = this.canvas.height;
    }

    const ctx = this.ctx2d;
    const { scaleX, scaleY } = this.getScalingFactors();

    // Clear the canvas
    ctx.clearRect(0, 0, this.canvas2d.width, this.canvas2d.height);

    // Scale for high-DPI displays
    ctx.scale(this.pixelRatio, this.pixelRatio);

    // Set text style
    ctx.font = "12px Arial";
    ctx.textAlign = "center";

    // Draw label for each box
    for (const box of this.boxes.values()) {
      const [x1, y1, x2, y2] = box.bbox;

      // Scale box coordinates
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;

      // Determine label text
      let labelText = box.name || "Unknown";

      // Skip drawing if label is empty (for add_person page)
      if (labelText.trim() === "") {
        // Only show confidence if enabled
        if (this.options.showConfidence && box.confidence) {
          labelText = `${Math.round(box.confidence * 100)}%`;
        } else {
          continue; // Skip this box's label entirely
        }
      } else {
        // Add status tags
        if (box.is_spoofed) labelText += " [SPOOF]";
        if (box.wearing_mask) labelText += " [MASK]";

        // Add confidence if available and enabled
        if (this.options.showConfidence && box.confidence) {
          labelText += ` ${Math.round(box.confidence * 100)}%`;
        }
      }

      // Skip empty labels
      if (labelText.trim() === "") continue;

      // Measure text width
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const textHeight = 16; // Approximate height

      // Calculate position (centered below box)
      const textX = (scaledX1 + scaledX2) / 2;
      const textY = scaledY2 + textHeight;

      // Draw text background
      ctx.fillStyle = `rgba(0, 0, 0, 0.7)`;
      ctx.fillRect(
        textX - textWidth / 2 - 4,
        textY - textHeight + 4,
        textWidth + 8,
        textHeight
      );

      // Draw text
      ctx.fillStyle = "white";
      ctx.fillText(labelText, textX, textY);
    }

    // Reset scale
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /**
   * Render boxes using Canvas 2D fallback
   */
  renderCanvas2D() {
    if (!this.ctx2d) return;

    const ctx = this.ctx2d;
    const { scaleX, scaleY } = this.getScalingFactors();

    // Clear the canvas
    ctx.clearRect(0, 0, this.canvas2d.width, this.canvas2d.height);

    // Scale for high-DPI displays
    ctx.scale(this.pixelRatio, this.pixelRatio);

    const lineWidth = this.options.lineWidth;

    // Draw each box
    for (const box of this.boxes.values()) {
      const [x1, y1, x2, y2] = box.bbox;

      // Scale box coordinates
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledWidth = (x2 - x1) * scaleX;
      const scaledHeight = (y2 - y1) * scaleY;

      // Determine box color based on properties
      if (box.is_spoofed) {
        ctx.strokeStyle = `rgba(${this.options.spoofColor[0] * 255}, ${
          this.options.spoofColor[1] * 255
        }, ${this.options.spoofColor[2] * 255}, ${this.options.spoofColor[3]})`;
      } else if (box.is_unknown) {
        ctx.strokeStyle = `rgba(${this.options.unknownColor[0] * 255}, ${
          this.options.unknownColor[1] * 255
        }, ${this.options.unknownColor[2] * 255}, ${
          this.options.unknownColor[3]
        })`;
      } else {
        ctx.strokeStyle = `rgba(${this.options.defaultColor[0] * 255}, ${
          this.options.defaultColor[1] * 255
        }, ${this.options.defaultColor[2] * 255}, ${
          this.options.defaultColor[3]
        })`;
      }

      // Draw box
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);

      // Draw labels if enabled
      if (this.options.showLabels) {
        // Determine label text
        let labelText = box.name || "Unknown";

        // Skip drawing if label is empty (for add_person page)
        if (labelText.trim() === "") {
          // Only show confidence if enabled
          if (this.options.showConfidence && box.confidence) {
            labelText = `${Math.round(box.confidence * 100)}%`;
          } else {
            continue; // Skip label
          }
        } else {
          // Add status tags
          if (box.is_spoofed) labelText += " [SPOOF]";
          if (box.wearing_mask) labelText += " [MASK]";

          // Add confidence if available and enabled
          if (this.options.showConfidence && box.confidence) {
            labelText += ` ${Math.round(box.confidence * 100)}%`;
          }
        }

        // Skip empty labels
        if (labelText.trim() === "") continue;

        // Measure text width
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;
        const textHeight = 16; // Approximate height

        // Calculate position (centered below box)
        const textX = scaledX1 + scaledWidth / 2;
        const textY = scaledY1 + scaledHeight + textHeight;

        // Draw text background
        ctx.fillStyle = `rgba(0, 0, 0, 0.7)`;
        ctx.fillRect(
          textX - textWidth / 2 - 4,
          textY - textHeight + 4,
          textWidth + 8,
          textHeight
        );

        // Draw text
        ctx.fillStyle = "white";
        ctx.textAlign = "center";
        ctx.font = "12px Arial";
        ctx.fillText(labelText, textX, textY);
      }
    }

    // Reset scale
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /**
   * Destroy the renderer and clean up resources
   */
  destroy() {
    // Cancel any pending animation frame
    if (this.frameId) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }

    // Disconnect resize observer
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    // Remove canvases
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }

    if (this.canvas2d && this.canvas2d.parentNode) {
      this.canvas2d.parentNode.removeChild(this.canvas2d);
    }

    // Clean up WebGL resources
    if (this.isWebGLSupported && this.gl) {
      this.gl.deleteBuffer(this.positionBuffer);
      this.gl.deleteBuffer(this.colorBuffer);
      this.gl.deleteProgram(this.program);
    }

    // Clear references
    this.canvas = null;
    this.canvas2d = null;
    this.gl = null;
    this.ctx2d = null;
    this.boxes.clear();
  }
}

// Export the renderer
window.WebGLBoxRenderer = WebGLBoxRenderer;
