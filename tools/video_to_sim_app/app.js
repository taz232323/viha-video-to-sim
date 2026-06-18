const defaultWindows = [
  {
    name: "scene_reference",
    start_sec: 0.5,
    end_sec: 2.0,
    purpose: "Full table, robot base, object, target, and calibration marker visible."
  },
  {
    name: "pre_pick",
    start_sec: 2.0,
    end_sec: 4.0,
    purpose: "Object visible before the human touches it."
  },
  {
    name: "pre_place",
    start_sec: 5.0,
    end_sec: 7.0,
    purpose: "Target visible before the object is placed."
  },
  {
    name: "post_place",
    start_sec: 7.0,
    end_sec: 9.0,
    purpose: "Object resting on/in the target with the hand out of frame."
  }
];

const pointOrder = ["front_left", "front_right", "back_right", "back_left", "pick_object", "target"];
const pointLabels = {
  front_left: "front_left",
  front_right: "front_right",
  back_right: "back_right",
  back_left: "back_left",
  pick_object: "pick_object",
  target: "target"
};

let session = null;
let representativePath = null;
let imageNaturalSize = null;
let points = {};
let currentPointIndex = 0;
let image = new Image();

const uploadForm = document.getElementById("uploadForm");
const uploadStatus = document.getElementById("uploadStatus");
const buildStatus = document.getElementById("buildStatus");
const windowsInput = document.getElementById("windowsInput");
const selectedFrames = document.getElementById("selectedFrames");
const canvas = document.getElementById("annotationCanvas");
const ctx = canvas.getContext("2d");
const currentPointLabel = document.getElementById("currentPointLabel");
const pointList = document.getElementById("pointList");
const results = document.getElementById("results");

windowsInput.value = JSON.stringify(defaultWindows, null, 2);
renderPointList();

function artifactUrl(path) {
  return `/artifact?path=${encodeURIComponent(path)}`;
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  if (label) button.textContent = busy ? label : button.dataset.originalLabel || button.textContent;
}

function setRepresentative(path, size) {
  representativePath = path;
  imageNaturalSize = size || imageNaturalSize;
  image = new Image();
  image.onload = () => {
    if (!imageNaturalSize) {
      imageNaturalSize = { width: image.naturalWidth, height: image.naturalHeight };
    }
    resizeCanvas();
    drawAnnotations();
  };
  image.src = artifactUrl(path);
}

function resizeCanvas() {
  if (!imageNaturalSize) return;
  const containerWidth = canvas.parentElement.clientWidth;
  const scale = Math.min(1, containerWidth / imageNaturalSize.width);
  canvas.width = Math.round(imageNaturalSize.width * scale);
  canvas.height = Math.round(imageNaturalSize.height * scale);
}

function toCanvasPoint(point) {
  return {
    x: point[0] * canvas.width / imageNaturalSize.width,
    y: point[1] * canvas.height / imageNaturalSize.height
  };
}

function toImagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * imageNaturalSize.width / rect.width;
  const y = (event.clientY - rect.top) * imageNaturalSize.height / rect.height;
  return [Math.round(x), Math.round(y)];
}

function drawAnnotations() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (image.complete) {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  }

  ctx.lineWidth = 3;
  ctx.font = "14px sans-serif";
  const corners = ["front_left", "front_right", "back_right", "back_left"];
  if (corners.every((key) => points[key])) {
    ctx.strokeStyle = "#00ffff";
    ctx.beginPath();
    corners.concat(["front_left"]).forEach((key, index) => {
      const point = toCanvasPoint(points[key]);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  for (const key of pointOrder) {
    if (!points[key]) continue;
    const point = toCanvasPoint(points[key]);
    ctx.fillStyle = key === "pick_object" ? "#ffff00" : key === "target" ? "#ff3333" : "#00ffff";
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = key === "target" ? "#ff7777" : "#ffffff";
    ctx.fillText(pointLabels[key], point.x + 12, point.y - 10);
  }
}

function renderPointList() {
  pointList.innerHTML = "";
  for (const key of pointOrder) {
    const item = document.createElement("div");
    item.className = `point-item ${points[key] ? "done" : ""}`;
    const value = points[key] ? `[${points[key][0]}, ${points[key][1]}]` : "not set";
    item.innerHTML = `<span>${key}</span><span>${value}</span>`;
    pointList.appendChild(item);
  }
  const current = pointOrder[currentPointIndex] || "done";
  currentPointLabel.textContent = current;
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.dataset.originalLabel = button.textContent;
  setBusy(button, true, "Processing...");
  uploadStatus.textContent = "Uploading video and extracting frames...";
  selectedFrames.innerHTML = "";
  results.innerHTML = "";
  points = {};
  currentPointIndex = 0;
  renderPointList();

  try {
    const formData = new FormData(uploadForm);
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "Upload failed");

    session = data.session_id;
    representativePath = data.representative_frame;
    imageNaturalSize = data.representative_size;
    uploadStatus.textContent = `Uploaded session ${session}\nRepresentative frame: ${representativePath}`;
    renderSelectedFrames(data.selected_frames);
    setRepresentative(representativePath, imageNaturalSize);
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    setBusy(button, false);
  }
});

function renderSelectedFrames(frames) {
  selectedFrames.innerHTML = "";
  for (const frame of frames) {
    const card = document.createElement("div");
    card.className = "frame-card";
    const button = document.createElement("button");
    button.type = "button";
    const img = document.createElement("img");
    img.src = artifactUrl(frame.path);
    button.appendChild(img);
    button.addEventListener("click", () => setRepresentative(frame.path));
    const meta = document.createElement("div");
    meta.textContent = `${frame.window} score=${Number(frame.score || 0).toFixed(4)}`;
    card.append(button, meta);
    selectedFrames.appendChild(card);
  }
}

canvas.addEventListener("click", (event) => {
  if (!imageNaturalSize || currentPointIndex >= pointOrder.length) return;
  const key = pointOrder[currentPointIndex];
  points[key] = toImagePoint(event);
  currentPointIndex += 1;
  renderPointList();
  drawAnnotations();
});

document.getElementById("undoPoint").addEventListener("click", () => {
  currentPointIndex = Math.max(0, currentPointIndex - 1);
  delete points[pointOrder[currentPointIndex]];
  renderPointList();
  drawAnnotations();
});

document.getElementById("resetPoints").addEventListener("click", () => {
  points = {};
  currentPointIndex = 0;
  renderPointList();
  drawAnnotations();
});

window.addEventListener("resize", () => {
  resizeCanvas();
  drawAnnotations();
});

document.getElementById("buildButton").addEventListener("click", async (event) => {
  if (!session) {
    buildStatus.textContent = "Upload a video first.";
    return;
  }
  const missing = pointOrder.filter((key) => !points[key]);
  if (missing.length) {
    buildStatus.textContent = `Missing points: ${missing.join(", ")}`;
    return;
  }

  const button = event.currentTarget;
  button.dataset.originalLabel = button.textContent;
  setBusy(button, true, "Building...");
  buildStatus.textContent = "Building MuJoCo scene, running sim, and rendering review PNG...";

  const payload = {
    session_id: session,
    task_type: document.getElementById("taskType").value,
    points,
    table: {
      width_m: Number(document.getElementById("tableWidth").value),
      depth_m: Number(document.getElementById("tableDepth").value),
      surface_z_m: Number(document.getElementById("tableZ").value)
    },
    robot: {
      base_x_m: Number(document.getElementById("robotX").value),
      base_y_m: Number(document.getElementById("robotY").value),
      base_z_m: Number(document.getElementById("robotZ").value)
    },
    object_radius_m: Number(document.getElementById("objectRadius").value),
    target_radius_m: Number(document.getElementById("targetRadius").value)
  };

  try {
    const response = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "Build failed");
    buildStatus.textContent = `SUCCESS=${data.success}\n${data.stdout}`;
    renderResults(data);
  } catch (error) {
    buildStatus.textContent = error.message;
  } finally {
    setBusy(button, false);
  }
});

function renderResults(data) {
  results.innerHTML = "";
  const links = document.createElement("div");
  links.className = "artifact-links";
  for (const [label, path] of [
    ["Review PNG", data.review_sheet_path],
    ["Annotation Overlay", data.annotation_overlay_path],
    ["Scene XML", data.scene_path],
    ["Result JSON", data.result_path],
    ["Spec JSON", data.spec_path]
  ]) {
    const a = document.createElement("a");
    a.href = artifactUrl(path);
    a.target = "_blank";
    a.textContent = label;
    links.appendChild(a);
  }

  const review = document.createElement("img");
  review.src = artifactUrl(data.review_sheet_path);
  const overlay = document.createElement("img");
  overlay.src = artifactUrl(data.annotation_overlay_path);

  results.append(links, review, overlay);
}
