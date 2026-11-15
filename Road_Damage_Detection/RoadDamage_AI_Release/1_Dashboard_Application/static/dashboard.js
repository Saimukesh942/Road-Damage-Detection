/* 
   RoadDamage AI — Dashboard Logic 
*/

const state = { file: null, busy: false, resultReady: false };

const allowedExtensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const demoSamplePath = "/static/assets/demo_test_batch0_labels.jpg";
const classGuide = {
  D00:    { title: "Longitudinal crack",     summary: "Cracking along the road direction." },
  D10:    { title: "Transverse crack",       summary: "Cracking across the lane width." },
  D20:    { title: "Alligator crack",        summary: "Web-like cracking pattern." },
  D40:    { title: "Pothole",                summary: "Pothole or severe surface defect." },
  Repair: { title: "Repair area",            summary: "Previously patched surface." },
};

// Helpers
const $ = id => document.getElementById(id);

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1024)) + " KB";
}

function classMeta(label) {
  return classGuide[label] || { title: label, summary: "Detected road damage." };
}

function setStatus(msg, tone = "neutral") {
  const el = $("status-msg");
  el.textContent = msg;
  el.dataset.status = tone;
}


// File management
function isAllowed(file) {
  if (!file) return false;
  const name = file.name.toLowerCase();
  return allowedExtensions.some(ext => name.endsWith(ext));
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    $("preview-image").src = reader.result;
    $("preview-image").classList.remove("hidden");
    $("preview-empty").classList.add("hidden");
    $("file-thumb").src = reader.result;
    $("dropzone-copy").textContent = "Drop another image to replace.";
  };
  reader.readAsDataURL(file);
}

function setFile(file) {
  if (!file) return;
  if (!isAllowed(file)) {
    setStatus("Only JPG, PNG, BMP, or WEBP images are accepted.", "error");
    return;
  }

  state.file = file;
  state.resultReady = false;

  showPreview(file);
  $("file-info").classList.remove("hidden");
  $("file-name").textContent = file.name;
  $("file-size").textContent = formatBytes(file.size);
  $("run-btn").disabled = false;

  // Reset result side
  $("result-image").classList.add("hidden");
  $("result-empty").classList.remove("hidden");
  $("result-loader").classList.add("hidden");
  $("download-btn").classList.add("hidden");
  $("det-count").textContent = "0";
  $("det-confidence").textContent = "0%";
  $("detection-list").innerHTML = '<div class="empty-results">Run detection to see results.</div>';

  setStatus("Image loaded. Click 'Run Detection' to analyze.", "ready");
}

function clearFile() {
  state.file = null;
  state.resultReady = false;
  $("image-input").value = "";
  $("preview-image").classList.add("hidden");
  $("preview-image").removeAttribute("src");
  $("preview-empty").classList.remove("hidden");
  $("file-info").classList.add("hidden");
  $("run-btn").disabled = true;
  $("dropzone-copy").textContent = "or click to browse files";

  // Reset results
  $("result-image").classList.add("hidden");
  $("result-empty").classList.remove("hidden");
  $("result-loader").classList.add("hidden");
  $("download-btn").classList.add("hidden");
  $("model-chip").classList.add("hidden");
  $("det-count").textContent = "0";
  $("det-confidence").textContent = "0%";
  $("detection-list").innerHTML = '<div class="empty-results">No detections yet.</div>';

  setStatus("Ready. Upload an image to begin.", "neutral");
}


// Detection
async function runDetection() {
  if (!state.file || state.busy) return;

  const data = new FormData();
  data.append("image", state.file);

  state.busy = true;
  $("run-btn").disabled = true;
  $("run-btn").textContent = "⏳ Analyzing...";
  $("result-empty").classList.add("hidden");
  $("result-loader").classList.remove("hidden");
  setStatus("Running YOLOv8 detection...", "running");

  try {
    const resp = await fetch("/api/predict", { method: "POST", body: data });
    const payload = await resp.json();

    if (!payload.ok) throw new Error(payload.error || "Detection failed.");

    renderResults(payload);
    setStatus(`Done! Found ${payload.count} damage${payload.count !== 1 ? "s" : ""}.`, "success");

    // Refresh recent searches
    loadHistory();
  } catch (err) {
    $("result-loader").classList.add("hidden");
    $("result-empty").classList.remove("hidden");
    setStatus("Error: " + err.message, "error");
  } finally {
    state.busy = false;
    $("run-btn").disabled = false;
    $("run-btn").textContent = "🔍 Run Detection";
  }
}

function renderResults(payload) {
  state.resultReady = true;

  // Show annotated image
  $("result-image").src = payload.annotatedImage;
  $("result-image").classList.remove("hidden");
  $("result-loader").classList.add("hidden");
  $("result-empty").classList.add("hidden");

  // Download button
  $("download-btn").href = payload.annotatedImage;
  $("download-btn").classList.remove("hidden");

  // Model chip
  $("model-chip").classList.remove("hidden");

  // Stats
  const sorted = [...payload.detections].sort((a, b) => b.confidence - a.confidence);
  const topConf = sorted.length ? Math.round(sorted[0].confidence * 100) : 0;

  $("det-count").textContent = payload.count;
  $("det-confidence").textContent = topConf + "%";

  // Detection list
  const list = $("detection-list");
  list.innerHTML = "";

  if (!sorted.length) {
    list.innerHTML = '<div class="empty-results">No road damage detected in this image.</div>';
    return;
  }

  sorted.forEach((det, i) => {
    const pct = (det.confidence * 100).toFixed(1);
    const meta = classMeta(det.label);
    const item = document.createElement("div");
    item.className = "detection-item";
    item.innerHTML = `
      <div class="det-header">
        <span class="det-rank">#${i + 1}</span>
        <span class="det-label">${det.label}</span>
      </div>
      <div class="det-name">${meta.title}</div>
      <div class="det-conf-bar"><div class="det-conf-fill" style="width:${pct}%"></div></div>
      <div class="det-conf-text">Confidence: ${pct}%</div>
      <div class="det-bbox">Box: [${det.box.map(v => Math.round(v)).join(", ")}]</div>
    `;
    list.appendChild(item);
  });
}



// Recent Searches (History)

function timeAgo(dateStr) {
  const now = new Date();
  const then = new Date(dateStr + "Z"); // SQLite timestamps are UTC
  const diffMs = now - then;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return then.toLocaleDateString();
}

async function loadHistory() {
  const container = $("history-container");
  try {
    const resp = await fetch("/api/history");
    const data = await resp.json();

    if (!data.ok) {
      container.innerHTML = '<div class="history-empty"><p>Could not load history.</p></div>';
      return;
    }

    if (!data.history.length) {
      container.innerHTML = `
        <div class="history-empty">
          <div class="empty-icon-lg">🕒</div>
          <h4>No searches yet</h4>
          <p>Upload and analyze an image to see it here.</p>
        </div>`;
      $("clear-history-btn").classList.add("hidden");
      return;
    }

    $("clear-history-btn").classList.remove("hidden");

    let html = '<div class="history-grid">';
    data.history.forEach(item => {
      const badgeClass = item.damageCount > 0 ? "damages" : "clean";
      const badgeText = item.damageCount > 0
        ? `${item.damageCount} damage${item.damageCount !== 1 ? "s" : ""}`
        : "No damage";
      html += `
        <div class="history-card" data-id="${item.id}">
          <button class="hist-delete-btn" onclick="deleteHistoryEntry(${item.id}, event)" title="Delete this search">✕</button>
          <div class="history-card-images">
            <img src="${item.thumbnail}" alt="Original" loading="lazy">
            <img src="${item.annotatedThumbnail}" alt="Detected" loading="lazy">
            <span class="img-label left">Original</span>
            <span class="img-label right">Detected</span>
          </div>
          <div class="history-card-body">
            <div class="hist-filename" title="${item.filename}">${item.filename}</div>
            <div class="hist-meta">
              <span class="hist-badge ${badgeClass}">${badgeText}</span>
              ${item.topConfidence > 0 ? `<span>${item.topConfidence}% conf</span>` : ""}
            </div>
            <div class="hist-time">${timeAgo(item.searchedAt)}</div>
          </div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch {
    container.innerHTML = '<div class="history-empty"><p>Could not load history.</p></div>';
  }
}

async function deleteHistoryEntry(id, event) {
  event.stopPropagation();
  if (!confirm("Delete this search from history?")) return;
  try {
    await fetch(`/api/history/delete/${id}`, { method: "POST" });
    loadHistory();
  } catch {
    // ignore
  }
}

async function clearHistory() {
  if (!confirm("Clear all search history?")) return;
  try {
    await fetch("/api/history/clear", { method: "POST" });
    loadHistory();
  } catch {
    // ignore
  }
}


document.addEventListener("DOMContentLoaded", () => {
  const input = $("image-input");
  const dropzone = $("dropzone");

  // Browse click
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    input.click();
  });

  // File input change
  input.addEventListener("change", () => {
    const [file] = input.files;
    setFile(file);
  });

  // Drag & drop
  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("drag-active"); });
  });
  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("drag-active"); });
  });
  dropzone.addEventListener("drop", e => {
    const [file] = e.dataTransfer.files;
    setFile(file);
  });

  // Buttons
  $("run-btn").addEventListener("click", runDetection);
  $("remove-file").addEventListener("click", clearFile);
  $("clear-history-btn").addEventListener("click", clearHistory);

  // Load recent searches on page load
  loadHistory();
});

