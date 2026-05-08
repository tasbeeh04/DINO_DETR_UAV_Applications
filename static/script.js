const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const yoloImg = document.getElementById("yolo-img");
const dinoImg = document.getElementById("dino-img");

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus("pick an image first", "error");
    return;
  }

  const fd = new FormData();
  fd.append("image", file);

  runBtn.disabled = true;
  setStatus("running...", "");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    const bust = `?t=${Date.now()}`;
    yoloImg.src = data.yolo_url + bust;
    dinoImg.src = data.dino_url + bust;
    setStatus("done", "done");
  } catch (err) {
    setStatus(`error: ${err.message}`, "error");
  } finally {
    runBtn.disabled = false;
  }
});
