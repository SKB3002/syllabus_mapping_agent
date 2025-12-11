// script.js - fully working version with FastAPI backend integration
document.addEventListener("DOMContentLoaded", () => {

  // DOM refs
  const boardA = document.getElementById("boardA");
  const boardB = document.getElementById("boardB");
  const fileA = document.getElementById("fileA");
  const fileB = document.getElementById("fileB");
  const runBtn = document.getElementById("runBtn");
  const loaderRow = document.getElementById("loaderRow");
  const progressWrapper = document.getElementById("progressWrapper");
  const progressFill = document.getElementById("progressFill");
  const progressPct = document.getElementById("progressPct");

  // old previewText stays but unused now — FIX: allow null safely
  const previewText = document.getElementById("previewText") || null;

  const resultPreview = document.getElementById("resultPreview");
  const downloadBtn = document.getElementById("downloadBtn");
  const stepFloat = document.getElementById("stepFloat");
  const fileANote = document.getElementById("fileA-note");
  const fileBNote = document.getElementById("fileB-note");

  // NEW — table preview container
  const previewTable = document.getElementById("previewTable");

  // ===============================
  // FIX-A: Force previewTable to scroll horizontally ONLY
  // ===============================
  previewTable.style.overflowX = "auto";
  previewTable.style.overflowY = "auto";
  previewTable.style.display = "block";
  previewTable.style.maxWidth = "100%";
  previewTable.style.position = "relative";

  // state vars
  window.__resultsReady = false;
  window.__running = false;

  function computeStep() {
    const aName = boardA.value.trim();
    const bName = boardB.value.trim();
    const aFile = fileA.files.length > 0;
    const bFile = fileB.files.length > 0;
    const resultsReady = window.__resultsReady;
    const running = window.__running;

    if (!aName) return 1;
    if (!aFile) return 2;
    if (!bName) return 3;
    if (!bFile) return 4;
    if (resultsReady) return 6;
    if (!running) return 5;

    return 5;
  }

  function updateStepUI() {
    const step = computeStep();

    document.body.classList.remove(
      ...Array.from(document.body.classList).filter(c => c.startsWith("step-"))
    );
    document.body.classList.add("step-" + step);

    const texts = {
      1: ["Step 1: Enter Board A Name", "Please provide the Board A display name."],
      2: ["Step 2: Upload Board A File", "Upload the CSV/XLSX for Board A."],
      3: ["Step 3: Enter Board B Name", "Provide the Board B display name."],
      4: ["Step 4: Upload Board B File", "Upload the CSV/XLSX for Board B."],
      5: ["Step 5: Run Comparison", "All set — press Run Comparison to start."],
      6: ["Step 6: Download Results", "Comparison done — download the CSV."]
    };

    stepFloat.querySelector(".step-label").textContent = texts[step][0];
    stepFloat.querySelector(".step-desc").textContent = texts[step][1];

    runBtn.disabled = !(step === 5 && !window.__running);

    if (step === 6) downloadBtn.classList.remove("hidden");
    else downloadBtn.classList.add("hidden");
  }

  [boardA, boardB].forEach(el => {
    el.addEventListener("input", () => {
      window.__resultsReady = false;
      updateStepUI();
    });
  });

  fileA.addEventListener("change", () => {
    fileANote.textContent = fileA.files.length > 0 ? fileA.files[0].name : "";
    window.__resultsReady = false;
    updateStepUI();
  });

  fileB.addEventListener("change", () => {
    fileBNote.textContent = fileB.files.length > 0 ? fileB.files[0].name : "";
    window.__resultsReady = false;
    updateStepUI();
  });

  async function runRealComparison() {

    const aName = boardA.value.trim();
    const bName = boardB.value.trim();
    const aFileObj = fileA.files[0];
    const bFileObj = fileB.files[0];

    if (!aName || !bName || !aFileObj || !bFileObj) {
      alert("Please supply both board names and both files before running.");
      return;
    }

    window.__running = true;
    updateStepUI();

    loaderRow.classList.remove("hidden");
    progressWrapper.classList.remove("hidden");

    // ===============================
    // FIX-C: Force progress bar to render
    // ===============================
    progressWrapper.style.display = "flex";
    progressFill.style.transition = "width 0.35s ease";

    progressFill.style.width = "5%";
    progressPct.textContent = "5%";

    setTimeout(() => {
      progressFill.style.width = "15%";
      progressPct.textContent = "15%";
    }, 300);

    const formData = new FormData();
    formData.append("boardA", aName);
    formData.append("boardB", bName);
    formData.append("fileA", aFileObj);
    formData.append("fileB", bFileObj);

    try {
      const res = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData
      });

      progressFill.style.width = "60%";
      progressPct.textContent = "60%";

      const data = await res.json();

      progressFill.style.width = "100%";
      progressPct.textContent = "100%";

      loaderRow.classList.add("hidden");
      progressWrapper.classList.add("hidden");

      if (previewText !== null) {
        previewText.style.display = "none";
      }

      previewTable.innerHTML = "";
      const previewRows = data.preview;

      if (previewRows.length > 0) {
        const table = document.createElement("table");

        // ===============================
        // FIX-B: Table behaves like scrollable content
        // ===============================
        table.style.display = "inline-block";
        table.style.width = "max-content";
        table.style.tableLayout = "auto";

        const headerRow = document.createElement("tr");
        Object.keys(previewRows[0]).forEach(key => {
          const th = document.createElement("th");
          th.textContent = key;
          headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        previewRows.forEach(row => {
          const tr = document.createElement("tr");
          Object.values(row).forEach(val => {
            const td = document.createElement("td");
            td.textContent = val;
            tr.appendChild(td);
          });
          table.appendChild(tr);
        });

        previewTable.appendChild(table);
        previewTable.classList.remove("hidden");
      }

      resultPreview.classList.remove("hidden");

      downloadBtn.href = "data:text/csv;base64," + data.csv_b64;
      downloadBtn.classList.remove("hidden");

      window.__resultsReady = true;

    } catch (err) {
      alert("Backend error: " + err);
    }

    window.__running = false;
    updateStepUI();
  }

  runBtn.addEventListener("click", () => {
    runRealComparison();
  });

  updateStepUI();
});
