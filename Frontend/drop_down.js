/* ================= CORE ELEMENTS ================= */
const multiSelect = document.getElementById("multiSelect");
const dropdown = document.getElementById("dropdown");
const selectedItems = document.getElementById("selectedItems");
const searchInput = document.getElementById("sym-search");
const addBtn = document.getElementById("add-btn");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const API_BASE = "http://127.0.0.1:8000";
const selectedVals = new Set();

/* =============== INITIALISE MASTER LIST =============== */
fetch(`${API_BASE}/symptoms`)
    .then((r) => r.json())
    .then((js) => {
        const dl = document.getElementById("sym-list");
        js.symptoms.sort().forEach((sym) => {
            const opt = document.createElement("option");
            opt.value = sym;
            dl.appendChild(opt);
        });
    });

/* ===== MULTI-SELECT TOGGLE ===== */
multiSelect.addEventListener("click", (e) => {
    if (e.target === searchInput) return;
    multiSelect.classList.toggle("active");
    searchInput.focus();
});

/* ===== ADD SYMPTOM BUTTON ===== */
addBtn.addEventListener("click", addSymptom);

function addSymptom() {
    const val = searchInput.value.trim();
    if (!val || selectedVals.has(val)) return;
    selectedVals.add(val);

    // remove placeholder
    const ph = selectedItems.querySelector(".placeholder");
    if (ph) ph.remove();

    const tag = document.createElement("span");
    tag.className = "pill";
    tag.textContent = val;
    tag.dataset.remove = val;
    selectedItems.appendChild(tag);
    searchInput.value = "";
}

/* ===== REMOVE TAG ON CLICK ===== */
selectedItems.addEventListener("click", (e) => {
    const val = e.target.dataset.remove;
    if (!val) return;
    selectedVals.delete(val);
    e.target.remove();

    if (selectedVals.size === 0) {
        const ph = document.createElement("span");
        ph.className = "placeholder";
        ph.textContent = "No symptoms yet…";
        selectedItems.appendChild(ph);
    }
});

/* ===== CLEAR ALL ===== */
clearBtn.addEventListener("click", () => {
    selectedVals.clear();
    selectedItems.innerHTML = '<span class="placeholder">No symptoms yet…</span>';
    document.getElementById("result").innerHTML = "";
    document.getElementById("meds-heading").style.display = "none";
    document.getElementById("meds").innerHTML = "";
});

searchInput.addEventListener("input", () => {
    const term = searchInput.value.toLowerCase();
    const opts = dropdown.querySelectorAll(".dropdown-option");
    opts.forEach((o) => {
        o.style.display = o.textContent.toLowerCase().includes(term) ?
            "block" :
            "none";
    });
});


document.addEventListener("click", (e) => {
    if (!multiSelect.contains(e.target)) {
        multiSelect.classList.remove("active");
    }
});

/* ===== PREDICT DISEASE BUTTON ===== */
predictBtn.addEventListener("click", () => {
    if (selectedVals.size < 4) {
        alert("Select at least 4 symptoms first.");
        return;
    }


    fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                symptoms: [...selectedVals]
            }),
        })
        .then((r) => r.json())
        .then((js) => {
            const disease = js.disease ?
                js.disease :
                js.replace("Predicted disease: ", "");
            document.getElementById("result").textContent = `Disease: ${disease}`;
            document.getElementById("result").classList.remove("hidden");
            // fetch medicines
            return fetch(
                `${API_BASE}/medicines/${encodeURIComponent(disease)}?limit=3`
            );
        })
        .then((r) => r.json())
        .then((js) => {
            document.getElementById("meds-heading").style.display = "block";
            const ul = document.getElementById("meds");
            ul.innerHTML = "";
            js.medicines.forEach((m) => {
                const li = document.createElement("li");
                li.textContent = m;
                ul.appendChild(li);
            });
        })
        .catch((err) => alert("Error: " + err));
});
