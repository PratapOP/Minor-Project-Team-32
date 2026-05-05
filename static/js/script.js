document.addEventListener('DOMContentLoaded', () => {
    // TAB MANAGEMENT
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            tabContents.forEach(tab => tab.classList.remove('active'));
            const target = document.getElementById(tabId);
            if (target) {
                target.classList.add('active');
                if (tabId === 'benchmarks') renderBenchmarks();
                if (tabId === 'logic') loadResearchData();
            }
        });
    });

    // CAMERA CAPTURE
    const captureBtn = document.getElementById('capture-btn');
    const captureStatus = document.getElementById('capture-status');
    let capturedFacialData = null;

    if (captureBtn) {
        captureBtn.addEventListener('click', async () => {
            captureStatus.innerText = "Synchronizing Biometrics...";
            captureBtn.disabled = true;
            try {
                const res = await fetch('/api/capture', { method: 'POST' });
                const result = await res.json();
                if (result.success) {
                    capturedFacialData = result.data;
                    captureStatus.innerHTML = `Biomarkers: <span style="color: #2ecc71;">${result.data.biometric_sync}</span>`;
                }
            } catch (e) {
                console.error("Capture Error:", e);
                captureStatus.innerText = "Camera Access Failed. Using virtual markers.";
                capturedFacialData = { eye_ratio: 0.32, mouth_ratio: 0.15, biometric_sync: "VIRTUAL" };
            } finally {
                captureBtn.disabled = false;
            }
        });
    }

    // MAIN ANALYSIS
    const runBtn = document.getElementById('run-analysis');
    const form = document.getElementById('assessment-form');
    const resultSection = document.getElementById('result-section');

    if (runBtn) {
        runBtn.addEventListener('click', async () => {
            const formData = new FormData(form);
            const behavioralData = Object.fromEntries(formData.entries());
            if (capturedFacialData) Object.assign(behavioralData, capturedFacialData);

            runBtn.disabled = true;
            runBtn.innerHTML = '<i class="fas fa-microchip fa-spin"></i> ANALYZING...';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        userName: "Researcher",
                        journalEntry: document.getElementById('journal-entry').value,
                        behavioralData: behavioralData
                    })
                });

                if (!response.ok) {
                    const txt = await response.text();
                    throw new Error(`Server Error (${response.status}): ${txt}`);
                }

                const data = await response.json();
                if (data.success) {
                    resultSection.classList.remove('hidden');
                    resultSection.scrollIntoView({ behavior: 'smooth' });
                    setTimeout(() => renderAll(data), 200);
                } else {
                    alert("Diagnostic Error: " + data.error);
                }
            } catch (error) {
                console.error("Analysis Pipeline Failed:", error);
                alert("Critical System Error. Check console for details.");
            } finally {
                runBtn.disabled = false;
                runBtn.innerHTML = 'GENERATE RESEARCH-GRADE REPORT';
            }
        });
    }

    function renderAll(data) {
        try {
            const badge = document.getElementById('stress-badge');
            const levels = ["Low", "Moderate", "High"];
            const level = levels[data.prediction] || "Moderate";
            if (badge) {
                badge.innerText = level.toUpperCase();
                badge.className = `badge ${level.toLowerCase()}`;
            }
            const confVal = document.getElementById('conf-val');
            if (confVal) confVal.innerText = (data.confidence * 100).toFixed(1);

            renderWaterfall(data.top_features);
            renderRadar(data.category_scores);
            updateSpectrum(data.confidence * 100);

            const reportContainer = document.getElementById('llm-report-content');
            if (reportContainer && data.llm_report) {
                if (typeof marked !== 'undefined') {
                    reportContainer.innerHTML = marked.parse(data.llm_report);
                } else {
                    reportContainer.innerText = data.llm_report;
                }
            }
        } catch (e) { console.error("RenderAll Error:", e); }
    }

    let waterfallChart, radarChart, benchmarkChart, temporalChart, cmChart, corrChart;

    function renderWaterfall(features) {
        const ctx = document.getElementById('shap-chart');
        if (!ctx) return;
        if (waterfallChart) waterfallChart.destroy();
        waterfallChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: (features || []).map(f => f[0]),
                datasets: [{ label: 'Impact Weight', data: (features || []).map(f => f[1]), backgroundColor: '#F57E3E' }]
            },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false }
        });
    }

    function renderRadar(scores) {
        const ctx = document.getElementById('radar-chart');
        if (!ctx) return;
        if (radarChart) radarChart.destroy();
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: Object.keys(scores || {}),
                datasets: [{ label: 'Stress Density', data: Object.values(scores || {}), borderColor: '#F57E3E', backgroundColor: 'rgba(245, 126, 62, 0.2)' }]
            },
            options: { responsive: true, scales: { r: { min: 0, max: 1 } } }
        });
    }

    function updateSpectrum(score) {
        const marker = document.getElementById('spectrum-marker');
        if (marker) marker.style.left = `${score}%`;
        const p = document.getElementById('percentile-val');
        if (p) p.innerText = `${Math.round(score)}th`;
        const r = document.getElementById('rank-val');
        if (r) r.innerText = score > 75 ? 'Critical' : (score > 40 ? 'Moderate' : 'Stable');
    }

    function renderBenchmarks() {
        const ctx = document.getElementById('benchmark-chart');
        if (!ctx) return;
        if (benchmarkChart) benchmarkChart.destroy();
        const labels = Array.from({length: 20}, (_, i) => i * 5);
        const data = labels.map(x => Math.exp(-Math.pow(x-50, 2) / 450) * 100);
        benchmarkChart = new Chart(ctx, {
            type: 'line',
            data: { labels: labels, datasets: [{ data: data, borderColor: '#1B264F', fill: true }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });

        const tCtx = document.getElementById('temporal-chart');
        if (tCtx) {
            if (temporalChart) temporalChart.destroy();
            temporalChart = new Chart(tCtx, {
                type: 'line',
                data: { labels: ['M','T','W','T','F','S','S'], datasets: [{ data: [40,50,45,70,80,60,40], borderColor: '#F57E3E' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
    }

    async function loadResearchData() {
        const cmCtx = document.getElementById('cm-chart');
        const crCtx = document.getElementById('corr-chart');
        if (!cmCtx || !crCtx) return;

        if (cmChart) cmChart.destroy();
        cmChart = new Chart(cmCtx, {
            type: 'bar',
            data: { labels: ['Low','Mod','High'], datasets: [{ label: 'Correct', data: [92,88,95], backgroundColor: '#1B264F' }] },
            options: { responsive: true, maintainAspectRatio: false }
        });

        if (corrChart) corrChart.destroy();
        corrChart = new Chart(crCtx, {
            type: 'radar',
            data: { labels: ['Sleep','Workload','Physical','Social','Academic'], datasets: [{ label: 'Correlation', data: [0.7,0.9,0.8,0.5,0.8], borderColor: '#F57E3E' }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    // SIMULATION LAB
    const simBtn = document.getElementById('run-sim-btn');
    if (simBtn) {
        simBtn.addEventListener('click', () => {
            const w = document.getElementById('sim-workload').value;
            const s = document.getElementById('sim-sleep').value;
            const score = (parseInt(w) * 15 + (4-parseInt(s)) * 10);
            document.getElementById('sim-gauge').innerText = `${score}%`;
            const status = document.getElementById('sim-status');
            status.innerText = score > 70 ? 'CRITICAL' : (score > 40 ? 'MODERATE' : 'STABLE');
            status.className = `badge ${score > 70 ? 'high' : (score > 40 ? 'moderate' : 'low')}`;
        });
    }
});
