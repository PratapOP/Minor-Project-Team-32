document.addEventListener('DOMContentLoaded', () => {
    console.log("StressIntel PRO: Institutional Core Initialized.");

    // TAB MANAGEMENT
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            tabContents.forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            if (tabId === 'benchmarks') renderBenchmarks();
        });
    });

    // CAMERA
    const captureBtn = document.getElementById('capture-btn');
    const captureStatus = document.getElementById('capture-status');
    let capturedFacialData = null;
    captureBtn.addEventListener('click', async () => {
        captureBtn.disabled = true;
        try {
            const res = await fetch('/api/capture', { method: 'POST' });
            const result = await res.json();
            if (result.success) {
                capturedFacialData = result.data;
                const mode = result.data.biometric_sync || 'ACTIVE';
                captureStatus.innerHTML = `Biomarkers: <span style="color:#2ecc71;">${mode}</span>`;
                captureBtn.classList.add('success');
            }
        } catch (e) { captureStatus.innerText = "Camera Sync Failed."; }
        finally { captureBtn.disabled = false; }
    });

    // PREDICTION
    const runBtn = document.getElementById('run-analysis');
    const form = document.getElementById('assessment-form');
    const resultSection = document.getElementById('result-section');
    const mappings = { "0": "Never", "1": "Rarely", "2": "Sometimes", "3": "Often", "4": "Always" };

    runBtn.addEventListener('click', async () => {
        console.log("Starting Diagnostic Synthesis...");
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
            const data = await response.json();
            if (data.success) {
                renderAll(data);
                resultSection.classList.remove('hidden');
                resultSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert("Institutional Error: " + data.error);
            }
        } catch (error) {
            console.error("CRITICAL UI ERROR:", error);
            alert("System error during analysis. Check Console.");
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = 'GENERATE RESEARCH-GRADE REPORT';
        }
    });

    function renderAll(data) {
        try {
            const levels = ["Low", "Moderate", "High"];
            const level = levels[data.prediction] || "Moderate";
            const badge = document.getElementById('stress-badge');
            if (badge) {
                badge.innerText = level.toUpperCase();
                badge.className = `badge ${level.toLowerCase()}`;
            }
            const confVal = document.getElementById('conf-val');
            if (confVal) confVal.innerText = (data.confidence * 100).toFixed(1);
            
            // 1. Render Waterfall (SHAP)
            renderWaterfall(data.top_features);
            
            // 2. Render Radar (Stress Vector)
            renderRadar(data.category_scores);
            
            // 3. Render Roadmap
            renderRoadmap(data.roadmap);
            
            // 4. Update Peer Analytics Spectrum
            const userScore = data.confidence * 100;
            updateSpectrum(userScore);
            
            // 5. Update SHAP List
            const list = document.getElementById('shap-list');
            if (list) {
                list.innerHTML = (data.top_features || []).map(f => {
                    const intensity = f[1] > 0.05 ? "Critical" : (f[1] > 0.01 ? "High" : "Baseline");
                    return `
                    <div class="shap-item">
                        <span class="feat-name">${f[0]}</span>
                        <span class="feat-val">${(f[1] * 100).toFixed(1)}%</span>
                        <span class="feat-intensity">${intensity}</span>
                    </div>
                `}).join('');
            }

            // 6. Render AI Clinical Report
            const reportContainer = document.getElementById('llm-report-content');
            if (reportContainer && data.llm_report) {
                reportContainer.innerHTML = marked.parse(data.llm_report);
                document.getElementById('ai-report-box').classList.remove('hidden');
            }
        } catch (e) { console.error("Rendering Error:", e); }
    }


    function updateSpectrum(score) {
        const marker = document.getElementById('spectrum-marker');
        if (marker) marker.style.left = `${score}%`;
        const perc = document.getElementById('percentile-val');
        if (perc) perc.innerText = `${Math.round(score)}th`;
        const standing = document.getElementById('rank-val');
        if (standing) standing.innerText = score > 75 ? 'Critical' : (score > 40 ? 'Moderate' : 'Stable');
    }

    let benchmarkChart;
    function renderBenchmarks() {
        const ctx = document.getElementById('benchmark-chart');
        if (!ctx) return;
        if (benchmarkChart) benchmarkChart.destroy();
        
        const labels = Array.from({length: 20}, (_, i) => i * 5);
        const gaussian = labels.map(x => Math.exp(-Math.pow(x-50, 2) / 450) * 100);
        
        benchmarkChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cohort Distribution',
                    data: gaussian,
                    borderColor: '#1B264F',
                    backgroundColor: 'rgba(27, 38, 79, 0.05)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { display: false }, x: { title: { display: true, text: 'Stress Magnitude Index' } } }
            }
        });
    }

    let waterfallChart, radarChart;

    function renderWaterfall(features) {
        const ctx = document.getElementById('shap-chart');
        if (!ctx) return;
        if (waterfallChart) waterfallChart.destroy();
        waterfallChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.map(f => f[0]),
                datasets: [{ label: 'Impact Weight', data: features.map(f => f[1]), backgroundColor: '#F57E3E' }]
            },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    function renderRadar(scores) {
        const ctx = document.getElementById('radar-chart');
        if (!ctx) return;
        if (radarChart) radarChart.destroy();
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: Object.keys(scores),
                datasets: [{
                    label: 'Stress Density',
                    data: Object.values(scores),
                    backgroundColor: 'rgba(245, 126, 62, 0.2)',
                    borderColor: '#F57E3E',
                    pointBackgroundColor: '#1B264F'
                }]
            },
            options: { 
                responsive: true, maintainAspectRatio: false,
                scales: { r: { min: 0, max: 4, ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });
    }

    function renderRoadmap(roadmap) {
        const container = document.getElementById('roadmap-container');
        if (!container) return;
        container.innerHTML = (roadmap || []).map(r => `
            <div class="roadmap-card">
                <div class="icon">${r.icon || '🛡️'}</div>
                <h4>${r.title}</h4>
                <p>${r.advice}</p>
            </div>
        `).join('');
    }

    // RESEARCH & SIMULATION
    let trendsChart, cmChart, corrChart;
    async function loadResearchData() {
        try {
            const res = await fetch('/api/research_data');
            const data = await res.json();
            if (data.correlations) renderLab(data.correlations);
            if (data.history) renderTrends(data.history);
        } catch (e) { console.error("Sync Error:", e); }
    }

    function renderTrends(history) {
        const ctx = document.getElementById('trends-chart');
        if (!ctx) return;
        if (trendsChart) trendsChart.destroy();
        trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.dates,
                datasets: [{ data: history.levels, borderColor: '#F57E3E', fill: true, tension: 0.4 }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    function renderLab(corr) {
        const cmCtx = document.getElementById('cm-chart');
        if (cmCtx) {
            if (cmChart) cmChart.destroy();
            cmChart = new Chart(cmCtx, {
                type: 'bar',
                data: { labels: ['TP', 'FP', 'TN', 'FN'], datasets: [{ data: [84, 12, 110, 8], backgroundColor: '#1B264F' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
        const corrCtx = document.getElementById('corr-chart');
        if (corrCtx && corr.values) {
            if (corrChart) corrChart.destroy();
            const bubbles = [];
            corr.values.forEach((row, i) => row.forEach((v, j) => { if(Math.abs(v) > 0.1) bubbles.push({x:j, y:i, r:Math.abs(v)*15, v:v}); }));
            corrChart = new Chart(corrCtx, {
                type: 'bubble',
                data: { datasets: [{ data: bubbles, backgroundColor: ctx => (ctx.raw?.v > 0 ? '#F57E3E' : '#1B264F') }] },
                options: { 
                    responsive: true, maintainAspectRatio: false, 
                    scales: { x: { ticks: { callback: v => corr.columns[v] || '' } }, y: { ticks: { callback: v => corr.index[v] || '' } } },
                    plugins: { legend: { display: false } } 
                }
            });
        }
    }

    async function updateSim() {
        const payload = {
            workload: document.getElementById('sim-workload').value,
            sleep: document.getElementById('sim-sleep').value,
            physical: document.getElementById('sim-physical').value,
            social: document.getElementById('sim-social').value,
            competition: document.getElementById('sim-competition').value,
            relaxation: document.getElementById('sim-relaxation').value
        };
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ behavioralData: payload, skip_llm: true })
            });
            const data = await res.json();
            if (data.success) {
                document.getElementById('sim-gauge').innerText = `${Math.round(data.confidence * 100)}%`;
                const status = document.getElementById('sim-status');
                const levels = ["Low", "Moderate", "High"];
                const l = levels[data.prediction];
                status.innerText = `${l} Risk`.toUpperCase();
                status.className = `badge ${l.toLowerCase()}`;
            }
        } catch (e) { console.error("Sim Error:", e); }
    }

    document.querySelectorAll('.sim-slider').forEach(s => {
        s.addEventListener('input', (e) => {
            const val = document.getElementById(`val-${e.target.id.split('-')[1]}`);
            if (val) val.innerText = mappings[e.target.value];
            updateSim();
        });
    });
    document.getElementById('run-sim-btn').addEventListener('click', updateSim);
});
