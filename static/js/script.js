document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------
    // TAB MANAGEMENT
    // ---------------------------
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            // Update active nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Update active tab
            tabContents.forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');

            // Load data if switching to trends or lab
            if (tabId === 'trends' || tabId === 'lab') {
                loadResearchData();
            }
        });
    });

    // ---------------------------
    // CAMERA LOGIC
    // ---------------------------
    const captureBtn = document.getElementById('capture-btn');
    const captureStatus = document.getElementById('capture-status');
    let capturedFacialData = null;

    captureBtn.addEventListener('click', async () => {
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> SCANNING...';
        captureStatus.innerText = "Analyzing facial geometry and micro-expressions...";

        try {
            const response = await fetch('/api/capture', { method: 'POST' });
            const result = await response.json();

            if (result.success) {
                capturedFacialData = result.data;
                captureStatus.innerHTML = `Biomarkers Synced: <span style="color:#2ecc71; font-weight:700;">ACTIVE</span> (Activity: ${result.data.eye_ratio.toFixed(2)})`;
                captureBtn.classList.add('success');
            } else {
                captureStatus.innerText = "Error: Webcam not found.";
            }
        } catch (error) {
            captureStatus.innerText = "Error connecting to Camera module.";
        } finally {
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<i class="fas fa-camera"></i> CAPTURE SUCCESS';
        }
    });

    // ---------------------------
    // PREDICTION LOGIC
    // ---------------------------
    const runBtn = document.getElementById('run-analysis');
    const form = document.getElementById('assessment-form');
    const resultSection = document.getElementById('result-section');

    let shapChart = null;

    runBtn.addEventListener('click', async () => {
        const formData = new FormData(form);
        const behavioralData = Object.fromEntries(formData.entries());
        const journalEntry = document.getElementById('journal-entry').value;

        // Merge captured facial data if available
        if (capturedFacialData) {
            behavioralData['eye_ratio'] = capturedFacialData.eye_ratio;
            behavioralData['mouth_ratio'] = capturedFacialData.mouth_ratio;
        }

        runBtn.disabled = true;
        runBtn.innerHTML = '<i class="fas fa-microchip fa-spin"></i> SYNTHESIZING INTELLIGENCE...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userName: "Researcher",
                    journalEntry: journalEntry,
                    behavioralData: behavioralData
                })
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
                saveToHistory(data); // Save for Trends tab
                resultSection.classList.remove('hidden');
                resultSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert("Error: " + data.error);
            }
        } catch (error) {
            console.error("Prediction failed:", error);
            alert("System error during analysis.");
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = 'GENERATE RESEARCH-GRADE REPORT';
        }
    });

    function displayResults(data) {
        const levels = ["Low", "Moderate", "High"];
        const level = levels[data.prediction];
        
        const badge = document.getElementById('stress-badge');
        badge.innerText = level;
        badge.className = `badge ${level.toLowerCase()}`;
        
        document.getElementById('conf-val').innerText = (data.confidence * 100).toFixed(1);
        
        // Use Marked.js for the report
        document.getElementById('llm-report').innerHTML = marked.parse(data.llm_output);

        // Render Waterfall Chart
        renderWaterfall(data.top_features);

        // Render Radar Chart (Stress Vector)
        renderRadar(data.category_scores);

        // Render Roadmap
        renderRoadmap(data.roadmap);

        // Update Top Priority Action Preview
        if (data.roadmap && data.roadmap.length > 0) {
            const topRecBox = document.getElementById('top-recommendation-box');
            const topRecText = document.getElementById('top-rec-text');
            topRecBox.classList.remove('hidden');
            topRecText.innerText = `${data.roadmap[0].title}: ${data.roadmap[0].advice}`;
        }
    }

    let radarChart = null;
    function renderRadar(scores) {
        const ctx = document.getElementById('radar-chart').getContext('2d');
        if (radarChart) radarChart.destroy();

        const labels = Object.keys(scores);
        const values = Object.values(scores);

        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Stress Vector Magnitude',
                    data: values,
                    backgroundColor: 'rgba(245, 126, 62, 0.4)',
                    borderColor: '#F57E3E',
                    pointBackgroundColor: '#1B264F',
                    borderWidth: 2
                }]
            },
            options: {
                plugins: {
                    title: { display: true, text: 'Stress Vector Analysis', font: { size: 14, family: 'Outfit' } }
                },
                scales: {
                    r: {
                        angleLines: { display: true },
                        suggestedMin: 0,
                        suggestedMax: 4
                    }
                }
            }
        });
    }

    function renderWaterfall(features) {
        const ctx = document.getElementById('shap-chart').getContext('2d');
        
        if (shapChart) shapChart.destroy();

        const labels = features.map(f => f[0]);
        const values = features.map(f => f[1]);

        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Influence',
                    data: values,
                    backgroundColor: values.map(v => v > 0 ? '#F57E3E' : '#1B264F'),
                    borderRadius: 8
                }]
            },
            options: {
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Factor Contribution (XAI)', font: { size: 14, family: 'Outfit' } }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { display: false } }
                }
            }
        });
    }

    function renderRoadmap(roadmap) {
        const container = document.getElementById('roadmap-container');
        container.innerHTML = '';

        roadmap.forEach(item => {
            const card = document.createElement('div');
            card.className = 'roadmap-card';
            card.innerHTML = `
                <div class="icon">${item.icon}</div>
                <h4>${item.title}</h4>
                <p>${item.advice}</p>
            `;
            container.appendChild(card);
        });
    }

    function saveToHistory(data) {
        const history = JSON.parse(localStorage.getItem('stress_history') || '[]');
        history.push({
            date: new Date().toLocaleDateString(),
            score: (data.prediction + 1) * 33,
            categories: data.category_scores, // High-detail history
            confidence: data.confidence
        });
        if (history.length > 10) history.shift();
        localStorage.setItem('stress_history', JSON.stringify(history));
    }

    // ---------------------------
    // RESEARCH DATA LOADING
    // ---------------------------
    let trendsChart = null;
    let cmChart = null;
    let corrChart = null;

    // ---------------------------
    // SIMULATION LAB LOGIC
    // ---------------------------
    const simSliders = document.querySelectorAll('.sim-slider');
    const simGauge = document.getElementById('sim-gauge');
    const simStatus = document.getElementById('sim-status');
    const mappings = ["Never", "Rarely", "Sometimes", "Often", "Always"];

    simSliders.forEach(slider => {
        slider.addEventListener('input', () => {
            const valId = 'val-' + slider.id.split('-')[1];
            document.getElementById(valId).innerText = mappings[slider.value];
            // We can still update real-time or wait for button
        });
    });

    const runSimBtn = document.getElementById('run-sim-btn');
    if (runSimBtn) {
        runSimBtn.addEventListener('click', updateSimulation);
    }

    async function updateSimulation() {
        const workload = mappings[document.getElementById('sim-workload').value];
        const sleep = mappings[document.getElementById('sim-sleep').value];
        const physical = mappings[document.getElementById('sim-physical').value];

        // Send a mini-batch for real-time inference
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userName: "Simulation",
                    behavioralData: {
                        "Do you feel overwhelmed with your academic workload?": workload,
                        "Do you face any sleep problems or difficulties falling asleep?": sleep,
                        "Have you been getting headaches more often than usual?": physical
                    }
                })
            });
            const data = await response.json();
            
            const score = (data.confidence * 100).toFixed(0);
            simGauge.innerText = score + '%';
            
            const levels = ["Low", "Moderate", "High"];
            simStatus.innerText = levels[data.prediction].toUpperCase();
            simStatus.className = `badge ${levels[data.prediction].toLowerCase()}`;
            
        } catch (e) { console.error("Sim error", e); }
    }

    async function loadResearchData() {
        try {
            const response = await fetch('/api/research_data');
            const data = await response.json();
            renderLabCharts(data.correlations);
        } catch (error) {
            console.error("Failed to load research data:", error);
        }
    }

    function renderTrends(history) {
        const ctx = document.getElementById('trends-chart').getContext('2d');
        if (trendsChart) trendsChart.destroy();

        // If we have categories, let's show a better chart (Stacked Bar)
        const datasets = [{
            label: 'Total Stress Index',
            data: history.levels,
            borderColor: '#F57E3E',
            backgroundColor: 'rgba(245, 126, 62, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: history.peaks.map(p => p ? 6 : 0),
            pointBackgroundColor: '#1B264F'
        }];

        trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.dates,
                datasets: datasets
            },
            options: {
                plugins: { 
                    legend: { display: true },
                    title: { display: true, text: 'Predictive Stress Trajectory', font: { family: 'Outfit', size: 16 } }
                },
                scales: {
                    y: { min: 0, max: 100, title: { display: true, text: 'Stress Magnitude (%)' } }
                }
            }
        });
    }

    function renderLabCharts(corr) {
        // Simple Confusion Matrix Bar Chart (for demo)
        const cmCtx = document.getElementById('cm-chart').getContext('2d');
        if (cmChart) cmChart.destroy();
        cmChart = new Chart(cmCtx, {
            type: 'bar',
            data: {
                labels: ['TP', 'FP', 'TN', 'FN'],
                datasets: [{
                    data: [84, 12, 110, 8],
                    backgroundColor: '#1B264F'
                }]
            },
            options: { plugins: { legend: { display: false } } }
        });

        // Heatmap simulation for correlations
        const corrCtx = document.getElementById('corr-chart').getContext('2d');
        if (corrChart) corrChart.destroy();
        corrChart = new Chart(corrCtx, {
            type: 'bubble',
            data: {
                datasets: corr.values.flatMap((row, i) => 
                    row.map((val, j) => ({
                        x: j,
                        y: i,
                        r: Math.abs(val) * 20,
                        v: val
                    }))
                ).map(point => ({
                    ...point,
                    backgroundColor: point.v > 0 ? '#F57E3E' : '#C1E6F1'
                }))
            },
            options: {
                scales: {
                    x: { ticks: { callback: (v) => corr.columns[v] } },
                    y: { ticks: { callback: (v) => corr.index[v] } }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Correlation: ${ctx.raw.v.toFixed(2)}`
                        }
                    }
                }
            }
        });
    }
});
