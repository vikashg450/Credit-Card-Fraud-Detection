document.addEventListener('DOMContentLoaded', () => {
    // Navigation Tabs Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Load Metrics on Startup
    loadMetrics();

    // Random Sample Loading
    const btnLoadLegit = document.getElementById('btn-load-legit');
    const btnLoadFraud = document.getElementById('btn-load-fraud');

    if (btnLoadLegit) {
        btnLoadLegit.addEventListener('click', () => loadRandomSample(0));
    }
    if (btnLoadFraud) {
        btnLoadFraud.addEventListener('click', () => loadRandomSample(1));
    }

    // Prediction Form Submission
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', handlePrediction);
    }
});

// Global metrics caching
let modelMetrics = {};

// Fetch and load metrics from backend
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Failed to fetch metrics');
        
        const data = await response.json();
        modelMetrics = data;
        
        // Populate Stats cards on Dashboard
        document.getElementById('lr-pr-auc').innerText = data["Logistic Regression"]["pr_auc"].toFixed(4);
        document.getElementById('xgb-pr-auc').innerText = data["XGBoost"]["pr_auc"].toFixed(4);
        
        // Populate Metrics Comparison Table
        populateMetricsTable(data);
        
        // Populate Feature Importances on Dashboard
        populateGlobalFeatureImportances(data["XGBoost"]["feature_importances"]);
        
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Populate the performance table
function populateMetricsTable(data) {
    const tbody = document.getElementById('metrics-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    const models = ["Logistic Regression", "XGBoost"];
    models.forEach(modelName => {
        const m = data[modelName];
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td style="font-weight: 700;">${modelName}</td>
            <td>${(m.accuracy * 100).toFixed(2)}%</td>
            <td class="metric-highlight">${(m.precision * 100).toFixed(2)}%</td>
            <td class="metric-highlight">${(m.recall * 100).toFixed(2)}%</td>
            <td>${m.f1_score.toFixed(4)}</td>
            <td style="font-weight: 700; color: var(--secondary);">${m.pr_auc.toFixed(4)}</td>
            <td>${m.roc_auc.toFixed(4)}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Populate Global Feature Importances list
function populateGlobalFeatureImportances(importances) {
    const container = document.getElementById('global-importances-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Get top 6 features
    const topFeatures = Object.entries(importances).slice(0, 6);
    
    topFeatures.forEach(([feature, value]) => {
        const percentage = (value * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'impact-row';
        row.innerHTML = `
            <div class="impact-feat-name" style="width: 80px;">${feature}</div>
            <div class="impact-bar-track">
                <div class="impact-bar-fill pos" style="width: ${percentage}%"></div>
            </div>
            <div class="impact-value" style="width: 60px;">${percentage}%</div>
        `;
        container.appendChild(row);
    });
}

// Load a random transaction sample from the backend
async function loadRandomSample(classType) {
    try {
        const statusText = document.getElementById('sample-status');
        if (statusText) statusText.innerText = "Loading sample...";
        
        const response = await fetch(`/api/random-sample?class_type=${classType}`);
        if (!response.ok) throw new Error('Failed to load sample');
        
        const sample = await response.json();
        
        // Populate Form Fields
        document.getElementById('input-time').value = sample.Time.toFixed(0);
        document.getElementById('input-amount').value = sample.Amount.toFixed(2);
        
        for (let i = 1; i <= 28; i++) {
            const input = document.getElementById(`input-v${i}`);
            if (input) {
                input.value = sample[`V${i}`].toFixed(5);
            }
        }
        
        if (statusText) {
            statusText.innerText = `Loaded random ${classType === 1 ? 'Fraudulent' : 'Legitimate'} transaction!`;
            statusText.style.color = classType === 1 ? 'var(--danger)' : 'var(--success)';
            
            // Clear status after 3s
            setTimeout(() => {
                statusText.innerText = "";
            }, 4000);
        }
        
        // Reset Result Box
        resetResultPanel();
        
    } catch (error) {
        console.error('Error loading sample:', error);
        const statusText = document.getElementById('sample-status');
        if (statusText) {
            statusText.innerText = "Error loading transaction sample.";
            statusText.style.color = 'var(--danger)';
        }
    }
}

// Reset prediction result display
function resetResultPanel() {
    const resultDiv = document.getElementById('prediction-result-area');
    if (!resultDiv) return;
    
    resultDiv.innerHTML = `
        <div class="result-placeholder">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.3;">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            <p>Fill out the parameters or load a sample, select a model, and click "Analyze Transaction" to predict fraud.</p>
        </div>
    `;
}

// Handle Prediction Form Submission
async function handlePrediction(event) {
    event.preventDefault();
    
    const resultDiv = document.getElementById('prediction-result-area');
    if (!resultDiv) return;
    
    // Show Loading
    resultDiv.innerHTML = `
        <div class="result-placeholder">
            <div style="width: 50px; height: 50px; border: 4px solid var(--primary-glow); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p>Analyzing transaction features against model...</p>
        </div>
    `;
    
    // Get Form Data
    const formData = {};
    formData.Time = parseFloat(document.getElementById('input-time').value);
    formData.Amount = parseFloat(document.getElementById('input-amount').value);
    
    for (let i = 1; i <= 28; i++) {
        formData[`V${i}`] = parseFloat(document.getElementById(`input-v${i}`).value);
    }
    
    const selectedModel = document.getElementById('model-select').value;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: selectedModel,
                features: formData
            })
        });
        
        if (!response.ok) throw new Error('Prediction API failed');
        
        const result = await response.json();
        renderPredictionResult(result, resultDiv);
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultDiv.innerHTML = `
            <div class="result-placeholder" style="color: var(--danger);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <p>Error calculating prediction. Check backend logs.</p>
            </div>
        `;
    }
}

// Render prediction result with gauge and explanation
function renderPredictionResult(result, container) {
    const isFraud = result.prediction === 1;
    const probPercent = (result.probability * 100).toFixed(1);
    
    // Circle math
    const radius = 70;
    const circumference = 2 * Math.PI * radius; // 439.8
    const offset = circumference - (result.probability * circumference);
    
    const strokeColor = isFraud ? 'var(--danger)' : 'var(--success)';
    const badgeClass = isFraud ? 'danger' : 'safe';
    const badgeText = isFraud ? 'Critical Alert: FRAUD' : 'Verified: LEGITIMATE';
    
    container.innerHTML = `
        <div class="results-panel">
            <div class="gauge-wrapper">
                <svg class="gauge-svg" viewBox="0 0 160 160">
                    <circle class="gauge-bg" cx="80" cy="80" r="70"></circle>
                    <circle class="gauge-fill" id="gauge-fill-anim" cx="80" cy="80" r="70" 
                            style="stroke: ${strokeColor}; stroke-dasharray: ${circumference}; stroke-dashoffset: ${circumference};"></circle>
                </svg>
                <div class="gauge-value" style="color: ${strokeColor};">${probPercent}%</div>
            </div>
            
            <div class="result-badge ${badgeClass}">${badgeText}</div>
            
            <div class="impact-bar-container">
                <div class="impact-bar-title">Key Feature Contributions</div>
                <div id="local-contributions-list"></div>
                <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 1rem; text-align: left;">
                    * Bars represent relative directional impact of features on this prediction. 
                    <span style="color: var(--danger);">Red</span> drives towards Fraud, 
                    <span style="color: var(--success);">Green</span> drives towards Legit.
                </p>
            </div>
        </div>
    `;
    
    // Trigger gauge stroke fill animation
    setTimeout(() => {
        const fill = document.getElementById('gauge-fill-anim');
        if (fill) {
            fill.style.strokeDashoffset = offset;
        }
    }, 100);
    
    // Render local contributions
    const contribsContainer = document.getElementById('local-contributions-list');
    if (contribsContainer) {
        contribsContainer.innerHTML = '';
        
        // Sort contributions by absolute value to show top 5 impacts
        const sortedContribs = Object.entries(result.explanation)
            .map(([feat, val]) => ({ name: feat, val: val }))
            .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
            .slice(0, 5);
            
        // Find max absolute value for normalization
        const maxVal = Math.max(...sortedContribs.map(c => Math.abs(c.val)), 0.001);
        
        sortedContribs.forEach(c => {
            const barWidth = Math.min(100, (Math.abs(c.val) / maxVal) * 100);
            const isPositive = c.val > 0; // positive drives fraud
            const colorClass = isPositive ? 'pos' : 'neg';
            const valueSign = isPositive ? '+' : '-';
            
            const row = document.createElement('div');
            row.className = 'impact-row';
            row.innerHTML = `
                <div class="impact-feat-name">${c.name}</div>
                <div class="impact-bar-track">
                    <div class="impact-bar-fill ${colorClass}" style="width: 0%;"></div>
                </div>
                <div class="impact-value" style="color: ${isPositive ? 'var(--danger)' : 'var(--success)'};">${valueSign}${Math.abs(c.val).toFixed(2)}</div>
            `;
            contribsContainer.appendChild(row);
            
            // Animate bar width
            setTimeout(() => {
                const fillBar = row.querySelector('.impact-bar-fill');
                if (fillBar) fillBar.style.width = `${barWidth}%`;
            }, 150);
        });
    }
}
