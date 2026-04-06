const API_URL = "https://resume-backend-hy56.onrender.com";

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const addResumeBtn = document.getElementById('add-resume-btn');
    const resumeInputsContainer = document.getElementById('resume-inputs');
    const screenBtn = document.getElementById('screen-btn');
    const candidatesList = document.getElementById('candidates-list');
    const resultsCount = document.getElementById('results-count');
    const retrainBtn = document.getElementById('retrain-btn');


    let distributionChart = null;

    function animateValue(id, start, end, duration) {
        const obj = document.getElementById(id);
        if (!obj) return;
        let startValueString = obj.innerHTML;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const val = Math.floor(progress * (end - start) + start);
            obj.innerHTML = id.includes('score') ? val + '%' : val;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Load initial model info
    fetchModelInfo();

    // Event Listeners
    addResumeBtn.addEventListener('click', addResumeInputRow);
    const fileInput = document.getElementById('resume-file-input');
    fileInput.addEventListener('change', handleFileUploads);
    screenBtn.addEventListener('click', handleScreening);
    retrainBtn.addEventListener('click', handleRetraining);

    // Drag and Drop Logic
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileUploads({ target: fileInput });
            }
        });
    }

    function addResumeInputRow() {
        const row = document.createElement('div');
        row.className = 'resume-input-row';
        row.innerHTML = `
            <input type="text" placeholder="Candidate Name" class="cand-id">
            <textarea placeholder="Paste resume text here..." class="cand-text" rows="2"></textarea>
        `;
        resumeInputsContainer.appendChild(row);
    }

    async function handleFileUploads(e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch(`${API_URL}/api/v1/extract-text`, {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const errorMsg = await res.json();
                    throw new Error(errorMsg.detail || 'Extract failed');
                }

                const data = await res.json();

                // Add new row with extracted text
                const row = document.createElement('div');
                row.className = 'resume-input-row';
                row.innerHTML = `
                    <input type="text" placeholder="Candidate Name" class="cand-id" value="${data.filename.split('.')[0]}">
                    <textarea placeholder="Paste resume text here..." class="cand-text" rows="2">${data.text}</textarea>
                `;
                resumeInputsContainer.appendChild(row);

            } catch (error) {
                alert(`Error processing ${file.name}: ` + error.message);
            }
        }

        // Clear input to allow uploading the same file again if needed
        e.target.value = "";
    }

    async function fetchModelInfo() {
        try {

            if (!res.ok) throw new Error('Failed to fetch model info');
            const data = await res.json();

            document.getElementById('model-name').textContent = data.active_model;
            document.getElementById('model-auc').textContent = data.validation_auc.toFixed(3);
            document.getElementById('model-version').textContent = data.version;

            const indicator = document.querySelector('.status-indicator');
            if (data.active_model.includes('unknown')) {
                indicator.style.background = 'var(--accent-danger)';
                indicator.style.boxShadow = '0 0 10px var(--accent-danger)';
                document.getElementById('model-name').textContent = 'Not Trained';
            } else {
                indicator.style.background = 'var(--accent-success)';
                indicator.style.boxShadow = '0 0 10px var(--accent-success)';
            }

            const dpf = document.getElementById('dash-pending-feedback');
            if (dpf) {
                dpf.textContent = data.samples_since_retrain;
            }
        } catch (error) {
            console.error(error);
        }
    }

    async function handleScreening() {
        const orgBtnText = screenBtn.innerHTML;
        screenBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Screening...';
        screenBtn.disabled = true;

        try {
            const jobId = document.getElementById('job-id').value.trim();
            const jobDesc = document.getElementById('job-description').value.trim();
            const skillsRaw = document.getElementById('required-skills').value;
            const requiredSkills = skillsRaw.split(',').map(s => s.trim()).filter(s => s);

            const resumeRows = Array.from(document.querySelectorAll('.resume-input-row'));
            const resumes = resumeRows.map((row, idx) => {
                const id = row.querySelector('.cand-id').value.trim() || `Candidate-${idx + 1}`;
                const text = row.querySelector('.cand-text').value.trim();
                return { candidate_id: id, text };
            }).filter(r => r.text.length >= 50);

            if (resumes.length === 0) {
                alert("Please add at least one resume with sufficient text (50+ chars).");
                return;
            }

            // Cache data for Deep Dive Analysis UI
            window.lastScreeningCache = {
                skills: requiredSkills,
                resumes: {}
            };
            resumes.forEach(r => window.lastScreeningCache.resumes[r.candidate_id] = r.text);

            const payload = {
                job_id: jobId || "job_default",
                job_description: jobDesc,
                required_skills: requiredSkills,
                resumes: resumes,
                model_flavor: "xgboost"
            };

            const res = await fetch(`${API_URL}/api/v1/screen`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Screening failed');
            }

            const data = await res.json();
            renderResults(data);

        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            screenBtn.innerHTML = orgBtnText;
            screenBtn.disabled = false;
        }
    }

    function renderResults(data) {
        resultsCount.textContent = `${data.total_candidates} candidates`;
        candidatesList.innerHTML = '';

        data.ranked_candidates.forEach((cand, idx) => {
            let scoreClass = 'low';
            let autoDecision = 'REJECT Candidate';
            let decisionColor = 'var(--accent-danger)';

            if (cand.hybrid_score >= 0.70) {
                scoreClass = 'high';
                autoDecision = 'SELECT Candidate';
                decisionColor = 'var(--accent-success)';
            } else if (cand.hybrid_score >= 0.40) {
                scoreClass = 'medium';
                autoDecision = 'REVIEW Candidate';
                decisionColor = 'var(--accent-warning)';
            }

            const card = document.createElement('div');
            card.className = 'candidate-card';
            card.style.animationDelay = `${idx * 0.1}s`;

            // Build Positive factors HTML
            const posHtml = cand.explanation.top_positive_factors.map(f => `<li>${f}</li>`).join('');
            const negHtml = cand.explanation.top_negative_factors.map(f => `<li>${f}</li>`).join('');

            card.innerHTML = `
                <div class="card-header">
                    <div class="cand-name-rank">
                        <div class="rank-badge">#${cand.rank}</div>
                        <div class="cand-name">${cand.candidate_id}</div>
                    </div>
                    <div class="score-display">
                        <div class="score-label">Hybrid Score</div>
                        <div class="score-main ${scoreClass}">${(cand.hybrid_score * 100).toFixed(1)}%</div>
                    </div>
                </div>
                
                <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: var(--radius-sm); margin-bottom: 16px; display: inline-flex; align-items: center; gap: 8px; border: 1px solid ${decisionColor}40;">
                    <i class="fa-solid fa-robot" style="color: ${decisionColor};"></i>
                    <span style="font-size: 13px; font-weight: 600; color: ${decisionColor};">AI Suggestion: ${autoDecision}</span>
                </div>
                
                <div class="score-breakdown-bars">
                    <div class="score-bar-row">
                        <div class="score-bar-labels"><span class="score-item-label">Hire Prob</span><span class="score-item-value">${(cand.hire_probability * 100).toFixed(1)}%</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${(cand.hire_probability * 100).toFixed(1)}%; background: #6366F1;"></div></div>
                    </div>
                    <div class="score-bar-row">
                        <div class="score-bar-labels"><span class="score-item-label">TF-IDF</span><span class="score-item-value">${(cand.component_scores.tfidf_score * 100).toFixed(1)}%</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${(cand.component_scores.tfidf_score * 100).toFixed(1)}%; background: #A855F7;"></div></div>
                    </div>
                    <div class="score-bar-row">
                        <div class="score-bar-labels"><span class="score-item-label">Semantic Match</span><span class="score-item-value">${(cand.component_scores.semantic_score * 100).toFixed(1)}%</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${(cand.component_scores.semantic_score * 100).toFixed(1)}%; background: #3B82F6;"></div></div>
                    </div>
                    <div class="score-bar-row">
                        <div class="score-bar-labels"><span class="score-item-label">Skill Match</span><span class="score-item-value">${(cand.component_scores.skill_match_score * 100).toFixed(1)}%</span></div>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${(cand.component_scores.skill_match_score * 100).toFixed(1)}%; background: #10B981;"></div></div>
                    </div>
                </div>
                
                <div class="explanation">
                    <div class="exp-section">
                        <div class="exp-title pos"><i class="fa-solid fa-circle-check"></i> Key Strengths</div>
                        <ul class="exp-list pos">${posHtml}</ul>
                    </div>
                    <div class="exp-section" style="margin-top: 12px;">
                        <div class="exp-title neg"><i class="fa-solid fa-circle-exclamation"></i> Gaps / Risks</div>
                        <ul class="exp-list neg">${negHtml}</ul>
                    </div>
                </div>
                
                <div class="feedback-actions" data-cand-id="${cand.candidate_id}" data-job-id="${data.job_id}">
                    <button class="btn-feedback shortlist"><i class="fa-solid fa-thumbs-up"></i> Shortlist</button>
                    <button class="btn-feedback hold"><i class="fa-solid fa-pause"></i> Hold</button>
                    <button class="btn-feedback reject"><i class="fa-solid fa-thumbs-down"></i> Reject</button>
                </div>
                
                <div style="margin-top: 12px;">
                    <button class="btn btn-outline btn-sm deep-dive-btn" data-cand-id="${cand.candidate_id}" data-stats='${JSON.stringify({ pos: cand.explanation.top_positive_factors, neg: cand.explanation.top_negative_factors })}' style="width: 100%; border-color: var(--accent-primary); color: var(--accent-primary);">
                        <i class="fa-solid fa-magnifying-glass-chart"></i> Deep Dive Analysis
                    </button>
                </div>
            `;

            candidatesList.appendChild(card);
        });

        // Add feedback event listeners
        document.querySelectorAll('.btn-feedback').forEach(btn => {
            btn.addEventListener('click', handleFeedback);
        });

        // Add deep dive event listeners
        document.querySelectorAll('.deep-dive-btn').forEach(btn => {
            btn.addEventListener('click', openDeepDive);
        });
    }

    function openDeepDive(e) {
        const btn = e.currentTarget;
        const candId = btn.dataset.candId;
        const stats = JSON.parse(btn.dataset.stats);
        const originalText = window.lastScreeningCache.resumes[candId] || "Text not found";
        const requiredSkills = window.lastScreeningCache.skills || [];

        // Accurate highlighting logic based on required skills
        let highlightedText = originalText;
        if (requiredSkills.length > 0) {
            const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const keywordsEscaped = requiredSkills.map(s => escapeRegExp(s)).join('|');
            const regex = new RegExp(`(${keywordsEscaped})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<span class="highlight-match">$1</span>');
        }

        document.getElementById('modal-cand-name').textContent = candId;
        document.getElementById('modal-resume-text').innerHTML = highlightedText;

        // Populate reasoning
        const reasoningList = document.getElementById('modal-reasoning-list');
        reasoningList.innerHTML = '';
        stats.pos.forEach(s => {
            reasoningList.innerHTML += `<li style="color: #34D399;"><i class="fa-solid fa-plus" style="margin-right: 8px;"></i> ${s}</li>`;
        });
        stats.neg.forEach(s => {
            reasoningList.innerHTML += `<li style="color: #F87171;"><i class="fa-solid fa-minus" style="margin-right: 8px;"></i> ${s}</li>`;
        });

        document.getElementById('deep-dive-modal').style.display = 'flex';
    }

    document.getElementById('close-modal-btn')?.addEventListener('click', () => {
        document.getElementById('deep-dive-modal').style.display = 'none';
    });

    async function handleFeedback(e) {
        const btn = e.currentTarget;
        const actionsContainer = btn.closest('.feedback-actions');
        const candId = actionsContainer.dataset.candId;
        const jobId = actionsContainer.dataset.jobId;

        let decision = "on_hold";
        if (btn.classList.contains('shortlist')) decision = "shortlisted";
        if (btn.classList.contains('reject')) decision = "rejected";

        try {
            const orgText = btn.innerHTML;
            btn.innerHTML = '...';

            const res = await fetch(`${API_URL}/api/v1/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    candidate_id: candId,
                    recruiter_id: "recruiter_dashboard_1",
                    decision: decision
                })
            });

            if (!res.ok) throw new Error('Failed to record feedback');

            // Visual feedback
            actionsContainer.querySelectorAll('.btn-feedback').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            btn.innerHTML = orgText;

            // Re-fetch model info to update actionable samples count
            fetchModelInfo();

        } catch (error) {
            alert(error.message);
        }
    }

    async function handleRetraining() {
        if (!confirm("Are you sure you want to trigger model retraining using collected feedback?")) return;

        const orgText = retrainBtn.innerHTML;
        retrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retraining...';
        retrainBtn.disabled = true;

        try {
            const res = await fetch(`${API_URL}/api/v1/retrain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_flavor: "xgboost",
                    force: false
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Retraining failed');
            }

            const data = await res.json();

            let msg = `Retraining completed!\nStatus: ${data.reason}\n`;
            if (data.promoted) {
                msg += `New AUC: ${data.new_auc}\nIncumbent AUC: ${data.incumbent_auc}`;
            }
            alert(msg);

            fetchModelInfo();
        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            retrainBtn.innerHTML = orgText;
            retrainBtn.disabled = false;
        }
    }

    // Initialization
    async function initSettings() {
        try {
            const res = await fetch(`${API_URL}/api/v1/settings`);
            if (res.ok) {
                const data = await res.json();
                const thresVal = parseInt(data.threshold * 100);
                document.getElementById('settings-threshold').value = thresVal;
                document.getElementById('settings-threshold-display').textContent = data.threshold.toFixed(2);
                document.getElementById('settings-mode').value = data.mode;
            }
        } catch (e) {
            console.error("Failed to load settings from API", e);
        }
    }

    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const navScreen = document.getElementById('nav-screen');
    const navDashboard = document.getElementById('nav-dashboard');
    const navSettings = document.getElementById('nav-settings');
    const screenView = document.getElementById('screen-view');
    const dashboardView = document.getElementById('dashboard-view');
    const settingsView = document.getElementById('settings-view');

    // UI Element References
    const dashLoading = document.getElementById('dash-loading');
    const dashEmpty = document.getElementById('dash-empty');
    const dashTableContainer = document.getElementById('dash-table-container');
    const dashTableBody = document.getElementById('dash-table-body');
    const dashTop5List = document.getElementById('dash-top-5-list');
    const thresSlider = document.getElementById('settings-threshold');
    const thresDisplay = document.getElementById('settings-threshold-display');
    const saveSettingsBtn = document.getElementById('btn-save-settings');

    // Initial load
    initSettings();

    if (thresSlider) {
        thresSlider.addEventListener('input', (e) => {
            thresDisplay.textContent = (e.target.value / 100).toFixed(2);
        });
    }

    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', async () => {
            const prevText = saveSettingsBtn.innerHTML;
            saveSettingsBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
            try {
                const payload = {
                    threshold: parseFloat(thresSlider.value) / 100,
                    mode: document.getElementById('settings-mode').value
                };
                const res = await fetch(`${API_URL}/api/v1/settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    saveSettingsBtn.innerHTML = '<i class="fa-solid fa-check"></i> Saved';
                    setTimeout(() => saveSettingsBtn.innerHTML = prevText, 2000);
                }
            } catch (e) {
                console.error("Failed to save", e);
                saveSettingsBtn.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Error';
                setTimeout(() => saveSettingsBtn.innerHTML = prevText, 2000);
            }
        });
    }

    window.switchView = function (viewName) {
        console.log("Switching to view:", viewName);
        // Reset all views
        const views = {
            'screen': screenView,
            'dashboard': dashboardView,
            'settings': settingsView
        };

        const navLinks = {
            'screen': navScreen,
            'dashboard': navDashboard,
            'settings': navSettings
        };

        // Hide all views with a clear reset
        Object.values(views).forEach(v => {
            if (v) {
                v.style.display = 'none';
                v.classList.remove('animate-spring-in');
            }
        });

        // Remove active class from all navs
        Object.values(navLinks).forEach(n => {
            if (n) n.classList.remove('active');
        });

        // Show requested view with spring animation
        const activeView = views[viewName];
        if (activeView) {
            activeView.style.display = viewName === 'dashboard' ? 'flex' : 'block';
            // Trigger reflow for animation
            void activeView.offsetWidth;
            activeView.classList.add('animate-spring-in');

            if (navLinks[viewName]) navLinks[viewName].classList.add('active');

            // Update page title
            const titles = {
                'screen': 'Screen Candidates',
                'dashboard': 'Analytics Intelligence',
                'settings': 'System Control Center'
            };
            if (pageTitle) pageTitle.innerText = titles[viewName];

            // Load data if needed
            if (viewName === 'dashboard') loadDashboardData();
        }
    }

    navScreen.addEventListener('click', (e) => { e.preventDefault(); switchView('screen'); });
    navDashboard.addEventListener('click', (e) => { e.preventDefault(); switchView('dashboard'); });

    function generateMockData() {
        const mockCands = [
            { candidate_id: "Mock_Dr_Jane_Doe_PhD", hybrid_score: 0.94, hire_probability: 0.95, explanation: { top_positive_factors: ["Distinguished ML & Research Background", "Perfect keyword symmetry"], top_negative_factors: [] }, component_scores: { tfidf_score: 0.95, semantic_score: 0.96, skill_match_score: 0.91 } },
            { candidate_id: "Mock_Alex_Software_Eng", hybrid_score: 0.81, hire_probability: 0.80, explanation: { top_positive_factors: ["Strong foundational coding", "Cloud certifications present"], top_negative_factors: ["Missing specialized AI history"] }, component_scores: { tfidf_score: 0.8, semantic_score: 0.85, skill_match_score: 0.78 } },
            { candidate_id: "Mock_Sam_Intern", hybrid_score: 0.38, hire_probability: 0.35, explanation: { top_positive_factors: ["Academic projects"], top_negative_factors: ["No industry experience", "Missing core requested technologies"] }, component_scores: { tfidf_score: 0.35, semantic_score: 0.45, skill_match_score: 0.34 } }
        ];
        return {
            total_count: 36,
            top_candidates: mockCands.slice(0, 3), // Show 3 to save vertical space
            all_candidates: mockCands
        };
    }

    async function loadDashboardData() {
        dashLoading.style.display = 'flex';
        dashEmpty.style.display = 'none';
        dashTableContainer.style.display = 'none';

        try {
            const res = await fetch(`${API_URL}/api/v1/dashboard`);
            if (res.ok) {
                const data = await res.json();
                if (data.total_count === 0 || !data.all_candidates || data.all_candidates.length === 0) {
                    renderDashboard(generateMockData()); // Fallback to mock data to show functionality
                } else {
                    renderDashboard(data);
                }
            } else {
                throw new Error("API returned failure");
            }
        } catch (error) {
            console.error("Dashboard fetch failed:", error);
            // Fallback mock behavior if backend is unavailable
            renderDashboard(generateMockData());
        }
    }

    function renderDashboard(data) {
        dashLoading.style.display = 'none';

        document.getElementById('dash-total-screened').textContent = data.total_count || 0;

        if (!data.all_candidates || data.all_candidates.length === 0) {
            document.getElementById('dash-avg-score').textContent = '0%';
            dashEmpty.style.display = 'block';
            dashTop5List.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">No telemetry found</div>';
            document.getElementById('dash-chart-container').innerHTML = '';
            return;
        }

        dashTableContainer.style.display = 'block';

        // Calculate average
        const avg = data.all_candidates.reduce((sum, c) => sum + c.hybrid_score, 0) / data.all_candidates.length;
        document.getElementById('dash-avg-score').textContent = (avg * 100).toFixed(1) + "%";

        // Render Distribution Chart using Chart.js
        const chartCanvas = document.getElementById('main-distribution-chart');
        if (chartCanvas) {
            const bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            data.all_candidates.forEach(c => {
                const idx = Math.min(Math.floor(c.hybrid_score * 10), 9);
                bins[idx]++;
            });

            if (distributionChart) distributionChart.destroy();

            const ctx = chartCanvas.getContext('2d');
            distributionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
                    datasets: [{
                        label: 'Candidates',
                        data: bins,
                        backgroundColor: bins.map((_, i) => {
                            if (i >= 7) return '#10B981'; // Success
                            if (i >= 4) return '#F59E0B'; // Warning
                            return '#F43F5E'; // Danger
                        }),
                        borderRadius: 6,
                        borderSkipped: false,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                        x: { grid: { display: false }, ticks: { color: '#94A3B8' } }
                    }
                }
            });
        }

        // Animate stats
        const prevTotal = parseInt(document.getElementById('dash-total-screened').dataset.prev || 0);
        animateValue('dash-total-screened', prevTotal, data.total_count, 1000);
        document.getElementById('dash-total-screened').dataset.prev = data.total_count;

        const currentAvg = parseFloat((avg * 100).toFixed(0));
        const prevAvg = parseInt(document.getElementById('dash-avg-score').dataset.prev || 0);
        animateValue('dash-avg-score', prevAvg, currentAvg, 1000);
        document.getElementById('dash-avg-score').dataset.prev = currentAvg;

        // Render Top Elite Performers
        dashTop5List.innerHTML = '';
        const elite = [...data.all_candidates].sort((a, b) => b.hybrid_score - a.hybrid_score).slice(0, 3);
        elite.forEach((cand, idx) => {
            let actionColor = cand.hybrid_score >= 0.7 ? 'var(--accent-success)' : (cand.hybrid_score >= 0.4 ? 'var(--accent-warning)' : 'var(--accent-danger)');
            dashTop5List.innerHTML += `
                <div style="background: rgba(255,255,255,0.02); padding: 12px 16px; border-radius: var(--radius-md); border: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 24px; height: 24px; border-radius: 50%; background: ${actionColor}20; color: ${actionColor}; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 800;">${idx + 1}</div>
                        <div style="font-weight: 600; font-size: 13px; color: #fff;">${cand.candidate_id.split('_').pop()}</div>
                    </div>
                    <div style="color: ${actionColor}; font-weight: 800; font-size: 13px;">${(cand.hybrid_score * 100).toFixed(0)}%</div>
                </div>
            `;
        });

        // Render Table
        dashTableBody.innerHTML = '';
        data.all_candidates.forEach(cand => {
            let autoDecision = cand.hybrid_score >= 0.70 ? 'SELECT' : (cand.hybrid_score >= 0.40 ? 'REVIEW' : 'REJECT');
            let decisionColor = cand.hybrid_score >= 0.70 ? 'var(--accent-success)' : (cand.hybrid_score >= 0.40 ? 'var(--accent-warning)' : 'var(--accent-danger)');
            let autoDescId = cand.candidate_id.replace(/'/g, ''); // Fix payload breaking quotes

            // Reconstruct stats string securely
            const statsPayload = JSON.stringify({
                pos: cand.explanation.top_positive_factors,
                neg: cand.explanation.top_negative_factors
            }).replace(/"/g, '&quot;');

            // Generate mock history resumes if using mock data
            if (!window.lastScreeningCache) {
                window.lastScreeningCache = { skills: ["Python", "Machine Learning"], resumes: {} };
            }
            if (!window.lastScreeningCache.resumes[cand.candidate_id]) {
                window.lastScreeningCache.resumes[cand.candidate_id] = "Simulated resume text for " + cand.candidate_id + ". Extensive history with Python, Machine Learning, and algorithms mapping to keyword metrics in the vector database.";
            }

            dashTableBody.innerHTML += `
                <tr>
                    <td>${cand.candidate_id}</td>
                    <td style="color: ${decisionColor}; font-weight: 700;">${(cand.hybrid_score * 100).toFixed(1)}%</td>
                    <td>
                        <span class="badge" style="background: ${decisionColor}20; color: ${decisionColor}; border: 1px solid ${decisionColor}40;">
                            ${autoDecision}
                        </span>
                    </td>
                    <td style="text-align: right;">
                        <button class="btn btn-outline btn-sm deep-dive-btn" data-cand-id="${cand.candidate_id}" data-stats="${statsPayload}" style="font-size: 11px; padding: 4px 12px; width: auto;">
                            <i class="fa-solid fa-microscope"></i> Inspect
                        </button>
                    </td>
                </tr>
            `;
        });

        // Re-attach deep dive listeners for the table
        document.querySelectorAll('#dash-table-body .deep-dive-btn').forEach(btn => {
            btn.addEventListener('click', openDeepDive);
        });
    }

    if (navSettings) {
        navSettings.addEventListener('click', (e) => {
            e.preventDefault();
            switchView('settings');
        });
    }

    // --- THREE.JS 3D BACKGROUND ---
    function initThreeBG() {
        const canvas = document.getElementById('bg-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);

        // Particles
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 2000;
        const posArray = new Float32Array(particlesCount * 3);

        for (let i = 0; i < particlesCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 10;
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        const material = new THREE.PointsMaterial({
            size: 0.005,
            color: '#6366F1',
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        const particlesMesh = new THREE.Points(particlesGeometry, material);
        scene.add(particlesMesh);

        camera.position.z = 3;

        // Mouse interaction for particles
        let mouseX = 0;
        let mouseY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth - 0.5);
            mouseY = (e.clientY / window.innerHeight - 0.5);
        });

        function animate() {
            requestAnimationFrame(animate);
            particlesMesh.rotation.y += 0.001;
            particlesMesh.rotation.x += 0.001;

            particlesMesh.rotation.y += mouseX * 0.05;
            particlesMesh.rotation.x += -mouseY * 0.05;

            renderer.render(scene, camera);
        }

        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    // --- 3D CARD TRACKING ---


    initThreeBG();
});
