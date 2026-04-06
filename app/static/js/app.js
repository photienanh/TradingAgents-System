let currentSessionId = null;
let statusCheckInterval = null;
let isReviewingHistoricalSession = false;
let activePollingSessionId = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setupFormHandlers();
    setupSidebarToggle();
    setupSessionReviewModalHandlers();
});

function sanitizeReportText(text) {
    if (!text) return text;
    return text
        .split('\n')
        .filter(line => {
            const normalized = line.trim().toLowerCase();
            return normalized !== 'portfolio management decision' && normalized !== 'portfolio manager decision';
        })
        .join('\n');
}

function safeStringify(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch (error) {
        console.warn('Unable to stringify value:', error);
        return String(value);
    }
}

// Setup sidebar toggle
function setupSidebarToggle() {
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const mainWrapper = document.querySelector('.main-wrapper');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            mainWrapper.classList.toggle('expanded');
        });
    }
}

// Setup form handlers
function setupFormHandlers() {
    const form = document.getElementById('analysis-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await startAnalysis();
    });

    // Add event listener for "Start New Analysis" button
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', startNewAnalysis);
    }

    const stopAnalysisBtn = document.getElementById('stop-analysis-btn');
    if (stopAnalysisBtn) {
        stopAnalysisBtn.addEventListener('click', cancelCurrentAnalysis);
    }
}

function setupSessionReviewModalHandlers() {
    const closeBtn = document.getElementById('close-session-review-btn');
    const backdrop = document.getElementById('session-review-backdrop');

    if (closeBtn) {
        closeBtn.addEventListener('click', closeSessionReviewModal);
    }
    if (backdrop) {
        backdrop.addEventListener('click', closeSessionReviewModal);
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeSessionReviewModal();
        }
    });
}

function openSessionReviewModal() {
    const modal = document.getElementById('session-review-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeSessionReviewModal() {
    const modal = document.getElementById('session-review-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

function clearStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

function clearAnalysisDisplay() {
    const sessionIdEl = document.getElementById('session-id');
    const tickerEl = document.getElementById('current-ticker');
    const statusEl = document.getElementById('current-status');
    const stepEl = document.getElementById('current-step');
    const progressFill = document.getElementById('analysis-progress-fill');
    const agentStatusEl = document.getElementById('agent-status');
    const currentReportEl = document.getElementById('current-report');
    const finalReportEl = document.getElementById('final-report');
    const decisionEl = document.getElementById('final-decision');

    if (sessionIdEl) sessionIdEl.textContent = '-';
    if (tickerEl) tickerEl.textContent = '-';
    if (statusEl) {
        statusEl.textContent = '';
        statusEl.className = 'status-badge';
    }
    if (stepEl) stepEl.textContent = 'Đang chờ bắt đầu phân tích...';
    if (progressFill) progressFill.style.width = '0%';
    if (agentStatusEl) agentStatusEl.innerHTML = '';
    if (currentReportEl) currentReportEl.textContent = 'Waiting for analysis to start...';
    if (finalReportEl) finalReportEl.textContent = '';
    if (decisionEl) decisionEl.textContent = '';
}

function renderSessionReview(sessionId, data) {
    const reviewModal = document.getElementById('session-review-modal');
    if (!reviewModal) return;
    openSessionReviewModal();

    const reviewSessionId = document.getElementById('review-session-id');
    const reviewTicker = document.getElementById('review-ticker');
    const reviewStatus = document.getElementById('review-status');
    const reviewCurrent = document.getElementById('review-current-report');
    const reviewFinal = document.getElementById('review-final-report');
    const reviewDecision = document.getElementById('review-final-decision');

    const ticker = sessionId.split('_')[0] || '-';

    if (reviewSessionId) reviewSessionId.textContent = sessionId;
    if (reviewTicker) reviewTicker.textContent = ticker;
    if (reviewStatus) {
        reviewStatus.textContent = data.status || '-';
        reviewStatus.className = `status-badge ${data.status || ''}`;
    }

    if (reviewCurrent) {
        if (data.current_report) {
            reviewCurrent.innerHTML = marked.parse(sanitizeReportText(data.current_report));
        } else {
            reviewCurrent.textContent = 'Không có báo cáo tạm thời.';
        }
    }

    if (reviewFinal) {
        if (data.final_report) {
            reviewFinal.innerHTML = marked.parse(sanitizeReportText(data.final_report));
        } else {
            reviewFinal.textContent = 'Chưa có báo cáo tổng hợp.';
        }
    }

    if (reviewDecision) {
        if (data.decision) {
            updateDecision(data.decision, 'review-final-decision');
        } else {
            reviewDecision.innerHTML = '<p>Chưa có quyết định.</p>';
        }
    }
}

// Start new analysis
async function startAnalysis() {
    isReviewingHistoricalSession = false;
    clearStatusPolling();
    activePollingSessionId = null;

    const form = document.getElementById('analysis-form');
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');

    // Disable button and show spinner
    submitBtn.disabled = true;
    btnText.textContent = 'Starting Analysis...';
    spinner.classList.remove('hidden');

    // Get selected analysts
    const selectedAnalysts = [];
    const analystCheckboxes = form.querySelectorAll('input[name="analysts"]:checked');
    analystCheckboxes.forEach(checkbox => {
        selectedAnalysts.push(checkbox.value);
    });

    // Validate at least one analyst is selected
    if (selectedAnalysts.length === 0) {
        alert('Please select at least one analyst');
        submitBtn.disabled = false;
        btnText.textContent = 'Start Analysis';
        spinner.classList.add('hidden');
        return;
    }

    // Prepare request data
    const requestData = {
        ticker: formData.get('ticker').toUpperCase(),
        analysis_date: formData.get('analysis_date') || null,
        analysts: selectedAnalysts,
        research_depth: parseInt(formData.get('research_depth')),
        deep_think_llm: formData.get('deep_think_llm'),
        quick_think_llm: formData.get('quick_think_llm'),
        max_debate_rounds: parseInt(formData.get('research_depth')),
        data_vendors: {
            core_stock_apis: "vnstock",
            technical_indicators: "vnstock",
            fundamental_data: "vnstock",
            news_data: "google",
            global_data: "vietstock",
            insider_transaction_data: "yfinance"
        }
    };

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMsg = errorData.error || 'Failed to start analysis';
            const errorDetails = errorData.details || '';
            console.error('Error starting analysis:', errorMsg, errorDetails);
            alert(`Failed to start analysis: ${errorMsg}`);
            submitBtn.disabled = false;
            btnText.textContent = '🚀 Start Analysis';
            spinner.classList.add('hidden');
            return;
        }

        const data = await response.json();
        currentSessionId = data.session_id;
        activePollingSessionId = data.session_id;

        switchSection('new-analysis');

        // Show progress card and hide form
        const formCard = document.getElementById('analysis-form-card');
        const progressCard = document.getElementById('analysis-progress-card');
        const sessionIdEl = document.getElementById('session-id');
        const tickerEl = document.getElementById('current-ticker');
        
        if (!formCard || !progressCard || !sessionIdEl || !tickerEl) {
            console.error('Missing elements:', { formCard, progressCard, sessionIdEl, tickerEl });
            alert('UI elements not found. Please refresh the page.');
            return;
        }
        
        formCard.classList.add('hidden');
        progressCard.classList.remove('hidden');
        const stopBtn = document.getElementById('stop-analysis-btn');
        if (stopBtn) {
            stopBtn.classList.remove('hidden');
            stopBtn.disabled = false;
        }

        // Update session info
        sessionIdEl.textContent = currentSessionId;
        tickerEl.textContent = requestData.ticker;

        // Start polling for status
        startStatusPolling();

    } catch (error) {
        console.error('Error starting analysis:', error);
        alert(`Failed to start analysis: ${error.message}. Please try again.`);
        submitBtn.disabled = false;
        btnText.textContent = 'Start Analysis';
        spinner.classList.add('hidden');
    }
}

// Start polling for analysis status
function startStatusPolling() {
    // Clear any existing interval
    clearStatusPolling();

    if (!activePollingSessionId || isReviewingHistoricalSession) {
        return;
    }

    // Check immediately
    checkAnalysisStatus();

    // Then check every 2 seconds
    statusCheckInterval = setInterval(checkAnalysisStatus, 2000);
}

// Check analysis status
async function checkAnalysisStatus() {
    if (!activePollingSessionId || isReviewingHistoricalSession) return;

    try {
        const response = await fetch(`/api/status/${activePollingSessionId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch status');
        }

        const data = await response.json();
        updateProgressUI(data);

        // Stop polling if analysis is completed or errored
        if (data.status === 'completed' || data.status === 'error' || data.status === 'cancelled') {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
            activePollingSessionId = null;
            onAnalysisComplete(data);
        }

    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Update progress UI
function updateProgressUI(data) {
    // Update status badge
    const statusBadge = document.getElementById('current-status');
    statusBadge.textContent = data.status;
    statusBadge.className = `status-badge ${data.status}`;

    const stepEl = document.getElementById('current-step');
    const progressFill = document.getElementById('analysis-progress-fill');
    const fallbackStep = data.status === 'completed'
        ? '✅ Hoàn thành phân tích'
        : data.status === 'cancelled'
            ? '🛑 Đã hủy phân tích theo yêu cầu'
            : data.status === 'error'
                ? '❌ Phân tích thất bại'
                : 'Đang xử lý...';
    const currentStep = data.current_step || fallbackStep;
    const progressPercent = Number.isFinite(data.progress_percent)
        ? Math.max(0, Math.min(100, data.progress_percent))
        : (data.status === 'completed' ? 100 : 0);

    if (stepEl) {
        stepEl.textContent = `${currentStep} (${progressPercent}%)`;
    }
    if (progressFill) {
        progressFill.style.width = `${progressPercent}%`;
    }

    const stopBtn = document.getElementById('stop-analysis-btn');
    if (stopBtn) {
        const canStop = data.status === 'running' || data.status === 'initializing';
        stopBtn.classList.toggle('hidden', !canStop);
        stopBtn.disabled = !canStop;
    }

    // Show error if present
    if (data.status === 'error' && data.error) {
        const reportDiv = document.getElementById('current-report');
        reportDiv.innerHTML = `
            <div class="error-message">
                <h3>❌ Error</h3>
                <p><strong>Error:</strong> ${data.error}</p>
                ${data.error_details ? `<details><summary>View Details</summary><pre>${data.error_details}</pre></details>` : ''}
            </div>
        `;
    }

    // Update agent status
    updateAgentStatus(data.agent_status);

    // Update current report
    const reportDiv = document.getElementById('current-report');
    if (data.current_report) {
        reportDiv.innerHTML = marked.parse(sanitizeReportText(data.current_report));
    } else if (data.status === 'completed' && data.final_report) {
        reportDiv.innerHTML = marked.parse(sanitizeReportText(data.final_report));
    }

    // Update final report if available
    if (data.final_report) {
        const finalReportDiv = document.getElementById('final-report');
        if (finalReportDiv) {
            finalReportDiv.innerHTML = marked.parse(sanitizeReportText(data.final_report));
        }
    }

    // Update decision if available
    if (data.decision) {
        updateDecision(data.decision);
    }
}

// Update agent status display
function updateAgentStatus(agentStatus) {
    const container = document.getElementById('agent-status');
    
    if (!agentStatus || Object.keys(agentStatus).length === 0) {
        container.innerHTML = '<p>No agent status available</p>';
        return;
    }

    container.innerHTML = '';

    const preferredOrder = [
        'Market Analyst',
        'Social Analyst',
        'News Analyst',
        'Fundamentals Analyst',
        'Bull Researcher',
        'Bear Researcher',
        'Research Manager',
        'Trader',
        'AlphaGPT Analyst',
        'Risky Analyst',
        'Safe Analyst',
        'Neutral Analyst',
        'Portfolio Manager'
    ];

    const orderedEntries = [];
    for (const agent of preferredOrder) {
        if (Object.prototype.hasOwnProperty.call(agentStatus, agent)) {
            orderedEntries.push([agent, agentStatus[agent]]);
        }
    }
    for (const [agent, status] of Object.entries(agentStatus)) {
        if (!preferredOrder.includes(agent)) {
            orderedEntries.push([agent, status]);
        }
    }

    for (const [agent, status] of orderedEntries) {
        const agentDiv = document.createElement('div');
        agentDiv.className = `agent-item ${status}`;
        agentDiv.innerHTML = `
            <div class="agent-item-name">${agent}</div>
            <div class="agent-item-status">${getStatusIcon(status)} ${status.replace('_', ' ')}</div>
        `;
        container.appendChild(agentDiv);
    }
}

// Get status icon
function getStatusIcon(status) {
    const icons = {
        'pending': '⏳',
        'in_progress': '⚙️',
        'completed': '✅',
        'not_selected': '⏭️'
    };
    return icons[status] || '⏳';
}

function buildDecisionHtml(decision) {
    let html = '';

    // Check if decision is a string or object
    if (typeof decision === 'string') {
        // Extract decision type (BUY, SELL, HOLD)
        const decisionUpper = decision.toUpperCase();
        let decisionType = 'HOLD';
        let decisionClass = 'hold';

        if (decisionUpper.includes('BUY')) {
            decisionType = 'BUY';
            decisionClass = 'buy';
        } else if (decisionUpper.includes('SELL')) {
            decisionType = 'SELL';
            decisionClass = 'sell';
        }

        html = `
            <div class="decision-highlight decision-${decisionClass}">
                <div class="decision-icon">${decisionClass === 'buy' ? '📈' : decisionClass === 'sell' ? '📉' : '⏸️'}</div>
                <div class="decision-text">
                    <h2>${decisionType}</h2>
                    <p>${decision}</p>
                </div>
            </div>
        `;
    } else if (typeof decision === 'object' && decision !== null) {
        // Extract main decision from object
        let mainDecision = 'HOLD';
        let decisionClass = 'hold';

        const normalizedDecision = safeStringify(decision).toUpperCase();
        if (normalizedDecision.includes('BUY')) {
            mainDecision = 'BUY';
            decisionClass = 'buy';
        } else if (normalizedDecision.includes('SELL')) {
            mainDecision = 'SELL';
            decisionClass = 'sell';
        }

        html = `
            <div class="decision-highlight decision-${decisionClass}">
                <div class="decision-icon">${decisionClass === 'buy' ? '📈' : decisionClass === 'sell' ? '📉' : '⏸️'}</div>
                <div class="decision-text">
                    <h2>${mainDecision}</h2>
                </div>
            </div>
            <div class="decision-details">
        `;

        // Object decision with details
        for (const [key, value] of Object.entries(decision)) {
            html += `
                <div class="decision-item">
                    <strong>${formatKey(key)}:</strong> ${formatValue(value)}
                </div>
            `;
        }

        html += '</div>';
    } else {
        html = '<p>Chưa có quyết định.</p>';
    }

    return html;
}

function updateDecision(decision, targetId = 'final-decision') {
    const decisionDiv = document.getElementById(targetId);
    if (!decisionDiv) return;

    decisionDiv.innerHTML = buildDecisionHtml(decision);
}

// Format decision key
function formatKey(key) {
    return key.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

// Format decision value
function formatValue(value) {
    if (typeof value === 'object' && value !== null) {
        return safeStringify(value);
    }
    return value;
}

// Handle analysis completion
function onAnalysisComplete(data) {
    activePollingSessionId = null;

    // Show "Start New Analysis" button
    document.getElementById('new-analysis-btn').classList.remove('hidden');
    
    // Refresh sessions list
    loadSessions();
}

// Start new analysis
function startNewAnalysis() {
    // Clear current session
    clearStatusPolling();
    currentSessionId = null;
    isReviewingHistoricalSession = false;
    activePollingSessionId = null;
    
    // Reset form
    document.getElementById('analysis-form').reset();
    
    // Hide progress card and button
    document.getElementById('analysis-progress-card').classList.add('hidden');
    document.getElementById('new-analysis-btn').classList.add('hidden');
    const stopBtn = document.getElementById('stop-analysis-btn');
    if (stopBtn) {
        stopBtn.classList.add('hidden');
        stopBtn.disabled = false;
    }
    
    // Show form
    document.getElementById('analysis-form-card').classList.remove('hidden');
    
    // Reset button state
    const submitBtn = document.querySelector('#analysis-form button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    submitBtn.disabled = false;
    btnText.textContent = '🚀 Start Analysis';
    spinner.classList.add('hidden');

    clearAnalysisDisplay();
}

async function cancelCurrentAnalysis() {
    if (!activePollingSessionId) return;
    const confirmed = confirm('Hủy phiên phân tích hiện tại? Hành động này sẽ dừng hẳn và không thể tiếp tục.');
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/sessions/${activePollingSessionId}/cancel`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to request cancellation');
        }

        const stopBtn = document.getElementById('stop-analysis-btn');
        if (stopBtn) {
            stopBtn.disabled = true;
        }

        await checkAnalysisStatus();
    } catch (error) {
        console.error('Error cancelling analysis:', error);
        alert('Không thể hủy phiên phân tích. Vui lòng thử lại.');
    }
}

// Load previous sessions
async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');
        if (!response.ok) {
            throw new Error('Failed to load sessions');
        }

        const data = await response.json();
        displaySessions(data.sessions);

    } catch (error) {
        console.error('Error loading sessions:', error);
        document.getElementById('sessions-list').innerHTML = 
            '<p>Failed to load sessions</p>';
    }
}

// Display sessions
function displaySessions(sessions) {
    const container = document.getElementById('sessions-list');
    const completedCounter = document.getElementById('completed-analyses-count');
    if (completedCounter) {
        const completedCount = (sessions || []).filter(s => s.status === 'completed').length;
        completedCounter.textContent = String(completedCount);
    }
    
    if (!sessions || sessions.length === 0) {
        container.innerHTML = '<p>No previous sessions</p>';
        return;
    }

    // Sort by date (newest first)
    sessions.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    container.innerHTML = '';
    
    sessions.forEach(session => {
        const sessionDiv = document.createElement('div');
        sessionDiv.className = 'session-item';
        sessionDiv.innerHTML = `
            <div class="session-item-info">
                <div class="session-item-ticker">📊 ${session.ticker}</div>
                <div class="session-item-meta">
                    ${session.analysis_date} | ${session.status} | ${formatDateTime(session.created_at)}
                </div>
            </div>
            <div class="session-item-actions">
                <button class="btn btn-secondary btn-small" onclick="viewSession('${session.session_id}')">
                    View
                </button>
                <button class="btn btn-danger btn-small" onclick="deleteSession('${session.session_id}')">
                    Delete
                </button>
            </div>
        `;
        container.appendChild(sessionDiv);
    });
}

// Format datetime
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// View session
async function viewSession(sessionId) {
    clearStatusPolling();
    isReviewingHistoricalSession = true;
    activePollingSessionId = null;
    currentSessionId = sessionId;

    // Keep user in Sessions section when reviewing an old session.
    switchSection('sessions');
    
    // Fetch and display session data
    try {
        const response = await fetch(`/api/status/${sessionId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch session');
        }

        const data = await response.json();
        
        renderSessionReview(sessionId, data);

    } catch (error) {
        console.error('Error viewing session:', error);
        alert('Failed to load session');
    }
}

// Delete session
async function deleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this session?')) {
        return;
    }

    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete session');
        }

        if (isReviewingHistoricalSession && currentSessionId === sessionId) {
            currentSessionId = null;
            isReviewingHistoricalSession = false;
            closeSessionReviewModal();
        }

        // Reload sessions
        loadSessions();

    } catch (error) {
        console.error('Error deleting session:', error);
        alert('Failed to delete session');
    }
}
