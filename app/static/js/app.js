let currentSessionId = null;
let statusCheckInterval = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setupFormHandlers();
    setupSidebarToggle();
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
}

// Start new analysis
async function startAnalysis() {
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
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    // Check immediately
    checkAnalysisStatus();

    // Then check every 2 seconds
    statusCheckInterval = setInterval(checkAnalysisStatus, 2000);
}

// Check analysis status
async function checkAnalysisStatus() {
    if (!currentSessionId) return;

    try {
        const response = await fetch(`/api/status/${currentSessionId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch status');
        }

        const data = await response.json();
        updateProgressUI(data);

        // Stop polling if analysis is completed or errored
        if (data.status === 'completed' || data.status === 'error') {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
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
    
    for (const [agent, status] of Object.entries(agentStatus)) {
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
        'completed': '✅'
    };
    return icons[status] || '⏳';
}

// Update decision display
function updateDecision(decision) {
    const decisionDiv = document.getElementById('final-decision');
    if (!decisionDiv) return;
    
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
        
        const decisionStr = JSON.stringify(decision).toUpperCase();
        if (decisionStr.includes('BUY')) {
            mainDecision = 'BUY';
            decisionClass = 'buy';
        } else if (decisionStr.includes('SELL')) {
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
    }
    
    decisionDiv.innerHTML = html;
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
        return JSON.stringify(value, null, 2);
    }
    return value;
}

// Handle analysis completion
function onAnalysisComplete(data) {
    // Show "Start New Analysis" button
    document.getElementById('new-analysis-btn').classList.remove('hidden');
    
    // Refresh sessions list
    loadSessions();
}

// Start new analysis
function startNewAnalysis() {
    // Clear current session
    currentSessionId = null;
    
    // Reset form
    document.getElementById('analysis-form').reset();
    
    // Hide progress card and button
    document.getElementById('analysis-progress-card').classList.add('hidden');
    document.getElementById('new-analysis-btn').classList.add('hidden');
    
    // Show form
    document.getElementById('analysis-form-card').classList.remove('hidden');
    
    // Reset button state
    const submitBtn = document.querySelector('#analysis-form button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    submitBtn.disabled = false;
    btnText.textContent = '🚀 Start Analysis';
    spinner.classList.add('hidden');
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
    currentSessionId = sessionId;
    
    // Switch to new-analysis section and show progress card
    switchSection('new-analysis');
    document.getElementById('analysis-form-card').classList.add('hidden');
    document.getElementById('analysis-progress-card').classList.remove('hidden');
    
    // Fetch and display session data
    try {
        const response = await fetch(`/api/status/${sessionId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch session');
        }

        const data = await response.json();
        
        // Update session info in progress card
        document.getElementById('session-id').textContent = sessionId;
        const ticker = sessionId.split('_')[0];
        document.getElementById('current-ticker').textContent = ticker;
        
        updateProgressUI(data);

        if (data.status === 'completed') {
            document.getElementById('new-analysis-btn').classList.remove('hidden');
        }

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

        // Reload sessions
        loadSessions();

    } catch (error) {
        console.error('Error deleting session:', error);
        alert('Failed to delete session');
    }
}
