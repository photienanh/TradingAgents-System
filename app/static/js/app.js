// app/static/js/app.js
// Tab setup đã chuyển sang index.html (setupTabs scoped per container).
// File này chỉ handle: analysis flow, session list, modal open/close.

let currentSessionId       = null;
let statusCheckInterval    = null;
let isReviewingHistoricalSession = false;
let activePollingSessionId = null;

document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setupFormHandlers();
    setupSidebarToggle();
    setupSessionReviewModalHandlers();
});

// ── Utilities ──────────────────────────────────────────────────────────────

function sanitizeReportText(text) {
    if (!text) return text;
    return text
        .split('\n')
        .filter(line => {
            const n = line.trim().toLowerCase();
            return n !== 'portfolio management decision' && n !== 'portfolio manager decision';
        })
        .join('\n');
}

function safeStringify(value) {
    try   { return JSON.stringify(value, null, 2); }
    catch { return String(value); }
}

// ── Sidebar ────────────────────────────────────────────────────────────────

function setupSidebarToggle() {
    const toggle  = document.getElementById('menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const wrapper = document.querySelector('.main-wrapper');
    if (toggle) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            wrapper.classList.toggle('expanded');
        });
    }
}

// ── Form handlers ──────────────────────────────────────────────────────────

function setupFormHandlers() {
    document.getElementById('analysis-form')
        .addEventListener('submit', async (e) => { e.preventDefault(); await startAnalysis(); });

    document.getElementById('new-analysis-btn')
        ?.addEventListener('click', startNewAnalysis);

    document.getElementById('stop-analysis-btn')
        ?.addEventListener('click', cancelCurrentAnalysis);
}

// ── Session Review Modal ───────────────────────────────────────────────────

function setupSessionReviewModalHandlers() {
    document.getElementById('close-session-review-btn')
        ?.addEventListener('click', closeSessionReviewModal);
    document.getElementById('session-review-backdrop')
        ?.addEventListener('click', closeSessionReviewModal);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeSessionReviewModal();
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

// ── Status polling ─────────────────────────────────────────────────────────

function clearStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

function clearAnalysisDisplay() {
    const fields = {
        'session-id':              '-',
        'current-ticker':          '-',
        'current-step':            'Đang chờ bắt đầu phân tích...',
        'current-report':          'Waiting for analysis to start...',
        'final-report':            '',
        'final-decision':          '',
    };
    for (const [id, val] of Object.entries(fields)) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    }
    const statusEl = document.getElementById('current-status');
    if (statusEl) { statusEl.textContent = ''; statusEl.className = 'status-badge'; }
    const fill = document.getElementById('analysis-progress-fill');
    if (fill) fill.style.width = '0%';
    const agentEl = document.getElementById('agent-status');
    if (agentEl) agentEl.innerHTML = '';
}

// ── Start analysis ─────────────────────────────────────────────────────────

async function startAnalysis() {
    isReviewingHistoricalSession = false;
    clearStatusPolling();
    activePollingSessionId = null;

    const form     = document.getElementById('analysis-form');
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const btnText   = submitBtn.querySelector('.btn-text');
    const spinner   = submitBtn.querySelector('.spinner');

    submitBtn.disabled = true;
    btnText.textContent = 'Starting Analysis...';
    spinner.classList.remove('hidden');

    const selectedAnalysts = [...form.querySelectorAll('input[name="analysts"]:checked')]
        .map(cb => cb.value);

    if (!selectedAnalysts.length) {
        alert('Please select at least one analyst');
        submitBtn.disabled = false;
        btnText.textContent = '🚀 Start Analysis';
        spinner.classList.add('hidden');
        return;
    }

    const requestData = {
        ticker:          formData.get('ticker').toUpperCase(),
        analysis_date:   formData.get('analysis_date') || null,
        analysts:        selectedAnalysts,
        research_depth:  parseInt(formData.get('research_depth')),
        deep_think_llm:  formData.get('deep_think_llm'),
        quick_think_llm: formData.get('quick_think_llm'),
        max_debate_rounds: parseInt(formData.get('research_depth')),
        data_vendors: {
            core_stock_apis:          "vnstock",
            technical_indicators:     "vnstock",
            fundamental_data:         "vnstock",
            news_data:                "google",
            global_data:              "vietstock",
            insider_transaction_data: "yfinance",
        },
    };

    try {
        const resp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            alert(`Failed to start analysis: ${err.error || 'Unknown error'}`);
            submitBtn.disabled = false;
            btnText.textContent = '🚀 Start Analysis';
            spinner.classList.add('hidden');
            return;
        }

        const data = await resp.json();
        currentSessionId       = data.session_id;
        activePollingSessionId = data.session_id;

        switchSection('new-analysis');

        document.getElementById('analysis-form-card').classList.add('hidden');
        document.getElementById('analysis-progress-card').classList.remove('hidden');

        const stopBtn = document.getElementById('stop-analysis-btn');
        if (stopBtn) { stopBtn.classList.remove('hidden'); stopBtn.disabled = false; }

        document.getElementById('session-id').textContent     = currentSessionId;
        document.getElementById('current-ticker').textContent = requestData.ticker;

        startStatusPolling();

    } catch (error) {
        console.error('Error starting analysis:', error);
        alert(`Failed to start analysis: ${error.message}`);
        submitBtn.disabled = false;
        btnText.textContent = '🚀 Start Analysis';
        spinner.classList.add('hidden');
    }
}

function startStatusPolling() {
    clearStatusPolling();
    if (!activePollingSessionId || isReviewingHistoricalSession) return;
    checkAnalysisStatus();
    statusCheckInterval = setInterval(checkAnalysisStatus, 2000);
}

async function checkAnalysisStatus() {
    if (!activePollingSessionId || isReviewingHistoricalSession) return;
    try {
        const resp = await fetch(`/api/status/${activePollingSessionId}`);
        if (!resp.ok) throw new Error('Failed to fetch status');
        const data = await resp.json();
        updateProgressUI(data);
        if (['completed', 'error', 'cancelled'].includes(data.status)) {
            clearInterval(statusCheckInterval);
            statusCheckInterval    = null;
            activePollingSessionId = null;
            onAnalysisComplete(data);
        }
    } catch (err) {
        console.error('Error checking status:', err);
    }
}

// ── Progress UI update ─────────────────────────────────────────────────────

function updateProgressUI(data) {
    const statusBadge = document.getElementById('current-status');
    statusBadge.textContent = data.status;
    statusBadge.className   = `status-badge ${data.status}`;

    const fallback = {
        completed: '✅ Hoàn thành phân tích',
        cancelled: '🛑 Đã hủy phân tích theo yêu cầu',
        error:     '❌ Phân tích thất bại',
    }[data.status] || 'Đang xử lý...';

    const step    = data.current_step || fallback;
    const percent = Number.isFinite(data.progress_percent)
        ? Math.max(0, Math.min(100, data.progress_percent))
        : (data.status === 'completed' ? 100 : 0);

    const stepEl = document.getElementById('current-step');
    if (stepEl) stepEl.textContent = `${step} (${percent}%)`;

    const fill = document.getElementById('analysis-progress-fill');
    if (fill) fill.style.width = `${percent}%`;

    const stopBtn = document.getElementById('stop-analysis-btn');
    if (stopBtn) {
        const canStop = ['running', 'initializing'].includes(data.status);
        stopBtn.classList.toggle('hidden', !canStop);
        stopBtn.disabled = !canStop;
    }

    if (data.status === 'error' && data.error) {
        document.getElementById('current-report').innerHTML = `
            <div class="error-message">
                <h3>❌ Error</h3>
                <p><strong>Error:</strong> ${data.error}</p>
                ${data.error_details
                    ? `<details><summary>View Details</summary><pre>${data.error_details}</pre></details>`
                    : ''}
            </div>`;
    }

    updateAgentStatus(data.agent_status);

    const reportDiv = document.getElementById('current-report');
    if (data.current_report) {
        reportDiv.innerHTML = marked.parse(sanitizeReportText(data.current_report));
    } else if (data.status === 'completed' && data.final_report) {
        reportDiv.innerHTML = marked.parse(sanitizeReportText(data.final_report));
    }

    const finalDiv = document.getElementById('final-report');
    if (finalDiv && data.final_report) {
        finalDiv.innerHTML = marked.parse(sanitizeReportText(data.final_report));
    }

    if (data.decision) updateDecision(data.decision);
}

// ── Agent status ───────────────────────────────────────────────────────────

const AGENT_ORDER = [
    'Market Analyst', 'Social Analyst', 'News Analyst', 'Fundamentals Analyst',
    'Bull Researcher', 'Bear Researcher', 'Research Manager', 'Trader',
    'AlphaGPT Analyst', 'Risky Analyst', 'Safe Analyst', 'Neutral Analyst',
    'Portfolio Manager',
];

function updateAgentStatus(agentStatus) {
    const container = document.getElementById('agent-status');
    if (!agentStatus || !Object.keys(agentStatus).length) {
        container.innerHTML = '<p>No agent status available</p>';
        return;
    }
    container.innerHTML = '';

    const ordered = [
        ...AGENT_ORDER.filter(a => a in agentStatus).map(a => [a, agentStatus[a]]),
        ...Object.entries(agentStatus).filter(([a]) => !AGENT_ORDER.includes(a)),
    ];

    for (const [agent, status] of ordered) {
        const div = document.createElement('div');
        div.className = `agent-item ${status}`;
        div.innerHTML = `
            <div class="agent-item-name">${agent}</div>
            <div class="agent-item-status">${getStatusIcon(status)} ${status.replace('_', ' ')}</div>`;
        container.appendChild(div);
    }
}

function getStatusIcon(status) {
    return { pending: '⏳', in_progress: '⚙️', completed: '✅', not_selected: '⏭️' }[status] || '⏳';
}

// ── Decision ───────────────────────────────────────────────────────────────

function buildDecisionHtml(decision) {
    if (typeof decision === 'string') {
        const upper = decision.toUpperCase();
        const type  = upper.includes('BUY') ? 'BUY' : upper.includes('SELL') ? 'SELL' : 'HOLD';
        const cls   = type.toLowerCase();
        return `
            <div class="decision-highlight decision-${cls}">
                <div class="decision-icon">${cls==='buy'?'📈':cls==='sell'?'📉':'⏸️'}</div>
                <div class="decision-text"><h2>${type}</h2><p>${decision}</p></div>
            </div>`;
    }
    if (typeof decision === 'object' && decision !== null) {
        const str  = safeStringify(decision).toUpperCase();
        const type = str.includes('BUY') ? 'BUY' : str.includes('SELL') ? 'SELL' : 'HOLD';
        const cls  = type.toLowerCase();
        let html = `
            <div class="decision-highlight decision-${cls}">
                <div class="decision-icon">${cls==='buy'?'📈':cls==='sell'?'📉':'⏸️'}</div>
                <div class="decision-text"><h2>${type}</h2></div>
            </div>
            <div class="decision-details">`;
        for (const [k, v] of Object.entries(decision)) {
            html += `<div class="decision-item"><strong>${formatKey(k)}:</strong> ${formatValue(v)}</div>`;
        }
        return html + '</div>';
    }
    return '<p>Chưa có quyết định.</p>';
}

function updateDecision(decision, targetId = 'final-decision') {
    const el = document.getElementById(targetId);
    if (el) el.innerHTML = buildDecisionHtml(decision);
}

function formatKey(key)   { return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '); }
function formatValue(val) { return typeof val === 'object' && val !== null ? safeStringify(val) : val; }

// ── Analysis complete ──────────────────────────────────────────────────────

function onAnalysisComplete() {
    activePollingSessionId = null;
    document.getElementById('new-analysis-btn').classList.remove('hidden');
    loadSessions();
}

function startNewAnalysis() {
    clearStatusPolling();
    currentSessionId             = null;
    isReviewingHistoricalSession = false;
    activePollingSessionId       = null;

    document.getElementById('analysis-form').reset();
    document.getElementById('analysis-progress-card').classList.add('hidden');
    document.getElementById('new-analysis-btn').classList.add('hidden');

    const stopBtn = document.getElementById('stop-analysis-btn');
    if (stopBtn) { stopBtn.classList.add('hidden'); stopBtn.disabled = false; }

    document.getElementById('analysis-form-card').classList.remove('hidden');

    const submitBtn = document.querySelector('#analysis-form button[type="submit"]');
    submitBtn.querySelector('.btn-text').textContent = '🚀 Start Analysis';
    submitBtn.querySelector('.spinner').classList.add('hidden');
    submitBtn.disabled = false;

    clearAnalysisDisplay();
}

async function cancelCurrentAnalysis() {
    if (!activePollingSessionId) return;
    if (!confirm('Hủy phiên phân tích hiện tại?')) return;
    try {
        const resp = await fetch(`/api/sessions/${activePollingSessionId}/cancel`, { method: 'POST' });
        if (!resp.ok) throw new Error('Failed to cancel');
        document.getElementById('stop-analysis-btn').disabled = true;
        await checkAnalysisStatus();
    } catch (err) {
        console.error('Error cancelling:', err);
        alert('Không thể hủy. Vui lòng thử lại.');
    }
}

// ── Session list ───────────────────────────────────────────────────────────

async function loadSessions() {
    try {
        const resp = await fetch('/api/sessions');
        if (!resp.ok) throw new Error('Failed to load sessions');
        const data = await resp.json();
        displaySessions(data.sessions);
    } catch (err) {
        console.error('Error loading sessions:', err);
        document.getElementById('sessions-list').innerHTML = '<p>Failed to load sessions</p>';
    }
}

function displaySessions(sessions) {
    const container = document.getElementById('sessions-list');
    const counter   = document.getElementById('completed-analyses-count');
    if (counter) {
        counter.textContent = String((sessions || []).filter(s => s.status === 'completed').length);
    }
    if (!sessions || !sessions.length) {
        container.innerHTML = '<p>No previous sessions</p>';
        return;
    }

    sessions.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    container.innerHTML = '';
    sessions.forEach(session => {
        const div = document.createElement('div');
        div.className = 'session-item';
        div.innerHTML = `
            <div class="session-item-info">
                <div class="session-item-ticker">📊 ${session.ticker}</div>
                <div class="session-item-meta">
                    ${session.analysis_date} | ${session.status} | ${formatDateTime(session.created_at)}
                </div>
            </div>
            <div class="session-item-actions">
                <button class="btn btn-secondary btn-small"
                    onclick="viewSession('${session.session_id}')">View</button>
                <button class="btn btn-danger btn-small"
                    onclick="deleteSession('${session.session_id}')">Delete</button>
            </div>`;
        container.appendChild(div);
    });
}

function formatDateTime(dateString) {
    return new Date(dateString).toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

// ── View / Delete session ──────────────────────────────────────────────────

function renderSessionReview(sessionId, data) {
    openSessionReviewModal();

    document.getElementById('review-session-id').textContent = sessionId;
    document.getElementById('review-ticker').textContent     = sessionId.split('_')[0] || '-';

    const statusEl = document.getElementById('review-status');
    statusEl.textContent = data.status || '-';
    statusEl.className   = `status-badge ${data.status || ''}`;

    const currentEl = document.getElementById('review-current-report');
    if (currentEl) {
        currentEl.innerHTML = data.current_report
            ? marked.parse(sanitizeReportText(data.current_report))
            : 'Không có báo cáo tạm thời.';
    }

    const finalEl = document.getElementById('review-final-report');
    if (finalEl) {
        finalEl.innerHTML = data.final_report
            ? marked.parse(sanitizeReportText(data.final_report))
            : 'Chưa có báo cáo tổng hợp.';
    }

    const decisionEl = document.getElementById('review-final-decision');
    if (decisionEl) {
        if (data.decision) updateDecision(data.decision, 'review-final-decision');
        else decisionEl.innerHTML = '<p>Chưa có quyết định.</p>';
    }
}

async function viewSession(sessionId) {
    clearStatusPolling();
    isReviewingHistoricalSession = true;
    activePollingSessionId       = null;
    currentSessionId             = sessionId;

    switchSection('sessions');
    try {
        const resp = await fetch(`/api/status/${sessionId}`);
        if (!resp.ok) throw new Error('Failed to fetch session');
        renderSessionReview(sessionId, await resp.json());
    } catch (err) {
        console.error('Error viewing session:', err);
        alert('Failed to load session');
    }
}

async function deleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this session?')) return;
    try {
        const resp = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Failed to delete session');

        // Nếu đang xem session này trong modal thì đóng modal
        if (isReviewingHistoricalSession && currentSessionId === sessionId) {
            currentSessionId             = null;
            isReviewingHistoricalSession = false;
            closeSessionReviewModal();
        }

        // Refresh danh sách
        loadSessions();
    } catch (err) {
        console.error('Error deleting session:', err);
        alert('Failed to delete session');
    }
}