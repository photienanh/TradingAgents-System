/**
 * analysis.js
 * Analysis flow: start, status polling, UI updates, cancel.
 */
import { renderReport, buildDecisionHtml } from './report_renderer.js';
import { updateAgentStatus } from './agent_status.js';
import { loadSessions } from './sessions.js';

const START_ANALYSIS_HTML = '<i class="fa-solid fa-play" aria-hidden="true"></i> Start Analysis';

let currentSessionId       = null;
let statusCheckInterval    = null;
let activePollingSessionId = null;

export function clearStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

export function getActivePollingSessionId() { return activePollingSessionId; }

// ── Status badge ──────────────────────────────────────────────────────────

function getStatusBadgeMeta(status) {
    const s = String(status || '').toLowerCase();
    const map = {
        initializing: { icon: 'fa-solid fa-hourglass-start', label: 'initializing' },
        running:      { icon: 'fa-solid fa-spinner fa-spin',   label: 'running' },
        completed:    { icon: 'fa-solid fa-circle-check',      label: 'completed' },
        cancelled:    { icon: 'fa-solid fa-ban',               label: 'cancelled' },
        error:        { icon: 'fa-solid fa-triangle-exclamation', label: 'error' },
    };
    return map[s] || { icon: 'fa-solid fa-circle-info', label: status || '-' };
}

// ── Start analysis ────────────────────────────────────────────────────────

export async function startAnalysis() {
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

    const selectedAnalysts = [...form.querySelectorAll('input[name="analysts"]:checked')].map(cb => cb.value);
    if (!selectedAnalysts.length) {
        alert('Please select at least one analyst');
        submitBtn.disabled = false;
        btnText.innerHTML = START_ANALYSIS_HTML;
        spinner.classList.add('hidden');
        return;
    }

    const requestData = {
        ticker:           formData.get('ticker').toUpperCase(),
        analysts:         selectedAnalysts,
        trading_horizon:  formData.get('trading_horizon') || 'short',
        research_depth:   parseInt(formData.get('research_depth')),
        deep_think_llm:   formData.get('deep_think_llm'),
        quick_think_llm:  formData.get('quick_think_llm'),
        max_debate_rounds: parseInt(formData.get('research_depth')),
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
            btnText.innerHTML = START_ANALYSIS_HTML;
            spinner.classList.add('hidden');
            return;
        }

        const data = await resp.json();
        currentSessionId       = data.session_id;
        activePollingSessionId = data.session_id;

        if (typeof window.switchSection === 'function') window.switchSection('new-analysis');

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
        btnText.innerHTML = START_ANALYSIS_HTML;
        spinner.classList.add('hidden');
    }
}

// ── Status polling ────────────────────────────────────────────────────────

function startStatusPolling() {
    clearStatusPolling();
    if (!activePollingSessionId) return;
    checkAnalysisStatus();
    statusCheckInterval = setInterval(checkAnalysisStatus, 2000);
}

async function checkAnalysisStatus() {
    if (!activePollingSessionId) return;
    try {
        const resp = await fetch(`/api/status/${activePollingSessionId}`);
        if (!resp.ok) throw new Error('Failed to fetch status');
        const data = await resp.json();
        await updateProgressUI(data);
        if (['completed', 'error', 'cancelled'].includes(data.status)) {
            clearInterval(statusCheckInterval);
            statusCheckInterval    = null;
            activePollingSessionId = null;
            onAnalysisComplete();
        }
    } catch (err) {
        console.error('Error checking status:', err);
    }
}

// ── UI update ─────────────────────────────────────────────────────────────

async function updateProgressUI(data) {
    const statusBadge = document.getElementById('current-status');
    const meta = getStatusBadgeMeta(data.status);
    statusBadge.innerHTML = `<i class="${meta.icon}" aria-hidden="true"></i> ${meta.label}`;
    statusBadge.className = `status-badge ${data.status}`;

    const fallback = { completed: 'Hoàn thành phân tích', cancelled: 'Đã hủy phân tích theo yêu cầu', error: 'Phân tích thất bại' }[data.status] || 'Đang xử lý...';
    const step    = String(data.current_step || fallback).trim();
    const percent = Number.isFinite(data.progress_percent) ? Math.max(0, Math.min(100, data.progress_percent)) : (data.status === 'completed' ? 100 : 0);

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
                <h3><i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> Error</h3>
                <p><strong>Error:</strong> ${data.error}</p>
                ${data.error_details ? `<details><summary>View Details</summary><pre>${data.error_details}</pre></details>` : ''}
            </div>`;
    }

    updateAgentStatus(data.agent_status, 'agent-status');

    const reportDiv = document.getElementById('current-report');
    if (data.current_report) {
        reportDiv.innerHTML = await renderReport(data.current_report);
    } else if (data.status === 'completed' && data.final_report) {
        reportDiv.innerHTML = await renderReport(data.final_report);
    }

    const finalDiv = document.getElementById('final-report');
    if (finalDiv && data.final_report) finalDiv.innerHTML = await renderReport(data.final_report);

    if (data.decision) {
        const decEl = document.getElementById('final-decision');
        if (decEl) decEl.innerHTML = buildDecisionHtml(data.decision);
    }
}

// ── Analysis complete / new / cancel ─────────────────────────────────────

function onAnalysisComplete() {
    activePollingSessionId = null;
    document.getElementById('new-analysis-btn')?.classList.remove('hidden');
    loadSessions();
}

export function startNewAnalysis() {
    clearStatusPolling();
    currentSessionId       = null;
    activePollingSessionId = null;

    document.getElementById('analysis-form').reset();
    window._setHorizon('short', document.querySelector('.horizon-btn[data-horizon="short"]'));
    document.getElementById('analysis-progress-card').classList.add('hidden');
    document.getElementById('new-analysis-btn')?.classList.add('hidden');

    const stopBtn = document.getElementById('stop-analysis-btn');
    if (stopBtn) { stopBtn.classList.add('hidden'); stopBtn.disabled = false; }

    document.getElementById('analysis-form-card').classList.remove('hidden');

    const submitBtn = document.querySelector('#analysis-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.querySelector('.btn-text').innerHTML = START_ANALYSIS_HTML;
        submitBtn.querySelector('.spinner').classList.add('hidden');
        submitBtn.disabled = false;
    }

    // Clear displays
    const fields = {
        'session-id':     '-', 'current-ticker':  '-',
        'current-step':   'Đang chờ bắt đầu phân tích...',
        'current-report': 'Waiting for analysis to start...',
        'final-report':   '', 'final-decision':   '',
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

export async function cancelCurrentAnalysis() {
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