/**
 * sessions.js
 * Session list management, view and delete.
 */
import { renderReport } from './report_renderer.js';
import { buildDecisionHtml } from './report_renderer.js';
import { clearStatusPolling } from './analysis.js';

let _isReviewingHistoricalSession = false;
let _currentSessionId = null;

export function setCurrentSession(id) { _currentSessionId = id; }
export function getCurrentSession()  { return _currentSessionId; }
export function isReviewing()        { return _isReviewingHistoricalSession; }

// ── Status badge helpers ──────────────────────────────────────────────────

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

function formatDateTime(dateString) {
    return new Date(dateString).toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

// ── Load / display sessions ───────────────────────────────────────────────

export async function loadSessions() {
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
                <div class="session-item-ticker"><i class="fa-solid fa-chart-column" aria-hidden="true"></i> ${session.ticker}</div>
                <div class="session-item-meta">${session.analysis_date} | ${session.status} | ${formatDateTime(session.created_at)}</div>
            </div>
            <div class="session-item-actions">
                <button class="btn btn-secondary btn-small" onclick="window._viewSession('${session.session_id}')">View</button>
                <button class="btn btn-danger btn-small"    onclick="window._deleteSession('${session.session_id}')">Delete</button>
            </div>`;
        container.appendChild(div);
    });
}

// ── View / delete ─────────────────────────────────────────────────────────

export async function viewSession(sessionId) {
    clearStatusPolling();
    _isReviewingHistoricalSession = true;
    _currentSessionId = sessionId;

    // Switch to sessions section
    if (typeof window.switchSection === 'function') window.switchSection('sessions');

    try {
        const resp = await fetch(`/api/status/${sessionId}`);
        if (!resp.ok) throw new Error('Failed to fetch session');
        await renderSessionReview(sessionId, await resp.json());
    } catch (err) {
        console.error('Error viewing session:', err);
        alert('Failed to load session');
    }
}

async function renderSessionReview(sessionId, data) {
    openSessionReviewModal();

    document.getElementById('review-session-id').textContent = sessionId;
    document.getElementById('review-ticker').textContent     = sessionId.split('_')[0] || '-';

    const statusEl = document.getElementById('review-status');
    const meta = getStatusBadgeMeta(data.status);
    statusEl.innerHTML = `<i class="${meta.icon}" aria-hidden="true"></i> ${meta.label}`;
    statusEl.className = `status-badge ${data.status || ''}`;

    const currentEl = document.getElementById('review-current-report');
    if (currentEl) currentEl.innerHTML = data.current_report ? await renderReport(data.current_report) : 'Không có báo cáo tạm thời.';

    const finalEl = document.getElementById('review-final-report');
    if (finalEl) finalEl.innerHTML = data.final_report ? await renderReport(data.final_report) : 'Chưa có báo cáo tổng hợp.';

    const decisionEl = document.getElementById('review-final-decision');
    if (decisionEl) decisionEl.innerHTML = data.decision ? buildDecisionHtml(data.decision) : '<p>Chưa có quyết định.</p>';
}

export async function deleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this session?')) return;
    try {
        const resp = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Failed to delete session');

        if (_isReviewingHistoricalSession && _currentSessionId === sessionId) {
            _currentSessionId = null;
            _isReviewingHistoricalSession = false;
            closeSessionReviewModal();
        }
        loadSessions();
    } catch (err) {
        console.error('Error deleting session:', err);
        alert('Failed to delete session');
    }
}

// ── Modal ─────────────────────────────────────────────────────────────────

export function openSessionReviewModal() {
    const modal = document.getElementById('session-review-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

export function closeSessionReviewModal() {
    const modal = document.getElementById('session-review-modal');
    if (!modal) return;
    modal.classList.add('hidden');
    document.body.style.overflow = '';
}