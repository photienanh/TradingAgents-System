/**
 * app.js — Entrypoint, wires modules together.
 *
 * Modules:
 *   report_renderer.js  – markdown → HTML, decision badges
 *   agent_status.js     – agent flow board rendering
 *   analysis.js         – start/poll/cancel analysis
 *   sessions.js         – session list, view, delete
 */
import { startAnalysis, startNewAnalysis, cancelCurrentAnalysis, clearStatusPolling } from './analysis.js';
import { loadSessions, viewSession, deleteSession, closeSessionReviewModal } from './sessions.js';
import { loadAlphaLibrary, runAlphaPipeline, stopAlphaPipeline, switchAlphaTab } from './alpha_panel.js';

// ── Expose to inline onclick handlers (HTML uses window.* calls) ──────────
window._viewSession   = viewSession;
window._deleteSession = deleteSession;
window._loadAlphaLibrary  = (force) => loadAlphaLibrary(force);
window._runAlphaPipeline  = runAlphaPipeline;
window._stopAlphaPipeline = stopAlphaPipeline;
window._switchAlphaTab    = switchAlphaTab;

// ── Tab setup ─────────────────────────────────────────────────────────────
function setupTabs(containerEl) {
    const btns  = containerEl.querySelectorAll('.tab-btn');
    const panes = containerEl.querySelectorAll('.tab-pane');
    btns.forEach(btn => {
        btn.addEventListener('click', function () {
            const tab = this.dataset.tab;
            btns.forEach(b  => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));
            this.classList.add('active');
            const pane = containerEl.querySelector('#tab-' + tab);
            if (pane) pane.classList.add('active');
        });
    });
}

// ── Section navigation ────────────────────────────────────────────────────
window.switchSection = function switchSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item[data-section]').forEach(n => n.classList.remove('active'));
    const sec = document.getElementById(name + '-section');
    if (sec) sec.classList.add('active');
    const nav = document.querySelector('[data-section="' + name + '"]');
    if (nav) nav.classList.add('active');
    const titles = {
        home: 'Home',
        'new-analysis': 'New Analysis',
        sessions: 'Sessions',
        alpha: 'Alpha'
    };
    const el = document.getElementById('page-title');
    if (el) el.textContent = titles[name] || name;

    // Entering Alpha section should always show and refresh the library tab state.
    if (name === 'alpha') switchAlphaTab('library');
};

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Setup tabs in progress card and review modal
    const mainProgress = document.getElementById('analysis-progress-card');
    if (mainProgress) setupTabs(mainProgress);
    const reviewModal = document.getElementById('session-review-modal');
    if (reviewModal) setupTabs(reviewModal);

    // Nav items
    document.querySelectorAll('.nav-item[data-section]').forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            window.switchSection(this.dataset.section);
        });
    });

    // Hamburger (mobile)
    const menuToggle = document.getElementById('menu-toggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            document.querySelector('.sidebar').classList.toggle('active');
        });
    }

    // Analysis form
    document.getElementById('analysis-form')
        ?.addEventListener('submit', async e => { e.preventDefault(); await startAnalysis(); });

    document.getElementById('new-analysis-btn')
        ?.addEventListener('click', startNewAnalysis);

    document.getElementById('stop-analysis-btn')
        ?.addEventListener('click', cancelCurrentAnalysis);

    // Modal close
    document.getElementById('close-session-review-btn')
        ?.addEventListener('click', closeSessionReviewModal);
    document.getElementById('session-review-backdrop')
        ?.addEventListener('click', closeSessionReviewModal);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeSessionReviewModal();
    });

    // Sidebar sidebar hover expansion on mobile via toggle
    document.querySelector('.sidebar')?.addEventListener('mouseenter', () => {
        document.querySelector('.main-wrapper')?.classList.add('expanded');
    });
    document.querySelector('.sidebar')?.addEventListener('mouseleave', () => {
        document.querySelector('.main-wrapper')?.classList.remove('expanded');
    });

    // Initial state
    window.switchSection('home');
    loadSessions();

    // Sync hidden counter → visible home counter
    const src = document.getElementById('completed-analyses-count');
    const dst = document.getElementById('home-analyses-count');
    if (src && dst) {
        const obs = new MutationObserver(() => { dst.textContent = src.textContent; });
        obs.observe(src, { childList: true, characterData: true, subtree: true });
    }
});