/**
 * alpha_panel.js
 * Quản lý section "Alpha" trong sidebar:
 *   - Tab 1: Library — hiển thị danh sách alpha từ alpha_library.json
 *   - Tab 2: Pipeline — chạy sinh alpha, stream log lên UI
 */

// ── Library ───────────────────────────────────────────────────────────────

let _libraryLoaded = false;

export async function loadAlphaLibrary(forceReload = false) {
    if (_libraryLoaded && !forceReload) return;
    const container = document.getElementById('alpha-library-body');
    const stats     = document.getElementById('alpha-library-stats');
    if (!container) return;

    container.innerHTML = '<tr><td colspan="6" class="alpha-loading">Đang tải...</td></tr>';
    try {
        const res  = await fetch('/api/alpha/library');
        const data = await res.json();

        stats.textContent = `${data.total} alpha • sorted by IC`;
        if (!data.alphas || data.alphas.length === 0) {
            container.innerHTML = '<tr><td colspan="6" class="alpha-loading">Chưa có alpha nào trong library.</td></tr>';
            return;
        }

        container.innerHTML = data.alphas.map((a, i) => {
            const ic     = a.ic_oos   != null ? a.ic_oos.toFixed(4)       : '—';
            const sh     = a.sharpe_oos != null ? a.sharpe_oos.toFixed(3) : '—';
            const ret    = a.return_oos != null ? (a.return_oos * 100).toFixed(1) + '%' : '—';
            const icCls  = a.ic_oos != null && a.ic_oos > 0 ? 'metric-pos' : 'metric-neg';
            return `
            <tr class="alpha-row" title="${escHtml(a.formula)}">
                <td class="alpha-rank">#${i + 1}</td>
                <td class="alpha-id"><code>${escHtml(a.id)}</code></td>
                <td class="alpha-desc">${escHtml(a.description || '—')}</td>
                <td class="alpha-metric ${icCls}">${ic}</td>
                <td class="alpha-metric">${sh}</td>
                <td class="alpha-metric">${ret}</td>
            </tr>
            <tr class="alpha-expr-row">
                <td colspan="6"><code class="alpha-expr-code">${escHtml(a.formula)}</code></td>
            </tr>`;
        }).join('');

        _libraryLoaded = true;
    } catch (err) {
        container.innerHTML = `<tr><td colspan="6" class="alpha-loading alpha-error">Lỗi: ${err.message}</td></tr>`;
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────

let _pipelineRunning = false;
let _currentEventSource = null;

export function runAlphaPipeline() {
    if (_pipelineRunning) return;

    const ideaInput  = document.getElementById('alpha-idea-input');
    const iterInput  = document.getElementById('alpha-iterations-input');
    const logEl      = document.getElementById('alpha-pipeline-log');
    const runBtn     = document.getElementById('alpha-run-btn');
    const statusEl   = document.getElementById('alpha-pipeline-status');

    const idea       = ideaInput ? ideaInput.value.trim() : '';
    const iterations = iterInput ? Math.max(1, Math.min(10, parseInt(iterInput.value) || 3)) : 3;

    logEl.textContent = '';
    _appendLog(logEl, idea
        ? `▶ Bắt đầu pipeline — idea: "${idea}" — ${iterations} vòng`
        : `▶ Bắt đầu pipeline — idea: [auto-generate] — ${iterations} vòng`
    );

    _setPipelineRunning(true, runBtn, statusEl);

    const params = new URLSearchParams({ iterations });
    if (idea) params.set('idea', idea);

    const url = `/api/alpha/pipeline/run?${params}`;
    const es  = new EventSource(url);
    _currentEventSource = es;

    es.onmessage = (e) => {
        const text = e.data;
        if (text === '__DONE__') {
            _setPipelineRunning(false, runBtn, statusEl);
            _appendLog(logEl, '✓ Xong.');
            es.close();
            _currentEventSource = null;
            // Reload library sau khi pipeline xong
            _libraryLoaded = false;
            if (_isLibraryTabActive()) loadAlphaLibrary(true);
            return;
        }
        _appendLog(logEl, text);
    };

    es.onerror = () => {
        _setPipelineRunning(false, runBtn, statusEl);
        _appendLog(logEl, '✗ Kết nối bị ngắt hoặc pipeline lỗi.');
        es.close();
        _currentEventSource = null;
    };
}

export function stopAlphaPipeline() {
    if (_currentEventSource) {
        _currentEventSource.close();
        _currentEventSource = null;
    }
    const runBtn   = document.getElementById('alpha-run-btn');
    const statusEl = document.getElementById('alpha-pipeline-status');
    _setPipelineRunning(false, runBtn, statusEl);
    const logEl = document.getElementById('alpha-pipeline-log');
    if (logEl) _appendLog(logEl, '⏹ Đã dừng theo yêu cầu.');
}

// ── Tab switching ─────────────────────────────────────────────────────────

export function switchAlphaTab(tab) {
    document.querySelectorAll('#alpha-section .alpha-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('#alpha-section .alpha-tab-pane').forEach(p => p.classList.remove('active'));
    const btn  = document.querySelector(`#alpha-section .alpha-tab-btn[data-tab="${tab}"]`);
    const pane = document.getElementById(`alpha-tab-${tab}`);
    if (btn)  btn.classList.add('active');
    if (pane) pane.classList.add('active');
    if (tab === 'library') loadAlphaLibrary();
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _appendLog(el, text) {
    const line = document.createElement('div');
    line.className = 'log-line';

    // Màu theo loại dòng
    if (text.includes('[OK]'))          line.classList.add('log-ok');
    else if (text.includes('[WEAK]'))   line.classList.add('log-weak');
    else if (text.includes('[ERROR]') || text.includes('ERROR')) line.classList.add('log-err');
    else if (text.includes('==='))      line.classList.add('log-header');
    else if (text.startsWith('  ['))    line.classList.add('log-indent');

    line.textContent = text;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
}

function _setPipelineRunning(running, runBtn, statusEl) {
    _pipelineRunning = running;
    if (runBtn) {
        runBtn.disabled    = running;
        runBtn.textContent = running ? 'Đang chạy...' : 'Chạy Pipeline';
    }
    if (statusEl) {
        statusEl.textContent  = running ? 'running' : 'idle';
        statusEl.className    = 'pipeline-status-badge ' + (running ? 'running' : 'idle');
    }
    const stopBtn = document.getElementById('alpha-stop-btn');
    if (stopBtn) stopBtn.style.display = running ? 'inline-flex' : 'none';
}

function _isLibraryTabActive() {
    const pane = document.getElementById('alpha-tab-library');
    return pane && pane.classList.contains('active');
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}