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
                <td class="alpha-metric" style="width:36px;text-align:center;">
                    <button class="alpha-delete-btn" title="Delete this alpha" onclick="window._deleteAlpha('${escHtml(a.id)}')">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </td>
            </tr>
            <tr class="alpha-expr-row">
                <td colspan="7"><code class="alpha-expr-code">${escHtml(a.formula)}</code></td>
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
    if (tab === 'signals') loadAlphaSignals();
}

export async function deleteAlpha(alphaId) {
    if (!confirm(`Delete this alpha "${alphaId}" from library?`)) return;
    try {
        const res = await fetch(`/api/alpha/library/${encodeURIComponent(alphaId)}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json();
            alert('Lỗi: ' + (err.error || res.status));
            return;
        }
        _libraryLoaded = false;
        loadAlphaLibrary(true);
    } catch (e) {
        alert('Lỗi kết nối: ' + e.message);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _appendLog(el, text) {
    const line = document.createElement('div');
    line.className = 'log-line';

    // Màu theo loại dòng
    if (text.includes('[OK]'))          line.classList.add('log-ok');
    else if (text.includes('[WEAK]'))   line.classList.add('log-weak');
    else if (text.includes('[ERR]') || text.includes('[ERROR]') || text.includes('ERROR')) line.classList.add('log-err');
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

// ── Alpha Signals Tab ─────────────────────────────────────────────────────

let _signalsLoaded  = false;
let _signalsData    = [];
let _signalsFilter  = 'all';
let _signalsSortKey = null;
let _signalsSortAsc = true;

export async function loadAlphaSignals(forceReload = false) {
    if (_signalsLoaded && !forceReload) return;
    const body   = document.getElementById('alpha-signals-body');
    const stats  = document.getElementById('alpha-signals-stats');
    const asofEl = document.getElementById('alpha-signals-asof');
    if (!body) return;

    body.innerHTML = '<tr><td colspan="8" class="alpha-loading">Đang tải...</td></tr>';
    try {
        const res  = await fetch('/api/alpha/signals');
        const data = await res.json();
        _signalsData   = data.signals || [];
        _signalsLoaded = true;

        if (stats) stats.textContent = `${data.total} mã`;
        if (asofEl && data.last_run_day) {
            const t = data.as_of ? new Date(data.as_of).toLocaleTimeString('vi-VN') : '';
            asofEl.textContent = `as of ${data.last_run_day}${t ? ' ' + t : ''}`;
        }
        _renderSignals();
    } catch (err) {
        body.innerHTML = `<tr><td colspan="8" class="alpha-loading alpha-error">Lỗi: ${err.message}</td></tr>`;
    }
}

export function filterSignals(filter, btnEl) {
    _signalsFilter = filter;
    document.querySelectorAll('.signals-filter-btn').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    _renderSignals();
}

export function sortSignals(key) {
    if (_signalsSortKey === key) {
        _signalsSortAsc = !_signalsSortAsc;
    } else {
        _signalsSortKey = key;
        _signalsSortAsc = true;
        ['ticker', 'signal'].forEach(k => {
            const el = document.getElementById(`sort-${k}-icon`);
            if (el) el.textContent = '↕';
        });
    }
    const icon = document.getElementById(`sort-${key}-icon`);
    if (icon) icon.textContent = _signalsSortAsc ? '↑' : '↓';
    _renderSignals();
}

function _renderSignals() {
    const body = document.getElementById('alpha-signals-body');
    if (!body) return;

    let filtered = _signalsFilter === 'all'
        ? [..._signalsData]
        : _signalsData.filter(s => s.side === _signalsFilter);

    if (_signalsSortKey === 'ticker') {
        filtered.sort((a, b) => {
            const cmp = (a.ticker || '').localeCompare(b.ticker || '');
            return _signalsSortAsc ? cmp : -cmp;
        });
    } else if (_signalsSortKey === 'signal') {
        filtered.sort((a, b) => {
            const av = a.signal_today ?? -Infinity;
            const bv = b.signal_today ?? -Infinity;
            return _signalsSortAsc ? av - bv : bv - av;
        });
    }

    if (!filtered.length) {
        body.innerHTML = '<tr><td colspan="5" class="alpha-loading">Không có dữ liệu.</td></tr>';
        return;
    }

    body.innerHTML = filtered.map((s, i) => {
        const sig    = s.signal_today != null ? s.signal_today.toFixed(4) : '—';
        const asof   = s.as_of ? s.as_of.slice(0, 10) : '—';
        const sideKey = (s.side || 'neutral').replace(/ /g, '-');
        const sigCls  = s.signal_today > 0 ? 'metric-pos' : s.signal_today < 0 ? 'metric-neg' : '';
        return `<tr>
            <td class="alpha-rank">${i + 1}</td>
            <td style="font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.82rem;">${escHtml(s.ticker)}</td>
            <td class="alpha-metric ${sigCls}" style="text-align:center;">${sig}</td>
            <td style="text-align:center;"><span class="side-badge side-${sideKey}">${escHtml(s.side || 'neutral')}</span></td>
            <td class="alpha-metric" style="font-size:0.72rem;color:var(--text-muted);">${asof}</td>
        </tr>`;
    }).join('');
}

// ── Data Refresh ──────────────────────────────────────────────────────────

let _refreshPollingTimer = null;

export async function triggerDataRefresh() {
    const btn      = document.getElementById('alpha-refresh-btn');
    const labelEl  = document.getElementById('alpha-refresh-label');

    if (btn)    { btn.disabled = true; btn.classList.add('running'); }
    if (labelEl) labelEl.textContent = 'Đang cập nhật...';

    try {
        const res  = await fetch('/api/alpha/refresh', { method: 'POST' });
        const data = await res.json();
        if (!data.accepted) {
            _setRefreshIdle(btn, labelEl, data.message || 'Không thể chạy');
            return;
        }
        _pollRefreshStatus(btn, labelEl);
    } catch (e) {
        _setRefreshIdle(btn, labelEl, 'Lỗi kết nối');
    }
}

function _pollRefreshStatus(btn, labelEl) {
    if (_refreshPollingTimer) clearInterval(_refreshPollingTimer);
    _refreshPollingTimer = setInterval(async () => {
        try {
            const res  = await fetch('/api/alpha/status');
            const data = await res.json();
            if (!data.running) {
                clearInterval(_refreshPollingTimer);
                _refreshPollingTimer = null;
                const lastAt = data.last_run_at
                    ? new Date(data.last_run_at).toLocaleTimeString('vi-VN')
                    : '';
                _setRefreshIdle(btn, labelEl, lastAt ? `Cập nhật lúc ${lastAt}` : 'Cập nhật dữ liệu');
                _signalsLoaded = false;
                _libraryLoaded = false;
            }
        } catch (_) {}
    }, 2000);
}

function _setRefreshIdle(btn, labelEl, message = 'Cập nhật dữ liệu') {
    if (btn)    { btn.disabled = false; btn.classList.remove('running'); }
    if (labelEl) labelEl.textContent = message;
}