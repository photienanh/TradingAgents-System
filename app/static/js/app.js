// app/static/js/app.js
// Tab setup đã chuyển sang index.html (setupTabs scoped per container).
// File này chỉ handle: analysis flow, session list, modal open/close.

let currentSessionId       = null;
let statusCheckInterval    = null;
let isReviewingHistoricalSession = false;
let activePollingSessionId = null;

// Keys cho các decision sections (hiển thị badge BUY/SELL/HOLD)
const DECISION_SECTION_KEYS = new Set([
    'investment_plan',
    'trader_investment_plan',
    'final_trade_decision',
    // Tên label tương ứng
    'Quyết định nhóm nghiên cứu',
    'Kế hoạch nhóm giao dịch',
    'Quyết định cuối cùng',
]);

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

let _sectionConfig = null;
 
async function loadSectionConfig() {
    if (_sectionConfig) return _sectionConfig;
    try {
        const r = await fetch('/api/config/section-titles');
        const d = await r.json();
        _sectionConfig = buildSectionConfig(d.section_titles);
    } catch {
        _sectionConfig = buildSectionConfig({
            market_report:          'Phân tích thị trường',
            sentiment_report:       'Phân tích tâm lý xã hội',
            news_report:            'Phân tích tin tức',
            fundamentals_report:    'Phân tích tài chính doanh nghiệp',
            quant_report:           'Phân tích định lượng (AlphaGPT)',
            investment_plan:        'Quyết định nhóm nghiên cứu',
            trader_investment_plan: 'Kế hoạch nhóm giao dịch',
            final_trade_decision:   'Quyết định cuối cùng',
        });
    }
    return _sectionConfig;
}
 
function buildSectionConfig(titles) {
    const KEY_COLORS = {
        market_report:          '#6366f1',
        sentiment_report:       '#f59e0b',
        news_report:            '#0ea5e9',
        fundamentals_report:    '#10b981',
        quant_report:           '#8b5cf6',
        investment_plan:        '#ef4444',
        trader_investment_plan: '#f97316',
        final_trade_decision:   '#dc2626',
    };
    // Chỉ các section này mới hiển thị badge BUY/SELL/HOLD
    const DECISION_KEYS = new Set(['investment_plan', 'trader_investment_plan', 'final_trade_decision']);
 
    const labelMap   = {};
    const labelOrder = [];
    for (const [key, label] of Object.entries(titles)) {
        labelMap[label] = {
            key,
            color:      KEY_COLORS[key] || '#6366f1',
            isDecision: DECISION_KEYS.has(key),
        };
        labelOrder.push(label);
    }
    labelOrder.sort((a, b) => b.length - a.length);
    return { labelMap, labelOrder, titles, KEY_COLORS, DECISION_KEYS };
}
 
async function renderReport(md) {
    if (!md) return '';
    const cfg = await loadSectionConfig();
    return _renderReport(md, cfg);
}
 
function _renderReport(md, cfg) {
    const text = sanitizeReportText(md);
 
    // ── Step 1: Tách ## ===== DECISION ===== ra TRƯỚC mọi thứ khác
    const decisionBlocks = [];
    let cleaned = text;
    const decisionHit = /^##\s+=====\s*/im.exec(cleaned);
    if (decisionHit) {
        const decisionPart = cleaned.slice(decisionHit.index).trim();
        cleaned            = cleaned.slice(0, decisionHit.index).trim();
        decisionPart.split(/(?=^##\s+=====)/im).forEach(blk => {
            if (blk.trim()) decisionBlocks.push(blk.trim());
        });
    }
 
    // ── Step 2: Xoá wrapper "## Báo cáo nhóm phân tích"
    cleaned = cleaned.replace(/^##\s+Báo cáo nhóm phân tích\s*\n?/im, '');

    // ── Step 2.5: Normalize legacy section labels
    // Hỗ trợ report cũ từng dùng "Phân tích cơ bản" để tránh bị dính vào section phía trên.
    const fundamentalsCanonical = cfg.titles?.fundamentals_report || 'Phân tích tài chính doanh nghiệp';
    cleaned = cleaned
        .replace(/^##\s+Phân\s+tích\s+cơ\s+bản\s*$/gim, `### ${fundamentalsCanonical}`)
        .replace(/^###\s+Phân\s+tích\s+cơ\s+bản\s*$/gim, `### ${fundamentalsCanonical}`);
 
    // ── Step 3: Normalize "## Tên section" → "### Tên section"
    cfg.labelOrder.forEach(label => {
        const esc = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        cleaned = cleaned.replace(
            new RegExp('^##\\s+(' + esc + ')\\s*$', 'gim'),
            '### $1'
        );
    });
 
    // ── Step 4: Split theo ### Section Label
    const escapedLabels = cfg.labelOrder.map(l => l.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const SPLIT_RE = new RegExp(
        '(?=^###\\s+(?:' + escapedLabels.join('|') + ')\\s*$)',
        'im'
    );
    const chunks = cleaned.split(SPLIT_RE);
 
    let html = '';
 
    for (const chunk of chunks) {
        if (!chunk.trim()) continue;
 
        let matchedLabel = null;
        for (const label of cfg.labelOrder) {
            const re = new RegExp('^###\\s+' + label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*$', 'im');
            if (re.test(chunk)) { matchedLabel = label; break; }
        }
 
        if (!matchedLabel) {
            const stripped = chunk.replace(/^#+\s*.+$/gm, '').replace(/[-\s|:]/g, '');
            if (stripped.length > 30) {
                html += `<div class="report-body" style="margin-bottom:1.5rem;">${marked.parse(chunk)}</div>`;
            }
            continue;
        }
 
        const meta    = cfg.labelMap[matchedLabel];
        const labelEsc = matchedLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const body    = chunk.replace(new RegExp('^###\\s+' + labelEsc + '\\s*\\n', 'im'), '').trim();
 
        html += renderSectionBlock(matchedLabel, body, meta.color, meta.isDecision, cfg);
    }
 
    // ── Step 5: Render decision blocks (luôn có badge)
    for (const blk of decisionBlocks) {
        const titleMatch = blk.match(/^##\s+=====\s*(.+?)\s*=====\s*$/im);
        if (!titleMatch) continue;
        const title = titleMatch[1].trim();
        const body  = blk.replace(/^##\s+=====.+=====\s*\n/im, '').trim();
 
        let color = '#ef4444';
        for (const [label, meta] of Object.entries(cfg.labelMap)) {
            if (title.toLowerCase().includes(label.toLowerCase().slice(0, 8))) {
                color = meta.color; break;
            }
        }
        if (/cuối cùng/i.test(title)) color = '#dc2626';
        else if (/giao dịch/i.test(title)) color = '#f97316';
 
        html += renderSectionBlock(title, body, color, true, cfg);
    }
 
    return html || '<p style="color:var(--text-muted);padding:2rem;text-align:center;">Không có nội dung.</p>';
}
 
function renderSectionBlock(label, body, color, isDecision, cfg) {
    let cleanBody = body;
    let badge = '';

    // Chỉ tìm và hiển thị badge BUY/SELL/HOLD cho decision sections
    // Analyst sections (market, news, sentiment, fundamentals, quant) KHÔNG có badge
    if (isDecision) {
        const actionRe = /\b(BUY|SELL|HOLD)\b/i;
        const explicitDecisionPatterns = [
            /^\s{0,3}>?\s*\*{0,2}\s*FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*RECOMMENDATION\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Khuyến\s*nghị(?:\s+cuối\s+cùng)?\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Khuyen\s*nghi(?:\s+cuoi\s+cung)?\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Quyết\s*định(?:\s+cuối\s+cùng)?\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Quyet\s*dinh(?:\s+cuoi\s+cung)?\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            // New structured format from updated prompts
            /^\s{0,3}####\s+(?:Quyết\s*Định|Đề\s*Xuất\s*Hành\s*Động)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}####\s+Quyết\s*Định\s*:\s*\*{1,2}(BUY|SELL|HOLD)\*{1,2}/gim,
        ];

        let latestDecision = '';
        const linesToRemove = new Set();
        const lines = String(body || '').split('\n');

        explicitDecisionPatterns.forEach((re) => {
            let m;
            while ((m = re.exec(String(body || ''))) !== null) {
                if (m[1]) latestDecision = m[1].toUpperCase();
                const matchedLine = m[0].trim();
                for (const line of lines) {
                    if (line.trim() === matchedLine) linesToRemove.add(line);
                }
            }
        });

        // Fallback: tìm dòng có nhãn quyết định + ':' + đúng 1 action
        if (!latestDecision) {
            for (const line of lines) {
                const normalized = line.replace(/[>*_`#-]/g, ' ').replace(/\s+/g, ' ').trim();
                const hasLabel = /(final\s+transaction\s+proposal|recommendation|khuyến\s*nghị|khuyen\s*nghi|quyết\s*định|quyet\s*dinh|đề\s*xuất\s*hành\s*động)/i.test(normalized);
                const hasColon = normalized.includes(':');
                const actions = [...normalized.matchAll(/\b(BUY|SELL|HOLD)\b/gi)].map(v => v[1].toUpperCase());
                const uniqueActions = [...new Set(actions)];
                if (hasLabel && hasColon && uniqueActions.length === 1) {
                    latestDecision = uniqueActions[0];
                    linesToRemove.add(line);
                }
            }
        }

        const keptLines = lines.filter(line => !linesToRemove.has(line));
        cleanBody = keptLines.join('\n').replace(/\n{3,}/g, '\n\n').trim();

        if (latestDecision) badge = makeDecisionBadge(latestDecision);
    }
    // isDecision === false: analyst sections - không tìm badge, không xóa dòng nào
 
    const bodyHtml = marked.parse(normalizeBodyHeadings(cleanBody));
    const pad = isDecision
        ? `background:${color}07;border-radius:0 12px 12px 0;padding:1.125rem 1.25rem;`
        : 'padding:0 0 0 1.125rem;';
 
    return `
<div class="report-section" style="margin-bottom:1.75rem;border-left:3px solid ${color};${pad}">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid ${color}22;">
        <span style="font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;color:${color};">${label}</span>
        ${badge}
        <span style="flex:1;height:1px;background:${color}20;"></span>
    </div>
    <div class="report-body">${bodyHtml}</div>
</div>`;
}
 
function makeDecisionBadge(v) {
    const c = { BUY: '#10b981', SELL: '#ef4444', HOLD: '#f59e0b' }[v] || '#6b7280';
    return `<span style="display:inline-flex;align-items:center;background:${c}18;border:1.5px solid ${c}55;color:${c};border-radius:8px;padding:3px 14px;font-size:0.78rem;font-weight:800;letter-spacing:0.06em;">${v}</span>`;
}
 
function normalizeBodyHeadings(md) {
    if (!md) return md;
    const matches = [...md.matchAll(/^(#{1,6})\s/gm)];
    if (!matches.length) return md;
    const minLevel = Math.min(...matches.map(m => m[1].length));
    const shift    = 3 - minLevel;
    if (shift === 0) return md;
    return md.replace(/^(#{1,6})(\s)/gm, (_, h, s) =>
        '#'.repeat(Math.min(6, Math.max(1, h.length + shift))) + s
    );
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

async function updateProgressUI(data) {
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
        reportDiv.innerHTML = await renderReport(data.current_report);
    } else if (data.status === 'completed' && data.final_report) {
        reportDiv.innerHTML = await renderReport(data.final_report);
    }

    const finalDiv = document.getElementById('final-report');
    if (finalDiv && data.final_report) {
        finalDiv.innerHTML = await renderReport(data.final_report);
    }

    if (data.decision) updateDecision(data.decision);
}

// ── Agent status ───────────────────────────────────────────────────────────

const AGENT_GROUPS = [
    {
        title: 'Analyst Team',
        agents: ['Market Analyst', 'Social Analyst', 'News Analyst', 'Fundamentals Analyst', 'AlphaGPT Analyst'],
    },
    {
        title: 'Researcher Team',
        agents: ['Bull Researcher', 'Bear Researcher', 'Research Manager'],
    },
    {
        title: 'Trader',
        agents: ['Trader'],
    },
    {
        title: 'Risk Management',
        agents: ['Risky Analyst', 'Safe Analyst', 'Neutral Analyst', 'Portfolio Manager'],
    },
];

const AGENT_ORDER = AGENT_GROUPS.flatMap(group => group.agents);

function formatStatusLabel(status) {
    return {
        pending: 'Pending',
        in_progress: 'In Progress',
        completed: 'Completed',
        not_selected: 'Not Selected',
    }[status] || status;
}

function createAgentStatusItem(agent, status) {
    const div = document.createElement('div');
    div.className = `agent-node ${status}`;
    div.dataset.agent = agent;
    div.dataset.status = status;
    div.innerHTML = `
        <div class="agent-node-name">${agent}</div>
        <div class="agent-node-status">${getStatusIcon(status)} ${formatStatusLabel(status)}</div>`;
    return div;
}

function updateAgentStatus(agentStatus) {
    const container = document.getElementById('agent-status');
    if (!agentStatus || !Object.keys(agentStatus).length) {
        container.innerHTML = '<p>No agent status available</p>';
        return;
    }

    container.className = 'agent-flow-board';
    container.innerHTML = '';

    for (const group of AGENT_GROUPS) {
        const entries = group.agents
            .filter(agent => agent in agentStatus)
            .map(agent => [agent, agentStatus[agent]]);

        if (!entries.length) continue;

        const completedCount = entries.filter(([, status]) => status === 'completed').length;

        const groupEl = document.createElement('section');
        groupEl.className = 'tier-lane';

        const headerEl = document.createElement('div');
        headerEl.className = 'tier-lane-header';
        headerEl.innerHTML = `
            <h5 class="tier-lane-title">${group.title}</h5>
            <span class="tier-lane-count">${completedCount}/${entries.length}</span>`;
        groupEl.appendChild(headerEl);

        const gridEl = document.createElement('div');
        gridEl.className = 'tier-lane-grid';
        for (const [agent, status] of entries) {
            gridEl.appendChild(createAgentStatusItem(agent, status));
        }

        groupEl.appendChild(gridEl);
        container.appendChild(groupEl);
    }

    const remaining = Object.entries(agentStatus).filter(([agent]) => !AGENT_ORDER.includes(agent));
    if (remaining.length) {
        const groupEl = document.createElement('section');
        groupEl.className = 'tier-lane';

        const headerEl = document.createElement('div');
        headerEl.className = 'tier-lane-header';
        headerEl.innerHTML = '<h5 class="tier-lane-title">Other</h5>';
        groupEl.appendChild(headerEl);

        const gridEl = document.createElement('div');
        gridEl.className = 'tier-lane-grid';
        for (const [agent, status] of remaining) {
            gridEl.appendChild(createAgentStatusItem(agent, status));
        }

        groupEl.appendChild(gridEl);
        container.appendChild(groupEl);
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

async function renderSessionReview(sessionId, data) {
    openSessionReviewModal();

    document.getElementById('review-session-id').textContent = sessionId;
    document.getElementById('review-ticker').textContent     = sessionId.split('_')[0] || '-';

    const statusEl = document.getElementById('review-status');
    statusEl.textContent = data.status || '-';
    statusEl.className   = `status-badge ${data.status || ''}`;

    const currentEl = document.getElementById('review-current-report');
    if (currentEl) {
        currentEl.innerHTML = data.current_report
            ? await renderReport(data.current_report)
            : 'Không có báo cáo tạm thời.';
    }

    const finalEl = document.getElementById('review-final-report');
    if (finalEl) {
        finalEl.innerHTML = data.final_report
            ? await renderReport(data.final_report)
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

        if (isReviewingHistoricalSession && currentSessionId === sessionId) {
            currentSessionId             = null;
            isReviewingHistoricalSession = false;
            closeSessionReviewModal();
        }

        loadSessions();
    } catch (err) {
        console.error('Error deleting session:', err);
        alert('Failed to delete session');
    }
}