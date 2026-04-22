/**
 * agent_status.js
 * Agent flow board rendering and status utilities.
 */

export const AGENT_GROUPS = [
    { title: 'Analyst Team',    agents: ['Market Analyst', 'Social Analyst', 'News Analyst', 'Fundamentals Analyst', 'AlphaGPT Analyst'] },
    { title: 'Researcher Team', agents: ['Bull Researcher', 'Bear Researcher', 'Research Manager'] },
    { title: 'Trader',          agents: ['Trader'] },
    { title: 'Risk Management', agents: ['Risky Analyst', 'Safe Analyst', 'Neutral Analyst', 'Portfolio Manager'] },
];

export const AGENT_ORDER = AGENT_GROUPS.flatMap(g => g.agents);

// ── Analyst tool metadata ─────────────────────────────────────────────────

const ANALYST_TOOLS = {
    'Market Analyst': {
        desc: 'Phân tích kỹ thuật & giá cổ phiếu',
        tools: [
            { name: 'get_stock_data(symbol, start_date, end_date)',        desc: 'Lấy dữ liệu OHLCV lịch sử.' },
            { name: 'get_indicators(ticker, indicators, start_date, end_date)', desc: 'Tính các chỉ báo kỹ thuật.' },
            { name: 'get_market_context(ticker, curr_date)',               desc: 'Lấy bối cảnh thị trường VN30.' },
        ],
    },
    'Social Analyst': {
        desc: 'Phân tích tâm lý & mạng xã hội',
        tools: [{ name: 'get_news(query, start_date, end_date)', desc: 'Thu thập tin tức/thảo luận theo mã.' }],
    },
    'News Analyst': {
        desc: 'Phân tích tin tức đa nguồn',
        tools: [
            { name: 'get_news(query, start_date, end_date)',                desc: 'Tin doanh nghiệp/ngành.' },
            { name: 'get_global_news(curr_date, look_back_days, limit)',    desc: 'Tin vĩ mô tổng quát.' },
        ],
    },
    'Fundamentals Analyst': {
        desc: 'Phân tích tài chính doanh nghiệp',
        tools: [
            { name: 'get_fundamentals(ticker, curr_date)',    desc: 'Tổng quan định giá.' },
            { name: 'get_balance_sheet(ticker, curr_date)',   desc: 'Bảng cân đối kế toán.' },
            { name: 'get_cashflow(ticker, curr_date)',        desc: 'Lưu chuyển tiền tệ.' },
            { name: 'get_income_statement(ticker, curr_date)', desc: 'Kết quả kinh doanh.' },
        ],
    },
};

// ── Helpers ───────────────────────────────────────────────────────────────

export function getStatusIcon(status) {
    const icons = {
        pending:      `<i class="fa-solid fa-clock status-fa-icon pending" aria-hidden="true"></i>`,
        in_progress:  `<i class="fa-solid fa-gear status-fa-icon in-progress" aria-hidden="true"></i>`,
        completed:    `<i class="fa-solid fa-circle-check status-fa-icon completed" aria-hidden="true"></i>`,
        not_selected: `<i class="fa-solid fa-forward status-fa-icon not-selected" aria-hidden="true"></i>`,
    };
    return icons[status] || icons.pending;
}

function formatStatusLabel(status) {
    return { pending: 'Pending', in_progress: 'In Progress', completed: 'Completed', not_selected: 'Not Selected' }[status] || status;
}

function formatToolSignature(name) {
    const match = /^([a-zA-Z_][a-zA-Z0-9_]*)(\(.*\))$/.exec(String(name || ''));
    if (!match) return `<span class="tooltip-tool-method">${name}</span>`;
    return `<span class="tooltip-tool-method">${match[1]}</span><span class="tooltip-tool-params">${match[2]}</span>`;
}

function createAnalystTooltip(agent) {
    const data = ANALYST_TOOLS[agent];
    if (!data) return '';
    const toolsHtml = data.tools.map(t => `
        <div class="tooltip-tool">
            <span class="tooltip-tool-name">${formatToolSignature(t.name)}</span>
            <span class="tooltip-tool-desc">${t.desc}</span>
        </div>`).join('');
    return `
        <div class="analyst-tooltip">
            <div class="tooltip-header">${agent}</div>
            <div class="tooltip-subdesc">${data.desc}</div>
            <div class="tooltip-divider"></div>
            <div class="tooltip-tools-label">Tools</div>
            ${toolsHtml}
        </div>`;
}

function createAgentStatusItem(agent, status) {
    const div = document.createElement('div');
    const hasTooltip = agent in ANALYST_TOOLS;
    div.className = `agent-node ${status}${hasTooltip ? ' has-tooltip' : ''}`;
    div.dataset.agent  = agent;
    div.dataset.status = status;
    div.innerHTML = `
        <div class="agent-node-name">${agent}</div>
        <div class="agent-node-status">${getStatusIcon(status)} ${formatStatusLabel(status)}</div>
        ${hasTooltip ? createAnalystTooltip(agent) : ''}`;
    return div;
}

// ── Main render ───────────────────────────────────────────────────────────

export function updateAgentStatus(agentStatus, containerId = 'agent-status') {
    const container = document.getElementById(containerId);
    if (!container) return;

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

        const completedCount = entries.filter(([, s]) => s === 'completed').length;
        const groupEl = document.createElement('section');
        groupEl.className = 'tier-lane';

        const headerEl = document.createElement('div');
        headerEl.className = 'tier-lane-header';
        headerEl.innerHTML = `<h5 class="tier-lane-title">${group.title}</h5><span class="tier-lane-count">${completedCount}/${entries.length}</span>`;
        groupEl.appendChild(headerEl);

        const gridEl = document.createElement('div');
        gridEl.className = 'tier-lane-grid';
        for (const [agent, status] of entries) gridEl.appendChild(createAgentStatusItem(agent, status));
        groupEl.appendChild(gridEl);
        container.appendChild(groupEl);
    }

    // Any remaining agents not in the defined groups
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
        for (const [agent, status] of remaining) gridEl.appendChild(createAgentStatusItem(agent, status));
        groupEl.appendChild(gridEl);
        container.appendChild(groupEl);
    }
}