/**
 * report_renderer.js
 * Report markdown → HTML rendering, section splitting, decision badge logic.
 */

// ── Config cache ──────────────────────────────────────────────────────────

let _sectionConfig = null;

export async function loadSectionConfig() {
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
            quant_report:           'Phân tích định lượng',
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
    const DECISION_KEYS = new Set(['investment_plan', 'trader_investment_plan', 'final_trade_decision']);

    const labelMap   = {};
    const labelOrder = [];
    for (const [key, label] of Object.entries(titles)) {
        labelMap[label] = { key, color: KEY_COLORS[key] || '#6366f1', isDecision: DECISION_KEYS.has(key) };
        labelOrder.push(label);
    }
    labelOrder.sort((a, b) => b.length - a.length);
    return { labelMap, labelOrder, titles, KEY_COLORS, DECISION_KEYS };
}

// ── Public render entry ───────────────────────────────────────────────────

export async function renderReport(md) {
    if (!md) return '';
    const cfg = await loadSectionConfig();
    return _renderReport(md, cfg);
}

function sanitizeReportText(text) {
    if (!text) return text;
    return text.split('\n').filter(line => {
        const n = line.trim().toLowerCase();
        return n !== 'portfolio management decision' && n !== 'portfolio manager decision';
    }).join('\n');
}

function _renderReport(md, cfg) {
    const text = sanitizeReportText(md);

    // Tách decision blocks ra trước
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

    cleaned = cleaned.replace(/^##\s+Báo cáo nhóm phân tích\s*\n?/im, '');

    const fundamentalsCanonical = cfg.titles?.fundamentals_report || 'Phân tích tài chính doanh nghiệp';
    cleaned = cleaned
        .replace(/^##\s+Phân\s+tích\s+cơ\s+bản\s*$/gim, `### ${fundamentalsCanonical}`)
        .replace(/^###\s+Phân\s+tích\s+cơ\s+bản\s*$/gim, `### ${fundamentalsCanonical}`);

    cfg.labelOrder.forEach(label => {
        const esc = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        cleaned = cleaned.replace(new RegExp('^##\\s+(' + esc + ')\\s*$', 'gim'), '### $1');
    });

    const escapedLabels = cfg.labelOrder.map(l => l.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const SPLIT_RE = new RegExp('(?=^###\\s+(?:' + escapedLabels.join('|') + ')\\s*$)', 'im');
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
        const meta     = cfg.labelMap[matchedLabel];
        const labelEsc = matchedLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const body     = chunk.replace(new RegExp('^###\\s+' + labelEsc + '\\s*\\n', 'im'), '').trim();
        html += renderSectionBlock(matchedLabel, body, meta.color, meta.isDecision, cfg);
    }

    for (const blk of decisionBlocks) {
        const titleMatch = blk.match(/^##\s+=====\s*(.+?)\s*=====\s*$/im);
        if (!titleMatch) continue;
        const title = titleMatch[1].trim();
        const body  = blk.replace(/^##\s+=====.+=====\s*\n/im, '').trim();
        let color = '#ef4444';
        for (const [label, meta] of Object.entries(cfg.labelMap)) {
            if (title.toLowerCase().includes(label.toLowerCase().slice(0, 8))) { color = meta.color; break; }
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

    if (isDecision) {
        const explicitDecisionPatterns = [
            /^\s{0,3}>?\s*\*{0,2}\s*FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*{0,2}\s*(NOT\s+BUY|BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*RECOMMENDATION\s*:\s*\*{0,2}\s*(NOT\s+BUY|BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Khuyến\s*nghị(?:\s+cuối\s+cùng)?\s*:\s*\*{0,2}\s*(NOT\s+BUY|BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}>?\s*\*{0,2}\s*Quyết\s*định(?:\s+cuối\s+cùng)?\s*:\s*\*{0,2}\s*(NOT\s+BUY|BUY|SELL|HOLD)\b.*$/gim,
            /^\s{0,3}####\s+(?:Quyết\s*Định|Đề\s*Xuất\s*Hành\s*Động)\s*:\s*\*{0,2}\s*(NOT\s+BUY|BUY|SELL|HOLD)\b.*$/gim,
        ];

        let latestDecision = '';
        const linesToRemove = new Set();
        const lines = String(body || '').split('\n');

        explicitDecisionPatterns.forEach(re => {
            let m;
            while ((m = re.exec(String(body || ''))) !== null) {
                if (m[1]) latestDecision = m[1].toUpperCase();
                const matchedLine = m[0].trim();
                for (const line of lines) {
                    if (line.trim() === matchedLine) linesToRemove.add(line);
                }
            }
        });

        if (!latestDecision) {
            for (const line of lines) {
                const normalized = line.replace(/[>*_`#-]/g, ' ').replace(/\s+/g, ' ').trim();
                const hasLabel = /(final\s+transaction\s+proposal|recommendation|khuyến\s*nghị|quyết\s*định|đề\s*xuất\s*hành\s*động)/i.test(normalized);
                const hasColon = normalized.includes(':');
                const actions = [...normalized.matchAll(/\bNOT\s+BUY\b|\b(BUY|SELL|HOLD)\b/gi)].map(v => v[0].toUpperCase().replace(/\s+/, ' '));
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
    const upper = v.toUpperCase();
    const label = upper.includes('NOT BUY') ? 'NOT BUY'
                : upper.includes('BUY')     ? 'BUY'
                : upper.includes('SELL')    ? 'SELL'
                : upper.includes('HOLD')    ? 'HOLD'
                : v;
    const c = {
        'BUY':     '#10b981',
        'NOT BUY': '#ef4444',
        'SELL':    '#ef4444',
        'HOLD':    '#f59e0b',
    }[label] || '#6b7280';
    return `<span style="display:inline-flex;align-items:center;background:${c}18;border:1.5px solid ${c}55;color:${c};border-radius:8px;padding:3px 14px;font-size:0.78rem;font-weight:800;letter-spacing:0.06em;">${label}</span>`;
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

// ── Decision HTML ─────────────────────────────────────────────────────────

export function buildDecisionHtml(decision) {
    const decisionIconClass = {
        buy:  'fa-solid fa-arrow-trend-up',
        sell: 'fa-solid fa-arrow-trend-down',
        hold: 'fa-solid fa-pause',
    };
    function safeStringify(v) { try { return JSON.stringify(v, null, 2); } catch { return String(v); } }
    function formatKey(key)   { return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '); }

    if (typeof decision === 'string') {
        const upper = decision.toUpperCase();
        const type  = upper.includes('NOT BUY') ? 'NOT BUY'
                    : upper.includes('BUY')     ? 'BUY'
                    : upper.includes('SELL')    ? 'SELL'
                    : 'HOLD';
        const cls   = type === 'NOT BUY' ? 'sell' : type.toLowerCase();
        return `
            <div class="decision-highlight decision-${cls}">
                <div class="decision-icon"><i class="${decisionIconClass[cls] || decisionIconClass.sell} decision-icon-${cls}" aria-hidden="true"></i></div>
                <div class="decision-text"><h2>${type}</h2></div>
            </div>`;
    }
    if (typeof decision === 'object' && decision !== null) {
        const str  = safeStringify(decision).toUpperCase();
        const type = str.includes('BUY') ? 'BUY' : str.includes('SELL') ? 'SELL' : 'HOLD';
        const cls  = type.toLowerCase();
        let html = `
            <div class="decision-highlight decision-${cls}">
                <div class="decision-icon"><i class="${decisionIconClass[cls]} decision-icon-${cls}" aria-hidden="true"></i></div>
                <div class="decision-text"><h2>${type}</h2></div>
            </div>
            <div class="decision-details">`;
        for (const [k, v] of Object.entries(decision)) {
            html += `<div class="decision-item"><strong>${formatKey(k)}:</strong> ${typeof v === 'object' && v !== null ? safeStringify(v) : v}</div>`;
        }
        return html + '</div>';
    }
    return '<p>Chưa có quyết định.</p>';
}