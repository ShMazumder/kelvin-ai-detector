/* ═══════════════════════════════════════════════════════════════════════════
   Kelvin AI Detector — Client-side JS
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Mobile sidebar toggle ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.mobile-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (toggle && sidebar) {
        toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
        document.addEventListener('click', (e) => {
            if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }
});

// ── Copy to clipboard ────────────────────────────────────────────────────────

function copyKey(text, btn) {
    navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        btn.style.background = '#10b981';
        setTimeout(() => {
            btn.textContent = orig;
            btn.style.background = '';
        }, 2000);
    });
}

// ── Score color helper ───────────────────────────────────────────────────────

function getScoreClass(score) {
    if (score >= 60) return 'score-high';
    if (score >= 30) return 'score-mid';
    return 'score-low';
}

function getScoreColor(score) {
    if (score >= 60) return 'var(--red)';
    if (score >= 30) return 'var(--yellow)';
    return 'var(--green)';
}

// ── Chat-style detection (AJAX) ──────────────────────────────────────────────

function initChatDetect() {
    const form = document.getElementById('chat-form');
    const messagesDiv = document.getElementById('chat-messages');
    const textarea = document.getElementById('chat-input');
    const submitBtn = document.getElementById('chat-submit');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = textarea.value.trim();
        if (text.length < 20) {
            addSystemMessage('Text too short — need at least 20 characters.', true);
            return;
        }

        // Add user bubble
        addUserMessage(text);
        textarea.value = '';
        textarea.style.height = 'auto';
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳';

        const selectEl = document.getElementById('detection-type-select');
        const detection_type = selectEl ? selectEl.value : 'all';

        try {
            const resp = await fetch('/dashboard/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json',
                },
                body: new URLSearchParams({ text, detection_type }),
            });

            if (!resp.ok) {
                const err = await resp.text();
                addSystemMessage('Error: ' + err, true);
                return;
            }

            const result = await resp.json();
            addResultBubble(result, text);

            // Update balance display
            const balEl = document.getElementById('balance-display');
            if (balEl && result.remaining_balance !== undefined) {
                balEl.textContent = result.remaining_balance.toFixed(1);
            }
        } catch (err) {
            addSystemMessage('Network error: ' + err.message, true);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = '→';
        }
    });

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    });

    // Ctrl+Enter to submit
    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            form.dispatchEvent(new Event('submit'));
        }
    });
}

function addUserMessage(text) {
    const div = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble user';
    bubble.innerHTML = `
        <div class="bubble-label">You</div>
        <div class="bubble-text">${escapeHtml(text.substring(0, 500))}${text.length > 500 ? '...' : ''}</div>
    `;
    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;
}

function addSystemMessage(text, isError = false) {
    const div = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble system';
    bubble.innerHTML = `
        <div class="bubble-label">Kelvin</div>
        <div class="bubble-text" style="${isError ? 'color: var(--red)' : ''}">${escapeHtml(text)}</div>
    `;
    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;
}

function addResultBubble(result, originalText) {
    const div = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble system result-bubble-wide';

    const score = result.final_score;
    const scoreClass = getScoreClass(score);
    const patterns = result.detected_patterns || {};
    const examples = result.pattern_examples || {};

    // SVG gauge ring
    const radius = 48;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    const verdictIcon = score >= 55 ? '🤖' : score >= 25 ? '⚠️' : '✅';
    const verdictText = result.final_verdict || 'Unknown';

    // Build patterns HTML with progress bars
    let patternsHtml = '';
    const sortedPatterns = Object.entries(patterns).sort((a, b) => b[1] - a[1]);
    if (sortedPatterns.length > 0) {
        patternsHtml = '<div class="patterns-section"><div class="patterns-title">Detected Patterns</div>';
        for (const [name, pts] of sortedPatterns) {
            const label = name.replace(/_/g, ' ');
            const barPct = Math.min((pts / 20) * 100, 100);
            const barClass = pts >= 10 ? 'bar-high' : pts >= 5 ? 'bar-mid' : 'bar-low';
            const color = PATTERN_COLORS[name] || PATTERN_COLORS._default;
            patternsHtml += `
                <div class="pattern-bar-item">
                    <div class="pattern-bar-header">
                        <span class="pattern-bar-name"><span class="pattern-color-dot" style="background:${color}"></span>${label}</span>
                        <span class="pattern-bar-score">${pts.toFixed(1)} pts</span>
                    </div>
                    <div class="pattern-bar-track">
                        <div class="pattern-bar-fill" style="width: 0%; background:${color}" data-width="${barPct}%"></div>
                    </div>
                </div>
            `;
        }
        patternsHtml += '</div>';
    } else {
        patternsHtml = '<div class="no-patterns">No AI patterns detected</div>';
    }

    // Build model scores breakdown HTML if available
    let modelScoresHtml = '';
    if (result.model_scores && Object.keys(result.model_scores).length > 0) {
        modelScoresHtml = '<div class="model-scores-section" style="margin-top: 16px; text-align: left; padding: 12px; background: rgba(255,255,255,0.03); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">';
        modelScoresHtml += '<div style="font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.03em;">Model Breakdown</div>';
        for (const [mName, mScore] of Object.entries(result.model_scores)) {
            const mLabel = mName.charAt(0).toUpperCase() + mName.slice(1);
            const subScoreClass = getScoreClass(mScore);
            modelScoresHtml += `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="color: var(--text-secondary);">${mLabel} Model:</span>
                    <span class="${subScoreClass}" style="font-weight: 600;">${mScore}%</span>
                </div>
            `;
        }
        modelScoresHtml += '</div>';
    }

    // Build highlighted text
    const highlightedHtml = buildHighlightedText(originalText || '', examples, patterns);

    // Compose copy text
    const copyText = `AI Detection Score: ${score}/100 — ${verdictText}\n` +
        `Model: ${result.model_used} | Words: ${result.word_count} | Time: ${result.processing_time_ms}ms\n` +
        (sortedPatterns.length > 0 ? `Patterns: ${sortedPatterns.map(([n, s]) => `${n.replace(/_/g, ' ')} (${s.toFixed(1)}pts)`).join(', ')}` : 'No AI patterns detected');

    bubble.innerHTML = `
        <div class="result-card">
            <div class="result-card-header">
                <div class="bubble-label">Kelvin Analysis</div>
                <div class="result-card-actions" style="display: flex; gap: 8px;">
                    <button class="result-action-btn" onclick="printResult(this)" title="Print analysis report">
                        🖨️ Print
                    </button>
                    <button class="result-action-btn" onclick="copyResult(this)" data-copy="${escapeAttr(copyText)}" title="Copy result summary">
                        📋 Copy
                    </button>
                </div>
            </div>

            <div class="result-two-col">
                <!-- Left: Score & Patterns -->
                <div class="result-col-score">
                    <div class="gauge-wrapper">
                        <svg class="gauge-svg" viewBox="0 0 120 120">
                            <circle class="gauge-bg" cx="60" cy="60" r="${radius}"/>
                            <circle class="gauge-fill ${scoreClass}" cx="60" cy="60" r="${radius}"
                                stroke-dasharray="${circumference}"
                                stroke-dashoffset="${circumference}"
                                data-target="${offset}"/>
                        </svg>
                        <div class="gauge-label">
                            <div class="gauge-score ${scoreClass}">${score}</div>
                            <div class="gauge-max">/ 100</div>
                        </div>
                    </div>

                    <div class="verdict-badge ${scoreClass}">
                        <span class="verdict-icon">${verdictIcon}</span>
                        ${verdictText}
                    </div>

                    <div class="model-badge" style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px; font-weight: 500; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: var(--radius-sm); display: inline-block;">
                        Model: <span style="color: var(--text-primary); font-weight: 600;">${result.model_used}</span>
                    </div>

                    <div class="result-stats">
                        <div class="result-stat">
                            <div class="result-stat-value">${result.word_count}</div>
                            <div class="result-stat-label">Words</div>
                        </div>
                        <div class="result-stat">
                            <div class="result-stat-value">${result.sentence_count || '—'}</div>
                            <div class="result-stat-label">Sentences</div>
                        </div>
                        <div class="result-stat">
                            <div class="result-stat-value">${result.processing_time_ms}ms</div>
                            <div class="result-stat-label">Speed</div>
                        </div>
                    </div>

                    ${modelScoresHtml}

                    ${patternsHtml}
                </div>

                <!-- Right: Highlighted Text -->
                <div class="result-col-text">
                    <div class="highlighted-text-header">
                        <span class="highlighted-text-title">Your Text — Highlighted</span>
                    </div>
                    <div class="highlighted-text-body">${highlightedHtml}</div>
                    <div class="highlight-legend">${buildLegendHtml(examples, patterns)}</div>
                </div>
            </div>

            <div class="result-disclaimer">${result.disclaimer || 'This is an estimate — not a definitive classification.'}</div>
        </div>
    `;

    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;

    // Animate gauge ring
    requestAnimationFrame(() => {
        const fill = bubble.querySelector('.gauge-fill');
        if (fill) fill.setAttribute('stroke-dashoffset', fill.dataset.target);
        bubble.querySelectorAll('.pattern-bar-fill').forEach(bar => {
            requestAnimationFrame(() => { bar.style.width = bar.dataset.width; });
        });
    });
}


// ── Pattern highlight colors ─────────────────────────────────────────────────

const PATTERN_COLORS = {
    ai_vocabulary:          '#ef4444',   // red
    ai_pattern_density:     '#ef4444',   // red (same group)
    inflated_significance:  '#f97316',   // orange
    editorializing:         '#eab308',   // yellow
    leftover_chat_artifacts:'#ec4899',   // pink
    letter_style_formality: '#a855f7',   // purple
    vague_attribution:      '#8b5cf6',   // violet
    compulsive_summary:     '#06b6d4',   // cyan
    negative_parallelism:   '#14b8a6',   // teal
    false_ranges:           '#22c55e',   // green
    low_sentence_variance:  '#64748b',   // gray (not highlightable)
    em_dash_overuse:        '#f59e0b',   // amber
    rule_of_three:          '#3b82f6',   // blue
    formatting_overkill:    '#6366f1',   // indigo
    media_puffery:          '#f43f5e',   // rose
    citation_bugs:          '#a21caf',   // magenta
    // ── New patterns (v2) ──
    promotional_superlatives:'#dc2626',  // darker red
    excessive_transitions:   '#0ea5e9',  // sky blue
    sentence_initial_adverbs:'#8b5cf6',  // violet
    temporal_vagueness:      '#d97706',  // amber
    deepseek_artifacts:      '#be123c',  // crimson
    title_case_headings:     '#0d9488',  // teal
    passive_voice:           '#7c3aed',  // purple
    passive_voice_overuse:   '#7c3aed',  // purple
    inline_header_lists:     '#2563eb',  // blue
    constructive_criticism:  '#e11d48',  // rose
    paragraph_uniformity:    '#64748b',  // gray (not highlightable)
    curly_quotes:            '#d946ef',  // fuchsia
    markdown_leak:           '#059669',  // emerald
    collective_we:           '#06b6d4',  // cyan-alt
    copulative_avoidance:    '#a3a3a3',  // neutral
    pua_citation_bugs:       '#a21caf',  // magenta-alt
    knowledge_cutoff_disclaimers:'#b45309', // dark amber
    utm_parameters:          '#991b1b',  // dark red
    // ── Wikipedia & General AI tells (v3) ──
    lead_proper_noun:        '#ea580c',  // dark orange
    thematic_break_headings: '#4f46e5',  // indigo
    skipped_headings:        '#db2777',  // dark pink
    consecutive_sentence_starters:'#06b6d4', // cyan
    list_item_sentence_count_uniformity:'#64748b', // gray (not highlightable)
    _default:               '#94a3b8',   // slate
};


function buildHighlightedText(text, examples, patterns) {
    if (!text) return '';

    // Collect all match spans with their pattern category
    const spans = []; // { start, end, pattern, match }

    for (const [patternName, matchList] of Object.entries(examples)) {
        if (!matchList || !Array.isArray(matchList)) continue;
        const color = PATTERN_COLORS[patternName] || PATTERN_COLORS._default;
        const label = patternName.replace(/_/g, ' ');

        for (const match of matchList) {
            if (!match || match.length < 1) continue;
            // Find all occurrences of this match in the text (case-insensitive)
            const lowerText = text.toLowerCase();
            const lowerMatch = match.toLowerCase();
            let searchFrom = 0;
            while (true) {
                const idx = lowerText.indexOf(lowerMatch, searchFrom);
                if (idx === -1) break;
                spans.push({
                    start: idx,
                    end: idx + match.length,
                    pattern: patternName,
                    label: label,
                    color: color,
                    match: text.substring(idx, idx + match.length),
                });
                searchFrom = idx + match.length;
            }
        }
    }

    if (spans.length === 0) {
        return '<span class="ht-clean">' + escapeHtml(text) + '</span>';
    }

    // Sort by start position, longer spans first for overlaps
    spans.sort((a, b) => a.start - b.start || b.end - a.end);

    // Remove overlapping spans (keep the first/longer one)
    const filtered = [];
    let lastEnd = 0;
    for (const s of spans) {
        if (s.start >= lastEnd) {
            filtered.push(s);
            lastEnd = s.end;
        }
    }

    // Build HTML
    let html = '';
    let cursor = 0;
    for (const s of filtered) {
        if (s.start > cursor) {
            html += escapeHtml(text.substring(cursor, s.start));
        }
        html += `<mark class="ht-mark" style="--ht-color:${s.color}" title="${escapeAttr(s.label)}">${escapeHtml(s.match)}</mark>`;
        cursor = s.end;
    }
    if (cursor < text.length) {
        html += escapeHtml(text.substring(cursor));
    }

    return html;
}


function buildLegendHtml(examples, patterns) {
    const seen = new Set();
    let html = '';
    for (const patternName of Object.keys(examples)) {
        if (!examples[patternName] || examples[patternName].length === 0) continue;
        if (seen.has(patternName)) continue;
        seen.add(patternName);
        const color = PATTERN_COLORS[patternName] || PATTERN_COLORS._default;
        const label = patternName.replace(/_/g, ' ');
        html += `<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${label}</span>`;
    }
    return html || '<span class="legend-item" style="color:var(--text-muted)">No patterns to highlight</span>';
}


function copyResult(btn) {
    const text = btn.dataset.copy;
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = '✓ Copied';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = '📋 Copy';
        }, 2000);
    });
}


function printResult(btn) {
    const card = btn.closest('.result-card');
    if (card) {
        printCard(card);
    }
}

function printModalResult() {
    const content = document.getElementById('modal-body-content');
    if (!content) return;
    const card = content.querySelector('.result-card');
    if (card) {
        printCard(card);
    }
}

function printCard(card) {
    const scoreVal = card.querySelector('.gauge-score') ? card.querySelector('.gauge-score').textContent : '—';
    const verdictEl = card.querySelector('.verdict-badge');
    const verdictHtml = verdictEl ? verdictEl.innerHTML : '';
    const modelEl = card.querySelector('.model-badge');
    const modelHtml = modelEl ? modelEl.innerHTML : '';
    
    const statsEl = card.querySelector('.result-stats');
    const statsHtml = statsEl ? statsEl.outerHTML : '';
    
    const breakdownEl = card.querySelector('.model-scores-section');
    const breakdownHtml = breakdownEl ? breakdownEl.outerHTML : '';
    
    const patternsEl = card.querySelector('.patterns-section');
    const patternsHtml = patternsEl ? patternsEl.outerHTML : '';
    
    const textEl = card.querySelector('.highlighted-text-body');
    const textHtml = textEl ? textEl.outerHTML : '';
    
    const legendEl = card.querySelector('.highlight-legend');
    const legendHtml = legendEl ? legendEl.outerHTML : '';

    const printWin = window.open('', '_blank');
    printWin.document.write(`
        <html>
        <head>
            <title>Kelvin AI Detection Report</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #1f2937;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 0 20px;
                }
                .header {
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 15px;
                    margin-bottom: 25px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .title {
                    font-size: 24px;
                    font-weight: 800;
                    color: #111827;
                    margin: 0;
                }
                .date {
                    font-size: 14px;
                    color: #6b7280;
                }
                .summary-box {
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 24px;
                    margin-bottom: 25px;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .score-row {
                    display: flex;
                    align-items: center;
                    gap: 24px;
                }
                .score-large {
                    font-size: 42px;
                    font-weight: 900;
                    color: #e11d48;
                    border: 3px solid #e11d48;
                    border-radius: 50%;
                    width: 90px;
                    height: 90px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .verdict-badge {
                    font-size: 18px;
                    font-weight: 700;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                }
                .model-badge {
                    font-size: 14px;
                    color: #4b5563;
                    margin-top: 4px;
                }
                .result-stats {
                    display: flex;
                    gap: 24px;
                    border-top: 1px solid #e5e7eb;
                    padding-top: 16px;
                }
                .result-stat {
                    display: flex;
                    flex-direction: column;
                }
                .result-stat-value {
                    font-size: 18px;
                    font-weight: 700;
                    color: #111827;
                }
                .result-stat-label {
                    font-size: 12px;
                    color: #6b7280;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .section-title {
                    font-size: 16px;
                    font-weight: 700;
                    color: #374151;
                    border-bottom: 2px solid #f3f4f6;
                    padding-bottom: 8px;
                    margin-top: 30px;
                    margin-bottom: 16px;
                }
                .highlighted-text-body {
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    padding: 20px;
                    border-radius: 8px;
                    white-space: pre-wrap;
                    font-size: 15px;
                    color: #374151;
                }
                mark {
                    background-color: var(--ht-color, #ffe4e6);
                    border-bottom: 2px solid rgba(0,0,0,0.1);
                    padding: 2px 4px;
                    border-radius: 3px;
                    color: #111827;
                }
                .model-scores-section {
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 16px;
                }
                .patterns-section {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .patterns-title {
                    display: none;
                }
                .pattern-bar-item {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .pattern-bar-header {
                    display: flex;
                    justify-content: space-between;
                    font-size: 14px;
                }
                .pattern-bar-track {
                    height: 8px;
                    background: #f3f4f6;
                    border-radius: 4px;
                }
                .pattern-bar-fill {
                    height: 100%;
                    border-radius: 4px;
                }
                .highlight-legend {
                    margin-top: 16px;
                    font-size: 12px;
                    color: #6b7280;
                }
                .legend-item {
                    display: inline-flex;
                    align-items: center;
                    margin-right: 16px;
                }
                .legend-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 6px;
                }
                .print-actions {
                    margin-top: 40px;
                    display: flex;
                    justify-content: center;
                }
                .print-btn {
                    padding: 12px 24px;
                    font-size: 16px;
                    background: #2563eb;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 700;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    transition: background 0.2s;
                }
                .print-btn:hover {
                    background: #1d4ed8;
                }
                @media print {
                    body {
                        margin: 0;
                        padding: 0;
                        max-width: 100%;
                    }
                    .print-actions {
                        display: none;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1 class="title">🔍 Kelvin AI Detection Report</h1>
                <div class="date">${new Date().toLocaleString()}</div>
            </div>
            
            <div class="summary-box">
                <div class="score-row">
                    <div class="score-large">${scoreVal}</div>
                    <div>
                        <div class="verdict-badge">${verdictHtml}</div>
                        <div class="model-badge">${modelHtml}</div>
                    </div>
                </div>
                ${statsHtml}
            </div>

            ${breakdownHtml ? `<div><h3 class="section-title">Model Breakdown</h3>${breakdownHtml}</div>` : ''}
            
            ${patternsHtml ? `<div><h3 class="section-title">Detected Patterns</h3>${patternsHtml}</div>` : ''}

            <div>
                <h3 class="section-title">Analyzed Text</h3>
                ${textHtml}
                ${legendHtml}
            </div>

            <div class="print-actions">
                <button class="print-btn" onclick="window.print()">🖨️ Print Report</button>
            </div>
        </body>
        </html>
    `);
    printWin.document.close();
}


// ── Details Modal Operations ───────────────────────────────────────────────

async function showLogDetails(logId) {
    const modal = document.getElementById('details-modal');
    const content = document.getElementById('modal-body-content');
    if (!modal || !content) return;

    // Show loading state
    content.innerHTML = '<div style="text-align:center; padding:40px; color:var(--text-secondary);">⏳ Fetching details...</div>';
    modal.style.display = 'flex';

    try {
        const resp = await fetch(`/dashboard/usage/${logId}`);
        if (!resp.ok) {
            throw new Error(await resp.text() || 'Failed to fetch details');
        }
        const data = await resp.json();
        
        // Reconstruct a result object compatible with result-card renderer
        const result = {
            final_score: data.score,
            final_verdict: data.verdict,
            model_used: data.model_used,
            word_count: data.word_count,
            processing_time_ms: data.details.processing_time_ms || '—',
            sentence_count: data.details.sentence_count || '—',
            detected_patterns: data.details.detected_patterns || {},
            pattern_examples: data.details.pattern_examples || {},
            model_scores: data.details.model_scores || {},
            disclaimer: data.details.disclaimer || 'Style diagnostic only — not proof of authorship.'
        };

        const score = result.final_score;
        const scoreClass = getScoreClass(score);
        const patterns = result.detected_patterns;
        const examples = result.pattern_examples;

        // SVG gauge ring
        const radius = 48;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;

        const verdictIcon = score >= 55 ? '🤖' : score >= 25 ? '⚠️' : '✅';
        const verdictText = result.final_verdict || 'Unknown';

        // Build patterns HTML
        let patternsHtml = '';
        const sortedPatterns = Object.entries(patterns).sort((a, b) => b[1] - a[1]);
        if (sortedPatterns.length > 0) {
            patternsHtml = '<div class="patterns-section" style="margin-top:20px;"><div class="patterns-title">Detected Patterns</div>';
            for (const [name, pts] of sortedPatterns) {
                const label = name.replace(/_/g, ' ');
                const barPct = Math.min((pts / 20) * 100, 100);
                const color = PATTERN_COLORS[name] || PATTERN_COLORS._default;
                patternsHtml += `
                    <div class="pattern-bar-item">
                        <div class="pattern-bar-header">
                            <span class="pattern-bar-name"><span class="pattern-color-dot" style="background:${color}"></span>${label}</span>
                            <span class="pattern-bar-score">${pts.toFixed(1)} pts</span>
                        </div>
                        <div class="pattern-bar-track">
                            <div class="pattern-bar-fill" style="width: ${barPct}%; background:${color}"></div>
                        </div>
                    </div>
                `;
            }
            patternsHtml += '</div>';
        } else {
            patternsHtml = '<div class="no-patterns" style="margin-top:20px;">No AI patterns detected</div>';
        }

        // Build model scores breakdown HTML if available
        let modelScoresHtml = '';
        if (result.model_scores && Object.keys(result.model_scores).length > 0) {
            modelScoresHtml = '<div class="model-scores-section" style="margin-top: 16px; text-align: left; padding: 12px; background: rgba(255,255,255,0.03); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">';
            modelScoresHtml += '<div style="font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.03em;">Model Breakdown</div>';
            for (const [mName, mScore] of Object.entries(result.model_scores)) {
                const mLabel = mName.charAt(0).toUpperCase() + mName.slice(1);
                const subScoreClass = getScoreClass(mScore);
                modelScoresHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
                        <span style="color: var(--text-secondary);">${mLabel} Model:</span>
                        <span class="${subScoreClass}" style="font-weight: 600;">${mScore}%</span>
                    </div>
                `;
            }
            modelScoresHtml += '</div>';
        }

        // Highlight original text
        const highlightedHtml = buildHighlightedText(data.input_text || '', examples, patterns);

        // Render card
        content.innerHTML = `
            <div class="result-card" style="box-shadow:none; border:none; padding:0; background:transparent;">
                <div class="result-two-col">
                    <!-- Left: Score & Patterns -->
                    <div class="result-col-score">
                        <div class="gauge-wrapper">
                            <svg class="gauge-svg" viewBox="0 0 120 120">
                                <circle class="gauge-bg" cx="60" cy="60" r="${radius}"/>
                                <circle class="gauge-fill ${scoreClass}" cx="60" cy="60" r="${radius}"
                                    stroke-dasharray="${circumference}"
                                    stroke-dashoffset="${offset}"/>
                            </svg>
                            <div class="gauge-label">
                                <div class="gauge-score ${scoreClass}">${score}</div>
                                <div class="gauge-max">/ 100</div>
                            </div>
                        </div>

                        <div class="verdict-badge ${scoreClass}">
                            <span class="verdict-icon">${verdictIcon}</span>
                            ${verdictText}
                        </div>

                        <div class="model-badge" style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px; font-weight: 500; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: var(--radius-sm); display: inline-block;">
                            Model: <span style="color: var(--text-primary); font-weight: 600;">${result.model_used}</span>
                        </div>

                        <div class="result-stats">
                            <div class="result-stat">
                                <div class="result-stat-value">${result.word_count}</div>
                                <div class="result-stat-label">Words</div>
                            </div>
                            <div class="result-stat">
                                <div class="result-stat-value">${result.sentence_count}</div>
                                <div class="result-stat-label">Sentences</div>
                            </div>
                            <div class="result-stat">
                                <div class="result-stat-value">${result.processing_time_ms}ms</div>
                                <div class="result-stat-label">Speed</div>
                            </div>
                        </div>

                        ${modelScoresHtml}

                        ${patternsHtml}
                    </div>

                    <!-- Right: Highlighted Text -->
                    <div class="result-col-text">
                        <div class="highlighted-text-header">
                            <span class="highlighted-text-title">Your Text — Highlighted</span>
                        </div>
                        <div class="highlighted-text-body" style="max-height: 400px; overflow-y: auto;">${highlightedHtml}</div>
                        <div class="highlight-legend">${buildLegendHtml(examples, patterns)}</div>
                    </div>
                </div>

                <div class="result-disclaimer" style="margin-top:20px;">${result.disclaimer}</div>
            </div>
        `;
    } catch (err) {
        content.innerHTML = `<div style="color:var(--red); text-align:center; padding:40px;">⚠️ Error: ${err.message}</div>`;
    }
}

function closeDetailsModal() {
    const modal = document.getElementById('details-modal');
    if (modal) modal.style.display = 'none';
}

// Close modal on escape key or clicking backdrop
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDetailsModal();
});
document.addEventListener('click', (e) => {
    const modal = document.getElementById('details-modal');
    if (e.target === modal) closeDetailsModal();
});


function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


function escapeAttr(str) {
    return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initChatDetect();
});
