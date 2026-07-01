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

        try {
            const resp = await fetch('/dashboard/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json',
                },
                body: new URLSearchParams({ text }),
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
                <div class="result-card-actions">
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
