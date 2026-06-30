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
            addResultBubble(result);

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

function addResultBubble(result) {
    const div = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble system';

    const score = result.final_score;
    const scoreClass = getScoreClass(score);
    const patterns = result.detected_patterns || {};
    const examples = result.pattern_examples || {};

    // SVG gauge ring
    const radius = 48;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    const verdictIcon = score >= 60 ? '🤖' : score >= 30 ? '⚠️' : '✅';
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
            const exList = examples[name];
            let exHtml = '';
            if (exList && exList.length > 0) {
                exHtml = `<div class="pattern-examples-inline">${exList.slice(0, 3).map(e => '"' + escapeHtml(e) + '"').join(', ')}</div>`;
            }
            patternsHtml += `
                <div class="pattern-bar-item">
                    <div class="pattern-bar-header">
                        <span class="pattern-bar-name">${label}</span>
                        <span class="pattern-bar-score">${pts.toFixed(1)} pts</span>
                    </div>
                    <div class="pattern-bar-track">
                        <div class="pattern-bar-fill ${barClass}" style="width: 0%;" data-width="${barPct}%"></div>
                    </div>
                    ${exHtml}
                </div>
            `;
        }
        patternsHtml += '</div>';
    } else {
        patternsHtml = '<div class="no-patterns">No AI patterns detected</div>';
    }

    // Compose full result summary for copy
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
                <div class="result-stat">
                    <div class="result-stat-value">${result.model_used}</div>
                    <div class="result-stat-label">Engine</div>
                </div>
            </div>

            ${patternsHtml}

            <div class="result-disclaimer">${result.disclaimer || 'This is an estimate — not a definitive classification.'}</div>
        </div>
    `;

    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;

    // Animate gauge ring
    requestAnimationFrame(() => {
        const fill = bubble.querySelector('.gauge-fill');
        if (fill) fill.setAttribute('stroke-dashoffset', fill.dataset.target);
        // Animate pattern bars
        bubble.querySelectorAll('.pattern-bar-fill').forEach(bar => {
            requestAnimationFrame(() => { bar.style.width = bar.dataset.width; });
        });
    });
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
