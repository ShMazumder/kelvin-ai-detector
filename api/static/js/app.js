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
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: new URLSearchParams({ text, _ajax: '1' }),
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

    const scoreClass = getScoreClass(result.final_score);
    const patterns = result.detected_patterns || {};
    const examples = result.pattern_examples || {};

    let patternsHtml = '';
    if (Object.keys(patterns).length > 0) {
        patternsHtml = '<div class="patterns-list">';
        for (const [name, score] of Object.entries(patterns)) {
            const label = name.replace(/_/g, ' ');
            const exList = examples[name];
            let exHtml = '';
            if (exList && exList.length > 0) {
                exHtml = `<div class="pattern-examples">${exList.slice(0, 3).map(e => '"' + escapeHtml(e) + '"').join(', ')}</div>`;
            }
            patternsHtml += `
                <div class="pattern-item">
                    <span class="pattern-name">${label}</span>
                    <span class="pattern-score" style="color: ${getScoreColor(score * 3)}">${score} pts</span>
                </div>
                ${exHtml}
            `;
        }
        patternsHtml += '</div>';
    }

    bubble.innerHTML = `
        <div class="bubble-label">Kelvin Analysis</div>
        <div class="score-display">
            <div class="score-circle ${scoreClass}">${result.final_score}</div>
            <div class="score-verdict">${result.final_verdict}</div>
            <div class="text-sm text-muted mt-16">
                ${result.model_used} · ${result.word_count} words · ${result.processing_time_ms}ms
            </div>
        </div>
        ${patternsHtml}
        <div class="text-sm text-muted mt-16" style="font-style: italic">${result.disclaimer}</div>
    `;

    div.appendChild(bubble);
    div.scrollTop = div.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initChatDetect();
});
