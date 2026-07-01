"""
AI-Text Likeness Detector — Core Detection Module

Extracted from detectors/ai_text_likeness_detector.ipynb.
Two layers:
  1. Heuristic pattern scorer (always available, zero ML deps)
  2. Optional ML classifier (loads pickled sklearn model if present)
"""

import re
import os
import math
import statistics
import pickle
from typing import Dict, List, Optional, Any

# ── Reference word / phrase lists ──────────────────────────────────────────────

AI_VOCAB = [
    "delve", "delves", "delving", "intricate", "intricacies", "tapestry",
    "pivotal", "underscore", "underscores", "underscoring", "landscape",
    "foster", "fosters", "fostering", "testament", "enhance", "enhances",
    "crucial", "robust", "seamless", "seamlessly", "leverage", "leveraging",
    "holistic", "paramount", "multifaceted", "myriad", "realm", "navigate",
    "navigating", "bolster", "bolstering", "cultivate", "cultivating",
    "elevate", "elevating", "unparalleled", "indelible", "ever-evolving",
    "boasts", "stands as a", "plays a vital role", "plays a crucial role",
    "serves as a testament", "rich history", "rich cultural heritage",
    "vibrant", "align with", "garner", "meticulous", "meticulously", "interplay",
    "profound", "exemplifies", "showcasing", "natural beauty", "nestled",
    "in the heart of", "groundbreaking", "renowned", "diverse array", "additionally",
    # ── Expanded vocabulary (v2) ──
    "spearhead", "spearheading", "embark", "embarking", "resonate", "resonates",
    "juxtaposition", "synergy", "burgeoning", "confluence",
    "catalyze", "catalyzing", "galvanize", "galvanizing", "nuanced",
    "cornerstone", "linchpin", "bulwark", "bedrock", "ethos",
    "pivoting", "underpin", "underpinning", "dovetail", "dovetails",
    "facet", "facets", "overarching", "underscored",
]

INFLATED_SIGNIFICANCE_PHRASES = [
    r"\bplays? an? (vital|crucial|pivotal|key|significant) role\b",
    r"\bserves? as an? testament\b",
    r"\bstands? as an? testament\b",
    r"\bleaves? a lasting (impact|impression|legacy)\b",
    r"\bwatershed moment\b",
    r"\bmarks? a (significant|pivotal|key) (turning point|moment)\b",
    r"\benduring legacy\b",
    r"\brich (cultural )?heritage\b",
    r"\bis a reminder\b",
    r"\bunderscores? (its|the) importance\b",
    r"\bhighlights? (its|the) importance\b",
    r"\bunderscores? (its|the) significance\b",
    r"\bhighlights? (its|the) significance\b",
    r"\breflects? broader\b",
    r"\bsymboliz(e|es|ing) its (ongoing|enduring|lasting)\b",
    r"\bsetting the stage for\b",
    r"\bmarking the\b",
    r"\bshaping the\b",
]

COMPULSIVE_SUMMARY_OPENERS = [
    r"\bin summary\b", r"\bin conclusion\b", r"\boverall\b,",
    r"\bto sum (it |things )?up\b", r"\bultimately\b,",
]

EDITORIALIZING_PHRASES = [
    r"\bit'?s important to note\b", r"\bit is important to note\b",
    r"\bno discussion (of|about) .{0,40} would be complete without\b",
    r"\bit is worth (noting|remembering|mentioning)\b",
    r"\bit'?s worth (noting|remembering|mentioning)\b",
]

VAGUE_ATTRIBUTION_PHRASES = [
    r"\bindustry experts (say|believe|argue)\b",
    r"\bsome critics (argue|say|believe)\b",
    r"\bobservers have (noted|argued|said)\b",
    r"\bmany (believe|argue|say)\b",
    r"\bexperts (say|agree|believe)\b",
    r"\bindustry reports?\b",
    r"\bseveral sources\b",
]

KNOWLEDGE_DISCLAIMER_PHRASES = [
    r"\bas of (?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
    r"\bup to my last training update\b",
    r"\bas of my last knowledge update\b",
    r"\bnot widely (?:available|documented|disclosed)\b",
    r"\bbased on available information\b",
    r"\bin the provided sources\b",
    r"\bwhile specific details are (?:limited|scarce)\b",
]

UTM_PARAMETER_PHRASES = [
    r"\butm_source=(?:openai|chatgpt\.com|copilot\.com)\b",
    r"\breferrer=grok\.com\b",
]

PUA_CITATION_ARTIFACTS = [
    r"\bturn0search\d+\b",
    r"\bturn0image\d+\b",
    r"【\d+†(?:L\d+)?(?:-\d+)?】",
    r"\[cite:\s*\d+(?:,\s*\d+)*\]",
]

LETTER_STYLE_PHRASES = [
    r"\bi hope this (message|email) finds you well\b",
    r"\bthank you for your time and consideration\b",
    r"\bi (am|'m) writing to\b",
    r"\bplease (do not hesitate|feel free) to (reach out|contact)\b",
]

LEFTOVER_CHAT_ARTIFACTS = [
    r"\bi hope this helps\b", r"^\s*of course!", r"^\s*certainly!",
    r"\blet me know if you (need|have)\b", r"\bas an ai (language model)?\b",
    r"\bi don'?t have personal (opinions|feelings)\b",
    r"\bhappy to help\b",
]

MEDIA_COVERAGE_PHRASES = [
    r"\bfeatured in (?:regional|local|national|major|prominent) media(?:\s+outlets)?\b",
    r"\bprominent media outlets\b",
    r"\bindependent coverage\b",
    r"\btrade publications\b",
    r"\bprofiled in\b",
    r"\bactive social media presence\b",
]

CITATION_BUG_PHRASES = [
    r"\bcontentReference\b",
    r"\boaicite(?::\d+)?\b",
    r"\boai_citation(?::\d+)?\b",
    r"\battached_file\b",
    r"\bgrok_card\b",
]

NEGATIVE_PARALLELISM_PATTERNS = [
    r"\bit'?s not (?:just )?[\w\s']{2,40}?, it'?s [\w\s']{2,40}",
    r"\bnot (?:only|just) [\w\s']{2,40}? but (?:also )?[\w\s']{2,40}",
    r"\b[\w\s']{2,30}? rather than [\w\s']{2,30}\b",
    r"\bnot [\w\s']{2,30}? but (?:rather )?[\w\s']{2,30}\b",
    r"\bno [\w\s']{2,30}?, no [\w\s']{2,30}?, just [\w\s']{2,30}\b",
]
FALSE_RANGE_PATTERN = r"\bfrom [\w\s]{2,30}? to [\w\s]{2,30}?(?=[.,;])"

PROMOTIONAL_SUPERLATIVE_PHRASES = [
    r"\bone of the most\b", r"\bwidely regarded as\b",
    r"\ba leading (figure|provider|company|authority|voice)\b",
    r"\bworld-class\b", r"\bindustry-leading\b",
    r"\binternationally (acclaimed|recognized|renowned)\b",
    r"\bcutting-edge\b", r"\bstate-of-the-art\b",
    r"\bpremier\b", r"\btrailblaz(er|ing)\b",
    r"\bunmatched\b", r"\bsecond to none\b",
    r"\bglobally recognized\b", r"\bpioneering\b",
]

EXCESSIVE_TRANSITION_PHRASES = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bconsequently\b",
    r"\bin addition(?:\s+to this)?\b", r"\bnevertheless\b",
    r"\bnonetheless\b", r"\bhence\b", r"\bthus\b,",
    r"\bthereby\b", r"\baccordingly\b",
]

SENTENCE_INITIAL_ADVERBS = [
    r"(?:^|(?<=\.\s))Notably\b",
    r"(?:^|(?<=\.\s))Importantly\b",
    r"(?:^|(?<=\.\s))Interestingly\b",
    r"(?:^|(?<=\.\s))Remarkably\b",
    r"(?:^|(?<=\.\s))Significantly\b",
    r"(?:^|(?<=\.\s))Crucially\b",
    r"(?:^|(?<=\.\s))Essentially\b",
    r"(?:^|(?<=\.\s))Fundamentally\b",
]

TEMPORAL_VAGUENESS_PHRASES = [
    r"\bin recent years\b", r"\bover the past decade\b",
    r"\bin the coming years\b", r"\bin recent times\b",
    r"\bthroughout history\b", r"\bover the years\b",
    r"\bfor (?:many|several) years\b", r"\bin modern times\b",
    r"\bhistorically\b,", r"\bas time (?:went|goes) on\b",
]

DEEPSEEK_ARTIFACT_PHRASES = [
    r"\battributableIndex\b",
    r":::writing\b",
    r":::thinking\b",
    r"\bsearch_result_\d+\b",
]

INLINE_HEADER_LIST_PATTERNS = [
    r"(?m)^\s*[-*\u2022]\s*\*\*[^*]+\*\*\s*[:—–\-]",
]

CONSTRUCTIVE_CRITICISM_PHRASES = [
    r"\bi welcome (?:any |your )?(?:feedback|suggestions|input)\b",
    r"\bfeel free to (?:suggest|provide|offer) (?:improvements|feedback|corrections)\b",
    r"\bopen to (?:suggestions|feedback|constructive criticism)\b",
    r"\bdon'?t hesitate to (?:point out|suggest|correct)\b",
    r"\bi(?:'m| am) happy to (?:make|incorporate) (?:any )?(?:changes|revisions|corrections)\b",
    r"\bplease (?:let me know|feel free) if (?:any|you have) (?:changes|corrections|suggestions)\b",
]

PASSIVE_VOICE_PATTERNS = [
    r"\b(?:is|are|was|were|been|being)\s+(?:\w+ly\s+)?\w+ed\b",
]

# ── Pattern weights (max points each pattern can contribute) ───────────────────

PATTERN_WEIGHTS = {
    "ai_vocabulary": 18,
    "inflated_significance": 12,
    "negative_parallelism": 10,
    "false_ranges": 8,
    "compulsive_summary": 10,
    "editorializing": 12,
    "vague_attribution": 10,
    "letter_style_formality": 8,
    "leftover_chat_artifacts": 18,  # near-certain tell if present
    "em_dash_overuse": 8,
    "rule_of_three": 8,
    "formatting_overkill": 6,
    "media_puffery": 10,
    "citation_bugs": 14,
    "curly_quotes": 8,
    "markdown_leak": 10,
    "collective_we": 6,
    "copulative_avoidance": 8,
    "pua_citation_bugs": 15,
    "knowledge_cutoff_disclaimers": 10,
    "utm_parameters": 16,
    # ── New patterns (v2) ──
    "promotional_superlatives": 10,
    "excessive_transitions": 8,
    "sentence_initial_adverbs": 8,
    "temporal_vagueness": 6,
    "deepseek_artifacts": 16,
    "title_case_headings": 6,
    "inline_header_lists": 8,
    "constructive_criticism": 8,
    # ── Wikipedia & General AI tells (v3) ──
    "lead_proper_noun": 8,
    "thematic_break_headings": 6,
    "skipped_headings": 6,
    "consecutive_sentence_starters": 8,
    "list_item_sentence_count_uniformity": 8,
}

# Minimum floor points if ANY match found (dead-giveaway patterns)
PATTERN_FLOORS = {
    "editorializing": 8,
    "leftover_chat_artifacts": 25,  # immediate possibly-AI threshold
    "letter_style_formality": 6,
    "inflated_significance": 6,
    "vague_attribution": 5,
    "compulsive_summary": 5,
    "citation_bugs": 10,
    "markdown_leak": 8,
    "pua_citation_bugs": 25,  # immediate possibly-AI threshold
    "utm_parameters": 25,  # immediate possibly-AI threshold
    # ── New floors (v2) ──
    "promotional_superlatives": 5,
    "deepseek_artifacts": 25,  # dead-giveaway, immediate possibly-AI
    "inline_header_lists": 5,
    "constructive_criticism": 6,
    # ── New floors (v3) ──
    "lead_proper_noun": 5,
}


# ── Individual pattern detectors ───────────────────────────────────────────────

def _find_all(patterns: List[str], text: str, flags=re.IGNORECASE) -> List[str]:
    spans = []
    for p in patterns:
        for m in re.finditer(p, text, flags):
            spans.append(m.group(0).strip())
    return spans


def detect_ai_vocab(text: str) -> Dict:
    words = re.findall(r"[a-zA-Z\-]+", text.lower())
    hits = [w for w in words if w in AI_VOCAB]
    phrase_hits = [p for p in AI_VOCAB if " " in p and p in text.lower()]
    all_hits = hits + phrase_hits
    return {"count": len(all_hits), "matches": all_hits[:25]}


def detect_inflated_significance(text: str) -> Dict:
    m = _find_all(INFLATED_SIGNIFICANCE_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_negative_parallelism(text: str) -> Dict:
    m = _find_all(NEGATIVE_PARALLELISM_PATTERNS, text)
    return {"count": len(m), "matches": m}


def detect_false_ranges(text: str) -> Dict:
    spans = re.findall(FALSE_RANGE_PATTERN, text, re.IGNORECASE)
    return {"count": len(spans), "matches": spans[:10]}


def detect_compulsive_summary(text: str) -> Dict:
    m = _find_all(COMPULSIVE_SUMMARY_OPENERS, text)
    return {"count": len(m), "matches": m}


def detect_editorializing(text: str) -> Dict:
    m = _find_all(EDITORIALIZING_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_vague_attribution(text: str) -> Dict:
    m = _find_all(VAGUE_ATTRIBUTION_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_letter_style(text: str) -> Dict:
    m = _find_all(LETTER_STYLE_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_leftover_chat_artifacts(text: str) -> Dict:
    m = _find_all(LEFTOVER_CHAT_ARTIFACTS, text, flags=re.IGNORECASE | re.MULTILINE)
    return {"count": len(m), "matches": m}


def detect_media_puffery(text: str) -> Dict:
    m = _find_all(MEDIA_COVERAGE_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_citation_bugs(text: str) -> Dict:
    m = _find_all(CITATION_BUG_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_em_dash_overuse(text: str) -> Dict:
    matches = re.findall(r"\u2014|--", text)
    words = max(len(text.split()), 1)
    rate_per_100w = len(matches) / words * 100
    return {
        "count": len(matches),
        "rate_per_100_words": round(rate_per_100w, 2),
        "matches": list(set(matches))
    }


def detect_rule_of_three(text: str) -> Dict:
    pattern = r"\b(?:[\w']+(?:\s+[\w']+){0,2}),\s+(?:[\w']+(?:\s+[\w']+){0,2}),?\s+and\s+(?:[\w']+(?:\s+[\w']+){0,2})\b"
    spans = re.findall(pattern, text, re.IGNORECASE)
    return {"count": len(spans), "matches": [s.strip() for s in spans][:10]}


def detect_curly_quotes(text: str) -> Dict:
    matches = re.findall(r"[“”‘’’]", text)
    return {"count": len(matches), "matches": list(set(matches))}


def detect_markdown_leak(text: str) -> Dict:
    markdown_link = r"\[[^\]\n]{2,80}\]\(https?://[^\s)]+\)"
    markdown_heading = r"(?m)^\s*#{2,4}\s+[^\n]+$"
    code_block = r"```\w*"
    links = re.findall(markdown_link, text)
    headings = re.findall(markdown_heading, text)
    blocks = re.findall(code_block, text)
    matches = links + headings + [b.strip() for b in blocks]
    return {"count": len(matches), "matches": matches[:10]}


def detect_collective_we(text: str) -> Dict:
    we_patterns = [
        r"\bas we (?:explore|examine|delve|navigate|see|look|discuss)\b",
        r"\blet us (?:explore|explore|examine|delve|navigate|see|look|discuss)\b",
        r"\bin this (?:section|chapter|article|essay) we\b",
    ]
    m = _find_all(we_patterns, text)
    return {"count": len(m), "matches": m}


def detect_copulative_avoidance(text: str) -> Dict:
    cop_patterns = [
        r"\bserves? as\b",
        r"\bstands? as\b",
        r"\brefers? to\b",
        r"\brepresents?\b",
    ]
    m = _find_all(cop_patterns, text)
    return {"count": len(m), "matches": m}


def detect_knowledge_cutoff_disclaimers(text: str) -> Dict:
    m = _find_all(KNOWLEDGE_DISCLAIMER_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_utm_parameters(text: str) -> Dict:
    m = _find_all(UTM_PARAMETER_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_pua_citation_bugs(text: str) -> Dict:
    m = _find_all(PUA_CITATION_ARTIFACTS, text)
    return {"count": len(m), "matches": m}


def detect_formatting_overkill(text: str) -> Dict:
    bolds = re.findall(r"\*\*[^*]+\*\*", text)
    bullets = re.findall(r"(?m)^\s*[-*\u2022]\s+", text)
    emojis = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text)
    matches = bolds + bullets + emojis
    return {
        "count": len(matches),
        "bold_count": len(bolds),
        "bullet_count": len(bullets),
        "emoji_count": len(emojis),
        "matches": matches[:20]
    }


def detect_burstiness(text: str) -> Dict:
    """Low sentence-length variance = weak AI signal."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    if len(lengths) < 3:
        return {"sentence_count": len(lengths), "stdev": None, "mean": None}
    return {
        "sentence_count": len(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "stdev": round(statistics.stdev(lengths), 2),
        "coefficient_of_variation": round(
            statistics.stdev(lengths) / max(statistics.mean(lengths), 1), 3
        ),
    }


def detect_lexical_diversity(text: str) -> Dict:
    """Type-token ratio: AI text often shows narrower vocabulary."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return {"ttr": None}
    ttr = len(set(words)) / len(words)
    return {"unique_words": len(set(words)), "total_words": len(words), "ttr": round(ttr, 3)}


# ── New pattern detectors (v2) ─────────────────────────────────────────────────

def detect_promotional_superlatives(text: str) -> Dict:
    m = _find_all(PROMOTIONAL_SUPERLATIVE_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_excessive_transitions(text: str) -> Dict:
    m = _find_all(EXCESSIVE_TRANSITION_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_sentence_initial_adverbs(text: str) -> Dict:
    m = _find_all(SENTENCE_INITIAL_ADVERBS, text, flags=re.MULTILINE)
    return {"count": len(m), "matches": m}


def detect_temporal_vagueness(text: str) -> Dict:
    m = _find_all(TEMPORAL_VAGUENESS_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_deepseek_artifacts(text: str) -> Dict:
    m = _find_all(DEEPSEEK_ARTIFACT_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_title_case_headings(text: str) -> Dict:
    """Detect AI-style title-cased headings like 'Early Life And Career'.
    Looks for lines where 3+ words are capitalized (excluding short articles/preps)."""
    MINOR_WORDS = {"a", "an", "the", "and", "or", "but", "in", "on", "at",
                   "to", "for", "of", "by", "with", "from", "as", "is", "nor"}
    heading_re = re.compile(r"(?m)^(?:#{1,4}\s+)?(.+)$")
    matches = []
    for m in heading_re.finditer(text):
        line = m.group(1).strip()
        words = line.split()
        if len(words) < 3:
            continue
        # Count words that are capitalized (excluding minor words in middle positions)
        cap_count = 0
        for i, w in enumerate(words):
            clean = re.sub(r"[^a-zA-Z]", "", w)
            if not clean:
                continue
            if clean[0].isupper() and clean.lower() not in MINOR_WORDS:
                cap_count += 1
            elif i > 0 and clean[0].isupper() and clean.lower() in MINOR_WORDS:
                # Minor word capitalized mid-heading = strong title-case signal
                cap_count += 1
        # Flag if most non-trivial words are capitalized
        if cap_count >= len(words) * 0.7 and cap_count >= 3:
            matches.append(line)
    return {"count": len(matches), "matches": matches[:10]}


def detect_inline_header_lists(text: str) -> Dict:
    m = _find_all(INLINE_HEADER_LIST_PATTERNS, text, flags=re.MULTILINE)
    return {"count": len(m), "matches": m}


def detect_constructive_criticism(text: str) -> Dict:
    m = _find_all(CONSTRUCTIVE_CRITICISM_PHRASES, text)
    return {"count": len(m), "matches": m}


def detect_passive_voice(text: str) -> Dict:
    """Detect passive voice constructions. Counts all matches for highlighting,
    but scoring only kicks in above a rate threshold (handled in heuristic_score)."""
    m = _find_all(PASSIVE_VOICE_PATTERNS, text)
    words = max(len(text.split()), 1)
    rate_per_100w = len(m) / words * 100
    return {"count": len(m), "rate_per_100_words": round(rate_per_100w, 2), "matches": m[:20]}


def detect_paragraph_uniformity(text: str) -> Dict:
    """Low variance in paragraph lengths suggests AI authorship.
    Silently skips texts with fewer than 3 paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    lengths = [len(p.split()) for p in paragraphs if len(p.split()) > 0]
    if len(lengths) < 3:
        return {"paragraph_count": len(lengths), "stdev": None, "mean": None}
    mean_len = statistics.mean(lengths)
    stdev_len = statistics.stdev(lengths)
    cv = stdev_len / max(mean_len, 1)
    return {
        "paragraph_count": len(lengths),
        "mean_length": round(mean_len, 1),
        "stdev": round(stdev_len, 1),
        "coefficient_of_variation": round(cv, 3),
    }


# ── Wikipedia & General AI Tells (v3) ──

def detect_lead_proper_noun(text: str) -> Dict:
    """Detect AI writing leads treating list/titles as proper nouns."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return {"count": 0, "matches": []}
    first_sent = sentences[0]
    pattern = r"^(?:List|History|Outline|Timeline) of\s+[A-Z]\w*(?:\s+[a-zA-Z0-9'\-]+){0,10}?\s+(?:is|was|are|were)\b"
    m = re.findall(pattern, first_sent)
    return {"count": len(m), "matches": m}


def detect_thematic_break_headings(text: str) -> Dict:
    """Detect thematic breaks directly preceding section headings."""
    pattern = r"(?m)^-{3,5}\s*\n\s*#{1,6}\s+.+$"
    m = re.findall(pattern, text)
    return {"count": len(m), "matches": m}


def detect_skipped_headings(text: str) -> Dict:
    """Detect skipped heading levels (e.g., H2 followed by H4)."""
    headings = re.findall(r"(?m)^\s*(#{1,6})\s+", text)
    skipped = []
    prev_level = None
    for h in headings:
        level = len(h)
        if prev_level is not None:
            if level > prev_level + 1:
                skipped.append(f"Skipped from H{prev_level} to H{level}")
        prev_level = level
    return {"count": len(skipped), "matches": skipped}


def detect_consecutive_sentence_starters(text: str) -> Dict:
    """Detect consecutive sentences starting with the same words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    matches = []
    prev_prefix = None
    for s in sentences:
        words = s.lower().split()
        if len(words) >= 3:
            prefix = " ".join(words[:3])
            prefix = re.sub(r"[^\w\s]", "", prefix).strip()
            if prev_prefix and prefix == prev_prefix:
                matches.append(prefix)
            prev_prefix = prefix
        else:
            prev_prefix = None
    return {"count": len(matches), "matches": list(set(matches))}


def detect_list_uniformity(text: str) -> Dict:
    """Detect uniform list item sentence counts."""
    bullets = re.findall(r"(?m)^\s*[-*\u2022]\s+(.+)$", text)
    if len(bullets) < 3:
        return {"count": 0, "matches": [], "cv": 0.0}
    
    sentence_counts = []
    for b in bullets:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", b) if s.strip()]
        sentence_counts.append(len(sents))
    
    if len(sentence_counts) >= 3:
        mean_len = statistics.mean(sentence_counts)
        stdev_len = statistics.stdev(sentence_counts)
        if mean_len > 0:
            cv = stdev_len / mean_len
            if cv < 0.2:
                return {"count": 1, "matches": [f"Uniform sentence count list: {sentence_counts}"], "cv": round(cv, 3)}
    return {"count": 0, "matches": [], "cv": 0.0}


# ── Aggregate extraction ──────────────────────────────────────────────────────

def extract_all_patterns(text: str) -> Dict:
    """Run every detector, return single results dict."""
    return {
        "ai_vocabulary": detect_ai_vocab(text),
        "inflated_significance": detect_inflated_significance(text),
        "negative_parallelism": detect_negative_parallelism(text),
        "false_ranges": detect_false_ranges(text),
        "compulsive_summary": detect_compulsive_summary(text),
        "editorializing": detect_editorializing(text),
        "vague_attribution": detect_vague_attribution(text),
        "letter_style_formality": detect_letter_style(text),
        "leftover_chat_artifacts": detect_leftover_chat_artifacts(text),
        "media_puffery": detect_media_puffery(text),
        "citation_bugs": detect_citation_bugs(text),
        "em_dash_overuse": detect_em_dash_overuse(text),
        "rule_of_three": detect_rule_of_three(text),
        "formatting_overkill": detect_formatting_overkill(text),
        "burstiness": detect_burstiness(text),
        "lexical_diversity": detect_lexical_diversity(text),
        "curly_quotes": detect_curly_quotes(text),
        "markdown_leak": detect_markdown_leak(text),
        "collective_we": detect_collective_we(text),
        "copulative_avoidance": detect_copulative_avoidance(text),
        "pua_citation_bugs": detect_pua_citation_bugs(text),
        "knowledge_cutoff_disclaimers": detect_knowledge_cutoff_disclaimers(text),
        "utm_parameters": detect_utm_parameters(text),
        # ── New patterns (v2) ──
        "promotional_superlatives": detect_promotional_superlatives(text),
        "excessive_transitions": detect_excessive_transitions(text),
        "sentence_initial_adverbs": detect_sentence_initial_adverbs(text),
        "temporal_vagueness": detect_temporal_vagueness(text),
        "deepseek_artifacts": detect_deepseek_artifacts(text),
        "title_case_headings": detect_title_case_headings(text),
        "inline_header_lists": detect_inline_header_lists(text),
        "constructive_criticism": detect_constructive_criticism(text),
        "passive_voice": detect_passive_voice(text),
        "paragraph_uniformity": detect_paragraph_uniformity(text),
        # ── Wikipedia & General AI tells (v3) ──
        "lead_proper_noun": detect_lead_proper_noun(text),
        "thematic_break_headings": detect_thematic_break_headings(text),
        "skipped_headings": detect_skipped_headings(text),
        "consecutive_sentence_starters": detect_consecutive_sentence_starters(text),
        "list_item_sentence_count_uniformity": detect_list_uniformity(text),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _saturating(rate: float, scale: float = 2.0) -> float:
    """Maps per-100-word rate to 0..1 with diminishing returns."""
    return 1 - math.exp(-rate / scale)


def heuristic_score(text: str, patterns: Optional[Dict] = None) -> Dict:
    if patterns is None:
        patterns = extract_all_patterns(text)

    word_count = max(len(text.split()), 1)
    contributions = {}

    for key, weight in PATTERN_WEIGHTS.items():
        count = patterns[key].get("count", 0)
        rate_per_100 = count / word_count * 100
        score_from_rate = weight * _saturating(rate_per_100)

        # Apply floor for dead-giveaway patterns (any single match = strong signal)
        if count > 0 and key in PATTERN_FLOORS:
            score_from_rate = max(score_from_rate, PATTERN_FLOORS[key])

        contributions[key] = round(score_from_rate, 2)

    # Burstiness: low sentence-length variance = weak AI signal
    burst = patterns["burstiness"]
    burstiness_contribution = 0.0
    if burst.get("coefficient_of_variation") is not None:
        cv = burst["coefficient_of_variation"]
        if cv < 0.35:
            burstiness_contribution = round((0.35 - cv) / 0.35 * 8, 2)
    contributions["low_sentence_variance"] = burstiness_contribution

    # AI vocabulary density bonus — stacking multiple AI words = much stronger signal
    vocab_count = patterns["ai_vocabulary"].get("count", 0)
    if vocab_count >= 2:
        density = vocab_count / word_count
        # If 5%+ of words are AI vocab, strong bonus
        density_bonus = min(density * 200, 15)  # up to 15 bonus points
        contributions["ai_pattern_density"] = round(density_bonus, 2)

    # Passive voice: only contribute when rate exceeds 5 per 100 words
    pv = patterns["passive_voice"]
    pv_rate = pv.get("rate_per_100_words", 0)
    pv_contribution = 0.0
    if pv_rate > 5.0:
        # Scale 0..6 based on how far above threshold
        pv_contribution = round(min((pv_rate - 5.0) / 5.0 * 6, 6), 2)
    elif pv.get("count", 0) > 0 and pv_rate > 3.0:
        # Mild signal between 3-5 per 100 words
        pv_contribution = round((pv_rate - 3.0) / 2.0 * 3, 2)
    contributions["passive_voice_overuse"] = pv_contribution

    # Paragraph uniformity: low CV = AI-like uniform paragraphs
    para = patterns["paragraph_uniformity"]
    para_contribution = 0.0
    if para.get("coefficient_of_variation") is not None:
        pcv = para["coefficient_of_variation"]
        if pcv < 0.30:
            para_contribution = round((0.30 - pcv) / 0.30 * 6, 2)
    contributions["paragraph_uniformity"] = para_contribution

    total = min(sum(contributions.values()), 100.0)
    detected = {k: v for k, v in contributions.items() if v > 0.5}
    detected_sorted = dict(sorted(detected.items(), key=lambda kv: -kv[1]))

    if total >= 55:
        verdict = "Likely AI-generated"
    elif total >= 25:
        verdict = "Possibly AI-assisted / mixed"
    else:
        verdict = "Likely human-written"

    return {
        "heuristic_score": round(total, 1),
        "verdict": verdict,
        "contributions": contributions,
        "top_detected_patterns": detected_sorted,
        "raw_patterns": patterns,
    }


# ── ML model loading ─────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def load_model(model_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load pickled sklearn model + vectorizer if they exist.
    Returns dict with 'classifier', 'vectorizer', 'heuristic_transformer' or None."""
    d = model_dir or _MODEL_DIR
    clf_path = os.path.join(d, "classifier.pkl")
    vec_path = os.path.join(d, "vectorizer.pkl")

    if not (os.path.exists(clf_path) and os.path.exists(vec_path)):
        return None

    with open(clf_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    heur_tf = None
    heur_path = os.path.join(d, "heuristic_transformer.pkl")
    if os.path.exists(heur_path):
        with open(heur_path, "rb") as f:
            heur_tf = pickle.load(f)

    return {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "heuristic_transformer": heur_tf,
    }


# ── Combined predictor ────────────────────────────────────────────────────────

def predict(text: str, ml_models: Optional[Dict[str, Any]] = None, detection_type: str = "all") -> Dict:
    """Run heuristic scorer, optionally blend with ML predictions from specified model type."""
    heur = heuristic_score(text)
    word_count = max(len(text.split()), 1)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    result = {
        "text_preview": text.strip()[:120] + ("..." if len(text.strip()) > 120 else ""),
        "word_count": word_count,
        "sentence_count": len(sentences),
        "heuristic_score": heur["heuristic_score"],
        "heuristic_verdict": heur["verdict"],
        "detected_patterns": heur["top_detected_patterns"],
        "pattern_examples": {
            k: heur["raw_patterns"][k]["matches"]
            for k in heur["raw_patterns"]
            if isinstance(heur["raw_patterns"][k], dict)
            and heur["raw_patterns"][k].get("matches")
        },
    }

    # Convert single ml_model dict to dict of models to preserve backward compatibility
    models_dict = {}
    if ml_models is not None:
        if isinstance(ml_models, dict) and "classifier" in ml_models:
            models_dict = {"default": ml_models}
        elif isinstance(ml_models, dict):
            models_dict = ml_models

    model_scores = {}
    ml_prob = None
    model_used = "heuristic"

    if models_dict:
        try:
            from scipy.sparse import hstack

            # Helper to run a single model prediction
            def _predict_single(model, heur_data):
                tfidf_vec = model["vectorizer"].transform([text])
                heur_tf = model.get("heuristic_transformer")
                if heur_tf is not None:
                    heur_vec = heur_tf.transform([text])
                else:
                    heur_vec = _build_heuristic_vector(heur_data)
                combined_vec = hstack([tfidf_vec, heur_vec])
                prob = float(model["classifier"].predict_proba(combined_vec)[0, 1])
                return prob

            if detection_type in ("scientific", "general", "wikipedia"):
                model = models_dict.get(detection_type)
                model_name = detection_type
                if model is None:
                    model = models_dict.get("default")
                    model_name = "default"

                if model is not None:
                    prob = _predict_single(model, heur)
                    ml_prob = prob * 100
                    model_used = f"heuristic+ml ({model_name})"
            else:
                # detection_type == "all" (or unknown, default to all)
                active_probs = []
                for name in ("scientific", "general", "wikipedia", "default"):
                    model = models_dict.get(name)
                    if model is not None:
                        prob = _predict_single(model, heur)
                        score_pct = round(prob * 100, 1)
                        model_scores[name] = score_pct
                        active_probs.append(score_pct)

                if active_probs:
                    ml_prob = sum(active_probs) / len(active_probs)
                    model_used = "heuristic+ml (blended)"
        except Exception:
            # Silently fall back to heuristic-only
            pass

    if ml_prob is not None:
        result["ml_probability_ai"] = round(ml_prob, 1)
        blended = 0.6 * ml_prob + 0.4 * heur["heuristic_score"]
        result["final_score"] = round(blended, 1)
        result["model_used"] = model_used
        if model_scores:
            result["model_scores"] = model_scores
    else:
        result["final_score"] = heur["heuristic_score"]
        result["model_used"] = "heuristic"

    score = result["final_score"]
    if score >= 55:
        result["final_verdict"] = "Likely AI-generated"
    elif score >= 25:
        result["final_verdict"] = "Possibly AI-assisted / mixed"
    else:
        result["final_verdict"] = "Likely human-written"

    result["disclaimer"] = "Style diagnostic only — not proof of authorship."
    return result


def _build_heuristic_vector(heur: Dict):
    """Fallback when no pickled HeuristicFeatureTransformer available."""
    import numpy as np

    row = [heur["contributions"].get(k, 0.0) for k in PATTERN_WEIGHTS.keys()]
    row.append(heur["contributions"].get("low_sentence_variance", 0.0))
    row.append(heur["contributions"].get("passive_voice_overuse", 0.0))
    row.append(heur["contributions"].get("paragraph_uniformity", 0.0))
    ttr = heur["raw_patterns"]["lexical_diversity"].get("ttr") or 0.0
    row.append(ttr)
    return np.array([row])
