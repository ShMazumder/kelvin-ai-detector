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

NEGATIVE_PARALLELISM_PATTERN = r"\bit'?s not (?:just )?[\w\s]{2,40}?, it'?s [\w\s]{2,40}"
FALSE_RANGE_PATTERN = r"\bfrom [\w\s]{2,30}? to [\w\s]{2,30}?(?=[.,;])"

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
}

# Minimum floor points if ANY match found (dead-giveaway patterns)
PATTERN_FLOORS = {
    "editorializing": 8,
    "leftover_chat_artifacts": 12,
    "letter_style_formality": 6,
    "inflated_significance": 6,
    "vague_attribution": 5,
    "compulsive_summary": 5,
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
    spans = re.findall(NEGATIVE_PARALLELISM_PATTERN, text, re.IGNORECASE)
    return {"count": len(spans), "matches": [s.strip() for s in spans][:10]}


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


def detect_em_dash_overuse(text: str) -> Dict:
    em_dashes = text.count("\u2014") + text.count("--")
    words = max(len(text.split()), 1)
    rate_per_100w = em_dashes / words * 100
    return {"count": em_dashes, "rate_per_100_words": round(rate_per_100w, 2)}


def detect_rule_of_three(text: str) -> Dict:
    pattern = r"\b(\w+),\s+(\w+),?\s+and\s+(\w+)\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    spans = re.findall(r"\b\w+,\s+\w+,?\s+and\s+\w+\b", text, re.IGNORECASE)
    return {"count": len(matches), "matches": spans[:10]}


def detect_formatting_overkill(text: str) -> Dict:
    bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
    bullet_count = len(re.findall(r"(?m)^\s*[-*\u2022]\s+", text))
    emoji_count = len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))
    return {"bold_count": bold_count, "bullet_count": bullet_count, "emoji_count": emoji_count}


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
        "em_dash_overuse": detect_em_dash_overuse(text),
        "rule_of_three": detect_rule_of_three(text),
        "formatting_overkill": detect_formatting_overkill(text),
        "burstiness": detect_burstiness(text),
        "lexical_diversity": detect_lexical_diversity(text),
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

def predict(text: str, ml_model: Optional[Dict] = None) -> Dict:
    """Run heuristic scorer, optionally blend with ML prediction."""
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

    if ml_model is not None:
        try:
            from scipy.sparse import hstack

            tfidf_vec = ml_model["vectorizer"].transform([text])
            heur_tf = ml_model.get("heuristic_transformer")
            if heur_tf is not None:
                heur_vec = heur_tf.transform([text])
            else:
                # Inline fallback — build heuristic feature vector
                heur_vec = _build_heuristic_vector(heur)
            combined_vec = hstack([tfidf_vec, heur_vec])
            ml_prob = float(ml_model["classifier"].predict_proba(combined_vec)[0, 1])
            result["ml_probability_ai"] = round(ml_prob * 100, 1)
            # Blend: 60% ML, 40% heuristic
            blended = 0.6 * (ml_prob * 100) + 0.4 * heur["heuristic_score"]
            result["final_score"] = round(blended, 1)
            result["model_used"] = "heuristic+ml"
        except Exception:
            result["final_score"] = heur["heuristic_score"]
            result["model_used"] = "heuristic"
    else:
        result["final_score"] = heur["heuristic_score"]
        result["model_used"] = "heuristic"

    score = result["final_score"]
    if score >= 60:
        result["final_verdict"] = "Likely AI-generated"
    elif score >= 30:
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
    ttr = heur["raw_patterns"]["lexical_diversity"].get("ttr") or 0.0
    row.append(ttr)
    return np.array([row])
