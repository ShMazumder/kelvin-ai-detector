import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ----------------------------------------------------------------------
md('''# AI-Text Likeness Detector

This notebook builds a tool that estimates whether a piece of text is likely
AI-generated, **and** explains *why* — which specific stylistic "tells" it
picked up on (em dash overuse, rule-of-three lists, inflated significance
phrasing, vague attribution, etc.), based on the patterns documented in
Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
guide.

**Two layers, used together:**

1. **Heuristic / rule-based scorer** — no training data required. Scans text
   for ~12 known stylistic tells and produces a 0–100 "AI-likeness" score
   plus a breakdown of exactly which patterns fired and where.
2. **Optional ML classifier** — if you have a labeled dataset (e.g. Kaggle's
   [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
   competition data), this trains a TF-IDF + Logistic Regression model on top
   of the heuristic features for a statistically grounded prediction.

**Important caveat — read before relying on this for anything consequential:**
No AI-text detector, including this one, is reliable enough to use as proof
of authorship. Detectors produce both false positives (flagging human
writing) and false negatives (missing AI writing) regularly, and that
unreliability is exactly why Wikipedia's own editors rely on a *catalog of
patterns reviewed by a human*, not an automated score, when assessing
content. Treat the output here as a writing-style diagnostic, not a verdict.
''')

# ----------------------------------------------------------------------
md('## Setup')

code('''import re
import math
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
''')

# ----------------------------------------------------------------------
md('''## Part 1 — Heuristic Pattern Detector

Each function below detects one specific "tell" and returns:
- a **raw count / rate**
- the **matched spans** (so you can see exactly what triggered it)

The patterns are grouped the way Wikipedia's guide groups them: language &
tone, structural habits, vocabulary, and formatting.
''')

code('''# ---- Reference word/phrase lists -------------------------------------------------

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
    r"\\bplays? an? (vital|crucial|pivotal|key|significant) role\\b",
    r"\\bserves? as an? testament\\b",
    r"\\bstands? as an? testament\\b",
    r"\\bleaves? a lasting (impact|impression|legacy)\\b",
    r"\\bwatershed moment\\b",
    r"\\bmarks? a (significant|pivotal|key) (turning point|moment)\\b",
    r"\\benduring legacy\\b",
    r"\\brich (cultural )?heritage\\b",
]

COMPULSIVE_SUMMARY_OPENERS = [
    r"\\bin summary\\b", r"\\bin conclusion\\b", r"\\boverall\\b,",
    r"\\bto sum (it |things )?up\\b", r"\\bultimately\\b,",
]

EDITORIALIZING_PHRASES = [
    r"\\bit'?s important to note\\b", r"\\bit is important to note\\b",
    r"\\bno discussion (of|about) .{0,40} would be complete without\\b",
    r"\\bit is worth (noting|remembering|mentioning)\\b",
    r"\\bit'?s worth (noting|remembering|mentioning)\\b",
]

VAGUE_ATTRIBUTION_PHRASES = [
    r"\\bindustry experts (say|believe|argue)\\b",
    r"\\bsome critics (argue|say|believe)\\b",
    r"\\bobservers have (noted|argued|said)\\b",
    r"\\bmany (believe|argue|say)\\b",
    r"\\bexperts (say|agree|believe)\\b",
]

LETTER_STYLE_PHRASES = [
    r"\\bi hope this (message|email) finds you well\\b",
    r"\\bthank you for your time and consideration\\b",
    r"\\bi (am|'m) writing to\\b",
    r"\\bplease (do not hesitate|feel free) to (reach out|contact)\\b",
]

LEFTOVER_CHAT_ARTIFACTS = [
    r"\\bi hope this helps\\b", r"^\\s*of course!", r"^\\s*certainly!",
    r"\\blet me know if you (need|have)\\b", r"\\bas an ai (language model)?\\b",
    r"\\bi don'?t have personal (opinions|feelings)\\b",
    r"\\bhappy to help\\b",
]

NEGATIVE_PARALLELISM_PATTERN = r"\\bit'?s not (?:just )?[\\w\\s]{2,40}?, it'?s [\\w\\s]{2,40}"
FALSE_RANGE_PATTERN = r"\\bfrom [\\w\\s]{2,30}? to [\\w\\s]{2,30}?(?=[\\.,;])"

# ---- Individual pattern detectors ------------------------------------------------

def _find_all(patterns, text, flags=re.IGNORECASE):
    spans = []
    for p in patterns:
        for m in re.finditer(p, text, flags):
            spans.append(m.group(0).strip())
    return spans

def detect_ai_vocab(text: str) -> Dict:
    words = re.findall(r"[a-zA-Z\\-]+", text.lower())
    hits = [w for w in words if w in AI_VOCAB]
    # also catch multi-word phrases
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
    em_dashes = text.count("\\u2014") + text.count("--")
    words = max(len(text.split()), 1)
    rate_per_100w = em_dashes / words * 100
    return {"count": em_dashes, "rate_per_100_words": round(rate_per_100w, 2)}

def detect_rule_of_three(text: str) -> Dict:
    # Looks for "X, Y, and Z" triplet list constructions
    pattern = r"\\b(\\w+),\\s+(\\w+),?\\s+and\\s+(\\w+)\\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    spans = re.findall(r"\\b\\w+,\\s+\\w+,?\\s+and\\s+\\w+\\b", text, re.IGNORECASE)
    return {"count": len(matches), "matches": spans[:10]}

def detect_formatting_overkill(text: str) -> Dict:
    bold_count = len(re.findall(r"\\*\\*[^*]+\\*\\*", text))
    bullet_count = len(re.findall(r"(?m)^\\s*[-*\\u2022]\\s+", text))
    emoji_count = len(re.findall(
        r"[\\U0001F300-\\U0001FAFF\\u2600-\\u27BF]", text))
    return {"bold_count": bold_count, "bullet_count": bullet_count, "emoji_count": emoji_count}

def detect_burstiness(text: str) -> Dict:
    """Human writing tends to vary sentence length more (high 'burstiness').
    AI text tends to be more uniform (low variance in sentence length)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    if len(lengths) < 3:
        return {"sentence_count": len(lengths), "stdev": None, "mean": None}
    return {
        "sentence_count": len(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "stdev": round(statistics.stdev(lengths), 2),
        "coefficient_of_variation": round(statistics.stdev(lengths) / max(statistics.mean(lengths), 1), 3),
    }

def detect_lexical_diversity(text: str) -> Dict:
    """Type-token ratio: AI text (especially short samples) often shows a
    narrower vocabulary range than equivalent human writing."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return {"ttr": None}
    ttr = len(set(words)) / len(words)
    return {"unique_words": len(set(words)), "total_words": len(words), "ttr": round(ttr, 3)}
''')

code('''def extract_all_patterns(text: str) -> Dict:
    """Run every detector and return a single results dict."""
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
''')

# ----------------------------------------------------------------------
md('''### Scoring

Each pattern contributes a weighted amount to a 0–100 "AI-likeness" score.
Weights are deliberately conservative — single matches barely move the
needle, since these phrases do occur in normal human writing too. Density
(matches per 100 words) is what actually drives the score up.
''')

code('''# Weight = max points this pattern can contribute, scaled by how often it fires
PATTERN_WEIGHTS = {
    "ai_vocabulary": 14,
    "inflated_significance": 10,
    "negative_parallelism": 10,
    "false_ranges": 8,
    "compulsive_summary": 8,
    "editorializing": 8,
    "vague_attribution": 8,
    "letter_style_formality": 6,
    "leftover_chat_artifacts": 14,   # near-certain tell if present at all
    "em_dash_overuse": 8,
    "rule_of_three": 8,
    "formatting_overkill": 6,
}

def _saturating(rate, scale=3.0):
    """Maps a per-100-word rate to 0..1, saturating so a few hits don't
    instantly max out the score but heavy repetition does."""
    return 1 - math.exp(-rate / scale)

def heuristic_score(text: str, patterns: Dict = None) -> Dict:
    if patterns is None:
        patterns = extract_all_patterns(text)

    word_count = max(len(text.split()), 1)
    contributions = {}

    for key, weight in PATTERN_WEIGHTS.items():
        count = patterns[key].get("count", 0)
        rate_per_100 = count / word_count * 100
        contributions[key] = round(weight * _saturating(rate_per_100), 2)

    # Burstiness: low sentence-length variance is a (weak) AI signal
    burst = patterns["burstiness"]
    burstiness_contribution = 0.0
    if burst.get("coefficient_of_variation") is not None:
        cv = burst["coefficient_of_variation"]
        # CV below ~0.35 is unusually uniform for human writing
        if cv < 0.35:
            burstiness_contribution = round((0.35 - cv) / 0.35 * 6, 2)
    contributions["low_sentence_variance"] = burstiness_contribution

    total = sum(contributions.values())
    total = min(total, 100.0)

    detected = {k: v for k, v in contributions.items() if v > 1.0}
    detected_sorted = dict(sorted(detected.items(), key=lambda kv: -kv[1]))

    if total >= 60:
        verdict = "Likely AI-generated"
    elif total >= 30:
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
''')

# ----------------------------------------------------------------------
md('''### Try it on two contrasting examples
''')

code('''ai_sounding_example = """
In today\\'s rapidly evolving digital landscape, it is important to note that effective
communication plays a pivotal role in fostering meaningful engagement. Our platform
boasts a robust, seamless, and intuitive experience, serving as a testament to our
unwavering commitment to excellence. From small startups to global enterprises, we
empower organizations to navigate the complexities of modern business. It\\'s not just
a tool, it\\'s a transformative solution. In summary, our holistic approach underscores
the crucial role that innovation, collaboration, and adaptability play in driving
sustainable growth. I hope this helps! Let me know if you need anything else.
"""

human_sounding_example = """
I spent most of Tuesday trying to get the printer to talk to the new laptop and honestly
gave up around 4pm. Took it apart, found a paper jam from two weeks ago that I never
noticed. Fixed that, printer still wouldn\\'t connect. Turned out the wifi password had
changed when Dad reset the router. Classic. Anyway it works now but I\\'m never buying
that brand again -- the setup app alone took 25 minutes and crashed twice.
"""

for label, txt in [("AI-sounding example", ai_sounding_example),
                    ("Human-sounding example", human_sounding_example)]:
    result = heuristic_score(txt)
    score = result["heuristic_score"]
    verdict = result["verdict"]
    print(f"--- {label} ---")
    print(f"Score: {score}/100  ->  {verdict}")
    print("Top detected patterns:")
    for k, v in result["top_detected_patterns"].items():
        print(f"   {k}: {v} pts")
    print()
''')

# ----------------------------------------------------------------------
md('''## Part 2 — Optional ML Classifier (needs labeled data)

This section trains a statistical model on top of the heuristic features.
It expects a labeled dataset with text + a 0/1 "is AI generated" column —
for example Kaggle's
[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
competition (`train_essays.csv`, columns: `text`, `generated`).

**If you're running this in a Kaggle notebook**, attach that dataset (or any
labeled essay-pair dataset) via *Add Data*, and update `DATA_PATH` below.

If no dataset is found, this cell builds a small synthetic fallback dataset
(using the heuristic patterns above to *construct* obviously-AI-styled vs.
plain text) purely so the rest of the notebook runs end-to-end as a demo.
**Do not treat the synthetic-data model's accuracy as meaningful — swap in
real data for anything you actually rely on.**
''')

code('''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import os

DATA_PATH = "/kaggle/input/llm-detect-ai-generated-text/train_essays.csv"  # adjust as needed
TEXT_COL = "text"
LABEL_COL = "generated"

def load_dataset(path=DATA_PATH, text_col=TEXT_COL, label_col=LABEL_COL):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df = df.dropna()
        print(f"Loaded real dataset: {len(df)} rows from {path}")
        return df, True
    print(f"No dataset found at {path}. Falling back to a small synthetic demo dataset.")
    return build_synthetic_dataset(), False

def build_synthetic_dataset(n_per_class=120, seed=42):
    rng = np.random.default_rng(seed)

    ai_templates = [
        "It is important to note that {topic} plays a pivotal role in fostering {benefit}.",
        "Our {topic} boasts a robust, seamless, and intuitive {benefit}, serving as a testament to excellence.",
        "From {a} to {b}, {topic} continues to underscore its crucial significance.",
        "In summary, {topic} represents a transformative and holistic approach to {benefit}.",
        "It's not just {a}, it's {b} -- a true testament to innovation and progress.",
    ]
    human_templates = [
        "I tried {topic} yesterday and honestly it was kind of a mess at first.",
        "So {topic} broke again last week, took me forever to figure out why.",
        "Not gonna lie, {topic} is way harder than people make it sound.",
        "We talked about {topic} for like twenty minutes and never actually decided anything.",
        "{topic} works fine most days but the {benefit} thing is still buggy.",
    ]
    topics = ["the platform", "our workflow", "the new feature", "this process",
              "the dataset", "the app", "the system", "our pipeline"]
    benefits = ["engagement", "growth", "collaboration", "efficiency", "reliability"]

    def fill(t):
        return t.format(topic=rng.choice(topics), benefit=rng.choice(benefits),
                         a=rng.choice(topics), b=rng.choice(benefits))

    ai_texts = [fill(rng.choice(ai_templates)) + " " + fill(rng.choice(ai_templates)) for _ in range(n_per_class)]
    human_texts = [fill(rng.choice(human_templates)) + " " + fill(rng.choice(human_templates)) for _ in range(n_per_class)]

    df = pd.DataFrame({
        "text": ai_texts + human_texts,
        "label": [1] * len(ai_texts) + [0] * len(human_texts),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

df, using_real_data = load_dataset()
df.head()
''')

code('''class HeuristicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Turns the rule-based pattern scores into a numeric feature matrix
    the ML model can use alongside TF-IDF."""

    feature_keys = list(PATTERN_WEIGHTS.keys()) + ["low_sentence_variance", "lexical_diversity_ttr"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            result = heuristic_score(text)
            row = [result["contributions"].get(k, 0.0) for k in PATTERN_WEIGHTS.keys()]
            row.append(result["contributions"].get("low_sentence_variance", 0.0))
            ttr = result["raw_patterns"]["lexical_diversity"].get("ttr") or 0.0
            row.append(ttr)
            rows.append(row)
        return np.array(rows)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

heuristic_tf = HeuristicFeatureTransformer()
X_train_heur = heuristic_tf.transform(X_train)
X_test_heur = heuristic_tf.transform(X_test)

from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, X_train_heur])
X_test_combined = hstack([X_test_tfidf, X_test_heur])

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_combined, y_train)

preds = clf.predict(X_test_combined)
probs = clf.predict_proba(X_test_combined)[:, 1]

print(f"Trained on {'REAL' if using_real_data else 'SYNTHETIC (demo only)'} data")
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
try:
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")
except ValueError:
    pass
print()
print(classification_report(y_test, preds, target_names=["human", "ai"]))
''')

# ----------------------------------------------------------------------
md('''## Part 3 — Combined Predictor

This is the function you actually call: it always runs the heuristic
scorer (so you get pattern explanations even with no model), and layers in
the ML probability when a trained classifier is available.
''')

code('''def predict(text: str, model=None, vectorizer=None, heuristic_transformer=None) -> Dict:
    heur = heuristic_score(text)

    result = {
        "text_preview": text.strip()[:120] + ("..." if len(text.strip()) > 120 else ""),
        "heuristic_score": heur["heuristic_score"],
        "heuristic_verdict": heur["verdict"],
        "detected_patterns": heur["top_detected_patterns"],
        "pattern_examples": {
            k: heur["raw_patterns"][k]["matches"]
            for k in heur["raw_patterns"]
            if isinstance(heur["raw_patterns"][k], dict) and heur["raw_patterns"][k].get("matches")
        },
    }

    if model is not None and vectorizer is not None:
        tfidf_vec = vectorizer.transform([text])
        heur_vec = (heuristic_transformer or HeuristicFeatureTransformer()).transform([text])
        combined_vec = hstack([tfidf_vec, heur_vec])
        ml_prob = float(model.predict_proba(combined_vec)[0, 1])
        result["ml_probability_ai"] = round(ml_prob * 100, 1)
        # Blend: 60% ML, 40% heuristic, since ML is trained on actual data
        blended = 0.6 * (ml_prob * 100) + 0.4 * heur["heuristic_score"]
        result["final_score"] = round(blended, 1)
    else:
        result["final_score"] = heur["heuristic_score"]

    score = result["final_score"]
    if score >= 60:
        result["final_verdict"] = "Likely AI-generated"
    elif score >= 30:
        result["final_verdict"] = "Possibly AI-assisted / mixed"
    else:
        result["final_verdict"] = "Likely human-written"

    return result
''')

code('''# Demo: combined prediction using the model trained above
for label, txt in [("AI-sounding example", ai_sounding_example),
                    ("Human-sounding example", human_sounding_example)]:
    res = predict(txt, model=clf, vectorizer=tfidf, heuristic_transformer=heuristic_tf)
    print(f"=== {label} ===")
    print(json.dumps(res, indent=2))
    print()
''')

# ----------------------------------------------------------------------
md('''## Notes, limits, and how to extend this

- **This is a style/pattern detector, not a lie detector.** It flags
  *statistical* resemblance to common AI writing habits. A careful human
  editor can pass it easily; a sloppy human writer can trip several of its
  patterns by accident (heavy formality, em dashes, three-item lists are all
  normal in plenty of human writing too).
- **The ML half is only as good as the data you train it on.** Swap
  `DATA_PATH` for a real labeled dataset before trusting the probability
  output — the synthetic fallback exists purely so the notebook runs without
  one attached, not as a usable model.
- **To extend pattern coverage**, add new regex/keyword lists to Part 1 and
  a corresponding entry in `PATTERN_WEIGHTS` — the rest of the pipeline
  (scoring, feature extraction for the ML model, the combined predictor)
  picks new patterns up automatically.
- **Don't use this to make accusations.** Wikipedia's own AI-cleanup project
  explicitly treats these signs as discussion points for human editors to
  review, not as automated proof — that's the right way to use this tool too.
''')

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("../detectors/ai_text_likeness_detector.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")