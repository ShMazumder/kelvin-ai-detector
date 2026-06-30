# Kelvin AI Text Detector

Detect AI-generated text using heuristic pattern analysis and optional ML classification. Full-stack application with REST API, admin panel, and user dashboard.

## Features

- **14 Heuristic Pattern Detectors** — vocabulary, inflated significance, em-dash overuse, rule-of-three, burstiness, lexical diversity, and more
- **Optional ML Classifier** — TF-IDF + Logistic Regression, loads automatically when trained model files exist
- **REST API** — single and batch text detection with JSON responses
- **Web Dashboard** — chat-style detection UI, usage history, API key management
- **Admin Panel** — user management, balance top-up, rate limit control, detection logs
- **Balance System** — credit-based usage tracking (1 credit per detection)
- **API Key Auth** — SHA-256 hashed keys with per-key rate limiting
- **SQLite Database** — zero-config, stores users, keys, logs, transactions

## Quick Start

```bash
# Install dependencies
cd api
pip install -r requirements.txt

# Start server
python detect-ai.py
```

On first run, default admin credentials are printed to console:
```
Email:    admin@kelvin.local
Password: admin123
API Key:  kad_...
```

> ⚠️ **Change these in production!** Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` environment variables before first run.

Open `http://localhost:8000` in your browser.

## API Endpoints

### Public
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check + model status |

### Auth (no key required)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user |
| `/api/auth/login` | POST | Login → JWT token |

### Detection (API key required via `X-API-Key` header)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Analyze single text |
| `/api/detect/batch` | POST | Analyze up to 20 texts |
| `/api/balance` | GET | Check balance |
| `/api/usage` | GET | Usage history |
| `/api/keys` | GET/POST | List/create own keys |
| `/api/keys/{id}` | DELETE | Revoke own key |

### Admin (admin API key required)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/users` | GET | List all users |
| `/api/admin/users/{id}/topup` | POST | Top up user balance |
| `/api/admin/users/{id}/toggle` | PUT | Enable/disable user |
| `/api/admin/keys` | GET | List all API keys |
| `/api/admin/keys/{id}/rate-limit` | PUT | Set rate limit |
| `/api/admin/logs` | GET | View detection logs |
| `/api/admin/stats` | GET | System statistics |

## API Usage Example

```bash
# Detect AI text
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"text": "Your text to analyze here..."}'
```

Response:
```json
{
  "final_score": 36.0,
  "final_verdict": "Possibly AI-assisted / mixed",
  "model_used": "heuristic",
  "detected_patterns": {
    "ai_vocabulary": 13.83,
    "leftover_chat_artifacts": 10.02
  },
  "pattern_examples": {
    "ai_vocabulary": ["landscape", "pivotal", "robust"],
    "leftover_chat_artifacts": ["I hope this helps"]
  },
  "remaining_balance": 99.0,
  "disclaimer": "Style diagnostic only — not proof of authorship."
}
```

## Adding ML Model

Train and export a classifier for improved accuracy:

```bash
# With labeled data (CSV with 'text' + 'generated' columns):
python api/export_model.py /path/to/train_essays.csv

# Restart server — model auto-loads
python api/detect-ai.py
```

## Project Structure

```
kelvin-ai-detector/
├── README.md
├── api/
│   ├── detect-ai.py          # FastAPI server
│   ├── detector.py            # Core detection logic
│   ├── database.py            # SQLAlchemy models
│   ├── auth.py                # Authentication
│   ├── export_model.py        # ML model trainer
│   ├── requirements.txt
│   ├── data/kelvin.db         # SQLite (auto-created)
│   ├── model/                 # ML model files (optional)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/
│       ├── base.html
│       ├── login.html
│       ├── register.html
│       ├── admin/             # Admin panel pages
│       └── user/              # User dashboard pages
└── detectors/
    └── ai_text_likeness_detector.ipynb  # Original notebook
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `SECRET_KEY` | (random) | JWT signing key |
| `ADMIN_EMAIL` | `admin@kelvin.local` | Default admin email |
| `ADMIN_PASSWORD` | `admin123` | Default admin password |

## Detection Patterns

| Pattern | Weight | Description |
|---------|--------|-------------|
| AI Vocabulary | 14 | Words like "delve", "robust", "seamless" |
| Leftover Chat Artifacts | 14 | "I hope this helps", "Happy to help" |
| Inflated Significance | 10 | "plays a pivotal role" |
| Negative Parallelism | 10 | "It's not just X, it's Y" |
| Em Dash Overuse | 8 | Excessive — use of em dashes |
| Rule of Three | 8 | "X, Y, and Z" triplet lists |
| Compulsive Summary | 8 | "In summary", "In conclusion" |
| Editorializing | 8 | "It's important to note" |
| Vague Attribution | 8 | "Experts say", "Many believe" |
| False Ranges | 8 | "From X to Y" constructions |
| Formatting Overkill | 6 | Excessive bold/bullets/emoji |
| Letter Style | 6 | "I hope this email finds you well" |
| Low Sentence Variance | 6 | Uniform sentence lengths |

## Disclaimer

This is a **style diagnostic tool**, not proof of authorship. AI text detectors produce both false positives and false negatives. Treat results as discussion points, not verdicts.

## License

MIT
