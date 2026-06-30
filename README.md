# Kelvin AI Text Detector

A full-stack AI-generated text detection system with a REST API, admin panel, and user dashboard. Uses heuristic pattern matching with optional ML model support.

## Features

### Detection Engine
- **Heuristic Analysis** — Pattern matching for 55+ AI writing markers (including Wikipedia's "Signs of AI writing" guidelines like proper-noun list leads, horizontal rule headers, skipped headings; and general tells like consecutive sentence starters, list item uniformity, filler phrases, hedging, overused transitions, vocabulary uniformity, and sentence rhythm)
- **Optional ML Layer** — Loads a trained classifier if model files are present; gracefully falls back to heuristic-only mode
- **Scoring** — 0-100 AI probability score with verdicts: *Likely Human*, *Possibly AI-assisted*, *Likely AI-generated*

### REST API
- `POST /api/detect` — Single text detection
- `POST /api/detect/batch` — Batch detection (up to 20 texts)
- `GET /api/balance` — Check credit balance
- `GET /api/usage` — Usage history
- `POST /api/keys` — Create API keys
- `DELETE /api/keys/{id}` — Revoke keys
- **Admin endpoints** — User management, key management, logs, stats

### Web Dashboard
- **User Panel** — AI text detection chat UI, API key management, usage stats, balance top-up
- **Admin Panel** — User management, global API key management, detection logs, system stats
- **Auth** — Registration, login, JWT sessions via secure HTTP-only cookies

### Balance & Rate Limiting
- Credit-based billing (1 credit per detection)
- Per-key rate limiting (configurable per key)
- New users get 100 free credits

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install & Run

```bash
cd api
pip install -r requirements.txt
python detect-ai.py
```

Server starts at **http://localhost:8000**

### First Run

On first startup, a default admin account is created:

| Field    | Value              |
|----------|--------------------|
| Email    | admin@kelvin.local |
| Password | admin123           |
| API Key  | (shown in logs)    |

> ⚠️ **Change these credentials in production!**

---

## API Usage

### Detect AI Text

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"text": "Your text to analyze here..."}'
```

**Response:**
```json
{
  "final_score": 72.5,
  "final_verdict": "Likely AI-generated",
  "model_used": "heuristic",
  "word_count": 150,
  "sentence_count": 8,
  "remaining_balance": 99.0,
  "heuristic": {
    "score": 72.5,
    "flags": ["filler_phrases", "hedging_language"],
    "details": { ... }
  }
}
```

### Register User

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret123", "display_name": "Test"}'
```

### Batch Detection

```bash
curl -X POST http://localhost:8000/api/detect/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"texts": ["Text one...", "Text two..."]}'
```

---

## Project Structure

```
kelvin-ai-detector/
├── api/
│   ├── detect-ai.py       # FastAPI server (API + Web routes)
│   ├── detector.py         # Detection logic (heuristic + ML)
│   ├── auth.py             # Auth, JWT, API key management
│   ├── database.py         # SQLAlchemy models (SQLite)
│   ├── requirements.txt    # Python dependencies
│   ├── static/
│   │   ├── css/style.css   # Dark theme design system
│   │   └── js/app.js       # Client-side detection UI
│   ├── templates/
│   │   ├── base.html       # Base layout with sidebar
│   │   ├── login.html      # Login page
│   │   ├── register.html   # Registration page
│   │   ├── user/           # User dashboard templates
│   │   │   ├── dashboard.html
│   │   │   ├── detect.html
│   │   │   ├── keys.html
│   │   │   ├── usage.html
│   │   │   └── topup.html
│   │   └── admin/          # Admin panel templates
│   │       ├── dashboard.html
│   │       ├── users.html
│   │       ├── keys.html
│   │       └── logs.html
│   └── data/               # SQLite database (auto-created)
├── detectors/              # Legacy detector scripts
└── generators/             # Test text generators
```

---

## Tech Stack

| Layer      | Technology                    |
|------------|-------------------------------|
| Backend    | FastAPI + Uvicorn             |
| Database   | SQLite via SQLAlchemy 2.0     |
| Auth       | bcrypt + python-jose (JWT)    |
| Templates  | Jinja2                        |
| Frontend   | Vanilla JS + CSS (dark theme) |

---

## Environment Variables

| Variable     | Default         | Description                 |
|--------------|-----------------|-----------------------------|
| `PORT`       | `8000`          | Server port                 |
| `HOST`       | `0.0.0.0`       | Bind address                |
| `SECRET_KEY` | Auto-generated  | JWT signing key             |
| `ADMIN_EMAIL`| `admin@kelvin.local` | Default admin email    |
| `ADMIN_PASS` | `admin123`      | Default admin password      |

---

## API Endpoints

### Public
| Method | Endpoint             | Description          |
|--------|----------------------|----------------------|
| POST   | `/api/auth/register` | Register new user    |
| POST   | `/api/auth/login`    | Login, get JWT       |
| GET    | `/api/health`        | Health check         |

### Authenticated (X-API-Key)
| Method | Endpoint              | Description          |
|--------|-----------------------|----------------------|
| POST   | `/api/detect`         | Detect AI text       |
| POST   | `/api/detect/batch`   | Batch detection      |
| GET    | `/api/balance`        | Check balance        |
| GET    | `/api/usage`          | Usage history        |
| POST   | `/api/keys`           | Create API key       |
| DELETE | `/api/keys/{id}`      | Revoke own key       |

### Admin (X-API-Key with admin role)
| Method | Endpoint                          | Description          |
|--------|-----------------------------------|----------------------|
| GET    | `/api/admin/users`                | List all users       |
| POST   | `/api/admin/users/{id}/topup`     | Add credits          |
| PUT    | `/api/admin/users/{id}/toggle`    | Enable/disable user  |
| GET    | `/api/admin/keys`                 | List all API keys    |
| PUT    | `/api/admin/keys/{id}/rate-limit` | Set rate limit       |
| DELETE | `/api/admin/keys/{id}`            | Revoke any key       |
| GET    | `/api/admin/logs`                 | Detection logs       |
| GET    | `/api/admin/stats`                | System statistics    |

---

## License

MIT
