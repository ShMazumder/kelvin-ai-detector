"""
Kelvin AI Text Detector — FastAPI Server

API endpoints + Web UI routes. Serves detection API with balance/rate limiting
and full admin panel + user dashboard via Jinja2 templates.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, Header, Depends, Request, Form, Query,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from detector import predict, load_model
from database import (
    init_db, get_db_session, User, APIKeyRecord, DetectionLog,
    BalanceTransaction, check_rate_limit, get_rate_limit_remaining,
)
import auth

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("kelvin-ai-detector")

# ── App state ──────────────────────────────────────────────────────────────────

ml_model = None

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model

    # Init database
    init_db()
    logger.info("Database initialized.")

    # Ensure admin exists
    db = next(get_db_session())
    try:
        admin_key = auth.ensure_admin(db)
        if admin_key:
            logger.info("=" * 60)
            logger.info("FIRST RUN — Default admin created:")
            logger.info(f"  Email:    admin@kelvin.local")
            logger.info(f"  Password: admin123")
            logger.info(f"  API Key:  {admin_key}")
            logger.info("  CHANGE THESE IN PRODUCTION!")
            logger.info("=" * 60)
        else:
            logger.info("Admin account exists.")
    finally:
        db.close()

    # Load ML model
    ml_model = load_model()
    if ml_model:
        logger.info("ML model loaded.")
    else:
        logger.info("Heuristic-only mode (no ML model found).")

    yield
    logger.info("Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kelvin AI Text Detector",
    description="Detect AI-generated text with heuristic patterns + optional ML.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files + templates
app.mount("/static", StaticFiles(directory=os.path.join(_BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))

# ── Constants ──────────────────────────────────────────────────────────────────

TEXT_MIN_LENGTH = 20
TEXT_MAX_LENGTH = 50_000
BATCH_MAX_SIZE = 20
COST_PER_DETECTION = 1.0


# ── Auth dependencies ─────────────────────────────────────────────────────────

async def require_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db_session),
) -> dict:
    """Validate API key. Returns key info dict."""
    key_info = auth.validate_api_key(db, x_api_key)
    if key_info is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")

    # Rate limit check
    if not check_rate_limit(key_info["key_hash"], key_info["rate_limit"]):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({key_info['rate_limit']} requests/minute).",
        )
    return key_info


async def require_admin_api_key(key_info: dict = Depends(require_api_key)) -> dict:
    if key_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return key_info


def get_current_web_user(request: Request, db: Session) -> Optional[User]:
    """Get user from session cookie. Returns None if not logged in."""
    token = request.cookies.get("session_token")
    if not token:
        return None
    decoded = auth.decode_access_token(token)
    if not decoded:
        return None
    user = auth.get_user_by_id(db, decoded["user_id"])
    if not user or not user.is_active:
        return None
    return user


# ── Request models ─────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=TEXT_MIN_LENGTH, max_length=TEXT_MAX_LENGTH)

class BatchDetectRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=BATCH_MAX_SIZE)

class CreateKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=60, ge=1, le=10000)

class TopUpRequest(BaseModel):
    amount: float = Field(..., gt=0, le=1000000)
    description: str = Field(default="Admin top-up")

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=100)

class LoginRequest(BaseModel):
    email: str
    password: str

class AdminCreateKeyRequest(BaseModel):
    role: str = Field(default="user", pattern="^(admin|user)$")
    label: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=60, ge=1, le=10000)

class AdminSetRateLimitRequest(BaseModel):
    rate_limit: int = Field(..., ge=1, le=10000)


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": ml_model is not None,
        "model_type": "heuristic+ml" if ml_model else "heuristic",
        "version": "2.0.0",
    }


# ── Auth API ───────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def api_register(req: RegisterRequest, db: Session = Depends(get_db_session)):
    user = auth.register_user(db, req.email, req.password, req.display_name)
    if not user:
        raise HTTPException(status_code=409, detail="Email already registered.")
    token = auth.create_access_token(user.id, user.role)
    return {
        "user": user.to_dict(),
        "token": token,
        "message": f"Registered with {auth.DEFAULT_BALANCE} free credits.",
    }


@app.post("/api/auth/login")
async def api_login(req: LoginRequest, db: Session = Depends(get_db_session)):
    user = auth.authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = auth.create_access_token(user.id, user.role)
    return {"user": user.to_dict(), "token": token}


# ── Detection ──────────────────────────────────────────────────────────────────

def _do_detect(text: str, key_info: dict, db: Session, ip: str = None) -> dict:
    """Core detection + logging + billing."""
    user_id = key_info["user_id"]

    # Balance check
    user = db.query(User).filter(User.id == user_id).first()
    if user.balance < COST_PER_DETECTION:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance ({user.balance} credits). Need {COST_PER_DETECTION}.",
        )

    # Run detection
    start = time.time()
    result = predict(text, ml_model=ml_model)
    elapsed_ms = round((time.time() - start) * 1000, 1)
    result["processing_time_ms"] = elapsed_ms

    # Deduct balance
    auth.deduct_balance(db, user_id, COST_PER_DETECTION, "AI text detection")

    # Refresh to get updated balance
    db.refresh(user)
    result["remaining_balance"] = user.balance

    # Log to DB
    log = DetectionLog(
        user_id=user_id,
        api_key_id=key_info.get("key_id"),
        input_text=text,
        output_json=json.dumps(result),
        score=result.get("final_score"),
        verdict=result.get("final_verdict"),
        model_used=result.get("model_used"),
        cost=COST_PER_DETECTION,
        word_count=result.get("word_count"),
        ip_address=ip,
    )
    db.add(log)
    db.commit()

    return result


@app.post("/api/detect")
async def detect_text(
    req: DetectRequest,
    request: Request,
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    ip = request.client.host if request.client else None
    return _do_detect(req.text, key_info, db, ip)


@app.post("/api/detect/batch")
async def detect_batch(
    req: BatchDetectRequest,
    request: Request,
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    for i, text in enumerate(req.texts):
        if len(text) < TEXT_MIN_LENGTH:
            raise HTTPException(422, f"Text at index {i} too short (min {TEXT_MIN_LENGTH}).")
        if len(text) > TEXT_MAX_LENGTH:
            raise HTTPException(422, f"Text at index {i} too long (max {TEXT_MAX_LENGTH}).")

    # Check total balance needed
    user = db.query(User).filter(User.id == key_info["user_id"]).first()
    total_cost = len(req.texts) * COST_PER_DETECTION
    if user.balance < total_cost:
        raise HTTPException(402, f"Insufficient balance. Need {total_cost}, have {user.balance}.")

    ip = request.client.host if request.client else None
    start = time.time()
    results = [_do_detect(text, key_info, db, ip) for text in req.texts]
    total_ms = round((time.time() - start) * 1000, 1)

    return {"count": len(results), "total_processing_time_ms": total_ms, "results": results}


# ── User API ───────────────────────────────────────────────────────────────────

@app.get("/api/balance")
async def get_balance(
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    user = db.query(User).filter(User.id == key_info["user_id"]).first()
    return {"balance": user.balance, "email": user.email}


@app.get("/api/usage")
async def get_usage(
    limit: int = Query(default=50, le=200),
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    logs = (
        db.query(DetectionLog)
        .filter(DetectionLog.user_id == key_info["user_id"])
        .order_by(desc(DetectionLog.created_at))
        .limit(limit)
        .all()
    )
    total = db.query(func.count(DetectionLog.id)).filter(
        DetectionLog.user_id == key_info["user_id"]
    ).scalar()

    return {
        "total_requests": total,
        "showing": len(logs),
        "logs": [l.to_dict() for l in logs],
    }


@app.get("/api/keys")
async def list_own_keys(
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    keys = (
        db.query(APIKeyRecord)
        .filter(APIKeyRecord.user_id == key_info["user_id"])
        .order_by(desc(APIKeyRecord.created_at))
        .all()
    )
    return {"keys": [k.to_dict() for k in keys]}


@app.post("/api/keys")
async def create_own_key(
    req: CreateKeyRequest,
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    raw_key = auth.create_api_key(db, key_info["user_id"], req.label, req.rate_limit)
    return {
        "key": raw_key,
        "label": req.label,
        "rate_limit": req.rate_limit,
        "message": "Save this key — shown only once.",
    }


@app.delete("/api/keys/{key_id}")
async def delete_own_key(
    key_id: int,
    key_info: dict = Depends(require_api_key),
    db: Session = Depends(get_db_session),
):
    if auth.revoke_api_key(db, key_id, key_info["user_id"]):
        return {"message": "Key revoked."}
    raise HTTPException(404, "Key not found or not yours.")


# ── Admin API ──────────────────────────────────────────────────────────────────

@app.get("/api/admin/users")
async def admin_list_users(
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    users = db.query(User).order_by(desc(User.created_at)).all()
    return {"users": [u.to_dict() for u in users]}


@app.post("/api/admin/users/{user_id}/topup")
async def admin_topup(
    user_id: int,
    req: TopUpRequest,
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    if auth.topup_balance(db, user_id, req.amount, req.description):
        user = db.query(User).filter(User.id == user_id).first()
        return {"message": f"Added {req.amount} credits.", "new_balance": user.balance}
    raise HTTPException(404, "User not found.")


@app.put("/api/admin/users/{user_id}/toggle")
async def admin_toggle_user(
    user_id: int,
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found.")
    if user.role == "admin" and user.id == key_info["user_id"]:
        raise HTTPException(400, "Cannot deactivate yourself.")
    user.is_active = not user.is_active
    db.commit()
    return {"message": f"User {'activated' if user.is_active else 'deactivated'}.", "is_active": user.is_active}


@app.get("/api/admin/keys")
async def admin_list_all_keys(
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    keys = db.query(APIKeyRecord).order_by(desc(APIKeyRecord.created_at)).all()
    result = []
    for k in keys:
        d = k.to_dict()
        user = db.query(User).filter(User.id == k.user_id).first()
        d["user_email"] = user.email if user else "unknown"
        result.append(d)
    return {"keys": result}


@app.put("/api/admin/keys/{key_id}/rate-limit")
async def admin_set_rate_limit(
    key_id: int,
    req: AdminSetRateLimitRequest,
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    record = db.query(APIKeyRecord).filter(APIKeyRecord.id == key_id).first()
    if not record:
        raise HTTPException(404, "Key not found.")
    record.rate_limit = req.rate_limit
    db.commit()
    return {"message": f"Rate limit set to {req.rate_limit}/min.", "key_id": key_id}


@app.delete("/api/admin/keys/{key_id}")
async def admin_revoke_key(
    key_id: int,
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    record = db.query(APIKeyRecord).filter(APIKeyRecord.id == key_id).first()
    if not record:
        raise HTTPException(404, "Key not found.")
    record.is_revoked = True
    db.commit()
    return {"message": "Key revoked."}


@app.get("/api/admin/logs")
async def admin_list_logs(
    limit: int = Query(default=100, le=500),
    user_id: Optional[int] = Query(default=None),
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    q = db.query(DetectionLog)
    if user_id:
        q = q.filter(DetectionLog.user_id == user_id)
    logs = q.order_by(desc(DetectionLog.created_at)).limit(limit).all()

    result = []
    for l in logs:
        d = l.to_dict()
        user = db.query(User).filter(User.id == l.user_id).first()
        d["user_email"] = user.email if user else "unknown"
        result.append(d)

    return {"logs": result, "count": len(result)}


@app.get("/api/admin/stats")
async def admin_stats(
    key_info: dict = Depends(require_admin_api_key),
    db: Session = Depends(get_db_session),
):
    total_users = db.query(func.count(User.id)).scalar()
    total_requests = db.query(func.count(DetectionLog.id)).scalar()
    total_keys = db.query(func.count(APIKeyRecord.id)).filter(
        APIKeyRecord.is_revoked == False
    ).scalar()
    total_revenue = db.query(func.sum(DetectionLog.cost)).scalar() or 0

    return {
        "total_users": total_users,
        "total_requests": total_requests,
        "active_keys": total_keys,
        "total_credits_used": total_revenue,
    }



# ══════════════════════════════════════════════════════════════════════════════
# WEB UI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def _render(request: Request, db: Session, template: str, **extra):
    """Render template with user context. Starlette 1.0 compatible."""
    user = get_current_web_user(request, db)
    ctx = {"user": user}
    ctx.update(extra)
    return templates.TemplateResponse(request, template, context=ctx)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db_session)):
    user = get_current_web_user(request, db)
    if user:
        if user.role == "admin":
            return RedirectResponse("/admin/dashboard", status_code=302)
        return RedirectResponse("/dashboard", status_code=302)
    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: Session = Depends(get_db_session)):
    return _render(request, db, "login.html")


@app.post("/login")
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db_session)):
    user = auth.authenticate_user(db, email, password)
    if not user:
        return _render(request, db, "login.html", error="Invalid credentials.")
    token = auth.create_access_token(user.id, user.role)
    dest = "/admin/dashboard" if user.role == "admin" else "/dashboard"
    response = RedirectResponse(dest, status_code=302)
    response.set_cookie("session_token", token, httponly=True, max_age=86400)
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, db: Session = Depends(get_db_session)):
    return _render(request, db, "register.html")


@app.post("/register")
async def register_submit(request: Request, email: str = Form(...), password: str = Form(...), display_name: str = Form(...), db: Session = Depends(get_db_session)):
    if len(password) < 6:
        return _render(request, db, "register.html", error="Password must be at least 6 characters.")
    user = auth.register_user(db, email, password, display_name)
    if not user:
        return _render(request, db, "register.html", error="Email already registered.")
    token = auth.create_access_token(user.id, user.role)
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("session_token", token, httponly=True, max_age=86400)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session_token")
    return response


def _require_web_user(request: Request, db: Session) -> User:
    user = get_current_web_user(request, db)
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return user


@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request, db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    recent_logs = db.query(DetectionLog).filter(DetectionLog.user_id == user.id).order_by(desc(DetectionLog.created_at)).limit(10).all()
    total_requests = db.query(func.count(DetectionLog.id)).filter(DetectionLog.user_id == user.id).scalar()
    return _render(request, db, "user/dashboard.html", recent_logs=recent_logs, total_requests=total_requests)


@app.get("/dashboard/detect", response_class=HTMLResponse)
async def user_detect_page(request: Request, db: Session = Depends(get_db_session)):
    _require_web_user(request, db)
    return _render(request, db, "user/detect.html")


@app.post("/dashboard/detect")
async def user_detect_submit(request: Request, text: str = Form(...), db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    if len(text.strip()) < TEXT_MIN_LENGTH:
        return _render(request, db, "user/detect.html", error=f"Text too short (min {TEXT_MIN_LENGTH} chars).")
    if user.balance < COST_PER_DETECTION:
        return _render(request, db, "user/detect.html", error="Insufficient balance.")
    key_info = {"user_id": user.id, "key_id": None, "key_hash": "web_ui", "role": user.role, "balance": user.balance, "rate_limit": 9999}
    result = _do_detect(text, key_info, db, request.client.host if request.client else None)
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return JSONResponse(result)
    return _render(request, db, "user/detect.html", result=result, input_text=text)


@app.get("/dashboard/keys", response_class=HTMLResponse)
async def user_keys_page(request: Request, db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    keys = db.query(APIKeyRecord).filter(APIKeyRecord.user_id == user.id).order_by(desc(APIKeyRecord.created_at)).all()
    return _render(request, db, "user/keys.html", keys=keys)


@app.post("/dashboard/keys/create")
async def user_create_key(request: Request, label: str = Form(...), db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    raw_key = auth.create_api_key(db, user.id, label)
    keys = db.query(APIKeyRecord).filter(APIKeyRecord.user_id == user.id).order_by(desc(APIKeyRecord.created_at)).all()
    return _render(request, db, "user/keys.html", keys=keys, new_key=raw_key)


@app.post("/dashboard/keys/{key_id}/delete")
async def user_delete_key(key_id: int, request: Request, db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    auth.revoke_api_key(db, key_id, user.id)
    return RedirectResponse("/dashboard/keys", status_code=302)


@app.get("/dashboard/usage", response_class=HTMLResponse)
async def user_usage_page(request: Request, db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    logs = db.query(DetectionLog).filter(DetectionLog.user_id == user.id).order_by(desc(DetectionLog.created_at)).limit(100).all()
    total = db.query(func.count(DetectionLog.id)).filter(DetectionLog.user_id == user.id).scalar()
    total_cost = db.query(func.sum(DetectionLog.cost)).filter(DetectionLog.user_id == user.id).scalar() or 0
    return _render(request, db, "user/usage.html", logs=logs, total=total, total_cost=total_cost)


@app.get("/dashboard/topup", response_class=HTMLResponse)
async def user_topup_page(request: Request, db: Session = Depends(get_db_session)):
    user = _require_web_user(request, db)
    transactions = db.query(BalanceTransaction).filter(BalanceTransaction.user_id == user.id).order_by(desc(BalanceTransaction.created_at)).limit(50).all()
    return _render(request, db, "user/topup.html", transactions=transactions)


def _require_admin_web(request: Request, db: Session) -> User:
    user = get_current_web_user(request, db)
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    total_users = db.query(func.count(User.id)).scalar()
    total_requests = db.query(func.count(DetectionLog.id)).scalar()
    active_keys = db.query(func.count(APIKeyRecord.id)).filter(APIKeyRecord.is_revoked == False).scalar()
    total_credits = db.query(func.sum(DetectionLog.cost)).scalar() or 0
    recent_logs = db.query(DetectionLog).order_by(desc(DetectionLog.created_at)).limit(10).all()
    for log in recent_logs:
        u = db.query(User).filter(User.id == log.user_id).first()
        log.user_email = u.email if u else "unknown"
    return _render(request, db, "admin/dashboard.html", total_users=total_users, total_requests=total_requests, active_keys=active_keys, total_credits=total_credits, recent_logs=recent_logs)


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request, db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    users = db.query(User).order_by(desc(User.created_at)).all()
    return _render(request, db, "admin/users.html", users=users)


@app.post("/admin/users/{user_id}/topup")
async def admin_topup_web(user_id: int, request: Request, amount: float = Form(...), db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    auth.topup_balance(db, user_id, amount, "Admin top-up via dashboard")
    return RedirectResponse("/admin/users", status_code=302)


@app.post("/admin/users/{user_id}/toggle")
async def admin_toggle_web(user_id: int, request: Request, db: Session = Depends(get_db_session)):
    admin = _require_admin_web(request, db)
    user = db.query(User).filter(User.id == user_id).first()
    if user and user.id != admin.id:
        user.is_active = not user.is_active
        db.commit()
    return RedirectResponse("/admin/users", status_code=302)


@app.get("/admin/keys", response_class=HTMLResponse)
async def admin_keys_page(request: Request, db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    keys = db.query(APIKeyRecord).order_by(desc(APIKeyRecord.created_at)).all()
    for k in keys:
        u = db.query(User).filter(User.id == k.user_id).first()
        k.user_email = u.email if u else "unknown"
    return _render(request, db, "admin/keys.html", keys=keys)


@app.post("/admin/keys/{key_id}/rate-limit")
async def admin_rate_limit_web(key_id: int, request: Request, rate_limit: int = Form(...), db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    record = db.query(APIKeyRecord).filter(APIKeyRecord.id == key_id).first()
    if record:
        record.rate_limit = rate_limit
        db.commit()
    return RedirectResponse("/admin/keys", status_code=302)


@app.post("/admin/keys/{key_id}/revoke")
async def admin_revoke_web(key_id: int, request: Request, db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    record = db.query(APIKeyRecord).filter(APIKeyRecord.id == key_id).first()
    if record:
        record.is_revoked = True
        db.commit()
    return RedirectResponse("/admin/keys", status_code=302)


@app.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs_page(request: Request, db: Session = Depends(get_db_session)):
    _require_admin_web(request, db)
    logs = db.query(DetectionLog).order_by(desc(DetectionLog.created_at)).limit(200).all()
    for log in logs:
        u = db.query(User).filter(User.id == log.user_id).first()
        log.user_email = u.email if u else "unknown"
    return _render(request, db, "admin/logs.html", logs=logs)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting Kelvin AI Detector on {host}:{port}")
    uvicorn.run("detect-ai:app", host=host, port=port, reload=True, log_level="info")
