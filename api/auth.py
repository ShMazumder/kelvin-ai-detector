"""
Authentication Module — DB-backed

User accounts (email/password with bcrypt), API keys, JWT sessions for web UI.
"""

import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

import bcrypt
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from database import User, APIKeyRecord, BalanceTransaction, get_db

# ── Config ─────────────────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_urlsafe(64))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
DEFAULT_BALANCE = 100.0
DEFAULT_RATE_LIMIT = 60


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── API key hashing ───────────────────────────────────────────────────────────

def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key(role: str = "user") -> str:
    prefix = "kad_" if role == "admin" else "kud_"
    return f"{prefix}{secrets.token_urlsafe(32)}"


# ── JWT tokens (for web UI sessions) ──────────────────────────────────────────

def create_access_token(user_id: int, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": int(payload["sub"]), "role": payload["role"]}
    except JWTError:
        return None


# ── User operations ───────────────────────────────────────────────────────────

def register_user(
    db: Session, email: str, password: str, display_name: str, role: str = "user"
) -> Optional[User]:
    """Register new user. Returns User or None if email exists."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return None

    user = User(
        email=email,
        password_hash=hash_password(password),
        display_name=display_name,
        role=role,
        balance=DEFAULT_BALANCE,
    )
    db.add(user)
    db.flush()

    # Log signup bonus
    tx = BalanceTransaction(
        user_id=user.id,
        amount=DEFAULT_BALANCE,
        type="signup_bonus",
        description="Welcome bonus credits",
        balance_after=DEFAULT_BALANCE,
    )
    db.add(tx)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Validate credentials. Returns User or None."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


# ── API key operations ────────────────────────────────────────────────────────

def create_api_key(
    db: Session, user_id: int, label: str, rate_limit: int = DEFAULT_RATE_LIMIT
) -> str:
    """Create new API key for user. Returns raw key (only shown once)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError("User not found")

    raw_key = generate_api_key(user.role)
    record = APIKeyRecord(
        user_id=user_id,
        key_hash=hash_api_key(raw_key),
        key_prefix=raw_key[:12],
        label=label,
        rate_limit=rate_limit,
    )
    db.add(record)
    db.commit()
    return raw_key


def validate_api_key(db: Session, raw_key: str) -> Optional[Dict]:
    """Validate API key. Returns dict with key info + user info, or None."""
    h = hash_api_key(raw_key)
    record = (
        db.query(APIKeyRecord)
        .filter(APIKeyRecord.key_hash == h, APIKeyRecord.is_revoked == False)
        .first()
    )
    if not record:
        return None

    user = db.query(User).filter(User.id == record.user_id).first()
    if not user or not user.is_active:
        return None

    # Update last used
    record.last_used_at = datetime.utcnow()
    db.commit()

    return {
        "key_id": record.id,
        "key_hash": record.key_hash,
        "user_id": user.id,
        "user_email": user.email,
        "role": user.role,
        "balance": user.balance,
        "rate_limit": record.rate_limit,
    }


def revoke_api_key(db: Session, key_id: int, user_id: int) -> bool:
    """Revoke API key. User can only revoke own keys (unless admin)."""
    record = db.query(APIKeyRecord).filter(APIKeyRecord.id == key_id).first()
    if not record:
        return False
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    # User can revoke own, admin can revoke any
    if record.user_id != user_id and user.role != "admin":
        return False
    record.is_revoked = True
    db.commit()
    return True


# ── Balance operations ────────────────────────────────────────────────────────

def deduct_balance(db: Session, user_id: int, amount: float, description: str) -> bool:
    """Deduct from user balance. Returns False if insufficient."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.balance < amount:
        return False

    user.balance -= amount
    tx = BalanceTransaction(
        user_id=user_id,
        amount=-amount,
        type="deduction",
        description=description,
        balance_after=user.balance,
    )
    db.add(tx)
    db.commit()
    return True


def topup_balance(db: Session, user_id: int, amount: float, description: str = "Admin top-up") -> bool:
    """Add credits to user balance."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False

    user.balance += amount
    tx = BalanceTransaction(
        user_id=user_id,
        amount=amount,
        type="topup",
        description=description,
        balance_after=user.balance,
    )
    db.add(tx)
    db.commit()
    return True


# ── Init ──────────────────────────────────────────────────────────────────────

def ensure_admin(db: Session) -> Optional[str]:
    """Create default admin if none exists. Returns raw API key if created."""
    admin = db.query(User).filter(User.role == "admin").first()
    if admin:
        return None

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@kelvin.local")
    admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")

    user = User(
        email=admin_email,
        password_hash=hash_password(admin_pass),
        display_name="Admin",
        role="admin",
        balance=999999.0,
    )
    db.add(user)
    db.flush()

    raw_key = generate_api_key("admin")
    key_record = APIKeyRecord(
        user_id=user.id,
        key_hash=hash_api_key(raw_key),
        key_prefix=raw_key[:12],
        label="default-admin-key",
        rate_limit=9999,
    )
    db.add(key_record)
    db.commit()
    return raw_key
