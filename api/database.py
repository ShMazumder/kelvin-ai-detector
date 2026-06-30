"""
Database Layer — SQLAlchemy + SQLite

Tables: users, api_keys, detection_logs, balance_transactions
Auto-creates on startup. DB file: api/data/kelvin.db
"""

import os
import time
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Text,
    DateTime, ForeignKey, Index, func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from contextlib import contextmanager

# ── Database setup ─────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(_DATA_DIR, "kelvin.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


@contextmanager
def get_db():
    """Context manager for DB sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session():
    """FastAPI dependency for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Models ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default="user")  # "admin" or "user"
    balance = Column(Float, nullable=False, default=100.0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    api_keys = relationship("APIKeyRecord", back_populates="user", lazy="dynamic")
    detection_logs = relationship("DetectionLog", back_populates="user", lazy="dynamic")
    transactions = relationship("BalanceTransaction", back_populates="user", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "balance": self.balance,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "key_count": self.api_keys.count() if self.api_keys else 0,
        }


class APIKeyRecord(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(12), nullable=False)  # first 12 chars for display
    label = Column(String(100), nullable=False)
    rate_limit = Column(Integer, nullable=False, default=60)  # requests per minute
    is_revoked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")
    detection_logs = relationship("DetectionLog", back_populates="api_key", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "key_prefix": self.key_prefix,
            "label": self.label,
            "rate_limit": self.rate_limit,
            "is_revoked": self.is_revoked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)
    input_text = Column(Text, nullable=False)
    output_json = Column(Text, nullable=False)  # JSON string
    score = Column(Float, nullable=True)
    verdict = Column(String(50), nullable=True)
    model_used = Column(String(30), nullable=True)
    cost = Column(Float, nullable=False, default=1.0)
    word_count = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="detection_logs")
    api_key = relationship("APIKeyRecord", back_populates="detection_logs")

    __table_args__ = (
        Index("idx_logs_user_created", "user_id", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "score": self.score,
            "verdict": self.verdict,
            "model_used": self.model_used,
            "cost": self.cost,
            "word_count": self.word_count,
            "input_preview": (self.input_text[:120] + "...") if len(self.input_text) > 120 else self.input_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BalanceTransaction(Base):
    __tablename__ = "balance_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)  # positive = credit, negative = debit
    type = Column(String(20), nullable=False)  # "topup", "deduction", "refund", "signup_bonus"
    description = Column(String(255), nullable=True)
    balance_after = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="transactions")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "amount": self.amount,
            "type": self.type,
            "description": self.description,
            "balance_after": self.balance_after,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Rate limit tracking (in-memory, resets on restart) ────────────────────────

_rate_limit_store: dict = {}  # key_hash -> list of timestamps


def check_rate_limit(key_hash: str, limit: int) -> bool:
    """Check if key is within rate limit. Returns True if allowed."""
    now = time.time()
    window = 60.0  # 1 minute window

    if key_hash not in _rate_limit_store:
        _rate_limit_store[key_hash] = []

    # Prune old entries
    _rate_limit_store[key_hash] = [
        t for t in _rate_limit_store[key_hash] if now - t < window
    ]

    if len(_rate_limit_store[key_hash]) >= limit:
        return False

    _rate_limit_store[key_hash].append(now)
    return True


def get_rate_limit_remaining(key_hash: str, limit: int) -> int:
    """Get remaining requests in current window."""
    now = time.time()
    if key_hash not in _rate_limit_store:
        return limit
    active = [t for t in _rate_limit_store[key_hash] if now - t < 60.0]
    return max(0, limit - len(active))


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
