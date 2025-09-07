from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean,
    func, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import secrets
import hashlib
from typing import List, Tuple

# ==== DB setup ====
engine = create_engine("sqlite:///promptgen.db", echo=False, future=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ==== Models ====
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=True)

    password_salt = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)

    role = Column(String, default="user", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    prompts = relationship("Prompt", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")
    choices = relationship("Choice", back_populates="user")


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True, index=True)
    tool = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    style = Column(String, nullable=True)

    platform_name = Column(String, nullable=True)
    color_palette = Column(String, nullable=True)
    mood = Column(String, nullable=True)

    generated_text = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)  # 'offline' or 'gemini'
    used_hybrid = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="prompts")
    feedbacks = relationship("Feedback", back_populates="prompt")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    rating = Column(Integer, nullable=False)  # 1 = thumbs up, 0 = thumbs down
    comments = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"))
    prompt_id = Column(Integer, ForeignKey("prompts.id"))

    user = relationship("User", back_populates="feedbacks")
    prompt = relationship("Prompt", back_populates="feedbacks")


class Choice(Base):
    """
    Records an explicit user choice when we generated both variants.
    Stores pointers to both prompt rows and which was selected.
    """
    __tablename__ = "choices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    offline_prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    gemini_prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    chosen_prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    chosen_model = Column(String, nullable=False)  # 'offline' or 'gemini'
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="choices")
    # relationships to Prompt are accessible via session.get if needed


# ==== Utilities: password hashing ====
PBKDF2_ITERS = 200_000
SALT_BYTES = 16  # bytes


def _generate_salt_hex() -> str:
    return secrets.token_hex(SALT_BYTES)


def _hash_password_hex(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return dk.hex()


# ==== DB initialization ====
def init_db():
    Base.metadata.create_all(bind=engine)


# ==== User management helpers ====
def register_user(username: str, password: str, email: str | None = None, role: str = "user"):
    """
    Registers a user if username not taken and email provided (non-empty).
    Returns (True, user_obj) on success.
    Returns (False, 'username_exists'|'invalid_username'|'email_required'|'email_exists'|'error').
    """
    username = (username or "").strip()
    email = (email or "").strip()
    if not username:
        return False, "invalid_username"
    if not email:
        return False, "email_required"

    with SessionLocal() as session:
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            return False, "username_exists"
        # Ensure email unique
        exist_e = session.query(User).filter(User.email == email).first()
        if exist_e:
            return False, "email_exists"

        salt_hex = _generate_salt_hex()
        pw_hash = _hash_password_hex(password, salt_hex)

        new = User(
            username=username,
            email=email,
            password_salt=salt_hex,
            password_hash=pw_hash,
            role=role
        )
        session.add(new)
        session.commit()
        session.refresh(new)
        return True, new


def authenticate_user(username: str, password: str):
    """Return User object if username/password correct, else None."""
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            return None
        computed = _hash_password_hex(password, user.password_salt)
        if secrets.compare_digest(computed, user.password_hash):
            return user
        return None


def change_password(username: str, old_password: str, new_password: str):
    """Return (True, None) on success, (False, reason) on failure."""
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            return False, "no_user"
        computed = _hash_password_hex(old_password, user.password_salt)
        if not secrets.compare_digest(computed, user.password_hash):
            return False, "incorrect_old_password"

        new_salt = _generate_salt_hex()
        new_hash = _hash_password_hex(new_password, new_salt)
        user.password_salt = new_salt
        user.password_hash = new_hash
        session.add(user)
        session.commit()
        return True, None


# ==== Recording Choices / building ranker dataset ====
def record_choice(user_id: int, offline_prompt_id: int, gemini_prompt_id: int, chosen_prompt_id: int, chosen_model: str):
    """Insert a Choice row linking the two variants and the selected prompt/model."""
    with SessionLocal() as session:
        # validate
        offline = session.get(Prompt, offline_prompt_id)
        gemini = session.get(Prompt, gemini_prompt_id)
        chosen = session.get(Prompt, chosen_prompt_id)
        if not (offline and gemini and chosen):
            return False, "prompt_not_found"
        if chosen_prompt_id not in (offline_prompt_id, gemini_prompt_id):
            return False, "chosen_not_in_pair"

        ch = Choice(
            user_id=int(user_id),
            offline_prompt_id=int(offline_prompt_id),
            gemini_prompt_id=int(gemini_prompt_id),
            chosen_prompt_id=int(chosen_prompt_id),
            chosen_model=str(chosen_model)
        )
        session.add(ch)
        session.commit()
        session.refresh(ch)
        return True, ch


def get_choice_dataset() -> List[Tuple[str, str]]:
    """
    Build dataset for ranker:
      For each Choice row, compose a single text that includes context and both variants:
      TEXT = "<topic> | <style> | OFFLINE: {offline_text} || GEMINI: {gemini_text}"
      LABEL = chosen_model ('offline'|'gemini')
    Returns list of (text, label).
    """
    rows = []
    with SessionLocal() as session:
        choices = session.query(Choice).order_by(Choice.timestamp).all()
        for c in choices:
            offline = session.get(Prompt, c.offline_prompt_id)
            gemini = session.get(Prompt, c.gemini_prompt_id)
            if not (offline and gemini):
                continue
            # context from offline (they share context)
            topic = offline.topic
            style = offline.style or ""
            text = f"{topic} | {style} | OFFLINE: {offline.generated_text} || GEMINI: {gemini.generated_text}"
            rows.append((text, c.chosen_model))
    return rows


# ==== Feedback helpers (keep for compatibility) ====
def add_feedback(user_id: int, prompt_id: int, rating: int, comments: str | None = None):
    """
    Store a feedback record.
    Server-side check: user_id must be provided (no anonymous feedback).
    Returns (True, feedback_obj) on success.
    Returns (False, reason) on failure.
    """
    if user_id is None:
        return False, "anonymous_not_allowed"

    with SessionLocal() as session:
        prompt = session.get(Prompt, prompt_id)
        if not prompt:
            return False, "prompt_not_found"

        # Optional: update existing feedback instead of duplicates? For now we allow duplicates.
        fb = Feedback(
            user_id=int(user_id),
            prompt_id=int(prompt_id),
            rating=int(bool(rating)),
            comments=comments
        )
        session.add(fb)
        session.commit()
        session.refresh(fb)
        return True, fb


# ==== Analytics helpers (unchanged) ====
def get_top_topics(limit: int = 10):
    with SessionLocal() as session:
        rows = (
            session.query(Prompt.topic, func.count(Prompt.id).label("count"))
            .group_by(Prompt.topic)
            .order_by(desc("count"))
            .limit(limit)
            .all()
        )
        return rows


def get_top_styles(limit: int = 10):
    with SessionLocal() as session:
        rows = (
            session.query(Prompt.style, func.count(Prompt.id).label("count"))
            .group_by(Prompt.style)
            .order_by(desc("count"))
            .limit(limit)
            .all()
        )
        return rows


def get_feedback_counts_by_model():
    with SessionLocal() as session:
        rows = (
            session.query(Prompt.model_used, Feedback.rating, func.count(Feedback.id))
            .join(Feedback, Feedback.prompt_id == Prompt.id)
            .group_by(Prompt.model_used, Feedback.rating)
            .all()
        )
        result = {}
        for model_used, rating, cnt in rows:
            entry = result.setdefault(model_used or "unknown", {"positive": 0, "negative": 0})
            if int(rating) == 1:
                entry["positive"] += cnt
            else:
                entry["negative"] += cnt
        return result


def get_active_users(limit: int = 20):
    with SessionLocal() as session:
        rows = (
            session.query(User.username, func.count(Prompt.id).label("cnt"))
            .join(Prompt, Prompt.user_id == User.id)
            .group_by(User.id)
            .order_by(desc("cnt"))
            .limit(limit)
            .all()
        )
        return rows


# Initialize DB on import
init_db()
