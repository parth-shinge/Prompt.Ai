from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# Create SQLite database file
engine = create_engine("sqlite:///promptgen.db", echo=True)  # echo=True logs queries
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# ==== Define Tables ====

# 1. Users
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=True)

    prompts = relationship("Prompt", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")

# 2. Prompts
class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True, index=True)
    tool = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    style = Column(String, nullable=True)

    # 🔥 Newly added fields
    platform_name = Column(String, nullable=True)
    color_palette = Column(String, nullable=True)
    mood = Column(String, nullable=True)

    generated_text = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)  # offline/gemini/hybrid
    timestamp = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="prompts")
    feedbacks = relationship("Feedback", back_populates="prompt")

# 3. Feedback
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

# ==== Create Tables ====
def init_db():
    Base.metadata.create_all(bind=engine)
