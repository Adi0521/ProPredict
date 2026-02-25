from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    run_id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="pending")
    progress_percent = Column(Integer, nullable=False, default=0)
    sequence = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)  # full result stored as JSON string


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist. Called on API startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a database session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
