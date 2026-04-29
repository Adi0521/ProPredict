from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, create_engine, text
from sqlalchemy.exc import OperationalError
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
    modal_call_id = Column(String, nullable=True)  # Modal FunctionCall object_id (production)


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist, and add any missing columns."""
    Base.metadata.create_all(bind=engine)
    # Add modal_call_id to existing deployments that predate Modal migration
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN modal_call_id VARCHAR"))
            conn.commit()
        except OperationalError:
            pass  # column already exists


def get_db():
    """FastAPI dependency: yields a database session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
