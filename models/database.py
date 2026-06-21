from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, create_engine, inspect, text
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
    """Create all tables if they don't exist, and add any missing columns.

    create_all() creates the `jobs` table with modal_call_id already present
    (it is declared on the model), so on a fresh database no migration is
    needed. The ALTER below exists only for deployments whose `jobs` table was
    created before modal_call_id was added to the model. create_all() does not
    alter existing tables, so we add the column conditionally — guarded by an
    inspector check rather than a try/except, because "column already exists"
    raises ProgrammingError (not OperationalError) on Postgres and would
    otherwise crash startup.
    """
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if inspector.has_table("jobs"):
        existing_cols = {col["name"] for col in inspector.get_columns("jobs")}
        if "modal_call_id" not in existing_cols:
            with engine.begin() as conn:  # begin() commits on success
                conn.execute(text("ALTER TABLE jobs ADD COLUMN modal_call_id VARCHAR"))


def get_db():
    """FastAPI dependency: yields a database session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()