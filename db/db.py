"""
PostgreSQL database module for GPR simulation parameter storage.

Provides:
- SQLModel ORM classes for `extraction_sessions` and `simulations` tables
- Engine / session factory targeting the docker-compose Postgres instance
- CRUD helpers: create sessions, batch-insert simulations, update signals, bulk-read for training
"""

import os
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlmodel import SQLModel, Field, Session, create_engine
from sqlalchemy import (
    Column,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ARRAY,
    Index,
    text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://myuser:mypassword@localhost:5432/my_app",
)


def pg_dsn() -> str:
    """The plain-libpq form of DATABASE_URL (no SQLAlchemy driver suffix) —
    used by psycopg3 consumers like the LangGraph Postgres checkpointer."""
    return DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},  # fail fast if PG is unreachable
)


def get_session() -> Session:
    """Return a new SQLModel session."""
    return Session(engine)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class ExtractionSession(SQLModel, table=True):
    """Stores the user's original parameter ranges from extraction agents.

    Each JSONB column is populated incrementally as its corresponding agent
    finishes extraction. A fresh row is created on the first agent POST;
    subsequent agents PATCH their section into the same row.
    """

    __tablename__ = "extraction_sessions"

    id: _uuid.UUID = Field(
        default_factory=_uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True),
    )
    user_id: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    # Four subagent extraction outputs — JSONB, nullable until that agent runs
    layers_ranges: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    antenna_waveform: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    model_config_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    advanced_params: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )

    num_samples_requested: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    status: str = Field(
        default="pending", sa_column=Column(Text, nullable=False)
    )


class Simulation(SQLModel, table=True):
    """One row = one complete simulation (parameters + signals).

    Scalar antenna/waveform/model fields are regular typed columns.
    Variable-length data (layers, objects) is stored as JSONB.
    Signal arrays are native PostgreSQL float8[].
    """

    __tablename__ = "simulations"

    # ── Identity ──────────────────────────────────────────────────────────
    id: _uuid.UUID = Field(
        default_factory=_uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True),
    )
    session_id: _uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            nullable=False,
        )
    )
    user_id: str = Field(sa_column=Column(Text, nullable=False))
    sample_index: int = Field(sa_column=Column(Integer, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    # ── Antenna / Waveform (scalar columns) ───────────────────────────────
    antenna_kind: str = Field(sa_column=Column(Text, nullable=False))
    antenna_axis: str = Field(sa_column=Column(Text, nullable=False))
    tx_rx_offset_m: float = Field(
        sa_column=Column(Float, nullable=False)
    )
    resistance: Optional[float] = Field(
        default=None, sa_column=Column(Float, nullable=True)
    )
    source_start_time: Optional[float] = Field(
        default=None, sa_column=Column(Float, nullable=True)
    )
    source_end_time: Optional[float] = Field(
        default=None, sa_column=Column(Float, nullable=True)
    )
    waveform_kind: str = Field(sa_column=Column(Text, nullable=False))
    waveform_amplitude: float = Field(
        sa_column=Column(Float, nullable=False)
    )
    waveform_center_freq_hz: float = Field(
        sa_column=Column(Float, nullable=False)
    )
    waveform_name: str = Field(sa_column=Column(Text, nullable=False))

    # ── Model Config (scalar columns) ─────────────────────────────────────
    model: str = Field(sa_column=Column(Text, nullable=False))
    title: str = Field(sa_column=Column(Text, nullable=False))
    source_height_m: float = Field(
        sa_column=Column(Float, nullable=False)
    )
    domain_x: float = Field(sa_column=Column(Float, nullable=False))
    domain_y: float = Field(sa_column=Column(Float, nullable=False))
    cells_per_wavelength: float = Field(
        sa_column=Column(Float, nullable=False)
    )
    max_cell_m: float = Field(sa_column=Column(Float, nullable=False))
    rx_same_height: bool = Field(
        default=True, sa_column=Column(Boolean, nullable=False)
    )
    temperature_c: float = Field(
        default=20.0, sa_column=Column(Float, nullable=False)
    )
    enforce_validity: bool = Field(
        default=True, sa_column=Column(Boolean, nullable=False)
    )

    # ── Advanced scalars ──────────────────────────────────────────────────
    pml_cells: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    num_threads: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    output_dir: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    # ── Layers (variable-length → JSONB) ──────────────────────────────────
    layers: List[Dict[str, Any]] = Field(
        sa_column=Column(JSONB, nullable=False)
    )
    num_layers: int = Field(sa_column=Column(Integer, nullable=False))

    # Derived dielectric labels, one entry per `layers` entry IN THE SAME ORDER:
    # {name, eps_r_dry, eps_r_wet, sigma_dry, sigma_wet}. eps is the in-band real
    # permittivity and sigma the effective conductivity (S/m) at the two edges of
    # the layer's theta_v band — both come straight from gprMax's own Peplinski
    # routine (backend.dataset_sampling.peplinski_derive). Nullable: an adopted or
    # legacy dataset can predate the derive manifest.
    derived_layers: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )

    # ── Geometry objects (variable-length → JSONB, nullable) ──────────────
    cylinders: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    boxes: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    spheres: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )

    # ── Optional config objects (JSONB, nullable) ─────────────────────────
    surface_roughness: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    snapshots: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    rx_array: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )

    # ── Signal output (populated after gprMax run) ────────────────────────
    signal_ex: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_ey: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_ez: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_hx: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_hy: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_hz: Optional[List[float]] = Field(
        default=None, sa_column=Column(ARRAY(Float), nullable=True)
    )
    signal_length: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    simulation_completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    # ── File references ───────────────────────────────────────────────────
    input_file_path: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    output_file_path: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class ChatSessionRow(SQLModel, table=True):
    """One row = one chat (= one simulation). Persists the whole WebSocket
    ChatSession as a JSONB blob so a chat survives browser close and backend
    restart; the promoted columns exist only for the per-user chat list.

    The LLM message history is NOT in the blob — it lives in the LangGraph
    Postgres checkpointer tables, keyed by `thread_id`.
    """

    __tablename__ = "chat_sessions"

    # Raw session-id string (frontend-minted uuid or legacy "session-..." form).
    id: str = Field(sa_column=Column(Text, primary_key=True))
    user_id: str = Field(sa_column=Column(Text, nullable=False))
    title: str = Field(default="New chat", sa_column=Column(Text, nullable=False))
    thread_id: str = Field(sa_column=Column(Text, nullable=False))
    complete: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, server_default=text("false"))
    )
    has_dataset: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, server_default=text("false"))
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    session_state: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    )


Index("ix_chat_sessions_user_updated", ChatSessionRow.user_id, ChatSessionRow.updated_at.desc())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Table creation (for use outside Alembic, e.g. quick dev setup)
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

# ── Section name → ExtractionSession column name mapping ──────────────────
_SECTION_TO_COLUMN = {
    "layers": "layers_ranges",
    "antenna_waveform": "antenna_waveform",
    "model_config": "model_config_data",
    "advanced_params": "advanced_params",
}


def upsert_extraction_section(
    session_id: _uuid.UUID,
    user_id: str,
    section: str,
    data: dict,
) -> ExtractionSession:
    """Create or update an extraction session row for the given section.

    Called by the FastAPI POST endpoint each time an agent stores parameters.
    First call creates the row; subsequent calls update their section column.
    """
    col_name = _SECTION_TO_COLUMN.get(section)
    if col_name is None:
        raise ValueError(f"Unknown section: {section}")

    with get_session() as db:
        row = db.get(ExtractionSession, session_id)
        if row is None:
            row = ExtractionSession(
                id=session_id,
                user_id=user_id,
                status="pending",
            )
            db.add(row)

        setattr(row, col_name, data)
        db.commit()
        db.refresh(row)
        return row


def batch_insert_simulations(rows: List[dict], chunk_size: int = 5000) -> int:
    """Bulk-insert simulation rows in chunked transactions.

    Args:
        rows: List of dicts, each matching the Simulation columns.
        chunk_size: Rows per INSERT batch (default 5000).

    Returns:
        Total number of rows inserted.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    total = 0
    with get_session() as db:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(Simulation).values(chunk)
            db.exec(stmt)  # type: ignore[arg-type]
            db.commit()
            total += len(chunk)
    return total


def bulk_update_signals(
    updates: List[dict],
    chunk_size: int = 1000,
) -> int:
    """Batch-update signal columns for completed simulations.

    Args:
        updates: List of dicts with keys: id, signal_ex, signal_ey, ...,
                 signal_hz, signal_length.
        chunk_size: Rows per UPDATE batch.

    Returns:
        Total rows updated.
    """
    total = 0
    with get_session() as db:
        for i in range(0, len(updates), chunk_size):
            chunk = updates[i : i + chunk_size]
            for u in chunk:
                sim = db.get(Simulation, u["id"])
                if sim is None:
                    continue
                sim.signal_ex = u.get("signal_ex")
                sim.signal_ey = u.get("signal_ey")
                sim.signal_ez = u.get("signal_ez")
                sim.signal_hx = u.get("signal_hx")
                sim.signal_hy = u.get("signal_hy")
                sim.signal_hz = u.get("signal_hz")
                sim.signal_length = u.get("signal_length")
                if "output_file_path" in u:
                    sim.output_file_path = u["output_file_path"]
                sim.simulation_completed_at = datetime.now(timezone.utc)
            db.commit()
            total += len(chunk)
    return total


def set_simulation_outputs(
    session_id: _uuid.UUID,
    outputs_by_sample: Dict[int, str],
) -> int:
    """Record forward-model results for one session: set output_file_path
    (+ completion timestamp) on its Simulation rows, keyed by sample_index.

    Returns the number of rows updated.
    """
    from sqlmodel import select

    total = 0
    with get_session() as db:
        rows = db.exec(
            select(Simulation).where(Simulation.session_id == session_id)
        ).all()
        for sim in rows:
            path = outputs_by_sample.get(sim.sample_index)
            if path is None:
                continue
            sim.output_file_path = path
            sim.simulation_completed_at = datetime.now(timezone.utc)
            total += 1
        db.commit()
    return total


def delete_simulations_for_session(session_id: _uuid.UUID) -> int:
    """Remove a session's Simulation rows (a re-finalize replaces them).

    Returns the number of rows deleted.
    """
    from sqlmodel import select

    with get_session() as db:
        rows = db.exec(
            select(Simulation).where(Simulation.session_id == session_id)
        ).all()
        for sim in rows:
            db.delete(sim)
        db.commit()
        return len(rows)


# ── Chat session persistence ───────────────────────────────────────────────

def get_chat_session(session_id: str) -> Optional[ChatSessionRow]:
    """Fetch one chat row (full blob) for hydration; None if unknown."""
    with get_session() as db:
        return db.get(ChatSessionRow, session_id)


def list_chat_sessions(user_id: str) -> List[dict]:
    """The per-user chat list — promoted columns only, newest first."""
    from sqlmodel import select

    with get_session() as db:
        rows = db.exec(
            select(
                ChatSessionRow.id,
                ChatSessionRow.title,
                ChatSessionRow.created_at,
                ChatSessionRow.updated_at,
                ChatSessionRow.complete,
                ChatSessionRow.has_dataset,
            )
            .where(ChatSessionRow.user_id == user_id)
            .order_by(ChatSessionRow.updated_at.desc())  # type: ignore[union-attr]
        ).all()
        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": r[2].isoformat() if r[2] else None,
                "updated_at": r[3].isoformat() if r[3] else None,
                "complete": r[4],
                "has_dataset": r[5],
            }
            for r in rows
        ]


def create_chat_stub(session_id: str, user_id: str, thread_id: str) -> None:
    """Insert the empty row for a freshly minted chat (visible in the list
    before its first WebSocket connect)."""
    with get_session() as db:
        db.add(ChatSessionRow(id=session_id, user_id=user_id, thread_id=thread_id))
        db.commit()


def upsert_chat_session(
    session_id: str,
    user_id: str,
    title: str,
    thread_id: str,
    complete: bool,
    has_dataset: bool,
    session_state: Dict[str, Any],
) -> None:
    """Whole-row rewrite of a chat's persisted state (once per turn)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    values = {
        "id": session_id,
        "user_id": user_id,
        "title": title,
        "thread_id": thread_id,
        "complete": complete,
        "has_dataset": has_dataset,
        "session_state": session_state,
        "updated_at": datetime.now(timezone.utc),
    }
    stmt = pg_insert(ChatSessionRow).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[ChatSessionRow.id],
        set_={k: v for k, v in values.items() if k != "id"},
    )
    with get_session() as db:
        db.exec(stmt)  # type: ignore[arg-type]
        db.commit()


def get_simulations_for_session(session_id: _uuid.UUID) -> List[Simulation]:
    """All Simulation rows for one session, ordered by sample_index."""
    from sqlmodel import select

    with get_session() as db:
        return list(
            db.exec(
                select(Simulation)
                .where(Simulation.session_id == session_id)
                .order_by(Simulation.sample_index)  # type: ignore[arg-type]
            ).all()
        )


def get_extraction_session(session_id: _uuid.UUID) -> Optional[ExtractionSession]:
    """Fetch one extraction-session row (the user's parameter ranges)."""
    with get_session() as db:
        return db.get(ExtractionSession, session_id)


def list_extraction_sessions() -> List[ExtractionSession]:
    """All extraction sessions — used by the similarity-index backfill CLI."""
    from sqlmodel import select

    with get_session() as db:
        return list(db.exec(select(ExtractionSession)).all())


def count_incomplete_simulations(session_id: _uuid.UUID) -> tuple:
    """(total, not-yet-completed) Simulation counts for one session.

    Reuse needs signal arrays, not just HDF5 output paths. A row with
    simulation_completed_at set but signal_length still NULL is not adoptable.
    """
    from sqlmodel import select

    with get_session() as db:
        total = db.exec(
            select(func.count()).select_from(Simulation).where(
                Simulation.session_id == session_id
            )
        ).one()
        incomplete = db.exec(
            select(func.count()).select_from(Simulation).where(
                Simulation.session_id == session_id,
                (
                    Simulation.simulation_completed_at.is_(None)  # type: ignore[union-attr]
                    | Simulation.signal_length.is_(None)  # type: ignore[union-attr]
                ),
            )
        ).one()
        return (int(total), int(incomplete))


def get_completed_simulations(
    model_filter: Optional[str] = None,
    num_layers_filter: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Simulation]:
    """Fetch completed simulations for NN training.

    Only returns rows where signal data has been populated.
    """
    from sqlmodel import select

    with get_session() as db:
        stmt = select(Simulation).where(
            Simulation.simulation_completed_at.is_not(None),  # type: ignore[union-attr]
            Simulation.signal_length.is_not(None),  # type: ignore[union-attr]
        )
        if model_filter:
            stmt = stmt.where(Simulation.model == model_filter)  # type: ignore[arg-type]
        if num_layers_filter:
            stmt = stmt.where(Simulation.num_layers == num_layers_filter)  # type: ignore[arg-type]
        if limit:
            stmt = stmt.limit(limit)

        return list(db.exec(stmt).all())
