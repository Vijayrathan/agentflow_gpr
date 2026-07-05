"""
PostgreSQL database module for GPR simulation parameter storage.

Provides:
- SQLModel ORM classes for `extraction_sessions` and `simulations` tables
- Engine / session factory targeting the docker-compose Postgres instance
- CRUD helpers: create sessions, batch-insert simulations, update signals, bulk-read for training
"""

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

DATABASE_URL = "postgresql+psycopg2://myuser:mypassword@localhost:5432/my_app"

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
            Simulation.simulation_completed_at.is_not(None)  # type: ignore[union-attr]
        )
        if model_filter:
            stmt = stmt.where(Simulation.model == model_filter)  # type: ignore[arg-type]
        if num_layers_filter:
            stmt = stmt.where(Simulation.num_layers == num_layers_filter)  # type: ignore[arg-type]
        if limit:
            stmt = stmt.limit(limit)

        return list(db.exec(stmt).all())
