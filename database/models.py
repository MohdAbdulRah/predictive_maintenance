"""Storage schema.

Raw observations are stored exactly as the agent reported them, keyed by
(device, instance_id, sequence) so replays and overlapping polls can never create
duplicates, and so an agent restart (new instanceId) is never confused with continuity.
Verified-failure and component-life tables are EMPTY until a human loads real records.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DataItemRow(Base):
    __tablename__ = "data_items"
    device_uuid: Mapped[str] = mapped_column(String(128), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    device_name: Mapped[str | None] = mapped_column(String(128))
    name: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(16))
    type: Mapped[str] = mapped_column(String(64))
    sub_type: Mapped[str | None] = mapped_column(String(64))
    units: Mapped[str | None] = mapped_column(String(32))
    native_units: Mapped[str | None] = mapped_column(String(32))
    representation: Mapped[str] = mapped_column(String(16), default="VALUE")
    component_path: Mapped[str | None] = mapped_column(String(512))
    component_type: Mapped[str | None] = mapped_column(String(64))


class ObservationRow(Base):
    __tablename__ = "observations"
    device_uuid: Mapped[str] = mapped_column(String(128), primary_key=True)
    instance_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    data_item_id: Mapped[str] = mapped_column(String(128))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    category: Mapped[str] = mapped_column(String(16))
    element: Mapped[str | None] = mapped_column(String(64))
    value_num: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(Text)
    unavailable: Mapped[bool] = mapped_column(Boolean, default=False)
    condition_level: Mapped[str | None] = mapped_column(String(16))
    native_code: Mapped[str | None] = mapped_column(String(64))
    condition_text: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (Index("ix_obs_item_ts", "device_uuid", "data_item_id", "ts"),
                      Index("ix_obs_ts", "device_uuid", "ts"))


class CollectorState(Base):
    __tablename__ = "collector_state"
    device_uuid: Mapped[str] = mapped_column(String(128), primary_key=True)
    instance_id: Mapped[int | None] = mapped_column(BigInteger)
    next_sequence: Mapped[int | None] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CollectionGap(Base):
    """Every known hole in the record. Downstream code MUST treat these as missing data."""
    __tablename__ = "collection_gaps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_uuid: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(24))   # BUFFER_OVERRUN | INSTANCE_RESET | COMM_FAILURE
    instance_id: Mapped[int | None] = mapped_column(BigInteger)
    from_sequence: Mapped[int | None] = mapped_column(BigInteger)
    to_sequence: Mapped[int | None] = mapped_column(BigInteger)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    gap_start_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    gap_end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)


# ---- Tables below are filled by humans / maintenance systems, never by the ML code ----

class OperatingRun(Base):
    """Independent operating history of a machine. observed_until defines where follow-up
    ends, which is what makes right-censoring (unknown labels) possible."""
    __tablename__ = "operating_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    machine_id: Mapped[str] = mapped_column(String(128), index=True)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    observed_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FailureEvent(Base):
    __tablename__ = "failure_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("operating_runs.run_id"))
    component: Mapped[str | None] = mapped_column(String(128))
    failure_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    failure_mode: Mapped[str | None] = mapped_column(String(128))
    verified_by: Mapped[str] = mapped_column(String(128))      # required: who confirmed it
    evidence_ref: Mapped[str | None] = mapped_column(String(256))  # work order / report id


class ComponentLife(Base):
    __tablename__ = "component_lives"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(128), index=True)
    component: Mapped[str] = mapped_column(String(128))         # e.g. spindle_bearing, tool_T05
    installed_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_reason: Mapped[str | None] = mapped_column(String(32))  # FAILED | REPLACED_WORN | PREVENTIVE | STILL_IN_SERVICE
    event_observed: Mapped[bool] = mapped_column(Boolean, default=False)  # True only for true degradation endpoint
    verified_by: Mapped[str | None] = mapped_column(String(128))
    __table_args__ = (UniqueConstraint("machine_id", "component", "installed_ts"),)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage: Mapped[str] = mapped_column(String(16))    # anomaly | failure | rul
    status: Mapped[str] = mapped_column(String(24))   # not_trained | insufficient_data | trained
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    detail_json: Mapped[str | None] = mapped_column(Text)
