"""Database models: upload and training job."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class JobStatus(str, enum.Enum):
    queued = "queued"
    preparing = "preparing"   # loading mocap / retargeting
    training = "training"
    exporting = "exporting"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    detected_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    upload_id: Mapped[int] = mapped_column(Integer)
    robot: Mapped[str] = mapped_column(String(32), default="g1_29dof")
    task: Mapped[str] = mapped_column(String(128), default="Unitree-G1-29dof-Mimic-Dance-102")
    max_iterations: Mapped[int] = mapped_column(Integer, default=1500)
    num_envs: Mapped[int] = mapped_column(Integer, default=4096)

    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    progress_iter: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    workspace_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    onnx_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    motion_bvh_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    motion_fbx_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    motion_csv_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
