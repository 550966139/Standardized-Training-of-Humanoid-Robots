"""REST endpoints: upload, create job, list, cancel, download."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import get_session
from ..loaders import detect_format
from ..models import Job, JobStatus, UploadedFile
from ..remote import health_check
from ..trainer.manager import manager

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parents[1] / "web" / "templates")


# ---------- pages ----------
@router.get("/", response_class=HTMLResponse)
async def index(request: Request, sess: AsyncSession = Depends(get_session)):
    result = await sess.execute(select(Job).order_by(Job.created_at.desc()))
    jobs = result.scalars().all()
    return templates.TemplateResponse(request, "index.html", {"jobs": jobs})


@router.get("/health")
async def health():
    return await health_check()


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(job_id: int, request: Request, sess: AsyncSession = Depends(get_session)):
    job = await sess.get(Job, job_id)
    if job is None:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "job_detail.html", {"job": job})


# ---------- uploads & job create ----------
@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    sess: AsyncSession = Depends(get_session),
):
    suffix = Path(file.filename or "").suffix.lower()
    target = settings.uploads_dir / f"{_new_uuid()}{suffix}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with target.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                target.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            fh.write(chunk)

    detected = detect_format(target)
    rec = UploadedFile(
        original_name=file.filename or target.name,
        stored_path=str(target),
        mime=file.content_type,
        size_bytes=total,
        detected_format=detected,
    )
    sess.add(rec)
    await sess.commit()
    await sess.refresh(rec)
    return templates.TemplateResponse(
        request, "partials/upload_card.html",
        {"upload": rec, "detected": detected},
    )


@router.post("/jobs", response_class=HTMLResponse)
async def create_job(
    request: Request,
    upload_id: int = Form(...),
    name: str = Form(...),
    max_iterations: int = Form(1500),
    num_envs: int = Form(4096),
    task: str = Form("Unitree-G1-29dof-Mimic-Dance-102"),
    sess: AsyncSession = Depends(get_session),
):
    up = await sess.get(UploadedFile, upload_id)
    if up is None:
        raise HTTPException(404, "upload missing")
    job = Job(
        name=name,
        upload_id=upload_id,
        task=task,
        max_iterations=max_iterations,
        num_envs=num_envs,
    )
    sess.add(job)
    await sess.commit()
    await sess.refresh(job)
    await manager.enqueue(job.id)
    return templates.TemplateResponse(
        request, "partials/job_row.html", {"job": job},
    )


@router.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job(job_id: int, request: Request, sess: AsyncSession = Depends(get_session)):
    job = await sess.get(Job, job_id)
    if job is None:
        raise HTTPException(404)
    ok = await manager.cancel(job_id)
    return templates.TemplateResponse(
        request, "partials/job_row.html", {"job": job},
    )


# ---------- downloads ----------
_DOWNLOADABLE = {
    "checkpoint": "checkpoint_path",
    "onnx": "onnx_path",
    "video": "video_path",
    "bvh": "motion_bvh_path",
    "fbx": "motion_fbx_path",
    "csv": "motion_csv_path",
}


@router.get("/jobs/{job_id}/download/{kind}")
async def download(
    job_id: int, kind: str, sess: AsyncSession = Depends(get_session)
) -> FileResponse:
    job = await sess.get(Job, job_id)
    if job is None:
        raise HTTPException(404)
    attr = _DOWNLOADABLE.get(kind)
    if attr is None:
        raise HTTPException(400, "unknown artifact")
    path = getattr(job, attr)
    if not path or not Path(path).exists():
        raise HTTPException(404, "artifact missing")
    return FileResponse(path, filename=Path(path).name)


def _new_uuid() -> str:
    import uuid
    return uuid.uuid4().hex
