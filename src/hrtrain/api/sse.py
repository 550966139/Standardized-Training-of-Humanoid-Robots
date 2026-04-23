"""Server-Sent Events for live job progress."""
from __future__ import annotations

import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..trainer.manager import manager

sse_router = APIRouter()


@sse_router.get("/sse/jobs/{job_id}")
async def stream(job_id: int):
    async def generator():
        async for msg in manager.listen(job_id):
            yield {"event": msg.get("event", "message"), "data": json.dumps(msg)}
    return EventSourceResponse(generator())
