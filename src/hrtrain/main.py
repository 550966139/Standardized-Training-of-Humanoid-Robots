"""FastAPI entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router, sse_router
from .config import settings
from .db import init_models
from .trainer.manager import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings.ensure_dirs()
    await init_models()
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(title="HR Train — Standardized Humanoid Training", lifespan=lifespan)

from pathlib import Path
_STATIC = Path(__file__).parent / "web" / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

app.include_router(router)
app.include_router(sse_router)


def cli() -> None:
    """Entrypoint for `hrtrain` console script."""
    import uvicorn

    uvicorn.run(
        "hrtrain.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
