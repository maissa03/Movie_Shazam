"""Application FastAPI + cycle de vie (``lifespan``).

Démarrage :  SQLite (WAL) -> thread écrivain -> nettoyage des jobs interrompus et des
             fichiers temporaires -> vérification de la bibliothèque -> pool de processus
             (préchauffé, PID relevés) -> pool de threads I/O -> gestionnaire de jobs asyncio.
Arrêt :      refus des nouveaux jobs -> annulation coopérative -> pool arrêté (FFmpeg tués)
             -> threads I/O -> écritures vidées puis thread écrivain arrêté -> base fermée.

Lancer **un seul** processus Uvicorn (pas d'option ``--workers``) : plusieurs workers web
dupliqueraient les pools de processus, la mémoire et l'écrivain SQLite.  Le parallélisme CPU
vient du pool interne, pas du nombre de réplicas du serveur.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import router
from app.concurrency.job_manager import JobManager
from app.concurrency.process_pool import LocalProcessPool
from app.concurrency.thread_io import IOExecutor
from app.core.logging import setup_logging
from app.core.settings import Settings, get_settings
from app.db import repo
from app.db.session import init_db, make_engine, make_session_factory
from app.db.writer import DbWriter, WriterBusy, WriterClosed
from app.services import library
from app.video.ffmpeg_io import FFmpegNotFound, find_binary
from app.video.matching import MatchParams

T = TypeVar("T")
log = logging.getLogger("movieshazam.app")


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    writer: DbWriter
    pool: LocalProcessPool
    io: IOExecutor
    params: MatchParams
    jobs: JobManager = None  # type: ignore[assignment]

    async def read(self, fn: Callable[[Session], T]) -> T:
        """Lecture SQLite courte dans un thread (session dédiée, jamais partagée)."""
        def _run() -> T:
            with self.session_factory() as s:
                return fn(s)
        return await asyncio.to_thread(_run)


def _clean_temp(settings: Settings) -> int:
    n = 0
    for p in settings.temp_dir.glob("*.part"):
        p.unlink(missing_ok=True)
        n += 1
    return n


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        settings.ensure_dirs()
        try:
            find_binary("ffmpeg")
            find_binary("ffprobe")
        except FFmpegNotFound as exc:
            log.error("%s", exc)
        engine = make_engine(settings.database_path)
        init_db(engine)
        sf = make_session_factory(engine)
        writer = DbWriter(sf)
        writer.start()
        interrupted = writer.run_sync(repo.fail_interrupted_jobs)
        removed = await asyncio.to_thread(_clean_temp, settings)
        report = await asyncio.to_thread(library.verify_library, sf, writer, settings)
        log.info("démarrage : %d job(s) interrompu(s), %d fichier(s) temporaire(s) supprimé(s), bibliothèque %s",
                 interrupted, removed, report)
        pool = LocalProcessPool(settings.effective_workers())
        await asyncio.to_thread(pool.start)
        io = IOExecutor(settings.io_threads)
        ctx = AppContext(settings, engine, sf, writer, pool, io, MatchParams())
        ctx.jobs = JobManager(ctx)
        await ctx.jobs.start()
        app.state.ctx = ctx
        log.info("MovieShazam prêt sur http://%s:%d (%d processus, PID %s)", settings.app_host, settings.app_port,
                 pool.workers, pool.pids)
        try:
            yield
        finally:
            log.info("arrêt en cours...")
            await ctx.jobs.stop()
            await asyncio.to_thread(pool.shutdown)
            await asyncio.to_thread(io.shutdown)
            await asyncio.to_thread(writer.close)
            engine.dispose()
            log.info("arrêt terminé")

    app = FastAPI(
        title="MovieShazam (local)",
        version="1.0.0",
        description="Identification locale d'un extrait vidéo (pHash + alignement temporel). "
                    "Tout s'exécute sur ce PC : aucune ressource distante.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):  # noqa: ANN001, ANN202
        """Refus précoce (413) d'un envoi trop volumineux, avant même de le lire."""
        length = request.headers.get("content-length")
        if request.method == "POST" and length and length.isdigit():
            path = request.url.path
            limit = (settings.max_upload_bytes if path == "/api/movies" else
                     settings.max_clip_bytes if path == "/api/search" else 10 * 1024**2)
            if int(length) > limit + 1024**2:
                return JSONResponse({"detail": f"envoi trop volumineux (limite {limit / 1024**2:.0f} Mo)"},
                                    status_code=413)
        return await call_next(request)

    @app.exception_handler(WriterBusy)
    async def _busy(_r: Request, exc: WriterBusy) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(WriterClosed)
    async def _closed(_r: Request, exc: WriterClosed) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    app.include_router(router)
    return app


app = create_app()
