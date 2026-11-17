"""Gestionnaire de jobs **asyncio** : coordonne sans calculer.

Rôle de l'event loop : recevoir les demandes, créer les jobs, réserver les ressources,
déclencher les traitements et publier leur état — **sans jamais bloquer**.  Le calcul lourd
part dans le ``LocalProcessPool`` (``asyncio.wrap_future``), les E/S bloquantes dans des
threads (``IOExecutor`` / ``asyncio.to_thread``), les écritures dans le ``DbWriter``.

Primitives utilisées (chacune pour un besoin concret) :
  * ``asyncio.Queue(maxsize)`` : file **bornée** des jobs d'indexation, consommée par
    ``MAX_ACTIVE_INDEX_JOBS`` tâches -> on ne lance jamais plusieurs indexations massives ;
  * ``asyncio.Semaphore`` : nombre maximal de recherches simultanées ;
  * ``asyncio.Condition`` : un benchmark exige l'usage **exclusif** du CPU ; il attend le
    prédicat « aucune indexation en cours » (``wait_for``) et les indexations attendent
    « aucun benchmark en cours » ;
  * ``asyncio.wait(FIRST_COMPLETED)`` : fenêtre glissante de segments en vol ;
  * ``loop.call_soon_threadsafe`` : un thread (benchmark) publie sa progression **dans**
    l'event loop, qui reste l'unique propriétaire de l'état des jobs.

État partagé : le dictionnaire ``jobs`` n'est modifié que depuis le thread de l'event loop
(coroutines coopératives, aucune préemption entre deux ``await``) : **aucun verrou n'est
nécessaire**.  Annulation **coopérative entre segments** : les segments non démarrés sont
annulés, ceux déjà engagés se terminent et sont enregistrés (reprise possible).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.db import repo
from app.db.models import utcnow
from app.db.search_repo import find_candidates
from app.services import library, searching
from app.services.indexing import build_task, plan_for
from app.video.ffmpeg_io import FFmpegError
from app.video.matching import identify
from app.video.pipeline import analyze_segment
from app.video.segmentation import format_hms, frame_interval_ms

if TYPE_CHECKING:
    from app.main import AppContext

log = logging.getLogger("movieshazam.jobs")
TERMINAL = ("completed", "failed", "cancelled")


class QueueFull(RuntimeError):
    pass


class NotAccepting(RuntimeError):
    pass


@dataclass
class JobState:
    id: str
    kind: str  # index | search | benchmark
    label: str
    movie_id: int | None = None
    state: str = "queued"
    stage: str = "en file d'attente"
    completed_units: int = 0
    total_units: int = 0
    fingerprints: int = 0
    frames_decoded: int = 0
    pids: set[int] = field(default_factory=set)
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    cancel_requested: bool = False
    force: bool = False
    search_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)  # signalé à la fin du job

    @property
    def progress(self) -> float:
        if self.total_units:
            return self.completed_units / self.total_units
        return 1.0 if self.state == "completed" else 0.0

    def to_dict(self) -> dict:
        end = self.finished_at or utcnow()
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "movie_id": self.movie_id, "state": self.state,
            "stage": self.stage, "completed_units": self.completed_units, "total_units": self.total_units,
            "progress": round(self.progress, 4), "fingerprints": self.fingerprints,
            "frames_decoded": self.frames_decoded, "workers_used": len(self.pids), "worker_pids": sorted(self.pids),
            "created_at": self.created_at.isoformat() + "Z",
            "started_at": self.started_at.isoformat() + "Z" if self.started_at else None,
            "finished_at": self.finished_at.isoformat() + "Z" if self.finished_at else None,
            "elapsed_ms": int((end - self.started_at).total_seconds() * 1000) if self.started_at else None,
            "error": self.error, "cancel_requested": self.cancel_requested, "search_id": self.search_id,
            "details": self.details,
        }


def job_row_to_dict(row) -> dict:  # noqa: ANN001
    """Job historique (lu en base) au même format que ``JobState.to_dict``."""
    pids = json.loads(row.worker_pids or "[]")
    elapsed = None
    if row.started_at:
        elapsed = int(((row.finished_at or utcnow()) - row.started_at).total_seconds() * 1000)
    details = json.loads(row.details_json or "{}")
    return {
        "id": row.id, "kind": row.kind, "label": row.label, "movie_id": row.movie_id, "state": row.state,
        "stage": row.stage, "completed_units": row.completed_units, "total_units": row.total_units,
        "progress": round(row.completed_units / row.total_units, 4) if row.total_units else
        (1.0 if row.state == "completed" else 0.0),
        "fingerprints": row.fingerprints_count, "frames_decoded": row.frames_decoded,
        "workers_used": len(pids), "worker_pids": pids,
        "created_at": row.created_at.isoformat() + "Z",
        "started_at": row.started_at.isoformat() + "Z" if row.started_at else None,
        "finished_at": row.finished_at.isoformat() + "Z" if row.finished_at else None,
        "elapsed_ms": elapsed, "error": row.error, "cancel_requested": False,
        "search_id": details.get("search_id"), "details": details,
    }


class JobManager:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.jobs: dict[str, JobState] = {}
        self._movie_jobs: dict[int, str] = {}
        self._tasks: set[asyncio.Task] = set()
        self._consumers: list[asyncio.Task] = []
        self._accepting = False
        self._index_queue: asyncio.Queue[str | None]
        self._search_sem: asyncio.Semaphore
        self._cond: asyncio.Condition
        self._heavy_running = 0
        self._exclusive_active = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------ cycle de vie
    async def start(self) -> None:
        s = self.ctx.settings
        self._loop = asyncio.get_running_loop()
        self._index_queue = asyncio.Queue(maxsize=s.max_queued_jobs)
        self._search_sem = asyncio.Semaphore(s.max_active_search_jobs)
        self._cond = asyncio.Condition()
        self._consumers = [asyncio.create_task(self._consumer(i), name=f"index-consumer-{i}")
                           for i in range(s.max_active_index_jobs)]
        self._accepting = True

    async def stop(self, timeout: float = 60.0) -> None:
        """Refuse les nouveaux jobs, annule ceux en attente, laisse finir proprement les autres."""
        self._accepting = False
        for job in self.jobs.values():
            if job.state not in TERMINAL:
                job.cancel_requested = True
        while not self._index_queue.empty():
            job_id = self._index_queue.get_nowait()
            if job_id and job_id in self.jobs:
                await self._finish(self.jobs[job_id], "cancelled", "arrêt de l'application")
        for _ in self._consumers:
            await self._index_queue.put(None)  # sentinelle : une par consommateur
        pending = [*self._consumers, *self._tasks]
        if pending:
            done, still = await asyncio.wait(pending, timeout=timeout)
            for t in still:
                t.cancel()
            if still:
                await asyncio.gather(*still, return_exceptions=True)

    @property
    def accepting(self) -> bool:
        return self._accepting

    def _spawn(self, coro: Awaitable, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        self._tasks.add(task)  # garder une référence forte (sinon la tâche peut être collectée)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------ persistance
    async def _persist(self, job: JobState) -> None:
        fields = dict(kind=job.kind, movie_id=job.movie_id, label=job.label[:300], state=job.state, stage=job.stage,
                      completed_units=job.completed_units, total_units=job.total_units,
                      fingerprints_count=job.fingerprints, frames_decoded=job.frames_decoded,
                      workers_used=len(job.pids), worker_pids=json.dumps(sorted(job.pids)),
                      created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
                      error=job.error, details_json=json.dumps({**job.details, "search_id": job.search_id}))
        await self.ctx.writer.run(lambda s: repo.upsert_job(s, job.id, **fields))

    async def _start(self, job: JobState, stage: str) -> None:
        job.state, job.stage, job.started_at = "running", stage, utcnow()
        await self._persist(job)

    async def _finish(self, job: JobState, state: str, error: str | None = None, stage: str | None = None) -> None:
        job.state = state
        job.error = error
        job.stage = stage or {"completed": "terminé", "failed": "échec", "cancelled": "annulé"}[state]
        job.finished_at = utcnow()
        if job.movie_id is not None and self._movie_jobs.get(job.movie_id) == job.id:
            del self._movie_jobs[job.movie_id]
        try:
            await self._persist(job)
        except Exception:  # noqa: BLE001 - la base peut être en fermeture
            log.exception("persistance du job impossible", extra={"job_id": job.id})
        finally:
            job.done.set()
        self._prune()

    def _prune(self, keep: int = 300) -> None:
        finished = [j for j in self.jobs.values() if j.state in TERMINAL]
        for j in sorted(finished, key=lambda j: j.finished_at or utcnow())[:-keep]:
            self.jobs.pop(j.id, None)

    # ------------------------------------------------------------ API publique
    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    def active_job_for_movie(self, movie_id: int) -> JobState | None:
        jid = self._movie_jobs.get(movie_id)
        return self.jobs.get(jid) if jid else None

    async def submit_index(self, movie_id: int, label: str, force: bool = False) -> tuple[JobState, bool]:
        """Met en file une indexation.  Retourne ``(job, créé)`` ; un film déjà en file ou en
        cours d'indexation n'est **jamais** indexé deux fois en parallèle (déduplication)."""
        if not self._accepting:
            raise NotAccepting("l'application s'arrête : nouveaux jobs refusés")
        existing = self.active_job_for_movie(movie_id)
        if existing is not None:
            return existing, False
        if self._index_queue.full():
            raise QueueFull("file d'attente pleine, réessayez plus tard")
        job = JobState(id=uuid.uuid4().hex, kind="index", label=label, movie_id=movie_id, force=force)
        self.jobs[job.id] = job
        self._movie_jobs[movie_id] = job.id
        await self._persist(job)
        self._index_queue.put_nowait(job.id)
        log.info("indexation en file", extra={"job_id": job.id, "movie_id": movie_id})
        return job, True

    async def submit_benchmark(self, label: str, runner: Callable[[Callable[[dict], None]], dict]) -> JobState:
        if not self._accepting:
            raise NotAccepting("l'application s'arrête")
        if any(j.kind == "benchmark" and j.state not in TERMINAL for j in self.jobs.values()):
            raise QueueFull("un benchmark est déjà en cours")
        job = JobState(id=uuid.uuid4().hex, kind="benchmark", label=label)
        self.jobs[job.id] = job
        await self._persist(job)
        self._spawn(self._run_benchmark(job, runner), f"benchmark-{job.id[:8]}")
        return job

    async def submit_search(self, clip_path: Path, clip_name: str, method: str) -> JobState:
        if not self._accepting:
            raise NotAccepting("l'application s'arrête")
        job = JobState(id=uuid.uuid4().hex, kind="search", label=clip_name)
        job.search_id = job.id
        self.jobs[job.id] = job
        await self._persist(job)
        await self.ctx.writer.run(lambda s: repo.save_search(s, id=job.id, job_id=job.id, status="queued",
                                                             clip_name=clip_name[:300], method=method))
        self._spawn(self._run_search(job, clip_path, clip_name, method), f"search-{job.id[:8]}")
        return job

    async def cancel(self, job_id: str) -> JobState | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.state in TERMINAL:
            return job
        job.cancel_requested = True
        if job.state == "queued" and job.kind == "index":
            await self._finish(job, "cancelled", None)
            await self.ctx.writer.run(lambda s: repo.update_movie(s, job.movie_id, status="cancelled")
                                      if (m := repo.get_movie(s, job.movie_id)) and m.status != "indexed" else None)
        else:
            job.stage = "annulation demandée (les segments déjà engagés se terminent)"
        return job

    async def wait_until_terminal(self, job_id: str, timeout: float = 60.0) -> JobState | None:
        """Attend la fin d'un job (ex. avant de supprimer son film) via son ``asyncio.Event``."""
        job = self.jobs.get(job_id)
        if job is None or job.state in TERMINAL:
            return job
        try:
            await asyncio.wait_for(job.done.wait(), timeout)
        except TimeoutError:
            pass
        return job

    # ------------------------------------------------------------ exclusivité CPU
    @asynccontextmanager
    async def _cpu_slot(self, exclusive: bool):
        async with self._cond:
            if exclusive:
                await self._cond.wait_for(lambda: self._heavy_running == 0 and not self._exclusive_active)
                self._exclusive_active = True
            else:
                await self._cond.wait_for(lambda: not self._exclusive_active)
                self._heavy_running += 1
        try:
            yield
        finally:
            async with self._cond:
                if exclusive:
                    self._exclusive_active = False
                else:
                    self._heavy_running -= 1
                self._cond.notify_all()

    # ------------------------------------------------------------ indexation
    async def _consumer(self, n: int) -> None:
        while True:
            job_id = await self._index_queue.get()
            try:
                if job_id is None:  # sentinelle d'arrêt
                    return
                job = self.jobs.get(job_id)
                if job is None or job.state in TERMINAL:
                    continue
                if job.cancel_requested:
                    await self._finish(job, "cancelled")
                    continue
                async with self._cpu_slot(exclusive=False):
                    try:
                        await self._run_index(job)
                    except Exception as exc:  # noqa: BLE001 - jamais de job bloqué en « running »
                        log.exception("indexation en échec", extra={"job_id": job.id, "movie_id": job.movie_id})
                        await self._fail_movie(job, f"erreur inattendue : {exc}")
            finally:
                self._index_queue.task_done()

    async def _fail_movie(self, job: JobState, message: str) -> None:
        mid = job.movie_id
        try:
            await self.ctx.writer.run(lambda s: repo.update_movie(s, mid, status="failed", error=message[:2000]))
        finally:
            await self._finish(job, "failed", message)

    async def _run_index(self, job: JobState) -> None:
        ctx, s = self.ctx, self.ctx.settings
        mid = job.movie_id
        assert mid is not None
        version = s.index_version
        movie = await ctx.read(lambda ss: repo.get_movie(ss, mid))
        if movie is None:
            await self._finish(job, "failed", "film supprimé avant l'indexation")
            return
        try:
            path = library.movie_path(movie, s)
        except ValueError:
            path = None
        if path is None or not path.exists():
            await self._fail_movie(job, "fichier vidéo introuvable dans le stockage")
            return
        await self._start(job, "préparation des segments")
        seg_ms, plans = plan_for(movie.duration_ms, s)
        todo = await ctx.writer.run(lambda ss: repo.prepare_index(ss, mid, version, seg_ms, plans, job.force))
        if todo is None:
            await self._finish(job, "failed", "film supprimé pendant la préparation")
            return
        refreshed = await ctx.read(lambda ss: repo.get_movie(ss, mid))
        job.total_units = len(plans)
        job.completed_units = len(plans) - len(todo)
        job.fingerprints = refreshed.fingerprints_count if refreshed else 0
        job.details.update({"segment_ms": seg_ms, "segments_reused": job.completed_units,
                            "process_workers": ctx.pool.workers, "index_version": version})
        job.stage = f"empreintes : {len(todo)} segment(s) à calculer sur {ctx.pool.workers} processus"
        await self._persist(job)
        log.info("indexation démarrée", extra={"job_id": job.id, "movie_id": mid})

        queue = [(t, build_task(path, t, s, mid, job.id)) for t in todo]
        window = ctx.pool.workers * s.max_inflight_segments_per_worker
        pending: dict[asyncio.Future, tuple[Any, repo.SegmentTodo]] = {}
        error: str | None = None
        t0 = time.perf_counter()
        while queue or pending:
            while queue and len(pending) < window and not job.cancel_requested and error is None:
                todo_item, task = queue.pop(0)
                cf = ctx.pool.submit(analyze_segment, task)
                pending[asyncio.wrap_future(cf)] = (cf, todo_item)
            if not pending:
                break
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for af in done:
                cf, t = pending.pop(af)
                if af.cancelled():
                    continue
                exc = af.exception()
                if exc is not None:
                    msg = (f"segment {t.segment_index} ({format_hms(t.start_ms)}–{format_hms(t.end_ms)}) : "
                           f"{exc}")
                    error = error or msg
                    log.error("segment en échec : %s", exc, extra={"job_id": job.id, "movie_id": mid,
                                                                   "segment": t.segment_index})
                    await ctx.writer.run(lambda ss, sid=t.segment_id, m=msg: repo.mark_segment_failed(ss, sid, m))
                    continue
                res = af.result()
                saved = await ctx.writer.run(
                    lambda ss, r=res, sid=t.segment_id: repo.save_segment_result(ss, mid, sid, version, r, job.id))
                if not saved:
                    error = error or "film supprimé pendant l'indexation"
                    continue
                job.completed_units += 1
                job.fingerprints += res.count
                job.frames_decoded += res.frames_decoded
                job.pids.add(res.pid)
                log.info("segment terminé", extra={"job_id": job.id, "movie_id": mid, "segment": t.segment_index,
                                                    "pid": res.pid, "elapsed_ms": int(res.wall_s * 1000)})
            if job.cancel_requested or error:
                queue.clear()
                for cf, _ in pending.values():
                    cf.cancel()  # n'annule que les segments pas encore démarrés
        job.details["compute_s"] = round(time.perf_counter() - t0, 3)

        if error:
            await self._fail_movie(job, error)
        elif job.cancel_requested:
            await ctx.writer.run(lambda ss: repo.update_movie(
                ss, mid, status="cancelled", error="indexation annulée (segments terminés conservés)"))
            await self._finish(job, "cancelled")
        else:
            ok, msg = await ctx.writer.run(lambda ss: repo.finalize_index(ss, mid, version))
            if ok:
                await self._finish(job, "completed")
                log.info("indexation terminée", extra={"job_id": job.id, "movie_id": mid})
            else:
                await self._finish(job, "failed", msg)

    # ------------------------------------------------------------ recherche
    async def _run_search(self, job: JobState, clip_path: Path, clip_name: str, method: str) -> None:
        ctx, s = self.ctx, self.ctx.settings
        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        async def stage(name: str, label: str, fn: Callable[[], Awaitable[Any]]) -> Any:
            if job.cancel_requested:
                raise asyncio.CancelledError
            job.stage = label
            t = time.perf_counter()
            out = await fn()
            timings[name] = round((time.perf_counter() - t) * 1000, 1)
            job.completed_units += 1
            return out

        try:
            async with self._search_sem:
                job.total_units = 4
                await self._start(job, "validation du clip")
                await ctx.writer.run(lambda ss: repo.save_search(ss, id=job.id, job_id=job.id, status="running",
                                                                 clip_name=clip_name[:300], method=method))
                info = await stage("validation_ms", "validation du clip",
                                   lambda: ctx.io.run(searching.validate_clip, clip_path, s))
                # Le décodage du clip (<= 30 s) est fait par un sous-processus FFmpeg ; le thread
                # attend surtout le pipe.  On n'attend donc pas derrière les segments d'indexation
                # déjà en file dans le pool de processus.
                clip = await stage("fingerprint_ms", "empreintes du clip",
                                   lambda: asyncio.to_thread(searching.fingerprint_clip, clip_path, s, info))

                def _candidates():
                    with ctx.session_factory() as ss:
                        return find_candidates(ss, clip.frames, ctx.params, s.index_version, method)

                cands, stats = await stage("candidates_ms", "recherche des candidats",
                                           lambda: asyncio.to_thread(_candidates))
                grid = frame_interval_ms(s.fingerprint_fps)
                decision = await stage("alignment_ms", "alignement temporel", lambda: asyncio.to_thread(
                    identify, clip.frames, cands, info.duration_ms, grid, clip.step_ms, ctx.params))
                payload = await asyncio.to_thread(searching.decision_payload, decision, ctx.session_factory)
                timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
                payload.update({
                    "clip": {"name": clip_name, "duration_ms": info.duration_ms, "width": info.width,
                             "height": info.height, "fps": info.fps, "codec": info.codec,
                             "frames": len(clip.frames), "sample_step_ms": clip.step_ms},
                    "candidate_stats": stats.to_dict(), "timings": timings, "method": method,
                    "params": {"max_hamming": ctx.params.max_hamming, "accept_score": ctx.params.accept_score,
                               "min_coverage": ctx.params.min_coverage,
                               "ambiguity_ratio": ctx.params.ambiguity_ratio},
                })
                best = decision.best
                await ctx.writer.run(lambda ss: repo.save_search(
                    ss, id=job.id, job_id=job.id, status=decision.status, clip_name=clip_name[:300], method=method,
                    query_duration_ms=info.duration_ms, best_movie_id=best.movie_id if best else None,
                    start_ms=best.start_ms if best else None, end_ms=best.end_ms if best else None,
                    score=best.score if best else (decision.candidates[0].score if decision.candidates else None),
                    elapsed_ms=int(timings["total_ms"]), result_json=json.dumps(payload)))
                job.details.update({"status": decision.status, "timings": timings})
                await self._finish(job, "completed", stage="résultat disponible")
        except asyncio.CancelledError:
            await self._search_failed(job, "cancelled", "recherche annulée")
            if not job.cancel_requested:
                raise
        except (searching.QueryRejected, FFmpegError, ValueError) as exc:
            await self._search_failed(job, "failed", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("recherche en échec", extra={"job_id": job.id})
            await self._search_failed(job, "failed", f"erreur inattendue : {exc}")
        finally:
            await asyncio.to_thread(lambda: clip_path.unlink(missing_ok=True))

    async def _search_failed(self, job: JobState, state: str, message: str) -> None:
        try:
            await self.ctx.writer.run(lambda ss: repo.save_search(
                ss, id=job.id, job_id=job.id, status="failed" if state == "failed" else "cancelled",
                clip_name=job.label[:300], result_json=json.dumps({"error": message})))
        finally:
            await self._finish(job, state, message)

    # ------------------------------------------------------------ benchmark
    async def _run_benchmark(self, job: JobState, runner: Callable[[Callable[[dict], None]], dict]) -> None:
        loop = asyncio.get_running_loop()

        def publish(update: dict) -> None:  # appelé depuis le thread du benchmark
            def apply() -> None:  # exécuté DANS l'event loop : propriétaire unique de l'état
                for k in ("stage", "completed_units", "total_units"):
                    if k in update:
                        setattr(job, k, update[k])
                job.pids.update(update.get("pids", ()))
            loop.call_soon_threadsafe(apply)

        try:
            job.stage = "attente de l'exclusivité CPU (fin des indexations en cours)"
            async with self._cpu_slot(exclusive=True):
                await self._start(job, "benchmark en cours")
                result = await asyncio.to_thread(runner, publish)
                job.details.update({"run_id": result.get("run_id")})
                await self._finish(job, "completed")
        except Exception as exc:  # noqa: BLE001
            log.exception("benchmark en échec", extra={"job_id": job.id})
            await self._finish(job, "failed", str(exc))
