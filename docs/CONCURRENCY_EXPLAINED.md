# La concurrence dans MovieShazam, expliquée avec le vrai code

Module *Systèmes distribués et parallèles* — ISAMM 2026-2027. Ce document relie chaque technique
du cours au **problème concret** qu'elle résout dans MovieShazam, avec des extraits du code et la
**preuve** (test ou mesure) qu'elle fonctionne. Tout s'exécute sur **un seul PC** : le chapitre 4
(sockets) n'est volontairement pas utilisé, il n'y a ni worker distant ni protocole réseau.

| Technique | Problème concret résolu | Où dans le code | Preuve |
|---|---|---|---|
| Séquentiel | Référence déterministe | `services/indexing.py` (`pool=None`) | `test_sequential_and_parallel_produce_identical_index`, colonne « séquentiel » des benchmarks |
| `multiprocessing` (`ProcessPoolExecutor`) | Indexer des segments de film sur plusieurs cœurs | `concurrency/process_pool.py`, `video/pipeline.py` | PID distincts, empreintes identiques, temps 1/2/4 processus |
| `threading` | E/S bloquantes (copie, SHA-256, ffprobe, vignettes) ; écrivain SQLite unique | `concurrency/thread_io.py`, `db/writer.py` | API réactive pendant l'indexation ; tests de course |
| `asyncio` | Coordonner jobs et requêtes sans bloquer | `concurrency/job_manager.py`, `api/routes.py` | `/api/health` < 5 ms pendant une indexation ; tests d'annulation |
| `queue.Queue`, `Lock`, `Event`, `Barrier`, `Semaphore`, `Condition` | État partagé, arrêt propre, limites | voir § 5 | tests `test_concurrency.py` |

---

## 1. CPU-bound ou I/O-bound ? Le critère qui décide de tout

* **CPU-bound** : le temps est passé à *calculer* (décoder une vidéo, réduire une image, calculer une DCT).
  Plus de cœurs = plus de calcul simultané… si le code peut s'exécuter en parallèle.
* **I/O-bound** : le temps est passé à *attendre* (disque, sous-processus, réseau local, base de données).
  On peut attendre plusieurs choses à la fois sans plusieurs cœurs.

En CPython, le **GIL** (Global Interpreter Lock) n'autorise qu'**un thread à la fois** à exécuter du
bytecode Python. Conséquences :

| | Threads (`threading`) | Processus (`multiprocessing`) | Coroutines (`asyncio`) |
|---|---|---|---|
| Mémoire | partagée | **séparée** (chaque processus a la sienne) | partagée (un seul thread) |
| Parallélisme CPU réel | non pour du Python pur (GIL) | **oui** | non |
| Coût de création | faible | élevé (Windows : *spawn* = nouvel interpréteur) | très faible |
| Communication | variables partagées + verrous | sérialisation *pickle* via files/pipes (IPC) | appels directs |
| Usage MovieShazam | E/S bloquantes, écrivain SQLite | indexation des segments | API + orchestration des jobs |

Le GIL est relâché pendant les appels système bloquants (lecture de fichier, attente d'un sous-processus) :
c'est pourquoi des threads conviennent très bien aux E/S, mais pas au calcul Python pur.

## 2. `multiprocessing` : l'indexation parallèle (chapitre 2)

### 2.1 Découper en unités de travail indépendantes

Un film est découpé en **segments** (`video/segmentation.py`), par exemple 180 s pour un long-métrage,
15 s pour un film de démonstration de 2 min. **Une tâche = un segment**, jamais une image : le coût
de lancement d'une tâche (sérialisation, file inter-processus) est amorti sur des centaines d'images.

Le découpage ne dépend **pas** du nombre de processus : séquentiel et parallèle traitent exactement les
mêmes segments. Toutes les bornes sont des multiples de 500 ms, donc les horodatages calculés par un
segment tombent sur la même grille globale que si le film était décodé d'un seul tenant
(vérifié : `test_segmented_decode_equals_whole_decode`).

### 2.2 La même fonction métier dans les deux modes

```python
# backend/app/video/pipeline.py — fonction de MODULE, arguments sérialisables
def analyze_segment(task: SegmentTask) -> SegmentResult:
    """Décode le segment en flux et calcule un pHash par image échantillonnée."""
    for k, frame in iter_gray_frames(task.video_path, start_ms=decode_start, ...):
        ts = decode_start + k * interval          # horodatage GLOBAL dans le film
        if ts < task.start_ms: continue          # pré-lecture : appartient au segment précédent
        if ts >= task.end_ms: break              # règle de propriété [start, end)
        value, info = phash_gray(frame)
        result.timestamps.append(ts); result.hashes.append(value) ...
    return result                                 # quelques Ko : jamais d'images brutes
```

```python
# backend/app/services/indexing.py
if pool is None:                                   # séquentiel : une simple boucle
    for seg_id, task in tasks.items():
        _save(seg_id, analyze_segment(task))
else:                                              # parallèle : mêmes tâches, fenêtre bornée
    pending[pool.submit(analyze_segment, task)] = seg_id
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
```

### 2.3 Le pool : créé une fois, compatible Windows

```python
# backend/app/concurrency/process_pool.py
self._ctx = mp.get_context("spawn")               # spawn partout (défaut Windows, sûr sous Linux)
ProcessPoolExecutor(max_workers=self.workers, mp_context=self._ctx,
                    initializer=worker_initializer, initargs=(self._abort, barrier))
```

* **`spawn`** : chaque processus démarre un *nouvel interpréteur* qui ré-importe le module principal.
  C'est pourquoi les fonctions envoyées au pool sont définies au niveau du module, et pourquoi les
  scripts qui créent des processus utilisent `if __name__ == "__main__":`.
* Le pool est créé **une seule fois** dans le *lifespan* FastAPI et réutilisé. On lance **un seul**
  processus Uvicorn : plusieurs workers web dupliqueraient le pool, la RAM et l'écrivain SQLite.
* `worker_initializer` fixe `cv2.setNumThreads(1)` : sans cela, chaque processus pourrait lancer
  12 threads OpenCV/NumPy (**sursouscription**). De même, chaque FFmpeg est lancé avec `-threads 1`.
* **Préchauffage avec une `multiprocessing.Barrier`** : N tâches attendent qu'il y ait N tâches en
  cours simultanément — preuve que N processus distincts existent (on relève leurs PID), sans `sleep()`.

### 2.4 Pourquoi pas des threads pour l'indexation ?

Chaque segment, c'est un décodage vidéo (dans le sous-processus FFmpeg piloté par le worker) puis
du calcul Python/NumPy (redimensionnement, DCT, médiane, empaquetage des bits). Avec des threads, la
partie Python serait sérialisée par le GIL ; avec des processus, chaque segment a son interpréteur,
sa mémoire, son FFmpeg, et un plantage dans un segment ne touche pas le serveur web.
Le coût : démarrage des processus (mesuré à part : ≈ 1–2 s pour 4 processus sous Windows),
sérialisation des résultats (minime : on ne renvoie que des entiers), et RAM (≈ 70–100 Mo par processus).

### 2.5 Annulation coopérative, erreurs, arrêt

```python
# backend/app/concurrency/job_manager.py (extrait)
while queue or pending:
    while queue and len(pending) < window and not job.cancel_requested and error is None:
        cf = ctx.pool.submit(analyze_segment, task)
        pending[asyncio.wrap_future(cf)] = (cf, todo_item)
    done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
    ...
    if job.cancel_requested or error:
        for cf, _ in pending.values():
            cf.cancel()      # n'annule que les segments PAS ENCORE démarrés
```

* Fenêtre **bornée** de segments en vol (`2 × processus`) : jamais 200 tâches empilées en mémoire.
* Un segment en échec → erreur explicite, les autres sont annulés, le film passe en `failed` ;
  `finalize_index` refuse de marquer `indexed` un index incomplet.
* Les segments terminés restent en base : une réindexation **reprend** sans recalcul
  (`test_cancel_then_resume_indexing`).
* Arrêt de l'application : un `multiprocessing.Event` transmis à chaque processus au démarrage est
  levé ; les segments en cours s'interrompent à l'image suivante et leur FFmpeg est tué
  (`test_backend_graceful_shutdown` vérifie qu'aucun PID du pool ne survit).

## 3. `threading` : les E/S bloquantes et l'écrivain unique (chapitre 1)

### 3.1 Pool de threads borné pour les fichiers

```python
# backend/app/concurrency/thread_io.py
self._executor = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="io")   # IO_THREADS=2
async def run(self, fn, *args):
    return await asyncio.get_running_loop().run_in_executor(self._executor, partial(fn, *args))
```

Utilisé pour : copie de l'upload **par blocs de 1 Mo avec SHA-256 calculé au passage**, `ffprobe`,
génération des vignettes. Borné à 2 pour ne pas saturer le disque.

### 3.2 Producteur–consommateur : un seul thread écrit dans SQLite

```python
# backend/app/db/writer.py
self._queue = queue.Queue(maxsize=maxsize)          # file bornée, déjà thread-safe : PAS de Lock
self._thread = threading.Thread(target=self._run, name="sqlite-writer", daemon=False)

def _run(self):
    while True:
        item = self._queue.get()
        if item is _STOP: return                    # sentinelle : arrêt après vidage de la file
        fn, fut = item
        with self._session_factory() as session, session.begin():   # une transaction
            fut.set_result(fn(session))
```

* SQLite n'accepte qu'un écrivain à la fois : plutôt que de laisser des threads se battre pour le
  verrou (`database is locked`), **toutes** les écritures passent par ce thread. Les processus du
  pool n'ouvrent jamais la base.
* Le mode **WAL** permet aux lectures (API, recherche) de se faire en parallèle de l'écriture.
* Thread **non daemon** + sentinelle FIFO + `join()` : à l'arrêt, toutes les écritures soumises sont
  vidées (`test_writer_close_drains_queue_and_refuses_new_work`).
* Bonus : « ce fichier existe-t-il déjà ? sinon l'insérer » s'exécute dans ce thread → deux imports
  simultanés du même film ne peuvent pas créer de doublon, sans verrou supplémentaire.

### 3.3 La condition de course, démontrée puis évitée

`test_race_condition_without_lock_loses_updates` : 8 threads font `v = x; x = v + 1` 2000 fois
→ le total final est **inférieur** à 16 000 (incréments perdus). Avec un `threading.Lock`
(`test_lock_preserves_consistency`) ou avec l'écrivain unique
(`test_single_writer_serializes_concurrent_sqlite_updates`, lecture-modification-écriture SQLite
depuis 8 threads) : **aucune perte**.

### 3.4 Pourquoi la recherche d'un clip utilise un thread et pas le pool ?

Le clip dure au plus 30 s : son décodage est fait par un **sous-processus FFmpeg** (parallèle par
nature), et Python ne calcule que ~150–300 pHash (quelques dizaines de ms). Le thread attend surtout
le pipe (GIL relâché). L'envoyer dans le pool le ferait attendre derrière les segments d'indexation
déjà en file : on garde le pool pour les traitements longs, la recherche reste réactive.

## 4. `asyncio` : orchestrer sans bloquer

L'event loop **ne calcule jamais** : elle reçoit les requêtes, crée les jobs, attend des *futures*.

```python
@router.post("/movies", status_code=202)            # réponse immédiate : 202 Accepted + id de job
async def import_movie(...):
    staged = await ctx.io.run(library.stage_stream, file.file, ...)     # thread E/S
    movie, created = await ctx.writer.run(lambda s: library.register_staged(s, staged, ...))  # écrivain
    job_state, _ = await ctx.jobs.submit_index(movie.id, movie.title)  # file asyncio bornée
```

* `asyncio.Queue(maxsize=…)` : file **bornée** des indexations, consommée par `MAX_ACTIVE_INDEX_JOBS`
  tâches ; `asyncio.Semaphore` : nombre maximal de recherches simultanées.
* `asyncio.Condition` : un benchmark exige l'exclusivité CPU ; il attend le prédicat
  « aucune indexation en cours » (`wait_for`), et les indexations attendent « pas de benchmark ».
* `asyncio.wrap_future` : attendre un `Future` du pool de processus ou du thread écrivain sans bloquer.
* `loop.call_soon_threadsafe` : le thread du benchmark publie sa progression **dans** l'event loop.
* **État des jobs sans verrou** : le dictionnaire `jobs` n'est modifié que depuis le thread de l'event
  loop ; entre deux `await` une coroutine ne peut pas être interrompue → pas de course possible.
* Le frontend consulte `GET /api/jobs/{id}` toutes les ~1 s (**polling HTTP**) : aucune socket ni
  WebSocket n'est nécessaire.

`asyncio` gère des **attentes** de manière coopérative ; il n'apporte aucun parallélisme multicœur.

## 5. Primitives de synchronisation : chacune pour un vrai besoin

| Primitive | Où | Pourquoi |
|---|---|---|
| `queue.Queue(maxsize)` | `DbWriter` | file producteur–consommateur bornée vers l'écrivain unique |
| `threading.Event` | `DbWriter._closed`, `ResourceSampler` | refuser les écritures à la fermeture ; arrêter le thread de mesure |
| `threading.Lock` | `KeyedLocks`, `ResourceSampler` | protéger un dictionnaire / des mesures réellement partagés entre threads |
| `threading.Timer` | `iter_gray_frames` | watchdog : tuer un FFmpeg qui dépasse son délai |
| `multiprocessing.Event` | pool | arrêt de l'application : abandon des segments en cours |
| `multiprocessing.Barrier` | pool | préchauffage : prouver N processus distincts sans `sleep()` |
| `asyncio.Queue/Semaphore/Condition/Event` | `JobManager` | file bornée, limites, exclusivité CPU, attente de fin de job |

Non utilisés car **inutiles ici** : `RLock` (aucune réentrance), `multiprocessing.Queue/Pipe` (les
futures du `ProcessPoolExecutor` suffisent), mémoire partagée (on ne partage que des résultats compacts),
sockets (application locale).

## 6. Limites du speedup : ce que les mesures montrent

`speedup(N) = T_séquentiel / T_N`, `efficacité(N) = speedup(N) / N`. Rien ne garantit qu'ils soient > 1 :

* **Loi d'Amdahl** : la partie non parallélisée (écriture SQLite séquentielle, démarrage des processus,
  agrégation) borne le gain.
* **Surcoûts** : création des processus (spawn), sérialisation pickle, file inter-processus.
* **Ressources partagées** : disque, cache L3 et surtout **bande passante mémoire**. Décoder de la
  vidéo HD lit/écrit plusieurs Mo par image ; sur un portable à **un seul canal mémoire**, plusieurs
  décodeurs se disputent cette bande passante.
* **Sursouscription** : plus de threads/processus actifs que de cœurs → commutations de contexte.

Les chiffres mesurés sur la machine de développement sont dans `docs/results/RESULTATS.md` (section
suivante pour le résumé).

## 7. Mesures (résumé — détail dans `docs/results/RESULTATS.md`)

Machine : Ryzen 5 5600H (6 cœurs / 12 threads), **8 Go sur un seul canal mémoire**, Windows 11.

| Corpus (345 s de vidéo, 30 segments) | Séquentiel | 2 processus | 4 processus |
|---|---|---|---|
| 1280×720 | 14,59 s | 10,52 s (**×1,39**, eff. 69 %) | 11,02 s (×1,32, eff. 33 %) |
| 480×270 | 3,56 s | 2,00 s (**×1,77**, eff. 89 %) | 1,46 s (**×2,43**, eff. 61 %) |

* Dans **tous** les modes, les empreintes sont strictement identiques à celles du séquentiel, et les PID sont distincts.
* FFmpeg seul, sans Python, montre le même plafond à 720p (×1,11 à 2, ×1,13 à 4 copies) et passe à l'échelle à
  480×270 (×2,61 à 4) : la limite à 720p vient du décodage HD exécuté en plusieurs exemplaires sur cette machine
  (contention mémoire la plus probable), pas du GIL, de l'IPC ni de SQLite.
* Réactivité : `GET /api/jobs/{id}` répond en 2–4 ms pendant une indexation (l'event loop ne calcule jamais).
* Recherche : index par bandes ×2,6 à ×3,0 plus rapide que l'exhaustive entre 50 000 et 500 000 empreintes, avec les
  mêmes verdicts (16/16).
