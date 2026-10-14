"""Онлайн-инференс: запрос и история пользователя в формате датасета -> top-20 m4a_id и текст ответа.

Весь пайплайн прогона (разбор -> кандгены -> отбор -> признаки -> CatBoost -> ответ) в одном процессе: индексы,
тексты треков, ранкер и модели башен грузятся один раз при старте, запрос — один проход Pipeline.run. Башни
кодируют запрос своими моделями (online: true): на GPU — десятки мс на башню, на CPU — секунды.

    python -m recsys.inference <команда>

  export   собрать папку для сервера из прогона: всё, что нужно инференсу, и ничего от обучения
           python -m recsys.inference export --run lazy3_best --out serving \\
               --model twotower=<model.pt v2> --model twotower_lazy=<model.pt lazy> \\
               --model item_text=<qwen3emb.npz> --model twotower_pavel=<папка экспорта Павла> --with-qwen
  serve    HTTP-сервис: POST /recommend (один запрос или список), GET /health
           python -m recsys.inference serve --bundle serving --port 8000
  predict  файл запросов -> файл ответов (как сабмит: query_id, top20, response)
           python -m recsys.inference predict --bundle serving --queries q.parquet --users u.parquet --out a.parquet
  chat     консоль: своя история (--track «артист название», несколько раз) и запросы с клавиатуры
           python -m recsys.inference chat --bundle serving --history britpop    # истории — examples/histories.yaml
           python -m recsys.inference chat --bundle serving --track "blur song 2" --track "oasis wonderwall"
  check    задержка и совпадение с эталонным сабмитом на n запросах
           python -m recsys.inference check --bundle serving --queries q.parquet --users u.parquet \\
               --reference runs/lazy3_best/submission_test_public.parquet -n 200

  --bundle — папка от export; вместо неё --config runs/<прогон>/config.yaml — прямо из репозитория (данные,
  кэш и векторы башен на своих местах; так сверяется инференс с сабмитом прогона).

Вход одного запроса — строки датасета (docs/dataset.md), лишние поля не мешают:
  {"query": {"query_id": "78ff70df6ca7c6df", "query": "90s british pop"},
   "user":  {"user_id": 35511, "history": ["tkLb1MuW7uxdZ9CW", ...], "history_ts": [1247864903, ...],
             "user_profile": "Listening since 2009; ...",
             "user_demographics": {"age": 21, "gender": "m", "country": "UK"}, "split": "test_public"}}
Выход: {"query_id": ..., "top20": [m4a_id, ...], "response": "I recommend ...", "timings": {шаг: секунды}}

history — все прослушивания по порядку, с повторами (как в train / test_public.parquet). Граница истории (cutoff,
от неё считается давность): user.split из data.windows (test_public, test_private — 2020-02-20, как при обучении),
иначе cutoff (unix) в query или user, иначе сразу после последнего прослушивания (живой пользователь).
query_type не нужен и не читается.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from recsys.config import config_path, git_commit, load_config, set_key
from recsys.data.crs import make_request, split_path
from recsys.pipeline import Pipeline
from recsys.resources import Resources
from recsys.schemas import Request

CONFIG = "config.yaml"
MANIFEST = "manifest.json"
QWEN = "models--Qwen--Qwen3-Embedding-0.6B"   # база всех трёх башен (у Павла — закреплённая ревизия)
PATHS = [("data", "dir"), ("data", "cache_dir"), ("data", "track_docs"), ("ranker", "model_dir"), ("semantic_ids", "dir")]
SOURCE_PATHS = ("dir", "item_text", "bundle", "hf_cache")   # пути в candgen.sources.<имя>


def resolve_paths(cfg: Dict[str, Any], root: str) -> Dict[str, Any]:
    """Относительные пути конфига папки инференса -> пути внутри root."""
    cfg = copy.deepcopy(cfg)

    def fix(d: Dict[str, Any], key: str) -> None:
        if d.get(key) and not os.path.isabs(str(d[key])):
            d[key] = os.path.join(root, str(d[key]))

    for section, key in PATHS:
        fix(cfg.get(section) or {}, key)
    for c in cfg["candgen"]["sources"].values():
        for key in SOURCE_PATHS:
            fix(c, key)
    return cfg


def use_bundle_hf(root: str) -> None:
    """Веса Qwen из папки инференса (export --with-qwen), если HF_HOME не задан: без интернета на сервере."""
    hf = os.path.join(root, "hf")
    if os.path.isdir(hf) and "HF_HOME" not in os.environ:
        if "huggingface_hub" in sys.modules:
            print("[inference] huggingface_hub уже загружен: HF_HOME из папки не подхватится", file=sys.stderr)
        os.environ["HF_HOME"] = hf
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------- инференс

class Recommender:
    """Пайплайн, загруженный один раз. recommend(query, user) — один запрос; потокобезопасно (по одному за раз)."""

    def __init__(self, cfg: Dict[str, Any], verbose: bool = True, warm: bool = True):
        t0 = time.time()
        self.cfg = cfg
        self.res = Resources(cfg, verbose=verbose)
        self.pipe = Pipeline(cfg, self.res)
        self.k = int(cfg["eval"].get("k", 20))
        if self.pipe.top_k < self.k:
            raise ValueError(f"ranker.top_k = {self.pipe.top_k} < {self.k}")
        self.windows = {s: self.res.dataset.cutoff(s) for s in (cfg["data"].get("windows") or {})}
        self.lock = threading.Lock()
        if warm:
            self.warm()
        self.res.log(f"инференс готов за {time.time() - t0:.0f} c; кандгены: {', '.join(self.pipe.sources)}")

    @classmethod
    def from_bundle(cls, root: str, overrides: Optional[Dict[str, Any]] = None, **kwargs) -> "Recommender":
        use_bundle_hf(root)
        cfg = load_config(os.path.join(root, CONFIG), overrides)
        return cls(resolve_paths(cfg, root), **kwargs)

    def warm(self) -> None:
        """Всё ленивое — сейчас, а не на первом запросе: тексты треков и индексы признаков, модели башен, разбор;
        затем пробный запрос (индексы кандгенов, ранкер, ответ)."""
        t0 = time.time()
        self.pipe.features.warm()
        for cg in self.pipe.candgens:
            cg.warm()
        tracks = list(self.res.catalog.track_ids[:3])
        with warnings.catch_warnings():   # без векторов в файлах (online: false) пробный запрос даст пустые списки
            warnings.simplefilter("ignore")
            self.recommend({"query": "calm acoustic songs"}, {"history": tracks, "history_ts": [1, 2, 3]})
        for cg in self.pipe.candgens:
            if hasattr(cg, "n_missing"):
                cg.n_missing = 0
        self.res.log(f"прогрев за {time.time() - t0:.0f} c")

    def cutoff(self, query: Dict[str, Any], user: Dict[str, Any]) -> Optional[int]:
        split = user.get("split") or query.get("split")
        if split in self.windows and self.windows[split] is not None:
            return self.windows[split]
        for src in (query, user):
            if src.get("cutoff") is not None:
                return int(src["cutoff"])
        return None

    def request(self, query: Dict[str, Any], user: Dict[str, Any]) -> Request:
        req = make_request(query, user, self.cutoff(query, user))
        # кэши по user_id (состояние пользователя, списки истории) в долгоживущем процессе только растут и
        # устаревают с новой историей: без user_id всё считается по запросу (это миллисекунды)
        req.meta["user_id"], req.user_id = req.user_id, None
        return req

    def recommend(self, query: Any, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """query — строка запроса или строка *_queries (query_id, query); user — строка сплита (history, history_ts,
        user_profile, user_demographics, split). -> {query_id, top20, response, timings}."""
        query = {"query": query} if isinstance(query, str) else dict(query)
        t0 = time.perf_counter()
        req = self.request(query, dict(user or {}))
        with self.lock:
            resp = self.pipe.run(req)
        timings = {k: round(v, 4) for k, v in resp.debug["timings"].items()}
        timings["total"] = round(time.perf_counter() - t0, 4)
        return {"query_id": req.request_id, f"top{self.k}": resp.track_ids[:self.k], "response": resp.text,
                "timings": timings}

    def predict(self, queries: pd.DataFrame, users: pd.DataFrame, log_every: int = 500) -> pd.DataFrame:
        """Запросы (строки *_queries) и их пользователи (строки сплита) -> query_id, top20, response."""
        by_user = {u: row for u, row in zip(users["user_id"], users.to_dict("records"))}
        rows, t0 = [], time.time()
        for i, q in enumerate(queries.to_dict("records"), start=1):
            if q["user_id"] not in by_user:
                raise KeyError(f"нет пользователя {q['user_id']} для запроса {q.get('query_id')}")
            out = self.recommend(q, by_user[q["user_id"]])
            rows.append({k: v for k, v in out.items() if k != "timings"})
            if i % log_every == 0 or i == len(queries):
                self.res.log(f"{i}/{len(queries)} запросов, {(time.time() - t0) / i * 1000:.0f} мс на запрос")
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- входы и выходы

def read_table(path: str, filter_users: Optional[List[Any]] = None) -> pd.DataFrame:
    """parquet | jsonl | json; filter_users — только строки этих user_id (история сплита — сотни МБ)."""
    if path.endswith(".parquet"):
        if filter_users is None:
            return pq.read_table(path).to_pandas()
        kind = pq.read_schema(path).field("user_id").type
        return pq.read_table(path, filters=pc.is_in(pc.field("user_id"), value_set=pa.array(
            sorted(set(filter_users)), type=kind))).to_pandas()
    df = pd.read_json(path, lines=path.endswith(".jsonl"))
    return df[df["user_id"].isin(set(filter_users))] if filter_users is not None else df


def write_table(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", lines=path.endswith(".jsonl"), force_ascii=False)


def make_server(rec: Recommender, host: str, port: int):
    """HTTP без зависимостей: POST /recommend {"query": {...}, "user": {...}} (или список таких) -> ответ (или
    список); GET /health. Запросы выполняются по одному (Recommender.lock)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "candgens": rec.pipe.sources})
            else:
                self._send(404, {"error": "GET /health, POST /recommend"})

        def do_POST(self) -> None:
            if self.path != "/recommend":
                return self._send(404, {"error": "POST /recommend"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                items = body if isinstance(body, list) else [body]
                out = [rec.recommend(x["query"], x.get("user")) for x in items]
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception as e:   # ошибка пайплайна: сервис живёт дальше
                rec.res.log(f"ошибка: {type(e).__name__}: {e}")
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, out if isinstance(body, list) else out[0])

        def log_message(self, fmt: str, *args) -> None:
            rec.res.log(f"{self.address_string()} {fmt % args}")

    return ThreadingHTTPServer((host, port), Handler)


def serve(rec: Recommender, host: str, port: int) -> None:
    server = make_server(rec, host, port)
    rec.res.log(f"слушаю http://{host}:{port} (POST /recommend, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def check(rec: Recommender, queries: pd.DataFrame, users: pd.DataFrame, reference: Optional[str] = None
          ) -> Dict[str, Any]:
    """Задержка по шагам и совпадение с эталоном (сабмит прогона: query_id, top20) на запросах queries."""
    by_user = {u: row for u, row in zip(users["user_id"], users.to_dict("records"))}
    ref = None
    if reference:
        r = pd.read_parquet(reference)
        ref = dict(zip(r["query_id"], zip(r[f"top{rec.k}"], r["response"] if "response" in r else [None] * len(r))))
    times: List[Dict[str, float]] = []
    same, top1, overlap, text, compared = 0, 0, 0.0, 0, 0
    for q in queries.to_dict("records"):
        out = rec.recommend(q, by_user[q["user_id"]])
        times.append(out["timings"])
        got = out[f"top{rec.k}"]
        if ref is not None and q["query_id"] in ref:
            want, response = list(ref[q["query_id"]][0]), ref[q["query_id"]][1]
            compared += 1
            same += got == want
            top1 += bool(got) and bool(want) and got[0] == want[0]
            overlap += len(set(got) & set(want)) / max(len(want), 1)
            text += out["response"] == response
    t = pd.DataFrame(times)
    out = {"queries": len(queries),
           "latency_ms": {c: {"p50": round(t[c].median() * 1000, 1), "p95": round(t[c].quantile(0.95) * 1000, 1)}
                          for c in t.columns}}
    if compared:
        out["reference"] = {"compared": compared, "same_top20": round(same / compared, 4),
                            "same_top1": round(top1 / compared, 4), "overlap@20": round(overlap / compared, 4),
                            "same_response": round(text / compared, 4)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def find_track(rec: Recommender, text: str) -> Optional[str]:
    """«артист — название» (или часть) -> m4a_id: все слова есть в «артист название», из таких — самый популярный."""
    from recsys.data.parts import normalize_query
    words = normalize_query(text).split()
    if not words:
        return None
    cat = rec.res.catalog
    if "_names" not in rec.__dict__:
        rec._names = np.array([f" {normalize_query(a)} {normalize_query(t)} " for a, t in
                               zip(cat.text("artist"), cat.df["title"])], dtype=object)
    hit = np.ones(len(cat), dtype=bool)
    for w in words:
        hit &= np.array([f" {w} " in n for n in rec._names])
    if not hit.any():
        return None
    pop = np.nan_to_num(cat.numeric("spotify_popularity"), nan=-1.0)
    return str(cat.track_ids[np.flatnonzero(hit)[np.argmax(pop[hit])]])


def chat(rec: Recommender, tracks: List[str], top: int = 10) -> None:
    """Консоль: история — найденные треки (по прослушиванию в день, последнее — вчера), дальше — запросы."""
    cat = rec.res.catalog
    name = lambda t: f"{cat.artist(t)} — {cat.df.at[cat.pos(t), 'title']}"
    history = []
    for text in tracks:
        t = text if text in cat else find_track(rec, text)
        print(f"  история: {text!r} -> {name(t) if t else 'не найден'}")
        if t:
            history.append(t)
    now = int(time.time())
    user = {"history": history, "history_ts": [now - (len(history) - i) * 86400 for i in range(len(history))]}
    print(f"\nИстория: {len(history)} треков. Пиши запрос (пустая строка или Ctrl+D — выход).")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query:
            break
        out = rec.recommend(query, user)
        for i, t in enumerate(out[f"top{rec.k}"][:top], 1):
            print(f"  {i:2d}. {name(t)}{'   [из истории]' if t in history else ''}")
        print(f"  ответ: {out['response']}")
        print(f"  время: {out['timings']['total']:.2f} с (кандгены {out['timings']['2_candgen']:.2f} с)")


# ---------------------------------------------------------------- папка для сервера

def _copy(src: str, dst: str) -> int:
    """Файл или папка; уже скопированное (тот же размер) не копируется заново. -> байт."""
    if os.path.isdir(src):
        return sum(_copy(os.path.join(src, f), os.path.join(dst, f)) for f in sorted(os.listdir(src)))
    size = os.path.getsize(src)
    if not (os.path.exists(dst) and os.path.getsize(dst) == size):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return size


def export(cfg: Dict[str, Any], out: str, models: Dict[str, str], lookup: bool = False,
           with_qwen: bool = False, log=print) -> List[str]:
    """Папка инференса из конфига прогона (runs/<прогон>/config.yaml): данные треков, индексы, документы треков,
    ранкер, файлы кандгенов, config.yaml с путями внутри папки. models — файлы моделей не на своих местах:
    twotower, twotower_lazy (model.pt), item_text (векторы названий lazy), twotower_pavel (папка экспорта Павла).
    lookup — ещё готовые векторы / списки запросов датасета (сверка с сабмитом без моделей; башни — online: false).
    -> чего не хватает (пусто — папка готова)."""
    from recsys.candgen import REGISTRY, enabled_sources
    from recsys.candgen.text import TEXT_INDEXES, index_key
    from recsys.ranker import model_dir
    from recsys.semantic_ids import semantic_paths

    res = Resources(cfg)
    dcfg = cfg["data"]
    sources = enabled_sources(cfg)
    new = copy.deepcopy(cfg)
    missing: List[str] = []
    sizes: Dict[str, int] = {}

    def put(src: str, rel: str, what: str) -> bool:
        if not src or not os.path.exists(src):
            missing.append(f"{what}: нет {src or '(путь не задан)'}")
            return False
        sizes[rel] = _copy(src, os.path.join(out, rel))
        log(f"{rel}: {sizes[rel] / 2 ** 20:.0f} МБ")
        return True

    # треки и каталог: tracks_meta целиком — тогда ключи кэшей (md5 файла) те же, что при обучении
    put(split_path(dcfg["dir"], "tracks_meta"), "data/tracks_meta.parquet", "tracks_meta")
    _ = res.catalog
    for f in glob.glob(os.path.join(dcfg["cache_dir"], f"catalog_{res.dataset.source_key}_{dcfg.get('max_tags', 20)}.parquet*")):
        put(f, os.path.join("cache", os.path.basename(f)), "каталог")
    # текстовые индексы кандгенов и признаков (bm25, tfidf) — собранные на части index
    for name in sorted(set(sources) & set(TEXT_INDEXES) | {"bm25", "tfidf"}):
        spec = cfg["candgen"]["sources"].get(name, {}).get("index")
        res.text_index(name, spec)
        key = index_key(name, {**TEXT_INDEXES[name], **(spec or {})}, res.dataset.source_key, dcfg["parts"])
        put(os.path.join(dcfg["cache_dir"], "indexes", f"{name}_{key}"), f"cache/indexes/{name}_{key}", f"индекс {name}")
    os.makedirs(out, exist_ok=True)
    res.track_docs.save(os.path.join(out, "track_docs.parquet"))
    sizes["track_docs.parquet"] = os.path.getsize(os.path.join(out, "track_docs.parquet"))
    log(f"track_docs.parquet: {sizes['track_docs.parquet'] / 2 ** 20:.0f} МБ (документы треков с query_document)")
    rdir = model_dir(cfg)
    for f in ("model.cbm", "features.json", "importance.csv"):
        put(os.path.join(rdir, f), f"ranker/{f}", "ранкер")
    new["data"].update(dir="data", cache_dir="cache", track_docs="track_docs.parquet")
    new["ranker"]["model_dir"] = "ranker"
    new["features"]["workers"] = 1

    # плотные кандгены: векторы треков и модели для запросов
    srcs = new["candgen"]["sources"]
    if {"twotower", "twotower_lazy"} & set(sources):
        p = semantic_paths(cfg)
        put(p["codes"], f"semantic_ids/{os.path.basename(p['codes'])}", "коды semantic ID")
        put(p["checkpoint"], f"semantic_ids/{os.path.basename(p['checkpoint'])}", "чекпоинт RQ-VAE")
        new["semantic_ids"]["dir"] = "semantic_ids"
    for name in sources:
        if REGISTRY[name].kind != "dense":
            continue
        c, rel = cfg["candgen"]["sources"][name], f"towers/{name}"
        if name == "twotower_pavel":
            srcs[name].update(online=not lookup, bundle=f"{rel}/model", dir=f"{rel}/lists" if lookup else "")
            if lookup:
                for f in sorted(glob.glob(os.path.join(c["dir"], "*.parquet"))):
                    put(f, f"{rel}/lists/{os.path.basename(f)}", "списки башни Павла")
            else:
                put(models.get(name, os.path.join(c["dir"], "bundle")), f"{rel}/model",
                    "экспорт башни Павла (config.json, tokenizer/, query_trainable.pt, history_features.npz, "
                    "catalog.json, track_vectors.npy) — --model twotower_pavel=<папка>")
            continue
        for f in ("item_vectors.npy", "item_ids.npy"):
            put(os.path.join(c["dir"], f), f"{rel}/{f}", f"векторы треков {name}")
        srcs[name].update(dir=rel, online=not lookup)
        if lookup:
            for f in sorted(glob.glob(os.path.join(c["dir"], c.get("query_files", "query_vectors_*.npz")))):
                put(f, f"{rel}/{os.path.basename(f)}", f"векторы запросов {name}")
            continue
        if name in ("twotower", "twotower_lazy"):
            ck = c.get("checkpoint", "model.pt")
            put(models.get(name, os.path.join(c["dir"], ck)), f"{rel}/{ck}", f"модель {name} — --model {name}=<model.pt>")
        if name == "twotower_lazy":
            put(models.get("item_text", "artifacts/item_text/qwen3emb.npz"), "towers/item_text/qwen3emb.npz",
                "векторы названий треков lazy (scripts/encode_item_text.py) — --model item_text=<qwen3emb.npz>")
            srcs[name]["item_text"] = "towers/item_text/qwen3emb.npz"
        if name == "hnsw":
            log("hnsw online: EmbeddingGemma-2 грузится с Hugging Face (нужен HF_TOKEN), в папку не кладётся")
    if with_qwen:
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", QWEN)
        if os.path.isdir(hub):   # снимки — ссылки на blobs: копируем как есть
            dst = os.path.join(out, "hf", "hub", QWEN)
            if not os.path.exists(dst):
                shutil.copytree(hub, dst, symlinks=True)
            log(f"hf/hub/{QWEN}: веса Qwen")
        else:
            missing.append(f"веса Qwen: нет {hub}")

    with open(os.path.join(out, CONFIG), "w", encoding="utf8") as f:
        f.write(f"# Конфиг инференса (recsys/inference.py) из прогона {cfg['run'].get('name')}; пути — внутри папки\n")
        yaml.safe_dump(new, f, allow_unicode=True, sort_keys=False)
    manifest = {"run": cfg["run"].get("name"), "commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_key": res.dataset.source_key, "candgens": sources, "lookup": lookup,
                "files_mb": {k: round(v / 2 ** 20, 1) for k, v in sorted(sizes.items())}, "missing": missing}
    with open(os.path.join(out, MANIFEST), "w", encoding="utf8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    total = sum(sizes.values()) / 2 ** 30
    if missing:
        log("НЕ ХВАТАЕТ (положите и запустите export ещё раз — готовое не копируется заново):\n  " + "\n  ".join(missing))
    log(f"{out}: {total:.1f} ГБ (без Qwen), {'не готова' if missing else 'готова'}")
    return missing


# ---------------------------------------------------------------- командная строка

def _overrides(items: List[str]) -> Dict[str, Any]:
    over: Dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item}: нужно KEY=VALUE")
        set_key(over, key, yaml.safe_load(value))
    return over


def _load(a: argparse.Namespace) -> Recommender:
    over = _overrides(a.set)
    if a.bundle:
        return Recommender.from_bundle(a.bundle, over)
    if not a.config:
        raise SystemExit("нужно --bundle <папка инференса> или --config runs/<прогон>/config.yaml")
    return Recommender(load_config(a.config, over))


def _inputs(a: argparse.Namespace, sample: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Запросы (первые --limit или sample случайных) и строки только их пользователей."""
    queries = read_table(a.queries)
    if a.limit:
        queries = queries.head(a.limit)
    if sample is not None and sample < len(queries):
        queries = queries.sample(n=sample, random_state=0).reset_index(drop=True)
    return queries, read_table(a.users, filter_users=list(queries["user_id"].unique()))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m recsys.inference", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("command", choices=("export", "serve", "predict", "check", "chat"))
    p.add_argument("--bundle", help="папка инференса (export --out)")
    p.add_argument("--config", help="вместо --bundle: конфиг прогона, всё — из репозитория")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="параметр конфига")
    # export
    p.add_argument("--run", help="export: прогон runs/<имя> (его config.yaml и ранкер)")
    p.add_argument("--out", help="export: папка инференса; predict: файл ответов (.parquet | .jsonl | .json)")
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="export: twotower, twotower_lazy, item_text, twotower_pavel — где лежат модели")
    p.add_argument("--lookup", action="store_true", help="export: готовые векторы запросов датасета вместо моделей")
    p.add_argument("--with-qwen", action="store_true", help="export: веса Qwen3-Embedding-0.6B в папку (hf/)")
    # serve
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    # predict, check
    p.add_argument("--queries", help="запросы: *_queries.parquet (query_id, user_id, query) | .jsonl")
    p.add_argument("--users", help="пользователи: train / test_public.parquet (user_id, history, history_ts, ...)")
    p.add_argument("--limit", type=int, help="первые N запросов")
    p.add_argument("--reference", help="check: эталонный сабмит (query_id, top20)")
    p.add_argument("-n", type=int, default=200, help="check: сколько запросов")
    p.add_argument("--history", help="chat: имя истории из --histories (examples/histories.yaml)")
    p.add_argument("--histories", default="examples/histories.yaml", help="chat: файл историй")
    p.add_argument("--track", action="append", default=[],
                   help="chat: трек истории — «артист название» (часть слов) или m4a_id; можно несколько")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if a.command == "export":
        if not (a.run and a.out):
            raise SystemExit("export: нужны --run и --out")
        cfg_file = os.path.join("runs", a.run, "config.yaml")
        cfg = load_config(cfg_file if os.path.exists(cfg_file) else config_path("default.yaml"), _overrides(a.set))
        models = dict(m.split("=", 1) for m in a.model)
        return 1 if export(cfg, a.out, models, lookup=a.lookup, with_qwen=a.with_qwen) else 0
    tracks = a.track
    if a.command == "chat" and a.history:
        with open(a.histories, encoding="utf8") as f:
            histories = yaml.safe_load(f)
        if a.history not in histories:
            raise SystemExit(f"нет истории {a.history!r} в {a.histories}; есть: {', '.join(histories)}")
        tracks = list(histories[a.history]) + tracks
    rec = _load(a)
    if a.command == "serve":
        serve(rec, a.host, a.port)
    elif a.command == "predict":
        if not (a.queries and a.users and a.out):
            raise SystemExit("predict: нужны --queries, --users, --out")
        queries, users = _inputs(a)
        write_table(rec.predict(queries, users), a.out)
        rec.res.log(f"ответы: {len(queries)} -> {a.out}")
    elif a.command == "chat":
        chat(rec, tracks)
    elif a.command == "check":
        if not (a.queries and a.users):
            raise SystemExit("check: нужны --queries, --users")
        queries, users = _inputs(a, sample=a.n)
        check(rec, queries, users, a.reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
