"""Кандгены по готовым спискам: модель посчитана где-то ещё (другой репозиторий, Colab), сюда приходят файлы
*.parquet в candgen.sources.<имя>.dir с колонками

  query_id   str
  top200     list<str> — m4a_id по убыванию (имя колонки — candgen.sources.<имя>.ids_column)
  scores     list<float>

  twotower_pavel  Qwen history-aware two-tower (ветка codex/qwen-history-candgen): train_top200.parquet,
                  test_public_top200.parquet. Учился на train, поэтому в признаки ранкера не идёт (features: false)

Списки не пересчитываются: новых запросов (другой сплит, реплика диалога) в файлах нет — их должна прогнать
сама модель. Нет списка для больше max_missing запросов части — прогон останавливается с подсказкой.
twotower_pavel умеет и сам: online: true — запроса нет в файлах, его кодирует модель из папки bundle
(экспорт Павла, recsys/twotower/pavel.py); файлов списков тогда может не быть вовсе (онлайн-инференс).
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

from recsys.candgen.base import Candgen, Cands, register
from recsys.schemas import Context


class _Precomputed(Candgen):
    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self.dir = cfg.get("dir") or ""
        self.files = sorted(glob.glob(os.path.join(self.dir, cfg.get("files", "*.parquet")))) if self.dir else []
        if not self.files and not cfg.get("online"):
            raise FileNotFoundError(f"candgen.sources.{self.name}.dir: нет *.parquet в {self.dir}")
        self.ids_column = cfg.get("ids_column", "top200")
        self.tables = [pq.read_table(f, columns=["query_id", self.ids_column, "scores"], memory_map=True)
                       for f in self.files]
        self.row: Dict[str, tuple] = {}   # query_id -> (номер файла, строка)
        for i, t in enumerate(self.tables):
            self.row.update((q, (i, j)) for j, q in enumerate(t.column("query_id").to_pylist()))
        self._pos = res.catalog._pos
        self.n_missing = 0

    def online(self, ctx: Context) -> Optional[Cands]:
        """Список для запроса, которого нет в файлах (online: true); None — не умеет."""
        return None

    def check_coverage(self, contexts: List[Context], part: str) -> None:
        if self.cfg.get("online"):
            return
        ids = [str(c.request.request_id) for c in contexts]
        missing = sum(q not in self.row for q in ids)
        if missing > float(self.cfg.get("max_missing", 0.01)) * len(ids):
            raise ValueError(f"{self.name}: нет списков для {missing} из {len(ids)} запросов части {part} в {self.dir} "
                             f"(их считает сама модель); выключить: --sources -{self.name}")

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        out = []
        for ctx in batch:
            at = self.row.get(str(ctx.request.request_id))
            if at is None and self.cfg.get("online"):
                got = self.online(ctx)
                if got is not None:
                    out.append(got)
                    continue
            if at is None:
                if self.n_missing == 0:
                    warnings.warn(f"{self.name}: нет списка для запроса {ctx.request.request_id}, кандидатов не будет")
                self.n_missing += 1
                out.append(Cands.empty())
                continue
            t = self.tables[at[0]]
            ids = t.column(self.ids_column)[at[1]].as_py()[:self.top_k]
            scores = t.column("scores")[at[1]].as_py()[:self.top_k]
            out.append(self.to_cands(ids, scores))
        return out

    def to_cands(self, ids, scores) -> Cands:
        """m4a_id по убыванию -> позиции каталога: без неизвестных каталогу и без повторов, порядок списка."""
        pos = [(self._pos.get(str(i)), s) for i, s in zip(ids, scores)]
        seen, uniq = set(), []
        for p, s in pos:
            if p is not None and p not in seen:
                seen.add(p)
                uniq.append((p, s))
        return Cands(np.array([p for p, _ in uniq], dtype=np.int32), np.array([s for _, s in uniq], dtype=np.float32))

    def cache_key(self) -> Dict[str, Any]:
        return {**super().cache_key(), "files": {os.path.basename(f): int(os.path.getmtime(f)) for f in self.files}}


@register("twotower_pavel", "dense")
class TwoTowerPavel(_Precomputed):
    """Qwen history-aware two-tower (codex/qwen-history-candgen): готовые top-200 по query_id; online: true — модель
    из папки bundle (device: cuda / cpu, hf_cache — кэш Hugging Face для Qwen)."""

    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self._tower = None

    def warm(self) -> None:
        if self.cfg.get("online") and self._tower is None:
            from recsys.twotower.pavel import PavelTower
            if not self.cfg.get("bundle"):
                raise ValueError(f"candgen.sources.{self.name}.bundle: папка экспорта башни Павла (online: true)")
            self._tower = PavelTower(self.cfg["bundle"], self.cfg.get("device"), self.cfg.get("hf_cache"))

    def online(self, ctx: Context) -> Cands:
        self.warm()
        messages = ctx.request.user_messages
        listens = ctx.request.listens
        if listens is None:   # нет сырых прослушиваний: история без повторов, по последнему прослушиванию
            history = [h.track_id for h in sorted(ctx.request.history, key=lambda h: h.timestamp or 0)]
        else:
            history = list(listens[0])
        ids, scores = self._tower.search(messages[-1] if messages else "", history, self.top_k)
        return self.to_cands(ids, scores)
