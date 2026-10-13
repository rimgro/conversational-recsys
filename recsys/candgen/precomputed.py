"""Кандгены по готовым спискам: модель посчитана где-то ещё (другой репозиторий, Colab), сюда приходят файлы
*.parquet в candgen.sources.<имя>.dir с колонками

  query_id   str
  top200     list<str> — m4a_id по убыванию (имя колонки — candgen.sources.<имя>.ids_column)
  scores     list<float>

  twotower_pavel  Qwen history-aware two-tower (ветка codex/qwen-history-candgen): train_top200.parquet,
                  test_public_top200.parquet. Учился на train, поэтому в признаки ранкера не идёт (features: false)

Списки не пересчитываются: новых запросов (другой сплит, реплика диалога) в файлах нет — их должна прогнать
сама модель. Нет списка для больше max_missing запросов части — прогон останавливается с подсказкой.
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq

from recsys.candgen.base import Candgen, Cands, register
from recsys.schemas import Context


class _Precomputed(Candgen):
    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self.dir = cfg.get("dir") or ""
        self.files = sorted(glob.glob(os.path.join(self.dir, cfg.get("files", "*.parquet"))))
        if not self.files:
            raise FileNotFoundError(f"candgen.sources.{self.name}.dir: нет *.parquet в {self.dir}")
        self.ids_column = cfg.get("ids_column", "top200")
        self.tables = [pq.read_table(f, columns=["query_id", self.ids_column, "scores"], memory_map=True)
                       for f in self.files]
        self.row: Dict[str, tuple] = {}   # query_id -> (номер файла, строка)
        for i, t in enumerate(self.tables):
            self.row.update((q, (i, j)) for j, q in enumerate(t.column("query_id").to_pylist()))
        self._pos = res.catalog._pos
        self.n_missing = 0

    def check_coverage(self, contexts: List[Context], part: str) -> None:
        ids = [str(c.request.request_id) for c in contexts]
        missing = sum(q not in self.row for q in ids)
        if missing > float(self.cfg.get("max_missing", 0.01)) * len(ids):
            raise ValueError(f"{self.name}: нет списков для {missing} из {len(ids)} запросов части {part} в {self.dir} "
                             f"(их считает сама модель); выключить: --sources -{self.name}")

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        out = []
        for ctx in batch:
            at = self.row.get(str(ctx.request.request_id))
            if at is None:
                if self.n_missing == 0:
                    warnings.warn(f"{self.name}: нет списка для запроса {ctx.request.request_id}, кандидатов не будет")
                self.n_missing += 1
                out.append(Cands.empty())
                continue
            t = self.tables[at[0]]
            ids = t.column(self.ids_column)[at[1]].as_py()[:self.top_k]
            scores = t.column("scores")[at[1]].as_py()[:self.top_k]
            pos = [(self._pos.get(str(i)), s) for i, s in zip(ids, scores)]
            pos = [(p, s) for p, s in pos if p is not None]
            seen, uniq = set(), []
            for p, s in pos:                      # без повторов, порядок файла
                if p not in seen:
                    seen.add(p)
                    uniq.append((p, s))
            out.append(Cands(np.array([p for p, _ in uniq], dtype=np.int32),
                             np.array([s for _, s in uniq], dtype=np.float32)))
        return out

    def cache_key(self) -> Dict[str, Any]:
        return {**super().cache_key(), "files": {os.path.basename(f): int(os.path.getmtime(f)) for f in self.files}}


@register("twotower_pavel", "dense")
class TwoTowerPavel(_Precomputed):
    """Qwen history-aware two-tower (codex/qwen-history-candgen): готовые top-200 по query_id."""
