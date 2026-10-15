"""Общее для шагов пайплайна: данные, индексы, LLM, разбор запроса, состояния пользователей. Всё лениво и один раз:
кандгены, отбор и признаки берут индекс отсюда, поэтому он строится (или читается из кэша) один раз на процесс.
"""
from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from recsys.data.catalog import Catalog
from recsys.data.dataset import Dataset
from recsys.data.history import build_profile
from recsys.data.users import UserStates
from recsys.schemas import Context, Request, UserProfile


class Resources:
    def __init__(self, cfg: Dict[str, Any], dataset: Optional[Dataset] = None, llm=None, verbose: bool = True):
        self.cfg = cfg
        self.dataset = dataset or Dataset(cfg, verbose=verbose)
        self.verbose = verbose
        self._llm = llm
        self._text_indexes: Dict[Tuple[str, str], Any] = {}
        self._profiles: Dict[Tuple[Optional[str], Any], UserProfile] = {}
        self.towers: Dict[str, Any] = {}   # плотные кандгены-башни пайплайна (Pipeline): ими пользуется entity.py

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[recsys] {msg}", flush=True)

    @property
    def catalog(self) -> Catalog:
        return self.dataset.catalog

    # ------------------------------------------------------------ модели и индексы

    @cached_property
    def llm(self):
        """LLM нужна разбору запроса (summarizer.type: llm) и ответу (explainer.type: llm); иначе — заглушка."""
        if self._llm is not None:
            return self._llm
        from recsys.llm import StubLLM, build_llm
        needs = "llm" in (self.cfg.get("summarizer", {}).get("type"), self.cfg.get("explainer", {}).get("type"))
        return build_llm(self.cfg) if needs else StubLLM()

    @cached_property
    def summarizer(self):
        from recsys.dialog import build_summarizer
        scfg = self.cfg.get("summarizer", {})
        resolver = None
        if scfg.get("resolve_entities"):
            from recsys.entity import EntityResolver
            resolver = EntityResolver(self, scfg.get("resolve"))
        return build_summarizer(self.cfg, self.catalog, self.llm, resolver)

    @cached_property
    def users(self) -> UserStates:
        return UserStates(self.catalog)

    @cached_property
    def track_docs(self):
        """Документы треков (candgen/text.py). data.track_docs — готовая таблица (папка инференса: на сервере
        нет запросов train, из которых строится query_document)."""
        from recsys.candgen.text import TrackDocs
        path = self.cfg["data"].get("track_docs")
        return TrackDocs.load(path) if path else TrackDocs.from_dataset(self.dataset)

    def text_index(self, name: str, spec: Optional[Dict[str, Any]] = None):
        """Текстовый индекс (candgen/text.py: bm25, tfidf, ...): кандгены и признаки ранкера пользуются одним."""
        from recsys.candgen.text import load_text_index
        key = (name, repr(sorted((spec or {}).items())))
        if key not in self._text_indexes:
            self._text_indexes[key] = load_text_index(name, spec, self, log=self.log)
        return self._text_indexes[key]

    @cached_property
    def card_index(self):
        from recsys.candgen.card import load_card_index
        c = self.cfg["candgen"].get("card_index", {})
        return load_card_index(self, k1=c.get("k1", 1.2), b=c.get("b", 0.75), log=self.log)

    # ------------------------------------------------------------ запросы

    def profile(self, request: Request) -> UserProfile:
        """Профиль вкуса для разбора запроса (data/history.py); один на пользователя и окно."""
        key = (request.user_id, request.meta.get("cutoff"))
        if request.user_id is None:
            return build_profile(request.history, self.catalog)
        if key not in self._profiles:
            self._profiles[key] = build_profile(request.history, self.catalog)
        return self._profiles[key]

    def context(self, request: Request) -> Context:
        """Запрос -> Context: разбор (правила или LLM), профиль, состояние пользователя, запрещённые треки."""
        profile = self.profile(request)
        summary = self.summarizer.summarize(request, profile)
        return Context.build(request, summary, profile, user=self.users.get(request))

    def contexts(self, requests: List[Request]) -> List[Context]:
        return [self.context(r) for r in requests]
