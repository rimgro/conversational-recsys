"""Шаг 5: top-k треков + контекст -> текстовый ответ пользователю.

TemplateExplainer  ответ про первый трек по метаданным и истории (как сабмит team1): до 120 слов, без LLM
StubExplainer      приветствие + список треков с причинами
LLMExplainer       LLM пишет ответ только по переданным фактам; ошибка -> шаблон
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from recsys.data.parts import normalize_query
from recsys.llm import BaseLLM
from recsys.schemas import Context, RankedTrack
from recsys.text import tag_matches


class BaseExplainer(ABC):
    """Шаг 5: top-k треков + контекст -> текстовый ответ пользователю."""

    @abstractmethod
    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        ...


def reasons_for(track_id: str, ctx: Context, catalog, max_reasons: int = 3) -> List[str]:
    """Короткие причины для объяснения (по-английски): совпавшие теги, артист, история."""
    tags = catalog.tags(track_id) + catalog.genres(track_id)
    artist = catalog.artist(track_id)
    matched = [t for t in ctx.summary.include_tags if tag_matches(t, tags)]
    out = [f"matches: {', '.join(matched)}"] if matched else []
    if artist and artist.lower() in {a.lower() for a in ctx.summary.seed_artists}:
        out.append(f"{artist}, as you asked")
    elif any(h.track_id == track_id for h in ctx.request.history):
        out.append("you have listened to it before")
    elif artist and artist in ctx.profile.artist_weights:
        out.append(f"you listen to {artist}")
    if not out:
        common = [t for t in ctx.profile.top_tags(10) if t in tags][:2]
        if common:
            out.append("close to your taste: " + ", ".join(common))
    return out[:max_reasons]


def _clean(value: Any, limit: int = 12) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split()[:limit])


def _split(value: Any) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip() and x.strip().lower() not in {"nan", "none"}]


def russian_request(query: str) -> bool:
    """Запрос по-русски (кириллическая цитата внутри английского запроса не считается)."""
    return bool(re.search("[а-яА-Я]", query)) and not re.search(
        r"\b(find|song|music|track|play|with|by|that|like|lyrics|rap)\b", query, flags=re.I)


class TemplateExplainer(BaseExplainer):
    """Ответ про первый трек: жанры из метаданных, теги, совпавшие с запросом, знаком ли артист по истории.
    Все утверждения о музыке — из метаданных, о вкусе — из истории; до 120 слов. Шаблон сабмита team1."""

    def __init__(self, res):
        self.res = res

    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        if not tracks:
            return StubExplainer().describe(tracks, ctx)
        res = self.res
        p = res.catalog.pos(tracks[0].track_id)
        meta = res.track_docs.meta.iloc[p]
        artists = res.users.artists
        title, artist = _clean(meta["m4a_song"]), _clean(meta["m4a_artist"])
        genres = list(dict.fromkeys(_split(meta["m4a_genres_full"])))[:3]
        tags = list(dict.fromkeys(genres + _split(meta["m4a_tags_full"])))
        query = ctx.request.user_messages[-1] if ctx.request.user_messages else ""
        query_text = f" {normalize_query(query)} "
        matching = [t for t in tags if f" {normalize_query(t)} " in query_text][:3]
        title_in_query = bool(normalize_query(meta["m4a_song"])) and f" {normalize_query(meta['m4a_song'])} " in query_text
        user = ctx.user
        familiar = user is not None and bool(user.artist_counts.get(artists[p], 0))
        history_artist = ""
        if user is not None and user.artist_counts:
            top = user.artist_counts.most_common(1)[0][0]
            hist = next((int(i) for i in user.positions if artists[i] == top), None)
            if hist is not None:
                history_artist = _clean(res.track_docs.meta.iloc[hist]["m4a_artist"])
        if russian_request(query):
            text = f"Предлагаю «{title}» — {artist}."
            if genres:
                text += " В метаданных указаны жанры: " + ", ".join(genres) + "."
            if matching:
                text += " С запросом совпадают теги: " + ", ".join(matching) + "."
            elif title_in_query:
                text += " Название трека встречается в вашем запросе."
            else:
                text += " Выбран с учётом текста запроса и вашей истории прослушиваний."
            if familiar:
                text += " Вы уже слушали этого исполнителя; это учитывает ваш музыкальный вкус."
            elif history_artist:
                text += f" Среди исполнителей в вашей истории — {history_artist}; эта история также учтена при выборе."
        else:
            text = f"I recommend “{title}” by {artist}."
            if genres:
                text += " Its metadata lists " + ", ".join(genres) + "."
            if matching:
                text += " It shares these tags with your request: " + ", ".join(matching) + "."
            elif title_in_query:
                text += " The track title appears in your request."
            else:
                text += " I selected it using your request and listening history."
            if familiar:
                text += " You have listened to this artist before, so the choice reflects that part of your taste."
            elif history_artist:
                text += f" Your history includes {history_artist}; that listening context also informed this choice."
        return " ".join(text.split()[:120])


class StubExplainer(BaseExplainer):
    def __init__(self, n_describe: int = 10):
        self.n_describe = n_describe

    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        if not tracks:
            return "Hi! Sorry, I could not find anything. Try rephrasing the request."
        greeting = "Hi! Here is what I picked"
        if ctx.summary.include_tags or ctx.summary.seed_artists:
            greeting += " (" + ", ".join(ctx.summary.include_tags + ctx.summary.seed_artists) + ")"
        lines = [greeting + ":"]
        for t in tracks[:self.n_describe]:
            tags = ", ".join(t.tags[:3])
            why = f" — {'; '.join(t.reasons)}" if t.reasons else ""
            lines.append(f"{t.rank}. {t.display_name} [{tags}]{why}")
        return "\n".join(lines)


EXPLAIN_SYSTEM = """You are a friendly music assistant in a chat.
Write a short reply in English (3-6 sentences) presenting the recommended tracks to the user.
Mention a few tracks by artist and title and say briefly why they fit the request.
Use ONLY the facts given below (artist, title, tags, reasons). Do not invent tracks, facts or links.
Do not reorder or drop tracks from the list: the full list is shown to the user separately."""


class LLMExplainer(BaseExplainer):
    def __init__(self, llm: BaseLLM, n_describe: int = 5, max_new_tokens: int = 300):
        self.llm = llm
        self.n_describe = n_describe
        self.max_new_tokens = max_new_tokens
        self.fallback = StubExplainer()
        self.last_raw = ""

    def build_messages(self, tracks: List[RankedTrack], ctx: Context) -> List[Dict[str, str]]:
        last = ctx.request.user_messages[-1] if ctx.request.user_messages else ""
        items = "\n".join(
            f"{t.rank}. {t.display_name} | tags: {', '.join(t.tags[:5])} | reasons: {'; '.join(t.reasons) or '-'}"
            for t in tracks[:self.n_describe]
        )
        user = (f"User's last message: {last}\nRequest summary: {ctx.summary.text}\n\n"
                f"Recommended tracks:\n{items}")
        return [{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}]

    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        if not tracks:
            return self.fallback.describe(tracks, ctx)
        try:
            self.last_raw = self.llm.generate(self.build_messages(tracks, ctx), max_new_tokens=self.max_new_tokens)
        except Exception as e:
            self.last_raw = f"<error: {e}>"
            return self.fallback.describe(tracks, ctx)
        return self.last_raw.strip() or self.fallback.describe(tracks, ctx)


def build_explainer(cfg: Dict[str, Any], llm: Optional[BaseLLM] = None, res=None) -> BaseExplainer:
    ecfg = cfg.get("explainer", {})
    kind = ecfg.get("type", "template")
    if kind == "template":
        if res is None:
            raise ValueError("explainer.type=template, но не переданы ресурсы (recsys.resources.Resources)")
        return TemplateExplainer(res)
    if kind == "stub":
        return StubExplainer(n_describe=ecfg.get("n_describe", 10))
    if kind == "llm":
        if llm is None:
            raise ValueError("explainer.type=llm, но llm не передана")
        return LLMExplainer(llm, n_describe=ecfg.get("n_describe", 5),
                            max_new_tokens=ecfg.get("max_new_tokens", 300))
    raise ValueError(f"Неизвестный explainer.type: {kind}")
