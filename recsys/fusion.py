"""Шаг 3: слияние списков кандидатов (RRF), жёсткие фильтры, ограничения запроса, дедупликация.

RRF: score(d) = sum_s w_s / (k + rank_s(d)); веса источников зависят от контекста запроса.
Ограничения (год, тональность, bpm, энергия, instrumental, пол вокала, язык; fusion.constraints): за каждое нарушение
из RRF вычитается penalty — нарушители уходят вниз списка, но не пропадают (hard: true — выкидываются).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from recsys.data.catalog import Catalog
from recsys.schemas import Candidate, Context, FusedCandidate
from recsys.text import tag_matches


def rrf(lists: Dict[str, List[Candidate]], k: int = 60,
        weights: Optional[Dict[str, float]] = None) -> List[FusedCandidate]:
    weights = weights or {}
    fused: Dict[str, FusedCandidate] = {}
    for source, cands in lists.items():
        w = float(weights.get(source, 1.0))
        if w <= 0:
            continue
        for c in cands:
            fc = fused.get(c.track_id)
            if fc is None:
                fc = fused[c.track_id] = FusedCandidate(track_id=c.track_id, score=0.0)
            if source in fc.ranks:  # дубль внутри одного источника: берём лучший ранг
                continue
            fc.ranks[source] = c.rank
            fc.scores[source] = c.score
            fc.score += w / (k + c.rank)
    return sorted(fused.values(), key=lambda f: -f.score)


class Filters:
    """Жёсткие фильтры запроса; reason(track_id) -> почему трек выкинут (None — проходит).

    banned          ctx.banned_ids: показанное, скипнутое, прослушанное для similar_to / novelty
    unknown         трека нет в каталоге
    exclude_artist  артист из summary.exclude_artists («like X by Y» -> Y, «no Coldplay»)
    new_artist      novelty: артист есть в истории пользователя
    exclude_tag     тег или жанр трека совпал с summary.exclude_tags («no rap»); exclude_top_tags > 0 — только
                    первые N тегов трека по весу Last.fm (без жанров): «no pop» не выкидывает прог-рок,
                    у которого pop — восьмой тег из двадцати
    """

    def __init__(self, ctx: Context, catalog: Catalog, exclude_top_tags: int = 0):
        self.catalog = catalog
        self.exclude_top_tags = exclude_top_tags
        self.banned = ctx.banned_ids
        self.ex_tags = ctx.summary.exclude_tags
        self.ex_artists = {a.lower() for a in ctx.summary.exclude_artists}
        self.history_artists = {a.lower() for a in ctx.profile.artist_weights} if ctx.summary.new_artists else set()

    def reason(self, track_id: str) -> Optional[str]:
        if track_id in self.banned:
            return "banned"
        if track_id not in self.catalog:
            return "unknown"
        artist = self.catalog.artist(track_id).lower()
        if artist in self.ex_artists:
            return "exclude_artist"
        if artist in self.history_artists:
            return "new_artist"
        if self.ex_tags:
            if self.exclude_top_tags:
                track_tags = self.catalog.tags(track_id)[:self.exclude_top_tags]
            else:
                track_tags = self.catalog.tags(track_id) + self.catalog.genres(track_id)
            if any(tag_matches(t, track_tags) for t in self.ex_tags):
                return "exclude_tag"
        return None


def apply_filters(fused: List[FusedCandidate], ctx: Context, catalog: Catalog,
                  exclude_top_tags: int = 0) -> List[FusedCandidate]:
    """Оставляет кандидатов, которых пропускают все фильтры (Filters)."""
    f = Filters(ctx, catalog, exclude_top_tags)
    return [fc for fc in fused if f.reason(fc.track_id) is None]


def source_weights(cfg: Dict[str, Any], ctx: Context) -> Dict[str, float]:
    """Базовые веса источников, поправленные по контексту запроса."""
    f = cfg.get("fusion", {})
    w = dict(f.get("weights", {}))
    if not ctx.summary.has_query:            # «ещё», «что-нибудь»: больше доверяем истории
        for s, m in f.get("no_query_boost", {}).items():
            w[s] = w.get(s, 1.0) * m
    if ctx.profile.n_known_tracks == 0:      # холодный старт
        for s, m in f.get("cold_start_boost", {}).items():
            w[s] = w.get(s, 1.0) * m
    return w


CONSTRAINTS = ("year", "mode", "bpm", "energy", "instrumental", "vocals", "lang", "country")
BPM_TOLERANCE = 10.0


def constraint_violations(positions: np.ndarray, ctx: Context, catalog: Catalog,
                          use: Sequence[str] = CONSTRAINTS) -> np.ndarray:
    """Сколько ограничений запроса нарушает каждый трек (позиции каталога). Нет значения в каталоге — не нарушение.

    year / mode / bpm / energy / instrumental / vocals — из summary.constraints (text.parse_constraints);
    lang — summary.languages (язык текста назван явно); country — summary.countries (страна артиста: в запросах
    это часто стиль, а не факт, цель ей удовлетворяет в ~80% — поэтому по умолчанию не используется).
    """
    s, c = ctx.summary, ctx.summary.constraints
    out = np.zeros(len(positions), dtype=np.int64)
    if not len(positions):
        return out
    if "year" in use and ("year_min" in c or "year_max" in c):
        y = catalog.numeric("release_year")[positions]
        known = y > 1900
        out += known & ((y < c.get("year_min", -np.inf)) | (y > c.get("year_max", np.inf)))
    if "mode" in use and "mode" in c:
        m = catalog.numeric("mode")[positions]
        out += ~np.isnan(m) & (m != c["mode"])
    if "bpm" in use and "bpm" in c:
        t, b = catalog.numeric("tempo")[positions], c["bpm"]
        dist = np.minimum.reduce([np.abs(t - b), np.abs(2 * t - b), np.abs(t / 2 - b)])  # темп вдвое — тот же бит
        out += (t > 0) & (dist > BPM_TOLERANCE)
    if "energy" in use and ("energy_min" in c or "energy_max" in c):
        e = catalog.numeric("energy")[positions]
        out += ~np.isnan(e) & ((e < c.get("energy_min", -np.inf)) | (e > c.get("energy_max", np.inf)))
    if "instrumental" in use and c.get("instrumental"):
        inst = catalog.text("is_instrumental")[positions]
        lang = catalog.text("lang")[positions]
        out += ~((inst == "True") | np.char.startswith(lang.astype(str), "INTRU"))
    if "vocals" in use and c.get("vocals"):
        other = "Male" if c["vocals"] == "female" else "Female"
        out += catalog.text("artist_gender")[positions] == other
    if "lang" in use and s.languages:
        lang = catalog.text("lang")[positions]
        out += (lang != "") & ~np.isin(lang, list(s.languages))
    if "country" in use and s.countries:
        country = catalog.text("artist_country")[positions]
        out += (country != "") & ~np.isin(country, list(s.countries))
    return out


def apply_constraints(fused: List[FusedCandidate], ctx: Context, catalog: Catalog,
                      cfg: Dict[str, Any]) -> List[FusedCandidate]:
    """fusion.constraints: {use: [...], penalty: 1.0, hard: false}. Нарушители — ниже (или прочь при hard)."""
    use = cfg.get("use") or []
    if not use or not fused or not (ctx.summary.constraints or ctx.summary.languages or ctx.summary.countries):
        return fused
    pos = catalog.positions(fc.track_id for fc in fused)  # apply_filters уже убрал треки не из каталога
    vio = constraint_violations(pos, ctx, catalog, use)
    if cfg.get("hard", False):
        return [fc for fc, v in zip(fused, vio) if v == 0]
    penalty = float(cfg.get("penalty", 1.0))
    for fc, v in zip(fused, vio):
        fc.violations = int(v)
        fc.score -= penalty * v
    return sorted(fused, key=lambda f: -f.score)
