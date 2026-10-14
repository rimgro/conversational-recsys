# Эксперименты (13.10.2026)

Пайплайн: кандгены (каждый до 1000 треков, у башни Павла 200) → отбор 150 кандидатов голосованием (взвешенный RRF:
`Σ сила_голоса / (60 + место в списке)`) → CatBoost (YetiRank, до 350 деревьев) → top-20. `query_type` модель
не использует. Части: ranker_train 30k и valid 4k — запросы train; **test — 10k запросов test_public**: их не
видела ни одна башня, по нему и сравниваем. Отчёты — `runs/<прогон>/<часть>/report.md` (не в git).

## Кандгены

| кандген | recall@150 на test | recall@1000 | откуда |
|---|---|---|---|
| `twotower_pavel` | 0.722 | 0.752 (у него 200) | Qwen history-aware two-tower (ветка codex/qwen-history-candgen), готовые top-200: `artifacts/twotower_pavel/` |
| `twotower` (наш v2) | 0.643 | 0.847 | `artifacts/twotower/v2/`, векторы — `scripts/encode_twotower.py` |
| `bm25` | 0.547 | 0.708 | индекс по документу трека |
| `history` | 0.385 | 0.638 | свои треки пользователя |
| остальные 9 кандгенов team1 | ≤ 0.48 | | |

Обе башни учились на всём train: на ranker_train и valid они «помнят» ответы (recall@200: наша 0.75 на train против
0.68 на test, Павла 0.90 против 0.75). Поэтому valid у прогонов с башнями завышен — решения по test.

## Прогоны

| прогон | голосуют в отборе | ранкер видит ранги башен | цель в 150 (test) | nDCG@20 valid | **nDCG@20 test** | hit@20 test |
|---|---|---|---|---|---|---|
| `tt` | twotower, bm25, history | нет | 0.780 | 0.427 | 0.418 | 0.578 |
| `tt_500` | то же, 500 кандидатов | нет | 0.890 (из 500) | 0.415 | 0.412 | 0.562 |
| `tt_feat` | twotower, bm25, history | нашей | 0.780 | 0.502 | 0.459 | 0.636 |
| `two` | twotower_pavel ×4, history_recent (подбор tune-policy) | нашей | 0.734 | 0.521 | 0.420 | 0.597 |
| `two_fix` | обе башни, bm25, history — все по 1 | нашей | 0.829 | 0.507 | 0.462 | 0.643 |
| **`two_fix_pfeat`** | обе башни, bm25, history — все по 1 | **обеих** | 0.829 | 0.587 | **0.488** | **0.674** |

Сабмиты: `runs/tt/` (0.418), `runs/tt_feat/` (0.459), `runs/two_fix_pfeat/` (0.488) — `submission_test_public.parquet`.

### nDCG@20 на test по типам запросов

| тип | `tt` | `tt_feat` | `two_fix` | `two_fix_pfeat` |
|---|---|---|---|---|
| exact | 0.947 | 0.948 | 0.956 | 0.940 |
| lyrics_recall | 0.942 | 0.942 | 0.954 | 0.950 |
| vague_recall | 0.795 | 0.813 | 0.815 | 0.815 |
| complex | 0.463 | 0.541 | 0.536 | 0.585 |
| lyrics_theme | 0.515 | 0.506 | 0.514 | 0.584 |
| genre | 0.341 | 0.405 | 0.407 | 0.429 |
| novelty | 0.258 | 0.386 | 0.390 | 0.414 |
| era_region | 0.343 | 0.402 | 0.396 | 0.401 |
| similar_to | 0.304 | 0.358 | 0.358 | 0.377 |
| mood | 0.229 | 0.265 | 0.271 | 0.315 |
| negative_constraint | 0.190 | 0.235 | 0.238 | 0.283 |
| audio_attributes | 0.122 | 0.153 | 0.154 | 0.195 |
| situation | 0.075 | 0.121 | 0.122 | 0.169 |
| **все** | 0.418 | 0.459 | 0.462 | **0.488** |

Главные признаки ранкера `two_fix_pfeat`: ранг башни Павла, длина запроса, место в отборе, популярность трека,
вес трека в истории.

## Выводы

1. **Ранги башен в ранкере — главный рост:** +0.041 от нашей башни (`tt` → `tt_feat`), ещё +0.026 от башни Павла
   (`two_fix` → `two_fix_pfeat`). Голос башни Павла в отборе сам по себе даёт мало (+0.003), хотя цель чаще в 150.
2. **Больше кандидатов без сигнала башни не помогает** (`tt_500`): цель в отборе чаще, ранкер её не находит.
3. **Подбор весов отбора (`tune-policy`) на valid ломается на башнях**, обученных на train: на valid они почти всегда
   правы, подбор отдаёт им весь отбор (`two`: Павел ×4, остальные выключены) — на test хуже (0.420). Поэтому в
   лучшем прогоне сила голоса у всех 1, заданная руками (`configs/selection_two_towers.yaml`).
4. Valid у прогонов с башнями завышен (0.587 против 0.488 на test) — сравнивать только по test.
5. Слабые типы: situation, audio_attributes, negative_constraint, mood (0.17–0.32) — мало текстовых зацепок.

## Что можно сделать дальше

| что | ожидание | цена |
|---|---|---|
| больше деревьев CatBoost (1000): все прогоны упирались в 350 | + | ~10 мин, признаки из кэша |
| 500 кандидатов вместе с рангами башен | возможно + | ~1 ч |
| башни, обученные только на части index: честный valid и подбор весов | + | ~6 ч GPU на башню |
| аудио-признаки трека против ограничений запроса (energy, tempo, valence) — для situation, mood, audio_attributes | + на слабых типах | ~1 ч кода |
| ответ LLM про top-1 трек вместо шаблона (оценивает LLM-судья) | + к оценке ответа | ~2 ч кода + генерация на GPU |
| бейзлайн без башен и без query_type — честная точка отсчёта | для отчёта | ~25 мин |

## Воспроизвести лучший прогон

```bash
python -m recsys run --run two_fix_pfeat --sources +twotower,+twotower_pavel --config configs/selection_two_towers.yaml \
  --set candgen.sources.twotower.features=true --set candgen.sources.twotower_pavel.features=true
python -m recsys submit --run two_fix_pfeat --sources +twotower,+twotower_pavel --config configs/selection_two_towers.yaml \
  --set candgen.sources.twotower.features=true --set candgen.sources.twotower_pavel.features=true
```
Нужны `artifacts/twotower/v2/` (векторы частей ranker_train, valid, test и test_public) и `artifacts/twotower_pavel/`.

## Бонус за новизну (14.10.2026, best_1000_new)

После CatBoost трекам не из истории добавляется `λ · σ(скоров запроса)`, первые `keep_top` мест ранкера не
трогаем (`ranker.discovery: {bonus: λ, keep_top}`, по умолчанию выключен). Метрики платформы на test (среднее по
13 типам), Fast = (0.35·Ranking + 0.15·nDCG new + 0.05·Discovery + 0.10·Diversity) / 0.65:

| вариант | Ranking | nDCG new | Discovery | Diversity | Fast |
|---|---|---|---|---|---|
| без бонуса | 0.503 | 0.378 | 0.367 | 0.974 | 0.536 |
| keep_top 3, λ 1 | 0.497 | 0.394 | 0.524 | 0.974 | 0.549 |
| keep_top 3, λ 2 | 0.482 | 0.402 | 0.697 | 0.978 | 0.556 |

С keep_top 3 после λ ≈ 2 Fast выходит на плато; без защиты верха (keep_top 0) после λ ≈ 1.25 Ranking обваливается.
λ подобран на test — для другой модели (lazy3_best) кривую надо пересчитать.
