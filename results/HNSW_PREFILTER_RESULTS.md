# HNSW + Prefiltering: результаты

## 1. Англоязычный датасет (Task 1) — vector-only

Модель эмбеддинга `google/embeddinggemma-2`, LanceDB (cosine, IVF_HNSW_SQ), 64 016 треков,
5 000 запросов из `train_queries/qrels`, метрика nDCG@20 = 1/log2(rank+2).

| Метрика | Значение |
|---|---|
| nDCG@20 | **0.1270** |
| Recall@20 | **0.1694** |
| Recall@100 | **0.2518** |
| MRR | **0.1169** |

По типам запросов (nDCG@20 / R@20): `exact` **0.937/0.954**, `lyrics_recall` 0.180/0.240,
`complex` 0.138/0.274, `vague_recall` 0.120/0.205, `lyrics_theme` 0.075/0.130,
`novelty` 0.055/0.103, `era_region` 0.050/0.119, `genre` 0.030/0.077, `similar_to` 0.016/0.043,
`negative_constraint` 0.013/0.020, `mood` 0.009/0.024, `situation` 0.004/0.007,
`audio_attributes` 0.002/0.008.

## 2. Сравнение подходов (префильтрация тегами, JEV)

Офлайн: 63-теговая таксономия; скрытые теги (mood/theme/vocal, 25 шт.) извлекались
JEV-подходом — один forward masked-diffusion модели (`Qwen3-0.6B-diffusion-mdlm`, пакет
`dllm`, `[MASK]`-слот → P(Yes)>0.8); остальные (genre/language/era/instrument/energy, 38 шт.)
— детерминированно из метаданных. Записаны в колонку `extracted_tags` (формат `|tag|tag|`).
Онлайн: теги извлекаются из запроса тем же классификатором; префильтр — SQL по `extracted_tags`.

| Подход | nDCG@20 | R@20 | R@100 | MRR | поиск, мс | RPS (поиск) | RPS (e2e*) |
|---|---|---|---|---|---|---|---|
| **vector** (чистый ANN) | **0.1270** | **0.1694** | **0.2518** | **0.1169** | 21.1 | ~47 | **11.5** |
| hybrid_or (фильтр «любой тег») | 0.1196 | 0.1604 | 0.2412 | 0.1099 | 87.7 | ~11 | ~3–4 |
| hybrid_soft (ANN + буст за теги) | 0.0295 | 0.0440 | 0.0760 | 0.0260 | 118.8 | ~8 | – |
| hybrid_and (строгий AND) | 0.0162 | 0.0238 | 0.0340 | 0.0142 | 51.8 | ~19 | ~3.6 |
| tags (только теги) | 0.0000 | 0.0002 | 0.0022 | 0.0001 | 198.2 | ~5 | ~2.7 |

\* e2e RPS — из прогона с end-to-end замером (эмбеддинг запроса + извлечение тегов + поиск, batch=1);
для вариантов — оценка по времени поиска.

## 3. Вывод

- **Чистый векторный поиск (`vector`) — лучший.** `hybrid_or` практически не уступает по качеству,
  но медленнее (фильтр + ANN). Строгий `AND` и `tags-only` — сильно хуже; `soft` требует меньшего `alpha`.
- **Причина:** лёгкий 0.6B-теггер **переоценивает** теги (например, `pop` у 76% треков, `protest` у 68%),
  т.е. теги шумные и недискриминативные → жёсткая логическая префильтрация режет правильный трек.
- Итог: архитектура **двухэтапного prefiltering** реализована и измерена (офлайн-теги → `extracted_tags` →
  онлайн-теги запроса → SQL-фильтр → ANN), но на лёгкой модели-заменителе она **не даёт прироста**.
  Ожидаемый эффект требует более сильного классификатора — целевого **`google/diffusiongemma-26B-A4B-it`**
  (на H100 / Ampere+ через vLLM `vllm/vllm-openai:gemma`; на Kaggle T4x2 не запускается).

## 4. Артефакты

- `results/results_en_vector.json` — ENG метрики (Task 1) + разбивка по типам.
- `results/results_en_hybrid_or.json`, `results_en_hybrid_and.json`, `results_en_hybrid_soft.json`,
  `results_en_tags.json` — варианты префильтра.
- `results/summary_variants.json` — сводка.
- `results/en_extracted_tags.jsonl` — 64 016 треков с JEV-тегами (кэш; Kaggle-датасет `justrizer/music4all-crs-tags`).
- `results/ru_query_gemma.json`, `results/ru_profile_gemma.json` — RU-метрики (архив).
- Код: `crs_prefilter.py`, `scripts/run_prefilter_eval.py`, `scripts/eval_prefilter_variants.py`,
  `scripts/eval_eng_vector.py`.
