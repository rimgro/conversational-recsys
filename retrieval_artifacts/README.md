# Генераторы последней модели

Два BM25-индекса и три TF-IDF-представления. `track_ids.npy` задаёт единый порядок каталога; `base_feature_names.json` и `generators.json` задают порядок фичей и источников. `candidate_cache/` хранит ранжированные кандидаты для validation/test-выборок обучающего ноутбука. Пересборка индексов — `../build_candidate_indexes.ipynb`.
