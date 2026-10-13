# Готовые топ-200 для итогового пайплайна

Полный `test_public`: 273 137 запросов. Полный `train`: 573 156 запросов, включая fit и внутренние validation-запросы. Это **инференс** best step 104000; веса не обновляются. Истории берутся из соответствующего сплита до его временного cutoff. Запросы заново токенизируются; дефектный training token cache не используется.

Основная выдача каждого сплита:

| Файл | Назначение |
|---|---|
| `candidates_top200.sqlite` | самостоятельный индексированный файл для быстрого получения примера по `query_id` |
| `candidates_top200.parquet` | те же выдачи для batch-обработки, обучения ранкера и аналитических инструментов |
| `discovery_top200.sqlite` / `.parquet` | отдельный вариант, исключающий всю историю прослушиваний |

В Parquet ровно три колонки: **`query_id`, `top200`, `scores`**. `top200` — упорядоченный список `m4a_id`, `scores` — соответствующие cosine scores. В SQLite та же информация хранится в сжатом JSON с `query_id` как primary key. В выдачах нет метрик, целей из qrels, target ranks или текста ответа LLM. Metadata SQLite/manifest фиксирует модель, split и схему, но не Recall/nDCG.

## Получить выдачу без модели

Этот reader использует только стандартную библиотеку Python; Qwen, PyTorch, API и GPU не нужны:

```python
from recsys.serving.cache import CandidateCache

with CandidateCache('/data/test_public/candidates_top200.sqlite',
                    expected_split='test_public') as cache:
    example = cache.get('query_id_из_test_public_queries.parquet')
    ids = example['top200']
    scores = example['scores']
    # Передать ids/scores в fusion / ranker / итоговый pipeline.
    batch = cache.get_many(['query_id_1', 'query_id_2'])
```

`get_many()` возвращает результаты в порядке входных ID, включая повторы. Неизвестный ID вызывает `KeyError`: reader не запускает скрытый инференс и не подменяет отсутствующий пример пустой выдачей. SQLite открывается read-only; один reader можно использовать из нескольких потоков. `expected_release_id` позволяет проверить, что скачан файл нужной модели.

В long-lived pipeline создайте reader на startup и закройте его через `close()` при shutdown. Для Discovery выберите `discovery_top200.sqlite`. Изолированные `train`/`test_public` файлы проверяют split, поэтому train не подмешивается в public. Spotify aliases объединены; `similar_to` исключает reference canonical-группу и reference artist. Если допустимых треков меньше 200, список короче; padding/дубликатов нет.

Для пакетного чтения:

```python
import pandas as pd
candidates = pd.read_parquet('/data/train/candidates_top200.parquet')
joined = examples.merge(candidates, on='query_id', how='left', validate='one_to_one')
```

Значения scores после чтения cache совпадают с сохранёнными; JSON не округляется дополнительно. `train` содержит предсказания уже обученной на части этих примеров модели, поэтому его нельзя выдавать за независимый test или out-of-fold кандидаты.

## Воспроизводимый прогон

`tools/prepare_split_inference.py` проверяет uniqueness/coverage/history cutoff и не включает target labels во входы энкодера. `tools/encode_split_queries.py` строит frozen query vectors на GPU: batch выбирается по короткому замеру, полный проход начинается только если оценка с запасом помещается в оставшийся лимит. Durable chunks фиксируются до продвижения cursor. `OfflineCandidates` ищет HNSW на CPU и сохраняет оба варианта выдачи. Обработанные chunks можно продолжить с теми же weights/input hashes/dtype.

Итоговые файлы проверяются по уникальным query IDs, coverage и SHA256. SQLite публикуется атомарно после полного заполнения; незавершённый `.building` не является результатом. Итоговый manifest перечисляет файлы и контрольные суммы. Веса, истории, файлы выдач и секреты не коммитятся в Git.
