# Текущий G4 runner

`jobs/colab_continue/` содержит рабочий remote runner текущего обучения; ядро
модели сохранено без изменения SHA256. В этой копии runtime-root дополнительно
задаётся через `RECSYS_RUNTIME_ROOT` (по умолчанию `/content/qwen_continue`),
а метаданные числа эпох передаются evaluator аргументами. Runtime настроен
под полный Music4All-CRS: 511 897 обучающих запросов, 61 259 holdout и 64 016 треков.

## Порядок выполнения

1. В уже выделенную G4 сессию оператор передаёт `launch.json` и отдельный
   `drive_credentials.json` с OAuth-доступом владельца к private Drive.
2. `bootstrap.py` скачивает private `payload.tar.gz` из Drive, проверяет SHA256
   архива и каждого исходного файла, запрещает escaping paths и links.
3. Ставит отдельный Python3.12, PyTorch2.7.1/cu128 и pinned requirements.
4. `run_continue.py` проверяет signature/data fingerprint, восстанавливает last.pt,
   optimizer/RNG/cursor и выполняет короткий smoke с rollback.
5. Продолжает ровно ещё пять проходов по fit данным от сохранённого cursor.
   Сохраняет best/last, небольшие progress reports, проверяет общий deadline.
6. Выбирает лучший macro-Ranking checkpoint, экспортирует bundle, завершает
   процесс обучения и в отдельном процессе оценивает весь holdout.
7. После сохранения всех отчётов освобождает только свой endpoint.

GPU выбирает внешний оператор через Google Colab CLI: bootstrap не выделяет
ресурсы, не покупает CU и не управляет чужими сессиями. Авторизация аккаунта,
allocation receipts, token/session store и рабочие submission/PID/log файлы
остаются вне Git. Для нового запуска нужны новые private folder/file/endpoint
идентификаторы и проверенный баланс. Не используйте конфигурацию живого запуска
для параллельного writer.

## Входы

```text
runtime-root/
  launch.json                 # private runtime configuration
  drive_credentials.json      # private OAuth, chmod 600
  payload.tar.gz              # скачивается bootstrap
  payload/
    payload_manifest.json     # sha256 для всех файлов ниже
    recsys/                   # ядро из этой ветки
    music4all_crs/             # пять parquet файлов датасета
    work/artifacts/
      cache/<data-fingerprint>/  # готовые item features/tokens/users
      runs/<signature>/
        last.pt, best.pt, config.json, identity.json, history.json
        best_metrics.json
    evaluation_inputs/        # queries.parquet, profiles.npz, manifest.json
    jobs/colab_continue/      # пять remote source/requirements файлов
```

`launch.json` содержит:

| Поле | Значение |
|---|---|
| `run_id` | уникальный ID этой продолженной работы |
| `signature`, `data_fingerprint` | из сохранённого identity.json |
| `initial_step` | step из исходного last.pt |
| `fit_queries` | 511897 |
| `folder_id`, `evaluation_folder_id` | private Drive folders для checkpoint и оценки |
| `quota_project` | Drive API consumer project |
| `payload_file_id`, `payload_sha256` | Drive file ID и SHA256 готового payload |
| `endpoint` | **свой** endpoint уже выделенной G4 |
| `deadline_unix` | абсолютный жёсткий предел времени с учётом текущего CU баланса и резерва |

Архив должен быть создан из выбранных файлов, а не из всей рабочей папки:
credentials, logs, виртуальные окружения и frozen HF model cache в него не входят.
После upload проверьте владельца folder и md5Checksum/size файла.

## Бюджет, бэкапы и ранняя остановка

Remote child processes ограничены watchdog до `deadline_unix`. Trainer учитывает
прошлые elapsed seconds из last.pt; smoke, подготовка, checkpoints и новые
обновления не обнуляют накопленное время. Отдельно резервируется время на export
и полную оценку. В текущем выделении резерв CU задаётся внешним оператором:
`min(11.5h, (balance - reserveCU) / accountRate)`; ставка и баланс проверены
после выделения. Это оценка расхода CU, не отдельная сумма счета Google.

Большие бэкапы — раз 1 800 секунд, также начальный и конечный. Каждые 1 000
шагов validation сохраняет локальные checkpoint и small metric reports;
это не отправка большого бэкапа в Drive каждый раз. Upload пропускает step-*.pt,
pre_smoke и незавершённые staging/tmp файлы. Progress marker обновляется после
успешных проверенных uploads.

Early stopping использует только фиксированные 2 600 validation queries:
min_delta=0.0005, patience=16 полных проверок, минимум одна новая эпоха.
Сохраняется настоящий максимум macro-nDCG@20, даже если улучшение меньше min_delta.
Additional holdout не используется для stopping или выбора checkpoint.

## Метрики и восстановление

Validation и итоговый evaluator считают Recall@1/5/10/20/50/100/200,
nDCG@20/100/200, MRR@200 и Discovery. Итоговая оценка отдельно сохраняет exact
и HNSW, micro/macro, 13 типов, checkpoint-selection и additional holdout.
Перед продвижением cursor оба candidate chunks сохраняются в Drive и проверяются.

Последний валидированный checkpoint может быть лучше или хуже предыдущего best:
последний нужен для продолжения optimizer/cursor, лучший — для финального поиска.
После сбоя сначала подтвердите остановку старого worker/endpoint, сохранность
last.pt и тот же input/bundle/scoring. Сохраните первоначальную целевую отметку,
накопленное время и CU. Этот snapshot не является автоматическим retry scheduler.

WebSocket может не доставить окончание уже успешного запуска. Сначала проверьте
durable Drive `reports/job_status.json`, `final_evaluation/status.json`,
`progress.json` и отсутствие собственного endpoint; не отправляйте fit повторно.
