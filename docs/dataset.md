# Music4All-CRS: разговорные запросы поверх истории прослушиваний

Датасет для разговорных рекомендаций музыки. Пользователь — история прослушиваний Last.fm (Music4All-Onion),
запрос — синтетический текстовый запрос к треку, который пользователь реально послушал в целевом окне.

Ключ трека — `m4a_id` (16-символьный ID Music4All), рядом `spotify_id`.
Время — unix timestamp (UTC, секунды).

## Файлы

`train.parquet`, `test_public.parquet`, `tracks_meta.parquet`, `track_embeddings.parquet`, `{train,test_public}_{queries,qrels}.parquet`; `test_private` — закрытый тест, хранится отдельно и не публикуется.
Примеры строк — `samples/`.

## Оценка: `query_id`, queries и qrels

У каждого запроса — и у обычных позитивов (`positives[].query_id`), и у `discovery[].query_id` — есть стабильный уникальный `query_id`
(`sha1("{train|test}|{user_id}|{positive|novelty|similar_to}|{m4a_id}")[:16]`, одинаковый при любой пересборке).

Для eval-системы есть плоские таблицы:

| файл | что | колонки |
|---|---|---|
| `train_queries.parquet` / `test_public_queries.parquet` | входы для модели | `query_id`, `user_id`, `split`, `query`, `source` (`positive` / `discovery`), `query_type` (11 типов + `novelty`, `similar_to`) |
| `train_qrels.parquet` / `test_public_qrels.parquet` | ответы для проверки | `query_id`, `target_m4a_id`, `target_spotify_id`, `exclude_m4a_id`, `exclude_artist`, `is_new`, `ts` |

История и профиль пользователя для запроса берутся из строки `train` / `test_public` с тем же `user_id`.
Для `test_private` такие же таблицы хранятся отдельно и не публикуются.

**Формат сабмита:** по строке на запрос — `query_id`, `top20` (список `m4a_id` или `spotify_id`, по убыванию), `response` (текст ответа LLM).

**Метрика:** nDCG@20 с одним релевантным треком (`target_*`): 1/log2(rank+1), если цель в топ-20, иначе 0.
Для `similar_to` перед подсчётом из выдачи удаляются `exclude_m4a_id` и все треки `exclude_artist`
(рекомендовать сам X или его артиста — ошибка). Метрики стоит считать отдельно по `query_type`.

Доли типов запросов одинаковы во всех сплитах: `similar_to` 11.19%, `novelty` 4.47% (discovery-записи прорежены детерминированно по `query_id`), 11 основных типов ≈ 7–7.9% каждый (±0.15 п.п. — шум случайного выбора типа).

| | запросов всего | positive | novelty | similar_to |
|---|---|---|---|---|
| train | 573,156 | 483,400 | 25,620 | 64,136 |
| test_public | 273,137 | 230,370 | 12,202 | 30,565 |
| test_private | 271,580 | 229,050 | 12,140 | 30,390 |

## Сплиты

| таблица | окно позитивов | история | строк (пользователей) | позитивов |
|---|---|---|---|---|
| `train` | 2020-01-21 — 2020-02-19 | все прослушивания до 2020-01-21 | 12,893 | 483,400 |
| `test_public` | 2020-02-20 — 2020-03-20 | все прослушивания до 2020-02-20 | 12,237 | 230,370 |
| `test_private` | 2020-02-20 — 2020-03-20 | все прослушивания до 2020-02-20 | 12,245 | 229,050 |

`test_public` / `test_private` — деление позитивов тестового месяца по `md5(query) % 2` (чётный → public):
одинаковый текст запроса всегда попадает в одну часть; вместе они дают весь тестовый месяц.
Окна идут встык: история теста включает train-месяц.

Только треки корпуса (64,016 треков, у которых есть аудио-эмбеддинг): прослушивания других треков из истории и окна удалены.

## `train`, `test_public`, `test_private` — одна строка = пользователь × сплит

| колонка | тип | описание |
|---|---|---|
| `user_id` | int | ID пользователя Last.fm (совпадает с LFM-2b / Music4All-Onion) |
| `split` | str | `train` / `test_public` / `test_private` |
| `history` | list[str] | **все** прослушивания пользователя до начала окна, `m4a_id` в хронологическом порядке (повторы = повторные прослушивания) |
| `history_ts` | list[int] | время каждого прослушивания из `history` |
| `history_len` | int | длина истории |
| `user_profile` | str | текстовое описание пользователя на английском по истории до начала окна: с какого года слушает, число прослушиваний, треков и артистов, топ-10 артистов, топ-8 жанров, доли эпох и языков песен, топ-5 артистов последних 3 месяцев |
| `user_demographics` | struct / null | `age` (int), `gender` (`m`/`f`), `country` (ISO-код) из LFM-2b; есть у ~63% пользователей, иначе null |
| `n_positives` | int | число позитивов в строке |
| `positives` | list[dict] | все треки, прослушанные в окне (уникальные), по времени первого прослушивания; поля ниже |

Поля элемента `positives`:

| поле | описание |
|---|---|
| `query_id` | уникальный ID запроса |
| `m4a_id`, `spotify_id` | трек |
| `artist`, `title` | артист и название |
| `ts` | время первого прослушивания трека в окне |
| `n_listens` | сколько раз трек прослушан в окне |
| `is_new` | `true`, если пользователь не слушал трек до начала окна (discovery) |
| `query` | синтетический запрос пользователя к этому треку (случайный из прошедших проверку) |
| `query_family` | тип запроса, см. ниже |

## `track_embeddings`

| колонка | описание |
|---|---|
| `m4a_id`, `spotify_id` | трек |
| `muq_embedding` | аудио-эмбеддинг MuQ, 128 float, L2-нормирован |

## `tracks_meta` — одна строка = трек (64,016)

Идентификаторы и базовая мета (Music4All):

| колонка | описание |
|---|---|
| `m4a_id`, `spotify_id` | ID трека |
| `isrc` | точный ISRC этой записи (по `spotify_id`: Spotify API / TalkPlayData-Extra); есть у 74.3% треков |
| `isrc_same_song` | ISRC той же песни из другого релиза (сопоставление по названию и артисту), только там, где нет точного; ещё ~20% треков. Не идентификатор этой записи |
| `m4a_artist`, `m4a_song`, `m4a_album` | артист, название, альбом |
| `release_year` | год релиза (для сборников — год сборника) |
| `lang` | язык текста (ISO-код; `INTRUMENTAL` — без слов) |
| `is_instrumental` | без вокала |
| `m4a_genres_full`, `m4a_tags_full` | жанры и теги Music4All (через запятую) |
| `spotify_popularity`, `danceability`, `energy`, `valence`, `tempo`, `key`, `mode`, `duration_ms` | популярность и аудиоатрибуты Spotify API (Music4All, 2019) |
| `lyrics` | полный текст песни |
| `lyrics_processed` | лемматизированный текст из Music4All-Onion |

Теги и описания:

| колонка | описание |
|---|---|
| `lastfm_tag_weights` | теги Last.fm с весами 0–100 (шумные); в parquet — JSON-строка `{"tag": weight}` |
| `title`, `artist`, `release`, `tags`, `m4a_genres`, `m4a_tags`, `pseudo_caption` | из `seungheondoh/enrich-music4all`; `pseudo_caption` — автоописание по тегам, бывает неточным |
| `onion_listens` | число прослушиваний трека во всём Music4All-Onion |

Альбом и артист (Music4All A+A, MusicBrainz + Last.fm):

| колонка | описание |
|---|---|
| `album_name`, `album_release_date`, `album_description`, `album_tags`, `album_genres`, `album_lastfm_listeners`, `album_lastfm_playcount`, `album_mbid` | альбом |
| `artist_mbid`, `artist_type`, `artist_gender`, `artist_country`, `artist_begin`, `artist_end`, `artist_genres` | артист |
| `artist_description` | биография артиста; в A+A имена и жанры замаскированы `<Person>` / `<Genre>` |
| `artist_description_masked` | флаг маскировки |
| `artist_wiki_en`, `artist_wiki_ru`, `artist_wikidata` | вступление статьи об артисте из английской / русской Википедии (без масок) и Wikidata ID; en есть у ~71% треков |

Аудио-эмбеддинг (то же, что в `track_embeddings`):

| колонка | описание |
|---|---|
| `muq_embedding` | MuQ, 128 float, L2-нормирован |

Свежая выгрузка Spotify API по совпавшей записи (заполнено у ~13% треков):

| колонка | описание |
|---|---|
| `spotify_id_2`, `spotify_title` | Spotify ID и название совпавшей записи (`spotify_id_2` может отличаться от `spotify_id` из Music4All — другая версия релиза) |
| `spotify_release_date` | дата оригинального релиза |
| `spotify_genres`, `spotify_genre_cnt` | жанры артиста в Spotify |
| `spotify2_popularity`, `spotify2_danceability`, `spotify2_energy`, `spotify2_valence`, `spotify2_tempo`, `spotify2_key`, `spotify2_mode`, `spotify2_loudness`, `spotify2_speechiness`, `spotify2_acousticness`, `spotify2_instrumentalness`, `spotify2_liveness`, `spotify2_time_signature`, `spotify2_duration_ms` | популярность и аудиоатрибуты Spotify API (свежие; аудиоатрибуты без префикса — выгрузка Music4All 2019 года) |
| `spotify2_language`, `spotify2_artist_cnt`, `spotify2_playlist_cnt` | язык, число артистов, в скольких плейлистах трек |

## `discovery` — запросы на открытие нового (в `train`, `test_public`, `test_private`)

Список у каждой строки пользователя (`n_discovery` — длина). Генерируется для пары пользователь × позитив окна, с учётом его истории.

| `kind` | позитив | запрос |
|---|---|---|
| `novelty` | новый трек **от артиста, которого нет в истории**, из нижней половины по сходству MuQ с центроидом истории (`sim_to_history` ≤ 0.425) | просьба о чём-то новом, вне привычного вкуса, с описанием того, чем трек отличается от любимого пользователем |
| `similar_to` | новый трек T; референс X — ближайший к T по MuQ трек **из истории пользователя**, другого артиста и не та же песня (`reference_sim` ≥ 0.58) | «похожее на X by artist(X), но от других артистов» + уточнения, верные для T |

Поля элемента: `query_id`, `kind`, `m4a_id`, `spotify_id`, `artist`, `title`, `ts`, `query`, `sim_to_history`, `genre_novelty`,
`reference_m4a_id`, `reference_spotify_id`, `reference_artist`, `reference_title`, `reference_sim` (референс только у `similar_to`).
При оценке `similar_to` из выдачи нужно исключать сам X и все треки `reference_artist`.

Для `novelty` трек и его артист **гарантированно отсутствуют** в истории пользователя (проверено по всем записям).
`genre_novelty`:
- `new_genre` — ни один жанр трека не входит в топ-10 жанров истории и на эти жанры приходится < 2% прослушиваний (новый артист из **нового** жанра);
- `same_genre_new_artist` — новый артист в уже знакомом жанре.

| | novelty (new_genre / same_genre_new_artist) | similar_to |
|---|---|---|
| train | 25,620 (9,190 / 16,430) | 64,136 |
| test_public | 12,202 (4,332 / 7,870) | 30,565 |
| test_private | 12,140 (4,374 / 7,766) | 30,390 |

Независимая проверка (Claude Opus 5.5, 160 случайных записей): трек — явно хороший ответ на запрос в 88% (`novelty`) / 90% (`similar_to`),
неверных 0%, галлюцинаций 0%; X и T действительно похожи в 75% пар `similar_to` (остальные — слабо связаны).

Тест делится на public/private по `md5(query) % 2`, как и основные позитивы.

## Синтетические запросы

Сгенерированы Qwen3.8-27B (FP8) по полной карточке трека (поля `tracks_meta`, включая полный текст песни и биографию артиста), отдельный промпт на каждый тип,
до 5 запросов на тип; каждый запрос проверен LLM-судьёй, непрошедшие перегенерированы.
В датасет попадают только прошедшие проверку. Язык запросов — английский (названия, имена и цитаты из текстов песен — на языке оригинала).

| `query_family` | что это |
|---|---|
| `exact` | точный поиск трека: артист и название, опечатки, транслит |
| `lyrics_recall` | по строчке текста песни |
| `lyrics_theme` | по теме текста своими словами |
| `genre` | по жанру/сцене с уточнением |
| `mood` | по настроению |
| `situation` | по ситуации/активности (дорога, тренировка, вечеринка…) |
| `era_region` | по эпохе, стране, языку |
| `audio_attributes` | по звучанию: темп, энергия, вокал, тональность, длительность |
| `complex` | 3–4 условия одновременно |
| `negative_constraint` | с исключением («…но не…», «без…») |
| `vague_recall` | размытое воспоминание о треке |

Выбор запроса к позитиву: сначала случайный тип, затем случайный запрос этого типа
(seed = `split|user_id|m4a_id`).

Независимая проверка русской версии разметки (Claude Opus 5.5, 2,012 запросов по 40 трекам): 89% — трек явно хороший ответ, 0.15% — неверно; галлюцинаций 2.5%. Английская версия по подбору промптов: 90–97% запросов проходят судью. Типы `situation`, `audio_attributes`, `mood`,
`negative_constraint` часто общие (подходят многим трекам) — метрики стоит считать по `query_family`.

## Источники

- Music4All (Santana et al., 2020), Music4All-Onion (Moscati et al., CIKM 2022, CC BY 4.0), Music4All A+A (2025, CC BY-SA)
- `seungheondoh/enrich-music4all`, `talkpl-ai/listening-history-filtered` (демография LFM-2b)
