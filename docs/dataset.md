# Music4All-CRS: разговорные запросы поверх истории прослушиваний

Датасет для разговорных рекомендаций музыки. Пользователь — история прослушиваний Last.fm (Music4All-Onion),
запрос — синтетический текстовый запрос к треку, который пользователь реально послушал в целевом окне.

Ключ трека — `m4a_id` (16-символьный ID Music4All), рядом `spotify_id`.
Время — unix timestamp (UTC, секунды).

## Сплиты

| таблица | окно позитивов | история | строк (пользователей) | позитивов |
|---|---|---|---|---|
| `train` | 2020-01-21 — 2020-02-19 | все прослушивания до 2020-01-21 | 12,891 | 483,006 |
| `test_public` | 2020-02-20 — 2020-03-20 | все прослушивания до 2020-02-20 | 12,247 | 231,316 |
| `test_private` | 2020-02-20 — 2020-03-20 | все прослушивания до 2020-02-20 | 12,235 | 227,800 |

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
| `user_profile` | str | текстовое описание пользователя по истории до начала окна: с какого года слушает, число прослушиваний, треков и артистов, топ-10 артистов, топ-8 жанров, доли эпох и языков песен, топ-5 артистов последних 3 месяцев |
| `user_demographics` | struct / null | `age` (int), `gender` (`m`/`f`), `country` (ISO-код) из LFM-2b; есть у ~63% пользователей, иначе null |
| `n_positives` | int | число позитивов в строке |
| `positives` | list[dict] | все треки, прослушанные в окне (уникальные), по времени первого прослушивания; поля ниже |

Поля элемента `positives`:

| поле | описание |
|---|---|
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

Аудио-эмбеддинг (то же, что в `track_embeddings`):

| колонка | описание |
|---|---|
| `muq_embedding` | MuQ, 128 float, L2-нормирован |

Свежая выгрузка Spotify API по совпавшей записи (заполнено у ~13% треков):

| колонка | описание |
|---|---|
| `spotify_id_2`, `spotify_title`, `isrc` | Spotify ID, название и ISRC совпавшей записи (`spotify_id_2` может отличаться от `spotify_id` из Music4All — другая версия релиза) |
| `spotify_release_date` | дата оригинального релиза |
| `spotify_genres`, `spotify_genre_cnt` | жанры артиста в Spotify |
| `spotify2_popularity`, `spotify2_danceability`, `spotify2_energy`, `spotify2_valence`, `spotify2_tempo`, `spotify2_key`, `spotify2_mode`, `spotify2_loudness`, `spotify2_speechiness`, `spotify2_acousticness`, `spotify2_instrumentalness`, `spotify2_liveness`, `spotify2_time_signature`, `spotify2_duration_ms` | популярность и аудиоатрибуты Spotify API (свежие; аудиоатрибуты без префикса — выгрузка Music4All 2019 года) |
| `spotify2_language`, `spotify2_artist_cnt`, `spotify2_playlist_cnt` | язык, число артистов, в скольких плейлистах трек |

## Синтетические запросы

Сгенерированы Qwen3.8-27B по полной карточке трека (все поля `tracks_meta`), отдельный промпт на каждый тип,
до 5 запросов на тип; каждый запрос проверен LLM-судьёй, непрошедшие перегенерированы.
В датасет попадают только прошедшие проверку. Язык — русский.

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

Независимая проверка (Claude Opus 5.5, 2,012 запросов по 40 случайным трекам): 89% — трек явно хороший ответ,
11% — допустимо, 0.15% — неверно; галлюцинаций 2.5%. Типы `situation`, `audio_attributes`, `mood`,
`negative_constraint` часто общие (подходят многим трекам) — метрики стоит считать по `query_family`.

## Источники

- Music4All (Santana et al., 2020), Music4All-Onion (Moscati et al., CIKM 2022, CC BY 4.0), Music4All A+A (2025, CC BY-SA)
- `seungheondoh/enrich-music4all`, `talkpl-ai/listening-history-filtered` (демография LFM-2b)
