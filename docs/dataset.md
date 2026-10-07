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
| `user_profile` | str | текстовое описание пользователя: демография (если есть: пол, возраст, страна, год регистрации) + описание по истории до окна: с какого года слушает, объём, топ-10 артистов, топ жанров, доли эпох и языков, артисты последних 3 месяцев |
| `user_demographics` | str (JSON) / null | `age`, `gender` (m/f), `country`, `created` (регистрация на Last.fm) из LFM-2b; есть у ~63% пользователей |
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
| `spotify_popularity`, `danceability`, `energy`, `valence`, `tempo`, `key`, `mode`, `duration_ms` | аудиоатрибуты Spotify API |
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

Признаки из каталога VK (без идентификаторов VK; заполнены у ~80–98% треков):

| колонка | описание |
|---|---|
| `vk_title`, `vk_subtitle`, `vk_main_artists`, `vk_artist_names`, `vk_featured_artists` | название, подзаголовок (версия/ремикс), артисты в каталоге VK |
| `vk_duration` | длительность версии в VK, с |
| `vk_release_date` | дата релиза в VK (unix) |
| `vk_album_type` | тип релиза (single / album / …) |
| `vk_is_cover`, `vk_is_well_known`, `vk_is_children_track` | флаги каталога |
| `vk_sad_mood_proba`, `vk_happy_mood_proba`, `vk_calm_mood_proba`, `vk_angry_mood_proba`, `vk_happy_russian_mood_proba` | вероятности настроений (аудиоклассификатор) |
| `vk_pop_genre_proba`, `vk_rock_genre_proba`, `vk_hip_hop_genre_proba`, `vk_electronic_genre_proba` | вероятности жанров (аудиоклассификатор) |
| `vk_instrumental_proba` | вероятность инструментала |
| `vk_en_text_coef` | доля английского в тексте |
| `vk_tags_probabilities` | вектор вероятностей тегов аудиоклассификатора |

Признаки из таблицы соответствий Spotify↔VK (заполнены у ~13% треков):

| колонка | описание |
|---|---|
| `vkpl_sp_id`, `vkpl_sp_title`, `vkpl_isrc` | Spotify ID, название и ISRC совпавшей записи |
| `vkpl_vk_title`, `vkpl_vk_title_lang`, `vkpl_vk_genres`, `vkpl_vk_genre_cnt`, `vkpl_vk_original_release_date`, `vkpl_original_release_date` | название, язык, жанры каталога VK, даты оригинального релиза |
| `vkpl_sp_genres`, `vkpl_sp_genre_cnt` | жанры Spotify |
| `vkpl_popularity`, `vkpl_danceability`, `vkpl_energy`, `vkpl_valence`, `vkpl_tempo`, `vkpl_key`, `vkpl_mode`, `vkpl_loudness`, `vkpl_speechiness`, `vkpl_acousticness`, `vkpl_instrumentalness`, `vkpl_liveness`, `vkpl_time_signature`, `vkpl_duration_ms`, `vkpl_af_null_response` | аудиоатрибуты Spotify (свежая выгрузка) |
| `vkpl_language`, `vkpl_artist_cnt`, `vkpl_playlist_cnt`, `vkpl_n_rows` | язык, число артистов, число плейлистов с треком, число совпавших строк |

Служебные поля сопоставления с аудио-каталогом:

| колонка | описание |
|---|---|
| `match_source` | как найдена аудио-версия: `spotify` / `isrc` / `name` (название + артист) |
| `n_candidates`, `n_emb_candidates` | сколько кандидатов-версий нашлось / из них с эмбеддингом |
| `dur_diff_s` | разница длительности Music4All и выбранной версии, с (>30 с — вероятно другая версия: live, ремикс) |

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
