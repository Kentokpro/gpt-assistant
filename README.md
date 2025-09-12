`Этот файл README доступен на двух языках: RU и EN.`
# 1. Leadinc AI Assistant — сервер умного менеджера-консультанта для B2B-продаж
<img width="2559" height="1238" alt="image_2025-09-12_13-02-56" src="https://github.com/user-attachments/assets/b75b51a7-7610-464b-880d-0ddcce7102ee" />

**Для бизнеса: ассистент помогает продавать быстрее и точнее. Он как дружелюбный менеджер:**

- подскажет, как работает сервис и с чего начать;

- мгновенно ответит на любой вопрос по услугам;

- проведёт быструю регистрацию;

- выдаст аналитику по 1 из 430 бизнес-ниш;

- снижает время на объяснения, квалификацию и уточнения → команда меньше отвлекается, лиды обрабатываются быстрее, меньше «утечек» на этапах воронки.

# 2. 🧭 TL;DR

- Что это: FastAPI-бэкенд ассистента Leadinc с RAG (2 коллекции: FAQ/Analytics), голосом (STT/TTS), Redis-лимитами, данными в Postgres, векторным поиском ChromaDB и Celery воркерами.

- Зачем бизнесу: ассистент как дружелюбный менеджер — быстро объясняет продукт, ведёт регистрацию, отвечает на FAQ, отдаёт аналитику по 1 из 430+ ниш → меньше рутины у команды, быстрее обработка лидов, ниже потери по воронке.

- Сценарии LLM: REGISTRATION, FAQ, ANALYTICS, OFFTOPIC. На каждый запрос — ровно одна RAG-коллекция (без смешивания).

- Запуск: локально через venv (Docker используется для инфраструктуры при желании).

- RAG-данные: приватные (в публичный репозиторий не входят).

- Статус: ALPHA тест (самоподписанный HTTPS для тестов).

- Куда смотреть: ⚙️ Быстрый старт → 🗂️ Структура → 🧩 Архитектура → 🎯 Сценарии.

# 3 ⚙️ Быстрый старт
**1) Клонирование**
git clone https://github.com/Kentokpro/gpt-assistant.git
cd ai-assistant

**2) Python venv**
source ~/ai-assistant/backend/venv/bin/activate

**3) Зависимости**
pip install -r ai-assistant/backend/requirements.txt

**4) Переменные окружения**
Создай ai-assistant/backend/.env.backend.

**5) База данных и миграции**
alembic upgrade head

**6) (Если требуется) запусти ChromaDB отдельно (версию из requirements.txt)**
пример: chroma run --host 000.0.0.0 --port 0000

**7) Запуск приложения**
uvicorn backend.main:app --host 0.0.0.0 --port 0000 --reload

**8) Тестовый фронт**
Открой в браузере frontend_test/index.html (локально) и проверь подключение к API.

# 4. 📂 Структура репозитория:
<img width="1455" height="1147" alt="image_2025-09-12_13-17-12" src="https://github.com/user-attachments/assets/a24d0dcf-b8bd-40a5-af2a-5643d7883ad1" />
<img width="1455" height="635" alt="image_2025-09-12_13-17-44" src="https://github.com/user-attachments/assets/5f9a7bc0-e6fa-4cd4-aa40-3870cff44e27" />

# 5. 🧩 Архитектура и ключевая логика
**Инфраструктура (ALPHA):**
- **OS:** Ubuntu 22.04.5 LTS (jammy), Python 3.10.12
- **Reverse-proxy:** Nginx (443/HTTPS; самоподписанный для тестов)
- **Backend:** FastAPI (Uvicorn) — `gpt-backend.service` на `000.0.0.0:0000`
- **DB:** PostgreSQL 14 (0000)
- **Cache/Rate-limit/State:** Redis 6.0.16 (0000)
- **Vector DB:** ChromaDB (HTTP 0000) — `leadinc-chroma.service`
- **Queues:** Celery — 2 воркера `text`, 2 воркера `audio` (раздельные очереди)
- **Codes:** `code-service` (0000) — одноразовые 6-значные коды (TTL 10 мин)
- **Static test front:** `/var/www/leadinc/static/chat.html`

**Тестовый Front (HTML+JS):**
- POST /ai/chat (текст → текст/голос),
- POST /ai/voice_upload (файл → STT),
- POST /ai/tts (текст → аудио).
Требует корректного CORS (разрешённый origin) и установки cookie в браузере (см. SECURE_COOKIES/SameSite).

**Маршрутизация сценариев: ассистент на каждый запрос выбирает один из 4 сценариев. В зависимости от сценария использует и обращается к одной RAG-коллекции.**
**Сценарии ассистента:**
- **FAQ** → коллекция `faq_leadinc` (RAG)
- **ANALYTICS** → коллекция `analytics_leadinc` (RAG)
- **REGISTRATION** → стадийная воронка 1→3, одноразовые коды (code-service), миграция истории по `user_id`
- **OFFTOPIC** → болталка с лимитами

> На каждый запрос используется **ровно одна** RAG-коллекция (FAQ **или** Analytics), без смешивания.

Stage-логика: используется при регистрации гость (1–2) → регистрация → авторизованный (3).

Голос: загрузка → STT → ответ + опциональный TTS (файл сохраняется и автоочищается).

# 6. 🎯 Сценарии и возможности

**Ценность: быстрый старт, меньше вопросов «как это работает», готовые ответы и аналитика по нише — быстрее к продажам.**

**REGISTRATION** - Воронка стадий (1→3): код подтверждения → город/ниша → авторегистрация. Создаёт пользователя, связывает текущую сессию, выдаёт JWT, предлагает перейти к аналитике.

**FAQ** - Ответы по сервису (как работает, условия, запуск, что входит). Источник — коллекция faq_leadinc (RAG).

**ANALYTICS** - Выдаёт аналитику по выбранной нише из analytics_leadinc (RAG).

**OFFTOPIC** - Просто поболтать, ненавязчиво возвращает диалог к цели (лимиты и антифлуд настраиваемые).

**Технически:**
- Гибридный RAG: ChromaDB, 2 коллекции (FAQ/Analytics), заголовки ниш кэшируются.
- Auth/лимиты: JWT-cookie, отдельная session-cookie, стадии/квоты и антифлуд на Redis. Проверка активной подписки (whitelist e-mail — байпас).
- Voice: STT (Whisper) и TTS (ElevenLabs), автоочистка по таймеру.
- Тест-фронт: статическая страница для проверки API (без прод-функций).

# 7. 🔐 **Переменные окружения (пример backend/.env.backend)**
**JWT и общие секреты**
SECRET=(secret)

SECRET_ADMIN_TOKEN=(secret)

OPENAI_API_KEY=(secret)

ELEVENLABS_API_KEY=(secret)

**Основной голос**

ELEVENLABS_VOICE_ID=(secret)

**Резервный голос на подмену основного**

ELEVENLABS_FALLBACK_VOICE_ID=(secret)

**Redis**

REDIS_URL=redis://localhost:0000

**Postgres connection**
DATABASE_URL=postgresql+asyncpg://(секрет):(секрет)@000.0.0.0:(порт)/(database_name)

POSTGRES_USER=(user_name)

POSTGRES_PASSWORD=(secret)

POSTGRES_DB=(database_name)

POSTGRES_HOST=000.0.0.0

POSTGRES_PORT=(port)

**ChromaDB (RAG)**

CHROMA_HOST=localhost

CHROMA_PORT=0000

**Безопасность и сессии**

SECURE_COOKIES=false

CSRF_ENABLED=true

SESSION_COOKIE_NAME=sessionid

ALLOWED_HOSTS=yourdomain.com,localhost,000.0.0.0

CORS_ORIGINS=https://yourdomain.com, http://localhost:0000

DEBUG=true

# 8 📚 **RAG-хранилище (ChromaDB) (`chroma_utils.py`)**
**Назначение:** тонкая обёртка над ChromaDB (HTTP) для RAG: подключение, поиск по эмбеддингам, фильтры по метаданным, чтение «полной статьи» из Markdown (fallback), перечисление коллекций.

- Коллекции: **`faq_leadinc`** и **`analytics_leadinc`**. Источники `.md` приватные → в публичный репозиторий не попадают.
- Нарезка/индексация: `scripts_ChromaDB/scripts_ChromaDB.py` (FAQ) и план/описание `analytic/script_2_collection_RAG.md` (Analytics).
- Кеш заголовков аналитики: `analytic/_titles_emb_cache.json`.
- Правило: на каждый запрос ассистент использует **ровно одну** коллекцию (FAQ **или** Analytics), без смешивания.
- Обновление корпуса: заменить `.md` → перегенерировать эмбеддинги → перезапустить backend.

**функции:**
- `connect_to_chromadb() -> chromadb.HttpClient`  
  Создаёт HTTP-клиент. Ошибки логируются и пробрасываются.
- `get_collection(collection_name: str)`  
  Возвращает коллекцию через клиент.
- `search_chunks_by_embedding(query_emb: List[float], n_results=5, collection_name=..., filters=None) -> List[Dict]`  
  Делает `collection.query(..., include=["documents","metadatas","distances"])`.  
  Внутри запрашивает **`internal_k = max(n_results, 12)`** для повышения релевантности, наружу отдаёт ровно `n_results`.  
  Нормализует `tags` к списку: `list | str | None → List[str]`.  
  **Выдаёт элементы со схемой:** `article_id`, `title`, `meta_tags`, `tags: List[str]`, `summary`, `text`.
- `filter_chunks(collection_name=..., article_id=None, meta_tags=None, tags=None, title=None, summary=None, limit=10) -> List[Dict]`  
  Делает `collection.get(where=filters, limit=limit)`. Возвращает ту же схему, что и `search_chunks_by_embedding`.
- `get_full_article(article_id: str, articles_file=ARTICLES_FILE) -> str`  
  Асинхронно читает общий Markdown, делит по `---`, ищет блок по шаблону `article_id: "<digits>"`.  
  Возвращает **полный текст** блока как есть.  
  Фолбэки: файла нет → `"Техническая ошибка: база знаний временно недоступна."`; не найдено → `"Статья не найдена."`  
  > Требуется **числовой** `article_id` в **кавычках**.
- `list_collections() -> list`  
  Возвращает список коллекций `client.list_collections()` и логирует результат.

**Асинхронность и производительность**
- Потенциально блокирующие вызовы Chroma — через `run_in_executor` (не блокируем event loop).
- Чтение Markdown — `aiofiles` (async).
- В `search_chunks_by_embedding` используется расширенный пул (**`internal_k`**) и последующая обрезка до `n_results`.

# 9 🧠 Модуль OpenAI и маршрутизация (`openai_utils.py`)

**Назначение:** инкапсулирует вызовы OpenAI (чат + эмбеддинги), маршрутизацию сценариев (`FAQ` / `ANALYTICS` / `REGISTRATION` / `OFFTOPIC`), работу с RAG-tools и **строгую валидацию** JSON-ответа LLM.

**Жёсткий JSON-контракт ответа LLM**
Ассистент **всегда** возвращает один JSON-объект.

**Требования:**
- scenario, action, reply — всегда присутствуют.
- fields — объект (может быть пустым).
- stage используется только в REGISTRATION.

**Основная функция**
- ask_openai(input, context, history, ...) -> dict
- Делает до 5 итераций tool-вызовов (auto tool-choice), затем возвращает финальный JSON.
- Если context.faq_article уже есть (confirm), инструменты FAQ не вызываются повторно.
- Нормализует ответ: гарантирует типы, переносит action в fields.action (если нужно), чистит stage вне REGISTRATION.
- Для ANALYTICS: таблица хранится локально и прикрепляется в dashboard.table только в финале (экономия токенов).

**Инструменты (Tools):**
read_file(path) — чтение сценарных .md из жёсткого белого списка:
- ~/ai-assistant/backend/scenario/scenario_faq.md,
- ~/ai-assistant/backend/scenario/scenario_analytics.md,
- ~/ai-assistant/backend/scenario/scenario_registration.md
- Кэш: TTL=2 часа, до 6 записей (LRU-подобно), инвалидация по mtime.

- faq_search(query, n_results=5, last_article_id=None) — семпоиск по FAQ (до 5 статей: article_id, title, summary, tags, meta_tags); last_article_id исключается из выдачи.
- faq_get_by_id(article_id) — полная FAQ-статья. Источник: Chroma (метаданные/док) или fallback на markdown (fulltext).
- analytics_titles_search(query, n_results<=25) — shortlist названий ниш (3–25 кандидатов) по эмбеддингам заголовков.
- analytics_titles_random(n=5..30) — случайные названия ниш из файла заголовков.
- analytics_search(query) — резервный семпоиск по аналитической коллекции.
- analytics_get_by_niche(niche) — точечная выдача аналитики по названию ниши:
    - where-фильтр → при неуспехе полный просмотр → при неуспехе fuzzy (SequenceMatcher, порог ≥ 0.72).
    - Возвращает {"analytic": {...}} (ключи: Бизнес ниша, analytics, table — при наличии).

Кэш заголовков аналитики (shortlist)

Источник заголовков: analytic/analytic_zagolovkov.md (_TITLES_PATH)

Кэш эмбеддингов: analytic/_titles_emb_cache.json (EMB_CACHE_PATH)

Инвалидация по mtime исходника; батчи эмбеддингов по 128 строк.

**Константы:**
- INTERNAL_POOL=25 (топ кандидатов по косинусу перед обрезкой),
- DEFAULT_TOOL_RETURN=25, MAX_TOOL_RETURN_HARD=25 (верхние лимиты выдачи),
- DEFAULT_SHOW=5 (ориентир «сколько показывать пользователю»).

# 10 🧵 Голосовой стек (STT/TTS). Celery-задачи
**(`backend/tasks/__init__.py`) Назначение:** точка инициализации подпакета задач и единый реэкспорт. Тут же — общий логгер задач и централизованная запись ошибок в файловый журнал.

- **OpenAI STT (Whisper)** — `openai==1.26.0` (пин `httpx==0.27.2`).
- **ElevenLabs TTS** — `elevenlabs==2.7.1`. Настроены 2 голоса (основной + резерв). Выход: `mp3` (web), `m4a` (macOS/iOS), `ogg` (Telegram).
- **Хранение/очистка** — файлы в `backend/media/audio`; автоочистка `scripts/cleanup_audio.sh` (файл должен иметь `+x`; таймер — раз в N дней).
- **Очереди** — фоновые задачи разделены (2 воркера `text`, 2 воркера `audio`).
- Количество воркеров в зависимости от ресурсов вашего сервера.

**Константы, тайминги, SLA (см. `audio_constants.py`)**

**Публичные объекты (реэкспорт)**
  - `stt_task` — распознавание речи (из `backend/tasks/stt.py`)
  - `tts_task` — синтез речи (из `backend/tasks/tts.py`)
  - `process_text` — обработка текстового сообщения/диалога (из `backend/tasks/chat.py`)
  - `log_error_to_db` — запись ошибок в `ErrorLog` (из `backend/utils/error_log_utils.py`)

**-  TTS-задача (`tasks/tts.py`)**
**Назначение:** Celery-задача `tts_task` генерирует голосовой ответ через ElevenLabs (очередь: `audio`, с SLA/тайм-лимитами).

**-  STT-задача (`backend/tasks/stt.py`)**
**Назначение:** Celery-задача `stt_task` распознаёт речь через OpenAI Whisper (очередь: `audio`, контроль SLA и тайм-лимитов).

**-  Константы аудио (`backend/utils/audio_constants.py`)**
**Назначение:** единая точка правды для аудиостека (STT/TTS): форматы, лимиты, SLA/тайм-ауты, пресеты кодеков, ретраи.

**Поддерживаемые форматы**
- **Вход/хранение:** `ALLOWED_EXTENSIONS = {".mp3", ".ogg", ".m4a", ".webm"}`
- **Выдача TTS (клиентам):** `SUPPORTED_TTS_FORMATS = ["mp3","ogg","m4a","webm"]`
> ℹ️ **Сейчас в коде `tts_task` разрешены только `mp3`/`ogg`**, а `stt_task` принимает только `mp3`/`ogg`.

**-  STT-утилиты (`utils/stt_utils.py`)**
**Назначение:** ассистируют голосовому стеку (Whisper/STT): сохраняют загруженные файлы, валидируют и **безопасно** конвертируют `ogg→mp3` через ffmpeg, отправляют аудио в OpenAI Whisper, логируют ошибки/метрики.

**-  TTS-утилиты (`utils/tts_utils.py`)**
**Назначение:** универсальный слой для генерации речи через **ElevenLabs**. Работает и из Celery, и из FastAPI: синхронный вызов SDK + асинхронная обёртка (`run_in_executor`). Единый контракт возврата (успех/фолбэк), логирование, SLA, хранение файлов.

**-  Chat-задача (`tasks/chat.py`)**
**Назначение:** обрабатывает текстовые запросы (чат) с вызовом LLM через `openai_utils.ask_openai`, сохраняет в БД вход/выход, меряет SLA и логирует ошибки.

**Конфигурация Celery (`celeryconfig.py`)**
**Назначение:** единая конфигурация Celery для Leadinc AI Assistant.  
Брокер и result backend — **Redis** (адрес берётся из `.env.backend` → `backend.config.REDIS_URL`).

**-  Celery-воркер (`celery_worker.py`)**
**Назначение:** основной процесс Celery-воркера Leadinc AI. Подхватывает конфиг из `backend.celeryconfig`, регистрирует задачи (`chat`, `stt`, `tts`) и логирует их жизненный цикл.

**-  Базовые аудио-утилиты (`utils/audio_utils.py`)**
**Назначение:** единая точка работы с аудио для FastAPI и Celery (STT/TTS): валидация формата/размера/длительности, конвертации (`mp3/ogg/m4a`), очистка старых файлов, логирование.

**Зависимости и требования**
- Требуются `ffmpeg` и `ffprobe` в `PATH` (используются через `ffmpeg-python` и `pydub.utils.mediainfo`)
- Логи — общий `leadinc-backend`; ошибки дополнительно идут в файловый журнал `ErrorLog` (`log_error_to_db`)

# 11🔌 Внешние сервисы
**Code-service:**
- `POST /api/generate-code` — выдача одноразового 6-значного кода (TTL 10 мин; хранение в Redis).
- `POST /api/verify-code` — проверка и «сжигание» кода.
- Используется в регистрации (Stage 1→2).

**Nginx:**
- Проксирует `/ai/*` и `/auth/*` → FastAPI
- Раздаёт `/static/` и `/media/`.
- Включены заголовки COOP/COEP (из сохранённого конфига).
- 80 → 443 редирект. Для тестов — самоподписанный SSL.

**CORS:**
- Разрешены домены из `ALLOWED_ORIGINS`.


# 12 📡 API обзор
- `POST /ai/chat` — центральная точка диалога (маршрутит сценарии; RAG, лимиты, стадии).
- `GET /health` — health-check
- - `GET /auth/users/me` — проверка текущего пользователя  
  ↳ `401` → `{"is_authenticated": false}`; при успехе → профиль `{is_authenticated, id, login}`
- `POST /auth/jwt/login_custom` — кастомный логин  
  - `sessionid` — **12 часов**, `HttpOnly`, `Secure=!DEBUG`, `SameSite=Strict|Lax`
  - `fastapiusersauth` — **JWT (7 дней)**, те же атрибуты
  > JWT-стратегия: `JWTStrategy(secret=SECRET, lifetime_seconds=604800, token_audience="fastapi-users")`- `POST /auth/register` — регистрация

- POST /ai/voice_upload — создаёт stt_task и возвращает task_id

- `POST /auth/jwt/logout` — логаут  
  Очищает Redis-состояние сессии и удаляет куки: `sessionid`, `fastapiusersauth`, `SESSION_COOKIE_NAME` (через `delete_cookie` + пустая кука `max_age=0`).

**Вспомогательные (внешний микросервис code-service):**
- `POST /api/generate-code` — выдача одноразового кода
- `POST /api/verify-code` — верификация кода

# 13 🗄️ ORM модели
**Назначение: единый слой данных для пользователей, сессий, сообщений и аудиособытий, чтобы чат/голос, RAG и стадийная логика жили на общей историях и лимитах.**

Как устроено:
- PK всех таблиц — UUID (UUID(as_uuid=True)).
- Время — DateTime без TZ; трактуем как UTC (используем datetime.utcnow()).
- Асинхронный доступ в FastAPI (engine asyncpg), синхронный — для Celery/миграций (psycopg2).
- Тяжёлые поля (метаданные, usage, трассинг) — JSONB.
- Индексация по «горячим» фильтрам: session_id, user_id, created_at, status.
- База: PostgreSQL 14. Миграции — Alembic (каталог alembic/versions).
- **ORM:** SQLAlchemy **2.x**, декларативные модели.

# 14 🧵 Async / Sync-сессии

- Основное приложение — **async** (FastAPI, async SQLAlchemy engine/session).
- **Celery** и **Alembic** используют **sync-сессии** (см. `backend/database.py`).
- **Правило:** не миксовать async-сессию в Celery и наоборот.  
  Для фоновых задач и миграций импортируй **sync** sessionmaker/engine.

# 15 🗃️ Модуль БД (`database.py`): движки и сессии

**Назначение:** единая точка создания двух SQLAlchemy-движков и фабрик сессий:

- **Async для FastAPI**  
  - `engine = create_async_engine(DATABASE_URL, echo=False)`  
  - `SessionLocal = sessionmaker(class_=AsyncSession, expire_on_commit=False, autoflush=False, autocommit=False)`  
  - Драйвер: **asyncpg**  
  - DSN: `postgresql+asyncpg://{USER}:{PASS}@{HOST}:{PORT}/{DB}`

- **Sync для Celery/CLI/Alembic**  
  - `engine_sync = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True, pool_recycle=3600)`  
  - `SessionLocalSync = sessionmaker(expire_on_commit=False, autoflush=False, autocommit=False)`  
  - Драйвер: **psycopg2**  
  - DSN: `postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB}`

- **База ORM:**  
  - `Base = declarative_base()` (используется в `models.py`)

**ENV-переменные (из `backend.config` / `.env.backend`):**  
`POSTGRES_HOST`, `POSTGRES_PORT` (**0000**), `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.

> В прод/стендах проверь, что `POSTGRES_PORT=....` в `.env.backend` и конфигурациях Alembic/systemd. Несоответствие порта ⇒ `Connection refused`.

# 16 🧬 Миграции БД (Alembic)

**Назначение:** версионирование схемы БД (PostgreSQL). Конфиг — `alembic.ini`, скрипты — в `alembic/` (ревизии в `alembic/versions/`).

** 🔧 Ключевые настройки `alembic.ini`**
- `script_location = %(here)s/alembic` — каталог миграций
- `prepend_sys_path = .` — добавляет корень репозитория в `PYTHONPATH` (чтобы `env.py` мог импортировать `backend.*`)
- `path_separator = os` — кроссплатформенный разделитель
- `sqlalchemy.url =` **(пусто)** — DSN подставляется **динамически** из `env.py`/переменных окружения. Секреты в git **не** храним.

** 🔗 Откуда берётся DSN**
Рекомендуется собирать DSN в `alembic/env.py` из переменных окружения:

**Пример sync-DSN (psycopg2) для Alembic:**
  - postgresql+psycopg2://....:***@000.0.0.0:0000/database_name

**сгенерировать ревизию по изменениям моделей**
alembic revision --autogenerate -m "описание изменений"

**применить все миграции**
alembic upgrade head

**откатиться на одну ревизию**
alembic downgrade -1

# 17 🧰 Модуль конфигурации (`config.py`)
**Назначение:** единая точка чтения переменных окружения и их типизации/дефолтов для всего бэкенда.

**Последовательность загрузки:**
- Пытается загрузить `/.env.backend` по пути: `/ai-assistant/backend/.env.backend`

**Обязательные секреты**
- `SECRET` — основной секрет приложения (JWT, reset/verify токены)
- `SECRET_ADMIN_TOKEN` — служебный админ-токен (если задействован)

**RAG / ChromaDB**
- `CHROMA_HOST` (дефолт `localhost`), `CHROMA_PORT` (int, дефолт)
- Нужны RAG-утилитам и health-check’у коллекций

**Веб-настройки и безопасность**
- `ALLOWED_HOSTS` — CSV-строка доменов (парсится в список)
- `CORS_ORIGINS` — **CSV** источников CORS (указывайте со схемой: `https://example.com`)
- `SESSION_COOKIE_NAME` — дефолт `sessionid`

**Почта**
- `EMAILS_FROM_EMAIL` / `EMAILS_FROM_NAME`
- `SMTP_HOST`, `SMTP_PORT` (int, дефолт), `SMTP_USER`, `SMTP_PASSWORD`
- `EMAIL_RESET_TOKEN_EXPIRE_HOURS`
- `EMAIL_TEMPLATES_DIR` — дефолт `./email-templates`
- `SUPPORT_EMAIL` — если не задан, берётся из `EMAILS_FROM_EMAIL`

**Логирование/мониторинг/метрики**
- `LOG_LEVEL` — дефолт `INFO`
- `SENTRY_DSN` — если используется
- `ADMIN_EMAIL` — для служебных уведомлений
- `GA_MEASUREMENT_ID`, `METRIKA_ID` — прокидываются в заголовки middleware
- `ENVIRONMENT` — дефолт `production`
- `TIMEZONE` — дефолт `Europe/Moscow`

**Формат и парсинг значений**
- Булевы: `os.getenv(..., "false").lower() == "true"`
- Порты (`SMTP_PORT`, `CHROMA_PORT`) приводятся к `int`

**Рекомендации**
- **Единый источник правды:** значения в `.env.backend` должны совпадать с `.env.docker` и тем, что проброшено в systemd/compose (особенно `POSTGRES_PORT=0000`)
- **Безопасность:** корректные `CORS_ORIGINS/ALLOWED_HOSTS` в проде
- **Отладочные `print(...)`:** временно включены при импорте модуля — **убрать в проде** или обернуть в `if DEBUG:`

# 18 👤 User Manager (fastapi-users)
**Назначение:** управляет жизненным циклом пользователя (регистрация, восстановление пароля, верификация) и используется как зависимость при инициализации `FastAPIUsers` (см. `auth.py`).

**Хранилище и сессии**
- Бэкэнд хранения: `SQLAlchemyUserDatabase(session, User)`
- Сессия: асинхронная из `SessionLocal`
- Фабрика зависимости: `get_user_manager()` — **async генератор**, каждый вызов открывает и корректно закрывает сессию (`async with SessionLocal()`)

**Секреты/токены**
- Токены fastapi-users используют **единый `SECRET`**:
  - `verification_token_secret = SECRET`
- Тот же `SECRET` используется для **JWT** (см. `auth.py`).
- ⚠️ Ротация `SECRET` должна планироваться с учётом **всех** типов токенов (JWT + reset + verify).

**Хуки жизненного цикла (callbacks)**
- `on_after_register(user, request)` → сейчас пишет в stdout: `✅ Зарегистрирован пользователь: {email}`
- `on_after_forgot_password(user, token, request)` → stdout: `🔐 ... токен: {token}`
- `on_after_request_verify(user, token, request)` → stdout: `📩 ... токен: {token}`

**Прод-заметка (безопасность)**
- Текущая реализация **логирует токены** в stdout. Для прод-режима заменить на:
  - отправку письма (SMTP/почтовый сервис) через `email_utils.py`, или
  - структурированное логирование **без значений токенов** (маскирование).

**Где используется**
- Экспортируется фабрика `get_user_manager()` → передаётся в `FastAPIUsers` в `auth.py`.
- Через `fastapi_users` поднимаются стандартные `/auth/*` роуты, доступен `current_user(...)` и др.

# 19🧯 **Централизованный error-логгер (`error_log_utils.py`)**
**Назначение:** единая точка записи **структурированных JSON-ошибок** для всего проекта (API, Celery-таски, утилиты). Пишет в **файловый журнал**, а не в БД.

- Файл журнала: **`/srv/leadinc-logs/error_log.log`**
- Логгер модуля: **`leadinc-errorlog`**
- Инициализация при импорте: базовая настройка `logging` (см. грабли ниже)

# 20 🧱 Грабли и частые ошибки

- JWT + HTTPS: в браузере cookie могут не сохраниться на http → локально ставь SECURE_COOKIE=false, для боевого домена — true и реальный HTTPS.
- Самоподписанный сертификат: часть браузеров/клиентов может блокировать запросы.
- ChromaDB: рассинхрон клиента/сервера ломает поиск.
- Alembic: битые миграции → очисти alembic/versions, проверь alembic_version в БД, пересобери ревизии.
- Кеш заголовков аналитики: _titles_emb_cache.json генерится автоматически.
- Размер аудио для STT: лимитируй длительность/размер (иначе таймауты и рост стоимости).
- CORS: добавь TEST_FRONTEND_ORIGIN для тест-клиента, иначе браузер зарежет запросы.
- **Async/Sync-сессии:** Celery/Alembic должны использовать **sync**-сессию. Ошибка “greenlet/await” = где-то подключили не тот sessionmaker.

- **Разные очереди Celery:** убедись, что `celery-text.service` и `celery-audio.service` реально слушают свои очереди.
- **Systemd-юниты:** старт/рестарт бэка и ChromaDB только через `systemctl` (иначе можно получить 2 инстанса и шаткий порт).

- **SQL echo в проде:** `echo=True` генерирует шумные логи и потенциально раскрывает SQL/данные.
- **Два источника DSN:** backend собирает из `POSTGRES_*`, alembic — из `DATABASE_URL`. Держите их **консистентными**.
- **Session TTL vs JWT TTL:** `sessionid` живёт **12ч**, а JWT — **7д**. Пользователь может иметь валидный JWT при истёкшей сессии → корректно обрабатывайте пересоздание `sessionid`.

# 21 🖼️ Скриншоты поведения
<img width="2559" height="1227" alt="image" src="https://github.com/user-attachments/assets/ab0990ba-b722-4f30-b6e1-95e1bb94c66d" />
<img width="2559" height="1232" alt="image" src="https://github.com/user-attachments/assets/2ddee8fb-dce9-4394-b298-b3b22772ec9b" />
<img width="2559" height="1228" alt="image" src="https://github.com/user-attachments/assets/294e5152-a52d-4cef-81d4-29abfae4a98e" />

# 22 📜 Лицензия

Проект распространяется под MIT License — разрешено всем использовать, изменять и распространять код.
Автор не несёт ответственности за использование кода любыми лицами.

**MIT-лицензия**

Авторское право (c) 2025 Блошко Константин Дмитриевич

Настоящим разрешается, бесплатно, любому лицу, получившему копию данного
программного обеспечения и сопутствующей документации (далее — «Программное обеспечение»),
без ограничений использовать Программное обеспечение, включая без ограничений права на
использование, копирование, изменение, слияние, публикацию, распространение, сублицензирование
и/или продажу копий Программного обеспечения, а также лицам, которым предоставляется это
Программное обеспечение, при соблюдении следующих условий:

Вышеупомянутое уведомление об авторском праве и настоящее уведомление о разрешении
должны быть включены во все копии или значимые части Программного обеспечения.

Программное обеспечение предоставляется «КАК ЕСТЬ», без каких-либо гарантий, явных или
подразумеваемых, включая, помимо прочего, гарантии товарной пригодности, соответствия
конкретной цели и ненарушения прав. Ни в коем случае авторы или правообладатели не несут
ответственности по каким-либо требованиям, убыткам или иным обязательствам, будь то по договору,
деликту или иным образом, возникшим из, из-за или в связи с Программным обеспечением
или использованием Программного обеспечения, либо иными действиями с Программным обеспечением.


# 23 Контакт 
Email: kdmitrievich1994@gmail.com
Telegram: Konstantikii

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


This README is provided in two languages: RU and EN. The section below is the EN.

# 1. Leadinc AI Assistant — a smart manager-consultant server for B2B sales
<img width="2559" height="1238" alt="image_2025-09-12_13-02-56" src="https://github.com/user-attachments/assets/b75b51a7-7610-464b-880d-0ddcce7102ee" />

For business: the assistant helps sell faster and more precisely. It’s like a friendly manager who:

explains how the service works and where to start;

instantly answers any questions about the services;

runs quick registration;

delivers analytics for 1 out of 430 business niches;

reduces time spent on explanations, qualification and clarifications → the team gets distracted less, leads are processed faster, fewer “leaks” across funnel stages.

# 2. 🧭 TL;DR

What it is: Leadinc assistant FastAPI backend with RAG (2 collections: FAQ/Analytics), voice (STT/TTS), Redis limits, data in Postgres, vector search in ChromaDB, and Celery workers.

Why it matters for business: the assistant acts like a friendly manager — quickly explains the product, handles registration, answers FAQs, provides analytics for 1 of 430+ niches → less routine for the team, faster lead processing, lower funnel loss.

LLM scenarios: REGISTRATION, FAQ, ANALYTICS, OFFTOPIC. For each request — exactly one RAG collection is used (no mixing).

Run: locally via venv (Docker is optional/for infrastructure if desired).

RAG data: private (not included in the public repository).

Status: ALPHA test (self-signed HTTPS for tests).

Where to look: ⚙️ Quick Start → 🗂️ Structure → 🧩 Architecture → 🎯 Scenarios.

# 3 ⚙️ Quick Start

1) Clone
git clone https://github.com/Kentokpro/gpt-assistant.git

cd ai-assistant

2) Python venv
source ~/ai-assistant/backend/venv/bin/activate

3) Dependencies
pip install -r ai-assistant/backend/requirements.txt

4) Environment variables
Create ai-assistant/backend/.env.backend.

5) Database & migrations
alembic upgrade head

6) (If required) run ChromaDB separately (version from requirements.txt)
example: chroma run --host 000.0.0.0 --port 0000

7) Start the app
uvicorn backend.main:app --host 0.0.0.0 --port 0000 --reload

8) Test front
Open frontend_test/index.html in your browser (locally) and verify connectivity to the API.

# 4. 📂 Repository structure:
<img width="1455" height="1147" alt="image_2025-09-12_13-17-12" src="https://github.com/user-attachments/assets/a24d0dcf-b8bd-40a5-af2a-5643d7883ad1" /> <img width="1455" height="635" alt="image_2025-09-12_13-17-44" src="https://github.com/user-attachments/assets/5f9a7bc0-e6fa-4cd4-aa40-3870cff44e27" />

# 5. 🧩 Architecture & key logic

Infrastructure (ALPHA):

OS: Ubuntu 22.04.5 LTS (jammy), Python 3.10.12

Reverse-proxy: Nginx (443/HTTPS; self-signed for tests)

Backend: FastAPI (Uvicorn) — gpt-backend.service on 000.0.0.0:0000

DB: PostgreSQL 14 (0000)

Cache/Rate-limit/State: Redis 6.0.16 (0000)

Vector DB: ChromaDB (HTTP 0000) — leadinc-chroma.service

Queues: Celery — 2 text workers, 2 audio workers (separate queues)

Codes: code-service (0000) — one-time 6-digit codes (TTL 10 min)

Static test front: /var/www/leadinc/static/chat.html

Test Front (HTML+JS):

POST /ai/chat (text → text/voice),

POST /ai/voice_upload (file → STT),

POST /ai/tts (text → audio).
Requires correct CORS (allowed origin) and cookie installation in the browser (see SECURE_COOKIES/SameSite).

Scenario routing: the assistant selects one of 4 scenarios per request. Depending on the scenario, it uses and queries exactly one RAG collection.
Assistant scenarios:

FAQ → collection faq_leadinc (RAG)

ANALYTICS → collection analytics_leadinc (RAG)

REGISTRATION → stage funnel 1→3, one-time codes (code-service), history migration by user_id

OFFTOPIC → small talk with limits

Exactly one RAG collection (FAQ or Analytics) is used per request, no mixing.

Stage logic: applied during registration guest (1–2) → registration → authorized (3).

Voice: upload → STT → response + optional TTS (file is saved and auto-cleaned).

# 6. 🎯 Scenarios & capabilities

Value: quick start, fewer “how it works” questions, ready answers and niche analytics — faster to sales.

REGISTRATION — Stage funnel (1→3): confirmation code → city/niche → auto-registration. Creates a user, links the current session, issues a JWT, offers to proceed to analytics.

FAQ — Answers about the service (how it works, terms, launch, what’s included). Source — faq_leadinc (RAG).

ANALYTICS — Provides analytics for the selected niche from analytics_leadinc (RAG).

OFFTOPIC — Just small talk, gently guides the dialog back to the goal (configurable limits and anti-flood).

Technically:

Hybrid RAG: ChromaDB, 2 collections (FAQ/Analytics), niche titles are cached.

Auth/limits: JWT cookie, separate session cookie, stages/quotas and anti-flood in Redis. Active subscription check (whitelist e-mail — bypass).

Voice: STT (Whisper) and TTS (ElevenLabs), auto-cleanup via timer.

Test front: static page to check the API (no production features).

# 7. 🔐 Environment variables (example backend/.env.backend)

JWT & general secrets
SECRET=(secret)

SECRET_ADMIN_TOKEN=(secret)

OPENAI_API_KEY=(secret)

ELEVENLABS_API_KEY=(secret)

Primary voice
ELEVENLABS_VOICE_ID=(secret)

Fallback voice substituting the primary
ELEVENLABS_FALLBACK_VOICE_ID=(secret)

Redis
REDIS_URL=redis://localhost:0000

Postgres connection
DATABASE_URL=postgresql+asyncpg://(secret):(secret)@000.0.0.0:(port)/(database_name)

POSTGRES_USER=(user_name)

POSTGRES_PASSWORD=(secret)

POSTGRES_DB=(database_name)

POSTGRES_HOST=000.0.0.0

POSTGRES_PORT=(port)

ChromaDB (RAG)
CHROMA_HOST=localhost

CHROMA_PORT=0000

Security & sessions
SECURE_COOKIES=false

CSRF_ENABLED=true

SESSION_COOKIE_NAME=sessionid

ALLOWED_HOSTS=yourdomain.com,localhost,000.0.0.0

CORS_ORIGINS=https://yourdomain.com
, http://localhost:0000

DEBUG=true

# 8 📚 RAG storage (ChromaDB) (chroma_utils.py)

Purpose: a thin wrapper over ChromaDB (HTTP) for RAG: connection, embedding search, metadata filters, reading the “full article” from Markdown (fallback), listing collections.

Collections: faq_leadinc and analytics_leadinc. Source .md files are private → not included in the public repository.

Chunking/indexing: scripts_ChromaDB/scripts_ChromaDB.py (FAQ) and plan/description analytic/script_2_collection_RAG.md (Analytics).

Analytics titles cache: analytic/_titles_emb_cache.json.

Rule: per request the assistant uses exactly one collection (FAQ or Analytics), no mixing.

Updating the corpus: replace .md → re-generate embeddings → restart backend.

functions:

connect_to_chromadb() -> chromadb.HttpClient
Creates the HTTP client. Errors are logged and re-raised.

get_collection(collection_name: str)
Returns a collection via the client.

search_chunks_by_embedding(query_emb: List[float], n_results=5, collection_name=..., filters=None) -> List[Dict]
Calls collection.query(..., include=["documents","metadatas","distances"]).
Internally requests internal_k = max(n_results, 12) to improve relevance; externally returns exactly n_results.
Normalizes tags to list: list | str | None → List[str].
Returns elements with schema: article_id, title, meta_tags, tags: List[str], summary, text.

filter_chunks(collection_name=..., article_id=None, meta_tags=None, tags=None, title=None, summary=None, limit=10) -> List[Dict]
Calls collection.get(where=filters, limit=limit). Returns the same schema as search_chunks_by_embedding.

get_full_article(article_id: str, articles_file=ARTICLES_FILE) -> str
Asynchronously reads the shared Markdown, splits by ---, searches for a block by pattern article_id: "<digits>".
Returns the full text of the block as-is.
Fallbacks: file missing → "Technical error: the knowledge base is temporarily unavailable."; not found → "Article not found."

Requires a numeric article_id in quotes.

list_collections() -> list
Returns the list of collections client.list_collections() and logs the result.

Async & performance

Potentially blocking Chroma calls — via run_in_executor (do not block the event loop).

Markdown reading — aiofiles (async).

search_chunks_by_embedding uses an extended pool (internal_k) and then truncates to n_results.

# 9 🧠 OpenAI module & routing (openai_utils.py)

Purpose: encapsulates OpenAI calls (chat + embeddings), scenario routing (FAQ / ANALYTICS / REGISTRATION / OFFTOPIC), RAG tools, and strict validation of the LLM JSON response.

Strict JSON contract of the LLM response
The assistant always returns a single JSON object.

Requirements:

scenario, action, reply — always present.

fields — an object (can be empty).

stage is used only in REGISTRATION.

Main function

ask_openai(input, context, history, ...) -> dict

Performs up to 5 tool-call iterations (auto tool-choice), then returns the final JSON.

If context.faq_article is already present (confirm), FAQ tools are not re-called.

Normalizes the response: guarantees types, moves action into fields.action (if needed), clears stage outside of REGISTRATION.

For ANALYTICS: the table is kept locally and attached to dashboard.table only at the very end (token savings).

Tools:
read_file(path) — read scenario .md from a strict allowlist:

~/ai-assistant/backend/scenario/scenario_faq.md,

~/ai-assistant/backend/scenario/scenario_analytics.md,

~/ai-assistant/backend/scenario/scenario_registration.md

Cache: TTL=2 hours, up to 6 entries (LRU-like), invalidation by mtime.

faq_search(query, n_results=5, last_article_id=None) — semantic search over FAQ (up to 5 articles: article_id, title, summary, tags, meta_tags); last_article_id is excluded from results.

faq_get_by_id(article_id) — full FAQ article. Source: Chroma (metadata/doc) or fallback to markdown (fulltext).

analytics_titles_search(query, n_results<=25) — shortlist of niche titles (3–25 candidates) by title embeddings.

analytics_titles_random(n=5..30) — random niche titles from the titles file.

analytics_search(query) — fallback semantic search over the analytics collection.

analytics_get_by_niche(niche) — targeted analytics by niche name:

where filter → if not found, full scan → if not found, fuzzy (SequenceMatcher, threshold ≥ 0.72).

Returns {"analytic": {...}} (keys: Бизнес ниша, analytics, table — if present).

Analytics titles cache (shortlist)

Titles source: analytic/analytic_zagolovkov.md (_TITLES_PATH)

Embeddings cache: analytic/_titles_emb_cache.json (EMB_CACHE_PATH)

Invalidation by source mtime; embeddings in batches of 128 lines.

Constants:

INTERNAL_POOL=25 (top candidates by cosine before truncation),

DEFAULT_TOOL_RETURN=25, MAX_TOOL_RETURN_HARD=25 (upper limits),

DEFAULT_SHOW=5 (hint “how many to show the user”).

# 10 🧵 Voice stack (STT/TTS). Celery tasks

(backend/tasks/__init__.py) Purpose: initialization point of the task subpackage and unified re-export. Also: shared task logger and centralized error writing to the file journal.

OpenAI STT (Whisper) — openai==1.26.0 (pin httpx==0.27.2).

ElevenLabs TTS — elevenlabs==2.7.1. Two voices are configured (primary + fallback). Output: mp3 (web), m4a (macOS/iOS), ogg (Telegram).

Storage/cleanup — files in backend/media/audio; auto-cleanup scripts/cleanup_audio.sh (file must have +x; timer — every N days).

Queues — background tasks split (2 text workers, 2 audio workers).

Number of workers depends on your server resources.

Constants, timings, SLA (see audio_constants.py)

Public objects (re-export)

stt_task — speech recognition (from backend/tasks/stt.py)

tts_task — speech synthesis (from backend/tasks/tts.py)

process_text — text message/dialog processing (from backend/tasks/chat.py)

log_error_to_db — write errors to the ErrorLog (from backend/utils/error_log_utils.py)

- TTS task (tasks/tts.py)
Purpose: Celery task tts_task generates a voice response via ElevenLabs (queue: audio, with SLA/time limits).

- STT task (backend/tasks/stt.py)
Purpose: Celery task stt_task recognizes speech via OpenAI Whisper (queue: audio, SLA and time-limit control).

- Audio constants (backend/utils/audio_constants.py)
Purpose: single source of truth for the audio stack (STT/TTS): formats, limits, SLA/timeouts, codec presets, retries.

Supported formats

Input/storage: ALLOWED_EXTENSIONS = {".mp3", ".ogg", ".m4a", ".webm"}

TTS output (clients): SUPPORTED_TTS_FORMATS = ["mp3","ogg","m4a","webm"]

ℹ️ Currently in code tts_task allows only mp3/ogg, and stt_task accepts only mp3/ogg.

- STT utils (utils/stt_utils.py)
Purpose: assist the voice stack (Whisper/STT): save uploaded files, validate and safely convert ogg→mp3 via ffmpeg, send audio to OpenAI Whisper, log errors/metrics.

- TTS utils (utils/tts_utils.py)
Purpose: a universal layer for speech generation via ElevenLabs. Works from both Celery and FastAPI: synchronous SDK call + async wrapper (run_in_executor). Unified return contract (success/fallback), logging, SLA, file storage.

- Chat task (tasks/chat.py)
Purpose: processes text requests (chat) by calling LLM via openai_utils.ask_openai, saves input/output to DB, measures SLA, and logs errors.

Celery configuration (celeryconfig.py)
Purpose: unified Celery configuration for Leadinc AI Assistant.
Broker and result backend — Redis (address from .env.backend → backend.config.REDIS_URL).

- Celery worker (celery_worker.py)
Purpose: main Celery worker process of Leadinc AI. Loads config from backend.celeryconfig, registers tasks (chat, stt, tts) and logs their lifecycle.

- Base audio utils (utils/audio_utils.py)
Purpose: unified audio tooling for FastAPI and Celery (STT/TTS): validation of format/size/duration, conversions (mp3/ogg/m4a), cleanup of old files, logging.

Dependencies & requirements

ffmpeg and ffprobe must be in PATH (used via ffmpeg-python and pydub.utils.mediainfo)

Logs — shared leadinc-backend; errors additionally go to the file journal ErrorLog (log_error_to_db)

# 11🔌 External services

Code-service:

POST /api/generate-code — issues a one-time 6-digit code (TTL 10 min; stored in Redis).

POST /api/verify-code — verifies and “burns” the code.

Used in registration (Stage 1→2).

Nginx:

Proxies /ai/* and /auth/* → FastAPI

Serves /static/ and /media/.

COOP/COEP headers enabled (from the saved config).

80 → 443 redirect. For tests — self-signed SSL.

CORS:

Allowed domains from ALLOWED_ORIGINS.

# 12 📡 API overview

POST /ai/chat — central dialog endpoint (routes scenarios; RAG, limits, stages).

GET /health — health check

GET /auth/users/me — check current user
↳ 401 → {"is_authenticated": false}; on success → profile {is_authenticated, id, login}

POST /auth/jwt/login_custom — custom login

sessionid — 12 hours, HttpOnly, Secure=!DEBUG, SameSite=Strict|Lax

fastapiusersauth — JWT (7 days), same attributes

JWT strategy: JWTStrategy(secret=SECRET, lifetime_seconds=604800, token_audience="fastapi-users")

POST /auth/register — registration

POST /ai/voice_upload — creates stt_task and returns task_id

POST /auth/jwt/logout — logout
Clears Redis session state and deletes cookies: sessionid, fastapiusersauth, SESSION_COOKIE_NAME (via delete_cookie + empty cookie max_age=0).

Helper (external microservice code-service):

POST /api/generate-code — issue a one-time code

POST /api/verify-code — verify the code

# 13 🗄️ ORM models

Purpose: a unified data layer for users, sessions, messages and audio events so that chat/voice, RAG, and stage logic live on shared histories and limits.

How it’s set up:

PK of all tables — UUID (UUID(as_uuid=True)).

Time — DateTime without TZ; treated as UTC (using datetime.utcnow()).

Async access in FastAPI (engine asyncpg), sync — for Celery/migrations (psycopg2).

Heavy fields (metadata, usage, tracing) — JSONB.

Indexing on “hot” filters: session_id, user_id, created_at, status.

DB: PostgreSQL 14. Migrations — Alembic (directory alembic/versions).

ORM: SQLAlchemy 2.x, declarative models.

# 14 🧵 Async / Sync sessions

Main app — async (FastAPI, async SQLAlchemy engine/session).

Celery and Alembic use sync sessions (see backend/database.py).

Rule: do not mix an async session in Celery and vice versa.
For background tasks and migrations import the sync sessionmaker/engine.

# 15 🗃️ DB module (database.py): engines & sessions

Purpose: single point for creating two SQLAlchemy engines and session factories:

Async for FastAPI

engine = create_async_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(class_=AsyncSession, expire_on_commit=False, autoflush=False, autocommit=False)

Driver: asyncpg

DSN: postgresql+asyncpg://{USER}:{PASS}@{HOST}:{PORT}/{DB}

Sync for Celery/CLI/Alembic

engine_sync = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True, pool_recycle=3600)

SessionLocalSync = sessionmaker(expire_on_commit=False, autoflush=False, autocommit=False)

Driver: psycopg2

DSN: postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB}

ORM base:

Base = declarative_base() (used in models.py)

ENV variables (from backend.config / .env.backend):
POSTGRES_HOST, POSTGRES_PORT (0000), POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB.

In prod/staging ensure POSTGRES_PORT=.... in .env.backend and in Alembic/systemd configs. Port mismatch ⇒ Connection refused.

# 16 🧬 DB migrations (Alembic)

Purpose: versioning of the DB schema (PostgreSQL). Config — alembic.ini, scripts — in alembic/ (revisions in alembic/versions/).

** 🔧 Key alembic.ini settings**

script_location = %(here)s/alembic — migrations directory

prepend_sys_path = . — adds repo root to PYTHONPATH (so env.py can import backend.*)

path_separator = os — cross-platform separator

sqlalchemy.url = (empty) — DSN is injected dynamically from env.py/environment variables. Do not store secrets in git.

** 🔗 Where DSN comes from**
Recommended to build DSN in alembic/env.py from environment variables:

Example sync DSN (psycopg2) for Alembic:

postgresql+psycopg2://....:***@000.0.0.0:0000/database_name

generate a revision from model changes
alembic revision --autogenerate -m "change description"

apply all migrations
alembic upgrade head

rollback one revision
alembic downgrade -1

# 17 🧰 Config module (config.py)

Purpose: single point for reading environment variables and their typing/defaults for the entire backend.

Load order:

Attempts to load /.env.backend at path: /ai-assistant/backend/.env.backend

Mandatory secrets

SECRET — main app secret (JWT, reset/verify tokens)

SECRET_ADMIN_TOKEN — service admin token (if used)

RAG / ChromaDB

CHROMA_HOST (default localhost), CHROMA_PORT (int, default)

Needed by RAG utils and the collections health-check

Web settings & security

ALLOWED_HOSTS — CSV string of domains (parsed into a list)

CORS_ORIGINS — CSV of CORS origins (specify with scheme: https://example.com)

SESSION_COOKIE_NAME — default sessionid

Mail

EMAILS_FROM_EMAIL / EMAILS_FROM_NAME

SMTP_HOST, SMTP_PORT (int, default), SMTP_USER, SMTP_PASSWORD

EMAIL_RESET_TOKEN_EXPIRE_HOURS

EMAIL_TEMPLATES_DIR — default ./email-templates

SUPPORT_EMAIL — if not set, taken from EMAILS_FROM_EMAIL

Logging/monitoring/metrics

LOG_LEVEL — default INFO

SENTRY_DSN — if used

ADMIN_EMAIL — for service notifications

GA_MEASUREMENT_ID, METRIKA_ID — injected into headers via middleware

ENVIRONMENT — default production

TIMEZONE — default Europe/Moscow

Format & parsing

Booleans: os.getenv(..., "false").lower() == "true"

Ports (SMTP_PORT, CHROMA_PORT) are cast to int

Recommendations

Single source of truth: values in .env.backend must match .env.docker and what’s injected into systemd/compose (especially POSTGRES_PORT=0000)

Security: correct CORS_ORIGINS/ALLOWED_HOSTS in prod

Debug print(...): temporarily enabled on module import — remove in prod or wrap with if DEBUG:

# 18 👤 User Manager (fastapi-users)

Purpose: manages user lifecycle (registration, password reset, verification) and is used as a dependency when initializing FastAPIUsers (see auth.py).

Storage & sessions

Storage backend: SQLAlchemyUserDatabase(session, User)

Session: async from SessionLocal

Dependency factory: get_user_manager() — async generator, each call opens and properly closes a session (async with SessionLocal())

Secrets/tokens

fastapi-users tokens use a single SECRET:

verification_token_secret = SECRET

The same SECRET is used for JWT (see auth.py).

⚠️ Rotating SECRET must be planned considering all token types (JWT + reset + verify).

Lifecycle hooks (callbacks)

on_after_register(user, request) → currently prints to stdout: ✅ User registered: {email}

on_after_forgot_password(user, token, request) → stdout: 🔐 ... token: {token}

on_after_request_verify(user, token, request) → stdout: 📩 ... token: {token}

Prod note (security)

Current implementation logs tokens to stdout. For prod, replace with:

sending mail (SMTP/mail service) via email_utils.py, or

structured logging without token values (masking).

Where used

Factory get_user_manager() is exported → passed into FastAPIUsers in auth.py.

Through fastapi_users, standard /auth/* routes are mounted; current_user(...) etc. are available.

# 19🧯 Centralized error logger (error_log_utils.py)

Purpose: single point for writing structured JSON errors for the whole project (API, Celery tasks, utilities). Writes to a file journal, not to DB.

Log file: /srv/leadinc-logs/error_log.log

Module logger: leadinc-errorlog

Initialization at import: basic logging setup (see pitfalls below)

# 20 🧱 Pitfalls & common errors

JWT + HTTPS: in the browser, cookies may not be saved over http → locally set SECURE_COOKIE=false; for a production domain — true and real HTTPS.

Self-signed certificate: some browsers/clients may block requests.

ChromaDB: client/server version mismatch breaks search.

Alembic: broken migrations → clear alembic/versions, check alembic_version in DB, rebuild revisions.

Analytics titles cache: _titles_emb_cache.json is generated automatically.

Audio size for STT: limit duration/size (otherwise timeouts and cost growth).

CORS: add TEST_FRONTEND_ORIGIN for the test client, otherwise the browser will drop requests.

Async/Sync sessions: Celery/Alembic must use sync session. “greenlet/await” error = the wrong sessionmaker was used somewhere.

Separate Celery queues: ensure celery-text.service and celery-audio.service actually listen to their queues.

systemd units: start/restart backend and ChromaDB only via systemctl (otherwise you may get 2 instances and a flaky port).

SQL echo in prod: echo=True produces noisy logs and can potentially expose SQL/data.

Two DSN sources: backend builds from POSTGRES_*, Alembic — from DATABASE_URL. Keep them consistent.

Session TTL vs JWT TTL: sessionid lives for 12h, JWT — 7d. A user can have a valid JWT while the session has expired → handle sessionid re-creation correctly.

# 21 🖼️ UI screenshots
<img width="2559" height="1227" alt="image" src="https://github.com/user-attachments/assets/ab0990ba-b722-4f30-b6e1-95e1bb94c66d" /> <img width="2559" height="1232" alt="image" src="https://github.com/user-attachments/assets/2ddee8fb-dce9-4394-b298-b3b22772ec9b" /> <img width="2559" height="1228" alt="image" src="https://github.com/user-attachments/assets/294e5152-a52d-4cef-81d4-29abfae4a98e" />
22 📜 License

The project is distributed under the MIT License — everyone is permitted to use, modify and distribute the code.
The author bears no responsibility for use of the code by any parties.

**MIT License**

Copyright (c) 2025 Konstantin Dmitrievich Bloshko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# 23 Contact

Email: kdmitrievich1994@gmail.com

Telegram: Konstantikii
