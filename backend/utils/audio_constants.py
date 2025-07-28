
"""
audio_constants.py — Глобальные константы для работы с аудиофайлами
Используется для унификации лимитов и поддерживаемых расширений по всему проекту.
"""

# --- Поддерживаемые расширения аудиофайлов ---
ALLOWED_EXTENSIONS = {".mp3", ".ogg", ".m4a", ".webm"}

SUPPORTED_TTS_FORMATS = ["mp3", "ogg", "m4a", "webm"]  # фронт может запросить любой из них

TTS_FORMAT_MAP = {
    "mp3":  "mp3_44100_128",     # для ElevenLabs API
    "ogg":  "ogg_44100_64",

    "m4a":  "aac_44100",         # для macOS .m4a (aac)
    "webm": "webm_44100_128"
}

DEFAULT_TTS_FORMAT = "mp3"

# --- Унифицированные константы для других модулей (прокидывать везде через импорт) ---
AUDIO_FORMAT = TTS_FORMAT_MAP[DEFAULT_TTS_FORMAT]    # "mp3_44100_128" — формат для фронта (gpt.leadinc)
OGG_FORMAT   = TTS_FORMAT_MAP["ogg"]                 # "ogg_44100_64"  — для Telegram-бота

TTS_SLA_LIMIT = 40           # SLA-порог, сек
TTS_SOFT_TIME_LIMIT = 50     # мягкий лимит задачи TTS
TTS_HARD_TIME_LIMIT = 60     # лимит времени для задачи TTS

STT_SLA_LIMIT = 40           # SLA-порог для STT
STT_SOFT_TIME_LIMIT = 50     # мягкий лимит задачи STT
STT_HARD_TIME_LIMIT = 60     # лимит времени для задачи STT

# --- Максимальная длительность аудиофайла (сек) ---
MAX_AUDIO_DURATION_SEC = 120   # 2 минуты

# --- Максимальный размер аудиофайла (МБ) ---
MAX_AUDIO_SIZE_MB = 10         # 15 мегабайт

# --- Количество попыток воркера ---
TTS_RETRY_COUNT = 3
