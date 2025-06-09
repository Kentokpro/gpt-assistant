import openai
from backend.config import OPENAI_API_KEY
import logging
import aiofiles
import tempfile
import uuid
import os
from pathlib import Path

openai.api_key = OPENAI_API_KEY
logger = logging.getLogger("leadinc-backend")

MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)

async def ask_openai(content, msg_type="text", file_bytes=None, scenario_image_path=None):
    try:
        if msg_type == "text":
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            usage = {
                "model": response.model,
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            return {"text": reply, "usage": usage}

        elif msg_type == "voice":
            # Распознаём голосовое сообщение пользователя (STT)
            audio_temp_path = MEDIA_DIR / f"{uuid.uuid4()}.ogg"
            async with aiofiles.open(audio_temp_path, "wb") as f:
                await f.write(file_bytes)
            stt_result = await openai.Audio.atranscribe(
                model="whisper-1",
                file=str(audio_temp_path),
                response_format="text"
            )
            text_query = stt_result.strip()
            # Запускаем чат для текста
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[{"role": "user", "content": text_query}],
                max_tokens=1024,
            )
            reply_text = response.choices[0].message.content
            usage = {
                "model": response.model,
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            # Генерируем голосовой ответ (TTS через SpeechKit/Google/ваш TTS)
            # Имитация: создаём пустой файл (реализация зависит от выбранного сервиса)
            tts_filename = f"{uuid.uuid4()}.ogg"
            tts_path = MEDIA_DIR / tts_filename
            # Тут должна быть интеграция с реальным TTS — сгенерируй файл!
            # Сюда добавь код вызова Yandex SpeechKit/Google TTS и сохранения в tts_path
            with open(tts_path, "wb") as f:
                f.write(b"")  # Заглушка, замени на реальное аудио TTS!
            # Возвращаем ссылку на голосовой файл
            return {
                "text": reply_text,
                "voice_url": f"/media/{tts_filename}",
                "usage": usage,
            }

        elif msg_type == "image":
            # Выдаём заранее подготовленное изображение (не генерируем!)
            # Пример: path к картинке из базы или сценария
            if scenario_image_path and Path(scenario_image_path).exists():
                filename = Path(scenario_image_path).name
                return {
                    "image_url": f"/media/{filename}",
                    "usage": {}
                }
            return {
                "text": "Картинка не найдена в сценарии.",
                "usage": {}
            }

        else:
            return {
                "text": "Тип данных не поддерживается.",
                "usage": {}
            }
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise
