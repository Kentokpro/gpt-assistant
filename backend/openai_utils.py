from openai import AsyncOpenAI
from backend.config import OPENAI_API_KEY
import logging
import aiofiles
import uuid
from pathlib import Path

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger("leadinc-backend")

MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)

async def ask_openai(content, msg_type="text", file_bytes=None, scenario_image_path=None):
    try:
        if msg_type == "text":
            response = await client.chat.completions.create(
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
            # Сохраняем аудиофайл
            audio_temp_path = MEDIA_DIR / f"{uuid.uuid4()}.ogg"
            async with aiofiles.open(audio_temp_path, "wb") as f:
                await f.write(file_bytes)
            # Новый способ для ASYNC audio transcription
            with open(audio_temp_path, "rb") as f:
                stt_result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )
            text_query = stt_result.strip()
            # Чат с OpenAI
            response = await client.chat.completions.create(
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
            # Генерация TTS (заглушка)
            tts_filename = f"{uuid.uuid4()}.ogg"
            tts_path = MEDIA_DIR / tts_filename
            with open(tts_path, "wb") as f:
                f.write(b"")  # TODO: заменить на реальное аудио от TTS!
            return {
                "text": reply_text,
                "voice_url": f"/media/{tts_filename}",
                "usage": usage,
            }

        elif msg_type == "image":
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
