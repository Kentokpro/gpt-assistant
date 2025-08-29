"""
Скрипт нарезки и загрузки второй коллекции аналитики (таблицы Leadinc) в ChromaDB.
- Читает markdown с JSON-аналитиками (разделитель ---)
- Каждый JSON = отдельная таблица = 1 чанк (chunk)
- Чанк установлен максимально большой (лимитируется возможностями embedding-модели OpenAI)
- Генерирует embedding для каждого чанка (аналитики)
- Загружает в отдельную коллекцию ChromaDB
"""

import time
import openai
import os
import json
import chromadb
from dotenv import load_dotenv

# === 1. Конфиги ===
load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/backend/.env.backend"))

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
COLLECTION_NAME = "analytics_leadinc"  # ← Имя коллекции для аналитики
MD_FILE = "/root/ai-assistant/analytic/analytic_zagolovkov.md"
CHUNK_SIZE = 8191 * 4  # Максимум для OpenAI ada-002 ≈ 20 000 символов

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 2. Парсинг JSON-аналитик из markdown ===
def parse_analytics(md_text):
    blocks = [b.strip() for b in md_text.split('---') if b.strip()]
    analytics = []
    for block in blocks:
        try:
            analytic = json.loads(block)
            analytics.append(analytic)
        except Exception as e:
            print("Ошибка парсинга аналитики:", e)
            print("Проблемный блок:\n", block[:200], "\n---\n")
    return analytics

# === 3. Чанкинг (1 аналитика = 1 чанк) ===
def chunk_analytic(analytic, chunk_size=CHUNK_SIZE):
    # Вся аналитика — это 1 чанк (дополнительно: если вдруг больше лимита, делим по кускам)
    text = json.dumps(analytic, ensure_ascii=False)
    if len(text) <= chunk_size:
        return [text]
    # Если вдруг очень длинная — делим по кускам (редко потребуется)
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        chunk_id += 1
        start += chunk_size
    return chunks

# === 4. Генерация embedding через OpenAI ===
def get_embedding(text, model="text-embedding-ada-002"):
    while True:
        try:
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print("Ошибка OpenAI:", e)
            print("Повтор через 5 секунд...")
            time.sleep(5)

# === 5. Загрузка markdown-файла ===
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print(f"Загрузка аналитики из: {MD_FILE}")

    # Подключение к ChromaDB
    client_db = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client_db.delete_collection(COLLECTION_NAME)
        print(f"Коллекция '{COLLECTION_NAME}' удалена для чистой загрузки!")
    except Exception as e:
        print(f"Коллекция не найдена или ошибка при удалении: {e}")

    md_text = load_markdown(MD_FILE)
    analytics = parse_analytics(md_text)

    all_chunks = []
    all_embeddings = []
    ids = []
    metas = []

    print(f"Найдено аналитик: {len(analytics)}")
    for idx, analytic in enumerate(analytics):
        # Чанкуем (по факту — просто сериализуем JSON, 1 чанк)
        chunks = chunk_analytic(analytic, chunk_size=CHUNK_SIZE)
        for c_id, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            emb = get_embedding(chunk_text)
            all_embeddings.append(emb)
            meta_key = analytic.get("meta") or analytic.get("Бизнес ниша", "")
            metas.append({
                "meta": meta_key,
                "chunk_id": c_id
            })
            ids.append(f"{meta_key}_{c_id}_{idx+1}")

    print(f"Всего чанков для загрузки: {len(all_chunks)}")

    # Создаём коллекцию и загружаем чанки
    collection = client_db.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=None
    )

    collection.add(
        documents=all_chunks,
        metadatas=metas,
        ids=ids,
        embeddings=all_embeddings
    )

    print(f"Загружено чанков: {len(all_chunks)} в коллекцию '{COLLECTION_NAME}' с embedding от OpenAI")
    print("Первые 2 чанка с метаданными из коллекции:")
    sample = collection.get(limit=2)
    for idx, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
        print(f"\nЧанк #{idx+1}:")
        print("Текст чанка:", doc[:100], "...")
        print("Метаданные:", meta)
