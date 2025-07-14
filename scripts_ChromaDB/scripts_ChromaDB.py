"""
Скрипт загрузки markdown-статей в ChromaDB с embedding через OpenAI.
- Читает статьи между ---
- Для каждой статьи делает chunking (4096 токенов с overlap = 0)
- Для каждого чанка делает embedding через OpenAI
- Каждый чанк содержит ВСЕ метаданные (article_id, title, summary, meta_tags, tags, chunk_id)
- Заменяет старую коллекцию на новую, чтобы не было “битых” чанков без метаданных
"""

import time
import openai
import os
import re
import ast
import tiktoken
import chromadb
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/backend/.env.backend"))

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
COLLECTION_NAME = "faq_leadinc"
MD_FILE = os.path.expanduser("~/ai-assistant/leadinc_Вопрос_ответ_готовый/FAQ_Leadinc.md")

# 1. Парсинг статей
def parse_articles(md_text):
    blocks = re.split(r'^---\s*$', md_text, flags=re.MULTILINE)
    articles = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header, *body = block.split('\n\n', 1)
        header_lines = header.strip().split('\n')
        meta = {}
        text = body[0] if body else ''
        for line in header_lines:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"')
            # Парсим теги в python-список
            if key == 'tags':
                try:
                    value = ast.literal_eval(value)
                    if not isinstance(value, list):
                        value = [value]
                except Exception:
                    value = []
            meta[key] = value
        meta['body'] = text.strip()
        articles.append(meta)
    return articles

# 2. Чанкинг статьи
def chunk_article(article, chunk_size=4096, overlap=0):
    enc = tiktoken.encoding_for_model("gpt-4-1106-preview")
    tokens = enc.encode(article['body'])
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_text = enc.decode(tokens[start:end])
        if chunk_text.strip():
            chunk = {
                "article_id": article.get("article_id"),
                "title": article.get("title"),
                "summary": article.get("summary"),
                "meta_tags": article.get("meta_tags"),
                "tags": article.get("tags", []),
                "chunk_id": chunk_id,
                "text": chunk_text,
            }
            chunks.append(chunk)
        chunk_id += 1
        start += chunk_size - overlap
    return chunks

# 3. Генерация embedding через OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    while True:
        try:
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print("Ошибка OpenAI:", e)
            print("Повтор через 5 секунд...")
            time.sleep(5)

# 4. Загрузка markdown-файла
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print(f"Загрузка статей из: {MD_FILE}")

    # УДАЛИ СТАРУЮ КОЛЛЕКЦИЮ (опционально, если не нужны старые чанки)
    client_db = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client_db.delete_collection(COLLECTION_NAME)
        print(f"Коллекция '{COLLECTION_NAME}' удалена для чистой загрузки!")
    except Exception as e:
        print(f"Коллекция не найдена или ошибка при удалении: {e}")

    md_text = load_markdown(MD_FILE)
    articles = parse_articles(md_text)

    all_chunks = []
    all_embeddings = []

    print(f"Найдено статей: {len(articles)}")
    for idx, article in enumerate(articles):
        chunks = chunk_article(article, chunk_size=4096, overlap=0)
        print(f"Статья [{idx+1}] '{article.get('title', '')}': {len(chunks)} чанков")
        for chunk in chunks:
            all_chunks.append(chunk)
            emb = get_embedding(chunk["text"])
            all_embeddings.append(emb)
    print(f"Всего чанков для загрузки: {len(all_chunks)}")

    # Подключение к ChromaDB
    collection = client_db.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=None
    )

    # Подготовка данных для ChromaDB
    ids = []
    docs = []
    metas = []
    for chunk in all_chunks:
        ids.append(f"{chunk['article_id']}_{chunk['chunk_id']}")
        docs.append(chunk["text"])
        metas.append({
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "summary": chunk["summary"],
            "meta_tags": chunk["meta_tags"],
            "tags": ", ".join(chunk["tags"]) if isinstance(chunk["tags"], list) else str(chunk["tags"]),  # <--- вот это исправление!
            "chunk_id": chunk["chunk_id"]
        })

    # Загрузка в ChromaDB с embedding
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=all_embeddings
    )

    print(f"Загружено чанков: {len(docs)} в коллекцию '{COLLECTION_NAME}' с embedding от OpenAI")
    print("Проверка: первые 3 чанка с метаданными из коллекции:")
    sample = collection.get(limit=3)
    for idx, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
        print(f"\nЧанк #{idx+1}:")
        print("Текст чанка:", doc[:80], "...")
        print("Метаданные:", meta)

