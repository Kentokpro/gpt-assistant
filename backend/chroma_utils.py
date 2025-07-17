import chromadb
import logging
from typing import Optional, List, Dict, Any
import os
import aiofiles
import asyncio
import re

# Настройка логирования
logger = logging.getLogger("leadinc-chroma")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [chroma_utils] %(message)s'))
    logger.addHandler(handler)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
ARTICLES_FILE = "/root/ai-assistant/leadinc_Вопрос_ответ_готовый/FAQ_Leadinc.md"

def connect_to_chromadb() -> chromadb.HttpClient:
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        logger.debug("ChromaDB client создан успешно.")
        return client
    except Exception as e:
        logger.error(f"Ошибка подключения к ChromaDB: {e}")
        raise

def get_collection(collection_name: str):
    client = connect_to_chromadb()
    try:
        collection = client.get_collection(collection_name)
        logger.debug(f"Коллекция '{collection_name}' получена.")
        return collection
    except Exception as e:
        logger.error(f"Ошибка получения коллекции '{collection_name}': {e}")
        raise

async def search_chunks_by_embedding(
    query_emb: List[float],
    n_results: int = 5,
    collection_name: str = "faq_leadinc",
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Поиск чанков в ChromaDB по embedding с возвращением всех метаданных.
    :return: Список словарей: [{article_id, title, meta_tags, tags, summary, text}, ...]
    """
    loop = asyncio.get_event_loop()

    def do_search():
        collection = get_collection(collection_name)
        try:
            result = collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                where=filters if filters else None
            )
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            enriched = []
            for doc, meta in zip(docs, metas):
                enriched.append({
                    "article_id": meta.get("article_id", "unknown"),
                    "title": meta.get("title", ""),
                    "meta_tags": meta.get("meta_tags", ""),
                    "tags": meta.get("tags", []),
                    "summary": meta.get("summary", ""),
                    "text": doc
                })
            logger.info(f"Поиск по базе '{collection_name}', найдено: {len(enriched)}")
            return enriched
        except Exception as e:
            logger.error(f"Ошибка поиска в ChromaDB: {e}")
            raise

    return await loop.run_in_executor(None, do_search)

async def filter_chunks(
    collection_name: str = "faq_leadinc",
    article_id: Optional[str] = None,
    meta_tags: Optional[str] = None,
    tags: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Фильтрация чанков по метаданным.
    :return: Список словарей: [{article_id, title, meta_tags, tags, summary, text}, ...]
    """
    loop = asyncio.get_event_loop()

    def do_filter():
        collection = get_collection(collection_name)
        filters = {}
        if article_id: filters["article_id"] = article_id
        if meta_tags:  filters["meta_tags"] = meta_tags
        if tags:       filters["tags"] = tags
        if title:      filters["title"] = title
        if summary:    filters["summary"] = summary
        try:
            result = collection.get(
                where=filters if filters else None,
                limit=limit
            )
            docs = []
            for doc, meta in zip(result['documents'], result['metadatas']):
                docs.append({
                    "article_id": meta.get("article_id", "unknown"),
                    "title": meta.get("title", ""),
                    "meta_tags": meta.get("meta_tags", ""),
                    "tags": meta.get("tags", []),
                    "summary": meta.get("summary", ""),
                    "text": doc
                })
            logger.info(f"Фильтрация по базе '{collection_name}', найдено: {len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"Ошибка фильтрации в ChromaDB: {e}")
            raise

    return await loop.run_in_executor(None, do_filter)

async def get_full_article(
    article_id: str,
    articles_file: str = ARTICLES_FILE
) -> str:
    """
    Поиск полной статьи по article_id в markdown-файле (ручной режим, не через ChromaDB).
    """
    try:
        async with aiofiles.open(articles_file, "r", encoding="utf-8") as f:
            content = await f.read()
    except FileNotFoundError:
        logger.error(f"Файл с базой статей не найден: {articles_file}")
        return "Техническая ошибка: база знаний временно недоступна."
    except Exception as e:
        logger.error(f"Ошибка при чтении файла статей: {e}")
        return "Техническая ошибка при получении статьи."
    
    blocks = re.split(r'\n?---\n', content)
    for block in blocks:
        if not block.strip():
            continue
        match = re.search(r'article_id:\s*"(\d+)"', block)
        if match and match.group(1) == str(article_id):
            return block.strip()
    logger.warning(f"Статья с article_id={article_id} не найдена в общем markdown-файле.")
    return "Статья не найдена."

def list_collections():
    """Список коллекций ChromaDB (для отладки)."""
    client = connect_to_chromadb()
    try:
        cols = client.list_collections()
        logger.info(f"Список коллекций: {cols}")
        return cols
    except Exception as e:
        logger.error(f"Ошибка при получении списка коллекций: {e}")
        return []
