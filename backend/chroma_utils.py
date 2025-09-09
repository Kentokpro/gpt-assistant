
import chromadb
import logging
from typing import Optional, List, Dict, Any
import os
import aiofiles
import asyncio
import re
from backend.config import ANALYTICS_COLLECTION_NAME, FAQ_COLLECTION_NAME
from backend.config import CHROMA_HOST, CHROMA_PORT
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(anonymized_telemetry=False)
client = chromadb.Client(CHROMA_SETTINGS)

# Настройка логирования
logger = logging.getLogger("leadinc-chroma")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [chroma_utils] %(message)s'))
    logger.addHandler(handler)

ARTICLES_FILE = "/root/ai-assistant/leadinc_Вопрос_ответ_готовый/FAQ_Leadinc.md"

def connect_to_chromadb() -> chromadb.HttpClient:
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False)  # <- ВАЖНО
        )
        logger.debug("ChromaDB HttpClient создан успешно (telemetry=disabled).")
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
    collection_name: str = FAQ_COLLECTION_NAME,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Поиск чанков в ChromaDB по embedding с возвращением всех метаданных.
    :return: Список словарей: [{article_id, title, meta_tags, tags, summary, text}, ...]
    """
    loop = asyncio.get_event_loop()

    def normalize_tags(v):
        # Приводит tags/meta_tags к удобному виду: list -> list, str -> [str], None -> []
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [str(v)]

    def do_search():
        collection = get_collection(collection_name)
        
        try:
            internal_k = max(int(n_results or 5), 12)
            result = collection.query(
                query_embeddings=[query_emb],
                n_results=internal_k,
                where=filters if filters else None,
                include=["documents", "metadatas", "distances"]
            )
            
            docs  = (result.get("documents")  or [[]])[0] or []
            metas = (result.get("metadatas")  or [[]])[0] or []
            enriched = []
            for doc, meta in zip(docs, metas):
                article_id = (meta or {}).get("article_id", "unknown")
                enriched.append({
                    "article_id": article_id,
                    "title": (meta or {}).get("title", "") or "",
                    "meta_tags": (meta or {}).get("meta_tags", "") or "",
                    "tags": normalize_tags((meta or {}).get("tags")),
                    "summary": (meta or {}).get("summary", "") or "",
                    "text": doc or ""
                })
            out = enriched[: int(n_results or 5)]
            logger.info(f"Поиск по базе '{collection_name}', запрошено={n_results}, internal_k={internal_k}, возвращаем={len(out)}")
            return out
        except Exception as e:
            logger.error(f"Ошибка поиска в ChromaDB: {e}")
            raise

    return await loop.run_in_executor(None, do_search)

async def filter_chunks(
    collection_name: str = FAQ_COLLECTION_NAME,
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
