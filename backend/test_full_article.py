# test_full_article.py
import asyncio
from chroma_utils import get_full_article

async def main():
    # Введи проблемный article_id!
    article_id = "75"
    text = await get_full_article(article_id)
    print(f"\n>>> TEXT for article_id={article_id}:\n{text}\n")

if __name__ == "__main__":
    asyncio.run(main())
