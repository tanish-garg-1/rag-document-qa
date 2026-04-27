import logging
from typing import List

from google import genai

from app.utils.constants import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using Gemini Embeddings API."""
    embeddings = []
    try:
        for text in texts:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        logger.info(f"Embedded {len(texts)} texts.")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    try:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=query
        )
        logger.info("Query embedded successfully.")
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Query embedding error: {e}")
        raise