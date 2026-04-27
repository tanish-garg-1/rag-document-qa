import logging
from typing import List, Dict

from app.utils.constants import MAX_MEMORY_MESSAGES, INFERENCE_MEMORY_MESSAGES

logger = logging.getLogger(__name__)

# In-memory store (resets on server restart)
_memory: List[Dict[str, str]] = []


def add_message(role: str, content: str):
    """Add a message to memory. Role must be 'user' or 'assistant'."""
    global _memory
    _memory.append({"role": role, "content": content})

    # Keep only last MAX_MEMORY_MESSAGES
    if len(_memory) > MAX_MEMORY_MESSAGES:
        _memory = _memory[-MAX_MEMORY_MESSAGES:]

    logger.info(f"Memory updated. Total messages: {len(_memory)}")


def get_recent_history(n: int = INFERENCE_MEMORY_MESSAGES) -> List[Dict[str, str]]:
    """Return last n messages for inference."""
    return _memory[-n:]


def get_history_as_text(n: int = INFERENCE_MEMORY_MESSAGES) -> str:
    """Return last n messages formatted as plain text for prompt injection."""
    history = get_recent_history(n)
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def clear_memory():
    """Clear all memory."""
    global _memory
    _memory = []
    logger.info("Memory cleared.")


def get_memory_size() -> int:
    """Return current number of messages in memory."""
    return len(_memory)