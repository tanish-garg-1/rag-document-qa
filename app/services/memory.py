import json
import logging
import os
from typing import List, Dict

from app.utils.constants import MAX_MEMORY_MESSAGES, INFERENCE_MEMORY_MESSAGES

logger = logging.getLogger(__name__)

# Path for JSON-based persistent memory storage
_MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "memory.json"
)


def _load_memory() -> List[Dict[str, str]]:
    """Load memory from disk. Returns empty list if file missing or corrupt."""
    try:
        if os.path.exists(_MEMORY_FILE):
            with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        logger.warning(f"Could not load memory file: {e}")
    return []


def _save_memory(memory: List[Dict[str, str]]):
    """Persist memory to disk using atomic write."""
    try:
        os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
        tmp = _MEMORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False)
        os.replace(tmp, _MEMORY_FILE)
    except Exception as e:
        logger.error(f"Could not save memory file: {e}")


def add_message(role: str, content: str):
    """Add a message to memory. Role must be 'user' or 'assistant'."""
    memory = _load_memory()
    memory.append({"role": role, "content": content})

    # Keep only last MAX_MEMORY_MESSAGES
    if len(memory) > MAX_MEMORY_MESSAGES:
        memory = memory[-MAX_MEMORY_MESSAGES:]

    _save_memory(memory)
    logger.info(f"Memory updated. Total messages: {len(memory)}")


def get_recent_history(n: int = INFERENCE_MEMORY_MESSAGES) -> List[Dict[str, str]]:
    """Return last n messages for inference."""
    if n <= 0:
        return []
    memory = _load_memory()
    return memory[-n:]


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
    _save_memory([])
    logger.info("Memory cleared.")


def get_memory_size() -> int:
    """Return current number of messages in memory."""
    return len(_load_memory())
