"""
TEST 08: CONVERSATION MEMORY
Tests in-memory conversation storage, rolling window, and history formatting.
No API key required.
Run: python tests/test_08_memory.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 08: CONVERSATION MEMORY")
print("=" * 60)

# --- 1. Import ---
print("\n--- Import memory ---")
try:
    from app.services.memory import (
        add_message,
        get_recent_history,
        get_history_as_text,
        clear_memory,
        get_memory_size,
    )
    print("  [PASS] memory module imported")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

from app.utils.constants import MAX_MEMORY_MESSAGES, INFERENCE_MEMORY_MESSAGES
print(f"  [INFO] MAX_MEMORY_MESSAGES      = {MAX_MEMORY_MESSAGES}")
print(f"  [INFO] INFERENCE_MEMORY_MESSAGES = {INFERENCE_MEMORY_MESSAGES}")

# --- 2. Clear and start fresh ---
print("\n--- Initial State ---")
clear_memory()
size = get_memory_size()
print(f"  Memory size after clear: {size}")
if size == 0:
    print("  [PASS] Memory starts empty after clear")
else:
    print(f"  [FAIL] Memory not empty after clear: {size} messages")

# --- 3. add_message ---
print("\n--- add_message() ---")
add_message("user", "Hello, what is RAG?")
add_message("assistant", "RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant documents with LLM generation.")
add_message("user", "How does it work?")
add_message("assistant", "It works by embedding your query, searching a vector store, and using retrieved context to answer.")

size = get_memory_size()
print(f"  Added 4 messages, memory size: {size}")
if size == 4:
    print("  [PASS] Message count correct")
else:
    print(f"  [FAIL] Expected 4 messages, got {size}")

# --- 4. get_recent_history ---
print("\n--- get_recent_history() ---")
history = get_recent_history(2)
print(f"  get_recent_history(2) returned: {len(history)} message(s)")
if len(history) == 2:
    print("  [PASS] Correct count returned")
    for i, msg in enumerate(history):
        print(f"    [{i}] role='{msg.get('role')}', content='{str(msg.get('content'))[:60]}'")
        if "role" not in msg:
            print(f"    [FAIL] Message {i} missing 'role' key")
        if "content" not in msg:
            print(f"    [FAIL] Message {i} missing 'content' key")
else:
    print(f"  [FAIL] Expected 2 messages, got {len(history)}")
    for m in history:
        print(f"    {m}")

# Check ordering — should be most recent last
if len(history) == 2:
    if history[-1].get("role") == "assistant":
        print("  [PASS] Last message is from assistant (most recent)")
    else:
        print(f"  [WARN] Last message role: {history[-1].get('role')}")

history_all = get_recent_history(100)
print(f"\n  get_recent_history(100) returned: {len(history_all)} (all messages)")
if len(history_all) == 4:
    print("  [PASS] All 4 messages returned when n > count")
else:
    print(f"  [WARN] Expected 4, got {len(history_all)}")

# --- 5. get_history_as_text ---
print("\n--- get_history_as_text() ---")
text = get_history_as_text(4)
print(f"  History text (n=4):\n{text}\n")
if "User:" in text or "user:" in text.lower():
    print("  [PASS] User role present in text")
else:
    print("  [WARN] 'User:' label not found in history text")
if "Assistant:" in text or "assistant:" in text.lower():
    print("  [PASS] Assistant role present in text")
else:
    print("  [WARN] 'Assistant:' label not found in history text")
if "Hello" in text:
    print("  [PASS] Message content visible in text")
else:
    print("  [FAIL] Expected content not found in text")

text_empty = get_history_as_text(0)
print(f"  get_history_as_text(0): '{text_empty}'")

# --- 6. Rolling window MAX_MEMORY_MESSAGES ---
print(f"\n--- Rolling Window (MAX={MAX_MEMORY_MESSAGES}) ---")
clear_memory()
# Add MAX + 5 messages
total_to_add = MAX_MEMORY_MESSAGES + 5
for i in range(total_to_add):
    role = "user" if i % 2 == 0 else "assistant"
    add_message(role, f"Message number {i} with some content for testing the rolling window.")

size = get_memory_size()
print(f"  Added {total_to_add} messages, memory size: {size}")
if size == MAX_MEMORY_MESSAGES:
    print(f"  [PASS] Rolling window working: capped at {MAX_MEMORY_MESSAGES}")
elif size == total_to_add:
    print(f"  [FAIL] Rolling window NOT working: stored all {total_to_add} messages (should cap at {MAX_MEMORY_MESSAGES})")
else:
    print(f"  [INFO] Memory size is {size} — check MAX_MEMORY_MESSAGES logic")

# Verify oldest messages dropped
history = get_recent_history(MAX_MEMORY_MESSAGES)
if history:
    first_content = history[0].get("content", "")
    print(f"  Oldest retained message: '{first_content[:60]}'")
    # Most recent message should be message number total_to_add-1
    last_content = history[-1].get("content", "")
    print(f"  Newest retained message: '{last_content[:60]}'")
    if f"Message number {total_to_add - 1}" in last_content:
        print("  [PASS] Newest message is present")
    else:
        print("  [WARN] Newest message not found at end of history")

# --- 7. clear_memory ---
print("\n--- clear_memory() ---")
clear_memory()
size = get_memory_size()
print(f"  Size after clear: {size}")
if size == 0:
    print("  [PASS] Memory cleared successfully")
else:
    print(f"  [FAIL] Memory not empty after clear: {size}")

# --- 8. Empty history text ---
print("\n--- get_history_as_text on empty memory ---")
clear_memory()
text = get_history_as_text(8)
print(f"  Empty memory text: '{text}'")
if not text or text.strip() == "":
    print("  [PASS] Empty memory returns empty/None text")
else:
    print(f"  [WARN] Non-empty text returned for empty memory: '{text}'")

# --- 9. INFERENCE_MEMORY_MESSAGES test ---
print(f"\n--- INFERENCE_MEMORY_MESSAGES = {INFERENCE_MEMORY_MESSAGES} ---")
for i in range(15):
    role = "user" if i % 2 == 0 else "assistant"
    add_message(role, f"Inference test message {i}")

infer_hist = get_recent_history(INFERENCE_MEMORY_MESSAGES)
print(f"  get_recent_history({INFERENCE_MEMORY_MESSAGES}) returns {len(infer_hist)} messages")
if len(infer_hist) == INFERENCE_MEMORY_MESSAGES:
    print(f"  [PASS] Correct inference window size")
else:
    print(f"  [INFO] Got {len(infer_hist)}, expected {INFERENCE_MEMORY_MESSAGES}")

clear_memory()
print("  [INFO] Memory cleared after test")

# --- 10. Persistence warning ---
print("\n--- PERSISTENCE WARNING ---")
print("  [WARN] Memory is stored in a Python global list — NOT persistent!")
print("  [WARN] All conversation history is LOST on server restart.")
print("  [WARN] This is a known architectural issue in the current codebase.")
print("  [INFO] Fix: Replace global list with Redis/SQLite/file-based storage.")

print("\n" + "=" * 60)
print("TEST 08 COMPLETE")
print("=" * 60)
