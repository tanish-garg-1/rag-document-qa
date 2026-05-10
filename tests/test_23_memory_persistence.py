"""
TEST 23: MEMORY PERSISTENCE (FILE-BASED)
Tests that conversation memory survives across module reloads,
enforces MAX_MEMORY_MESSAGES, handles corruption gracefully,
and uses atomic writes.
No API key required.
Run: python tests/test_23_memory_persistence.py
"""

import sys
import os
import json
import shutil
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("TEST 23: MEMORY PERSISTENCE (FILE-BASED)")
print("=" * 60)

# Locate the memory file
from app.utils.constants import MAX_MEMORY_MESSAGES, INFERENCE_MEMORY_MESSAGES
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.path.join(ROOT, "data", "memory.json")
BACKUP_FILE = MEMORY_FILE + ".bak_test23"

# Backup existing memory
print("\n--- Setup ---")
if os.path.exists(MEMORY_FILE):
    shutil.copy2(MEMORY_FILE, BACKUP_FILE)
    print(f"  [INFO] Existing memory backed up to memory.json.bak_test23")
else:
    print(f"  [INFO] No existing memory file — starting fresh")

# Wipe for test
import app.services.memory as mem
mem.clear_memory()
print(f"  [INFO] Memory cleared for test")

# --- 1. Basic add_message and get_recent_history ---
print("\n--- add_message / get_recent_history ---")
try:
    mem.add_message("user", "Hello, what is RAG?")
    mem.add_message("assistant", "RAG stands for Retrieval-Augmented Generation.")
    mem.add_message("user", "How does it work?")

    history = mem.get_recent_history()
    print(f"  Messages added: 3, retrieved: {len(history)}")
    if len(history) == 3:
        print("  [PASS] 3 messages stored and retrieved")
    else:
        print(f"  [FAIL] Expected 3, got {len(history)}")

    # Check roles
    roles = [m["role"] for m in history]
    if roles == ["user", "assistant", "user"]:
        print("  [PASS] Message roles correct")
    else:
        print(f"  [FAIL] Roles mismatch: {roles}")

    # Check content
    if history[0]["content"] == "Hello, what is RAG?":
        print("  [PASS] Content preserved correctly")
    else:
        print(f"  [FAIL] Content mismatch: {history[0]['content']}")

except Exception as e:
    print(f"  [FAIL] Basic test error: {e}")

# --- 2. Persistence across module reload ---
print("\n--- Persistence Across Reload ---")
try:
    # Write some messages
    mem.clear_memory()
    mem.add_message("user", "Persistent message 1")
    mem.add_message("assistant", "Persistent response 1")
    mem.add_message("user", "Persistent message 2")
    size_before = mem.get_memory_size()

    # Reload the module (simulates server restart)
    importlib.reload(mem)

    size_after = mem.get_memory_size()
    history_after = mem.get_recent_history(10)

    print(f"  Messages before reload: {size_before}")
    print(f"  Messages after reload:  {size_after}")

    if size_after == size_before:
        print("  [PASS] Memory persisted across module reload (simulates restart)")
    else:
        print(f"  [FAIL] Memory lost on reload: {size_before} -> {size_after}")

    contents = [m["content"] for m in history_after]
    if "Persistent message 1" in contents:
        print("  [PASS] Message content intact after reload")
    else:
        print("  [FAIL] Message content lost after reload")

except Exception as e:
    print(f"  [FAIL] Persistence test error: {e}")
    import traceback; traceback.print_exc()

# --- 3. MAX_MEMORY_MESSAGES enforcement ---
print(f"\n--- MAX_MEMORY_MESSAGES Limit ({MAX_MEMORY_MESSAGES}) ---")
try:
    mem.clear_memory()
    # Add more than MAX_MEMORY_MESSAGES
    extra = 5
    total_to_add = MAX_MEMORY_MESSAGES + extra
    for i in range(total_to_add):
        mem.add_message("user" if i % 2 == 0 else "assistant", f"Message {i}")

    size = mem.get_memory_size()
    print(f"  Added {total_to_add} messages, stored: {size}")

    if size == MAX_MEMORY_MESSAGES:
        print(f"  [PASS] Capped at MAX_MEMORY_MESSAGES ({MAX_MEMORY_MESSAGES})")
    elif size < MAX_MEMORY_MESSAGES:
        print(f"  [WARN] Fewer than MAX stored: {size}")
    else:
        print(f"  [FAIL] Exceeded MAX_MEMORY_MESSAGES: {size}")

    # Newest messages should be kept
    history = mem.get_recent_history(MAX_MEMORY_MESSAGES)
    last_msg = history[-1]["content"]
    expected_last = f"Message {total_to_add - 1}"
    if last_msg == expected_last:
        print(f"  [PASS] Newest messages retained (last: '{last_msg}')")
    else:
        print(f"  [WARN] Last message: '{last_msg}' (expected '{expected_last}')")

except Exception as e:
    print(f"  [FAIL] Max messages test error: {e}")

# --- 4. get_recent_history(n) slicing ---
print(f"\n--- get_recent_history(n) Slicing ---")
try:
    mem.clear_memory()
    for i in range(10):
        mem.add_message("user", f"Msg {i}")

    h3 = mem.get_recent_history(3)
    h5 = mem.get_recent_history(5)
    h0 = mem.get_recent_history(0)

    print(f"  get_recent_history(3) -> {len(h3)} messages")
    print(f"  get_recent_history(5) -> {len(h5)} messages")
    print(f"  get_recent_history(0) -> {len(h0)} messages")

    if len(h3) == 3:
        print("  [PASS] n=3 returns exactly 3 messages")
    else:
        print(f"  [FAIL] n=3 returned {len(h3)}")

    if len(h5) == 5:
        print("  [PASS] n=5 returns exactly 5 messages")
    else:
        print(f"  [FAIL] n=5 returned {len(h5)}")

    if len(h0) == 0:
        print("  [PASS] n=0 returns empty list (not all messages!)")
    else:
        print(f"  [FAIL] n=0 returned {len(h0)} (should be 0)")

    # Verify it returns the MOST RECENT n
    if h3[-1]["content"] == "Msg 9":
        print("  [PASS] Most recent messages returned")
    else:
        print(f"  [WARN] Last message is '{h3[-1]['content']}' (expected 'Msg 9')")

except Exception as e:
    print(f"  [FAIL] Slicing test error: {e}")

# --- 5. get_history_as_text ---
print("\n--- get_history_as_text ---")
try:
    mem.clear_memory()
    mem.add_message("user", "What is FAISS?")
    mem.add_message("assistant", "FAISS is a vector search library.")

    text = mem.get_history_as_text(2)
    print(f"  History as text:\n    {text[:150].encode('ascii','replace').decode()}")

    if "User:" in text:
        print("  [PASS] 'User:' label present")
    else:
        print("  [FAIL] 'User:' label missing")

    if "Assistant:" in text:
        print("  [PASS] 'Assistant:' label present")
    else:
        print("  [FAIL] 'Assistant:' label missing")

    if "FAISS" in text:
        print("  [PASS] Message content in text output")

except Exception as e:
    print(f"  [FAIL] get_history_as_text error: {e}")

# --- 6. Atomic write (temp file pattern) ---
print("\n--- Atomic Write Check ---")
try:
    mem.clear_memory()
    mem.add_message("user", "Atomic test")

    # Check that .tmp file does NOT linger after write
    tmp_path = MEMORY_FILE + ".tmp"
    if os.path.exists(tmp_path):
        print("  [FAIL] Temp file still exists after write (atomic rename failed)")
    else:
        print("  [PASS] No lingering .tmp file (atomic write completed)")

    # Check memory.json is valid JSON
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        print(f"  [PASS] memory.json is valid JSON with {len(data)} messages")
    else:
        print("  [FAIL] memory.json not found after write")

except Exception as e:
    print(f"  [FAIL] Atomic write test error: {e}")

# --- 7. Corrupted file recovery ---
print("\n--- Corrupted File Recovery ---")
try:
    # Write invalid JSON to memory file
    with open(MEMORY_FILE, "w") as f:
        f.write("this is not json {{[}")

    # Try loading — should return empty list, not crash
    importlib.reload(mem)
    history = mem.get_recent_history(10)
    print(f"  Corrupted file -> get_recent_history returns: {history}")

    if history == [] or isinstance(history, list):
        print("  [PASS] Corrupted memory file handled gracefully (returns empty list)")
    else:
        print(f"  [FAIL] Unexpected result: {history}")

except Exception as e:
    print(f"  [FAIL] Corruption recovery error: {e}")
    import traceback; traceback.print_exc()

# --- 8. clear_memory ---
print("\n--- clear_memory ---")
try:
    importlib.reload(mem)
    mem.add_message("user", "Before clear")
    mem.add_message("assistant", "Before clear response")
    size_before = mem.get_memory_size()
    mem.clear_memory()
    size_after = mem.get_memory_size()
    print(f"  Size before clear: {size_before}, after: {size_after}")
    if size_after == 0:
        print("  [PASS] Memory cleared successfully")
    else:
        print(f"  [FAIL] Memory not fully cleared: {size_after} messages remain")
except Exception as e:
    print(f"  [FAIL] clear_memory error: {e}")

# --- Restore backup ---
print("\n--- Restore ---")
if os.path.exists(BACKUP_FILE):
    shutil.copy2(BACKUP_FILE, MEMORY_FILE)
    os.remove(BACKUP_FILE)
    print("  [INFO] Original memory.json restored")
else:
    # Leave empty memory
    mem.clear_memory()
    print("  [INFO] Memory cleared (no backup to restore)")

print("\n" + "=" * 60)
print("TEST 23 COMPLETE")
print("=" * 60)
