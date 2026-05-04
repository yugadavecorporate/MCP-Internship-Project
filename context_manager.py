"""
Week 4: Context Management and Memory
======================================
A robust, minimalist context management module for the Groq + MCP Streamlit chatbot.
Implements:
  1. ConversationBufferMemory  - Tracks and manages chat history.
  2. inject_context_to_system_prompt - Injects tool outputs into the system prompt.
  3. _run_verification           - A test routine to validate multi-turn memory retention.
"""

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# 1. CONVERSATION BUFFER MEMORY
# ─────────────────────────────────────────────

class ConversationBufferMemory:
    """
    Maintains a rolling, ordered list of conversation messages that can be
    fed directly to the Groq API.

    In Streamlit, attach this to session state:
        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationBufferMemory()
    """

    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: Maximum number of *user+assistant pairs* to retain.
                       Older turns are automatically pruned to avoid
                       exceeding token limits.
        """
        self._messages: List[Dict[str, Any]] = []
        self.max_turns = max_turns

    # ── Write helpers ──────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        """Append a user turn to memory."""
        self._messages.append({"role": "user", "content": content})
        self._prune()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
    ) -> None:
        """
        Append an assistant turn to memory.
        Supply `tool_calls` when the model decided to invoke a tool instead
        of replying in plain text.
        """
        msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        """
        Append a tool-result turn returned by the MCP server so the LLM
        can reason over the fetched data in subsequent messages.
        """
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    # ── Read helpers ───────────────────────────────────────────────────────

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all buffered messages (no system prompt included)."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset the buffer (e.g., on 'New Chat' button press)."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    # ── Internal ───────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """
        Drop the oldest *pair* of messages once the buffer exceeds max_turns.
        We only prune user/assistant turns, never tool results, to prevent
        orphaned tool_call_id references that would upset the Groq API.
        """
        user_turns = [i for i, m in enumerate(self._messages) if m["role"] == "user"]
        while len(user_turns) > self.max_turns:
            drop_idx = user_turns.pop(0)
            # Remove user message + immediately following assistant message (if present)
            to_remove = [drop_idx]
            if drop_idx + 1 < len(self._messages) and self._messages[drop_idx + 1]["role"] == "assistant":
                to_remove.append(drop_idx + 1)
            for i in reversed(to_remove):
                self._messages.pop(i)
            # Recalculate indices after mutation
            user_turns = [i for i, m in enumerate(self._messages) if m["role"] == "user"]


# ─────────────────────────────────────────────
# 2. CONTEXT INJECTOR
# ─────────────────────────────────────────────

def inject_context_to_system_prompt(
    memory: ConversationBufferMemory,
    dynamic_context: str = "",
    base_instruction: str = (
        "You are a highly precise, context-aware AI assistant. "
        "You have access to a local database (SQLite), local files, and web fetch capabilities via MCP tools. "
        "CRITICAL INSTRUCTION: You MUST actively utilize your tools to find factual answers to user questions before responding. "
        "If the user asks a factual question (like current events, facts, or database queries), immediately execute the corresponding tool (e.g. `search_web` or `query_local_database`) to get real-time information. "
        "Do NOT hallucinate or guess. If no tools succeed or you cannot find the answer, explicitly state that you do not have the information."
    ),
) -> List[Dict[str, Any]]:
    """
    Construct the full message array ready to be sent to the Groq API.

    Layout:
        [system_prompt]  ← base instruction + optional dynamic context
        [history…]       ← buffered user / assistant / tool messages

    Args:
        memory:           The active ConversationBufferMemory instance.
        dynamic_context:  Any extra background text to prepend in the system
                          block (e.g., current user data, session metadata).
        base_instruction: The core persona / capability description for the LLM.

    Returns:
        A list of message dicts compatible with client.chat.completions.create().
    """
    # Build system block
    system_content = base_instruction

    if dynamic_context:
        system_content += (
            "\n\n--- Injected Background Context ---\n"
            f"{dynamic_context.strip()}"
            "\n-----------------------------------"
        )

    full_payload: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content}
    ]

    # Append rolling history so the LLM retains full conversation awareness
    full_payload.extend(memory.get_history())

    return full_payload


# ─────────────────────────────────────────────
# 3. VERIFICATION HANDLER
# ─────────────────────────────────────────────

def _run_verification() -> None:
    """
    Simulates a two-turn conversation exercising both tool-result injection
    and dynamic context injection, then asserts all payloads are intact.
    """
    print("=" * 60)
    print("  Week 4 – Context Memory Verification")
    print("=" * 60)

    memory = ConversationBufferMemory(max_turns=10)

    # ── Turn 1: User asks a question, LLM calls a tool ──────────────────

    print("\n[Turn 1] User queries the database via MCP tool.")
    memory.add_user_message("Who are the admin users stored in the local database?")

    mock_tool_calls = [
        {
            "id": "call_db_001",
            "type": "function",
            "function": {
                "name": "query_local_database",
                "arguments": json.dumps({"sql_query": "SELECT * FROM users WHERE role='admin'"}),
            },
        }
    ]
    memory.add_assistant_message(content=None, tool_calls=mock_tool_calls)

    mock_db_result = json.dumps(
        [{"id": 1, "username": "alice_admin", "email": "alice@example.com", "role": "admin"}]
    )
    memory.add_tool_result(
        tool_call_id="call_db_001",
        name="query_local_database",
        content=mock_db_result,
    )
    memory.add_assistant_message(
        content="The only admin user is **alice_admin** (alice@example.com)."
    )

    assert len(memory) == 4, "Expected 4 messages after turn 1."
    print("  ✅ Turn 1 memory length correct:", len(memory))

    # ── Turn 2: Follow-up with dynamic context injected ─────────────────

    print("\n[Turn 2] User follows up; dynamic background context is injected.")
    memory.add_user_message("How many API calls does alice have left?")

    dynamic_ctx = "SESSION METADATA: alice_admin has consumed 42 / 100 API calls this billing cycle."
    payload = inject_context_to_system_prompt(memory, dynamic_context=dynamic_ctx)

    # Verify structure
    assert payload[0]["role"] == "system", "First message must be the system prompt."
    assert dynamic_ctx[:20] in payload[0]["content"], "Dynamic context must appear in system prompt."
    assert any(m["role"] == "user" and "alice" in m["content"] for m in payload), \
        "Latest user turn must be present in payload."
    assert any(m["role"] == "tool" for m in payload), \
        "Tool result from Turn 1 must still persist in payload."

    print("  ✅ System prompt contains injected dynamic context.")
    print("  ✅ Tool result from Turn 1 is retained in payload.")
    print("  ✅ Latest user message is correctly appended.")
    print("\n--- Final Payload Preview (formatted) ---")
    for i, msg in enumerate(payload):
        role = msg["role"].upper()
        preview = str(msg.get("content") or msg.get("tool_calls", ""))[:80]
        print(f"  [{i}] {role}: {preview}...")

    print("\n✅ All assertions passed. Memory and context injection are working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    _run_verification()
