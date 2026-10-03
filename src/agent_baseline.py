from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, offline_response
from model_provider import invoke_chat, BASE_SYSTEM_PROMPT
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """A naive offline agent with independent histories for each thread.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a deterministic reply using only the selected thread."""
        if self.langchain_agent is not None:
            session = self.sessions.setdefault(thread_id, SessionState())
            prompt = session.messages + [{"role": "user", "content": message}]
            response, input_tokens, output_tokens = invoke_chat(self.langchain_agent, [{"role": "system", "content": BASE_SYSTEM_PROMPT}] + prompt)
            session.messages = prompt + [{"role": "assistant", "content": response}]
            session.prompt_tokens_processed += input_tokens
            session.token_usage += output_tokens
            return {"response": response, "tokens": output_tokens}
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative response tokens, or zero for a new thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens, including repeated history."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Append one turn and account for the full uncompressed prompt."""
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)

        facts = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        response = offline_response(message, facts)
        session.messages.append({"role": "assistant", "content": response})
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "tokens": response_tokens}

    def _maybe_build_langchain_agent(self):
        return build_chat_model(self.config.model)
