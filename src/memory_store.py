from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate tokens using one token per four stripped characters."""
    return len(text.strip()) // 4 if text else 0


@dataclass
class UserProfileStore:
    """Store one UTF-8 User.md per user under root_dir (state/profiles)."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id or user_id in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', user_id):
            raise ValueError("user_id must be a non-empty directory name")
        return self.root_dir / user_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


# Explicit self-declarations only: event locations and third-party facts are excluded.
_PROFILE_PATTERNS = {
    "name": r"(?:mình|tôi)\s+tên là|tên (?:mình|tôi) là",
    "location": r"(?:mình|tôi)\s+(?:(?:hiện|hiện tại|vẫn|đang|hiện đang)\s+)*(?:ở|sống ở|đang sống ở|đang làm việc ở)|hiện ở|nơi ở hiện tại (?:của (?:mình|tôi) )?là",
    "profession": r"(?:mình|tôi)\s+(?:đang |vẫn |vẫn đang )?(?:làm|là)\s+(?:nghề )?|đang làm|(?:giờ )?chuyển sang|nghề(?: nghiệp)?(?: hiện tại)?(?: của (?:mình|tôi))?(?: thì)?(?: vẫn)? là",
    "interest": r"(?:mình|tôi)\s+(?:vẫn |còn )?(?:thích|đang quan tâm(?: nhiều)? đến)|sở thích(?: của (?:mình|tôi))? là",
    "drink": r"(?:đồ uống yêu thích(?: của (?:mình|tôi))? là|(?:mình|tôi) (?:thích|hay|vẫn) uống)",
    "food": r"món ăn yêu thích(?: của (?:mình|tôi))? là",
    "pet": r"(?:mình|tôi) nuôi(?:\s+(?:một bé|một|bé))?",
}


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative rule-based entity extraction, with latest declaration winning.

    Questions, hypothetical clauses and negated facts cannot overwrite a profile.
    These rules are a deterministic lab implementation, not general Vietnamese NLU.
    """
    updates = {}
    for sentence in re.findall(r"[^.!?;\n]+[.!?;]?", message):
        if sentence.rstrip().endswith("?"):
            continue
        interest = re.search(r"(?:mình|tôi)\s+(?:vẫn |còn )?(?:thích|đang quan tâm(?: nhiều)? đến)\s+([^.!?;]+)", sentence, re.I)
        if interest and not re.search(r"trả lời|giải thích|tin này|markdown|uống", interest.group(1), re.I):
            updates["interest"] = interest.group(1).strip()
        # Break contrast clauses so an old negated value cannot enter the new fact.
        for clause in re.split(r",|\bchứ\b|\bnhưng\b|\bdù\b|\bcòn\b|\bvà\s+(?=(?:mình|tôi|đang|style|nghề)\b)", sentence, flags=re.I):
            if re.search(r"\b(nếu|giả sử|hay là|đùa|không còn|đừng|không phải)\b", clause, re.I):
                continue
            for key, prefix in _PROFILE_PATTERNS.items():
                for match in re.finditer(rf"(?:{prefix})\s+([^.!?;]+)", clause, flags=re.I):
                    value = re.split(r"\s+(?:cho|để|vì|trong giai đoạn|vài tháng|mỗi ngày|như cũ|thì|và đang)\b", match.group(1), maxsplit=1, flags=re.I)[0].strip()
                    if re.search(r"^(?:gì|ai|đâu|con gì)\b", value, re.I):
                        continue
                    if key == "profession":
                        # Only occupational declarations, never 'làm sạch data'.
                        if not re.search(r"engineer|developer|manager|teacher|bác sĩ|giáo viên|kỹ sư|lập trình viên|sinh viên|designer|researcher", value, re.I):
                            continue
                    if key == "name" and re.search(r"engineer|developer|manager", value, re.I):
                        continue
                    if key == "interest" and re.search(r"trả lời|giải thích|markdown|tin này", value, re.I):
                        continue
                    if value and not (key == "interest" and "interest" in updates):
                        updates[key] = value
            if re.search(r"(?:mình|tôi|hãy).*(?:trả lời|giải thích)|style trả lời", clause, re.I):
                if re.search(r"ngắn|gọn|bullet", clause, re.I):
                    style = "ngắn gọn"
                    if re.search(r"3 bullet", clause, re.I):
                        style += ", 3 bullet"
                    elif "bullet" in clause.lower():
                        style += ", bullet"
                    if re.search(r"ví dụ", message, re.I):
                        style += ", có ví dụ thực tế / thực chiến"
                    updates["style"] = style
    return updates


FACT_LABELS = {
    "name": "Tên", "location": "Nơi ở hiện tại", "profession": "Nghề nghiệp",
    "interest": "Mối quan tâm", "drink": "Đồ uống", "style": "Style trả lời",
    "food": "Món ăn", "pet": "Thú nuôi",
}


def offline_response(message: str, facts: dict[str, str]) -> str:
    """Shared recall behavior: both agents differ only in available memory."""
    query = message.casefold()
    if "?" not in query and not re.search(r"nhắc|nhớ lại|tóm tắt|hãy.*mô tả", query):
        return "Mình đã nhận thông tin."
    patterns = {
        "name": r"tên|là ai", "location": r"nơi ở|ở đâu|đang ở|còn ở|hiện ở",
        "profession": r"nghề|công việc|làm nghề|làm gì",
        "interest": r"mối quan tâm|kỹ thuật chính|thích gì|sở thích",
        "drink": r"đồ uống|uống", "style": r"style|kiểu trả lời|trả lời.*thế nào",
        "food": r"món ăn", "pet": r"nuôi|thú cưng",
    }
    keys = [key for key, pattern in patterns.items() if re.search(pattern, query)]
    if not keys and "tóm tắt" in query:
        keys = list(FACT_LABELS)
    if not keys:
        return "Mình chưa có đủ thông tin để trả lời câu hỏi này."
    return "\n".join(
        f"- {FACT_LABELS[key]}: {facts[key]}." if key in facts
        else f"- {FACT_LABELS[key]}: mình chưa biết."
        for key in keys
    )


def summarize_messages(messages: list[dict[str, str]], previous: str = "", max_chars: int = 1600) -> str:
    """Merge previous summary with new content under a fixed character budget.

    Explicit facts survive repeated compaction; bounded snippets retain topics.
    Older snippets are evicted when the budget is exhausted. No LLM cost here.
    """
    facts = dict(re.findall(r"^- (\w+): ([^\n]+)", previous, re.M))
    for item in messages:
        if item["role"] == "user":
            facts.update(extract_profile_updates(item["content"]))
    head = "Summary:\n" + "\n".join(f"- {key}: {value[:180]}" for key, value in facts.items())
    old_notes = previous.split("\nTopics:\n", 1)[-1] if "\nTopics:\n" in previous else ""
    notes = old_notes.splitlines() + [item["content"][:180] for item in messages if item["role"] == "user"]
    selected = []
    remaining = max_chars - len(head) - len("\nTopics:\n")
    for note in reversed(notes):
        if len(note) + 1 <= remaining:
            selected.insert(0, note)
            remaining -= len(note) + 1
    return (head + "\nTopics:\n" + "\n".join(selected))[:max_chars]


@dataclass
class CompactMemoryManager:
    """Move older messages into a summary when a thread exceeds its budget."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 0 or self.keep_messages < 0:
            raise ValueError("threshold_tokens and keep_messages must be non-negative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        messages = thread["messages"]
        messages.append({"role": role, "content": content})
        tokens = estimate_tokens(thread["summary"]) + sum(estimate_tokens(message["content"]) for message in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_at = len(messages) - self.keep_messages
            thread["summary"] = summarize_messages(messages[:split_at], previous=thread["summary"])
            thread["messages"] = messages[split_at:]
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
