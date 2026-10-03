from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON conversations."""
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Score all matches as 1, at least half as 0.5, otherwise 0."""
    if not expected:
        return 0.0
    normalized = answer.casefold()
    found = sum(item.casefold() in normalized for item in expected)
    if found == len(expected):
        return 1.0
    if found > 0 and found * 2 >= len(expected):
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score continuous fact coverage, penalizing verbosity and echo replies.

    This heuristic cannot verify semantics or detect every contradiction.
    """
    text = answer.strip()
    if not text or not expected or text.casefold().startswith("đã nhận:"):
        return 0.0
    coverage = sum(item.casefold() in text.casefold() for item in expected) / len(expected)
    # Allow labels and natural phrasing around each expected fact.
    budget = max(160, sum(len(item) for item in expected) * 2 + 80)
    brevity = min(1.0, budget / len(text))
    return coverage * brevity


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Measure training and fresh-thread recall, counting only this run's work."""
    initial_counts = {}
    users = {conv["user_id"] for conv in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_size = sum(memory_size(user_id) for user_id in users)
    recall_scores = []
    quality_scores = []
    recall_prefix = f"recall-{uuid4().hex}"
    training_threads = {conv["id"] for conv in conversations}

    def reply(user_id, thread_id, message):
        if thread_id not in initial_counts:
            initial_counts[thread_id] = (
                agent.token_usage(thread_id),
                agent.prompt_token_usage(thread_id),
                agent.compaction_count(thread_id),
            )
        return agent.reply(user_id, thread_id, message)

    for index, conv in enumerate(conversations):
        user_id = conv["user_id"]
        for turn in conv["turns"]:
            reply(user_id, conv["id"], turn)
        for question_index, question in enumerate(conv.get("recall_questions", [])):
            thread_id = f"{recall_prefix}-{index}-{question_index}"
            while thread_id in training_threads:
                thread_id += "-recall"
            result = reply(user_id, thread_id, question["question"])
            expected = question["expected_contains"]
            recall_scores.append(recall_points(result["response"], expected))
            quality_scores.append(heuristic_quality(result["response"], expected))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread) - counts[0] for thread, counts in initial_counts.items()),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread) - counts[1] for thread, counts in initial_counts.items()),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(memory_size(user_id) for user_id in users) - initial_size,
        compactions=sum(agent.compaction_count(thread) - counts[2] for thread, counts in initial_counts.items()),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the six benchmark metrics in a Markdown table."""
    lines = [
        "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        name = row.agent_name.replace("|", r"\|").replace("\n", " ")
        lines.append(
            f"| {name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
            f"{row.recall_score:.2f} | {row.response_quality:.2f} | "
            f"{row.memory_growth_bytes} | {row.compactions} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run both agents with fresh state; offline by default."""
    parser = argparse.ArgumentParser(description="Compare memory agents with isolated benchmark state.")
    parser.add_argument("--live", action="store_true", help="Call real LLM APIs; incurs provider usage.")
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        rows = []
        # Isolate persisted profiles so reruns and other suites cannot seed recall.
        with TemporaryDirectory(prefix="memory-benchmark-") as directory:
            for name, agent_class in [("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)]:
                isolated = replace(config, state_dir=Path(directory) / name.lower())
                agent = agent_class(isolated, force_offline=not args.live)
                rows.append(run_agent_benchmark(name, agent, conversations, isolated))
        print(f"\n{title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
