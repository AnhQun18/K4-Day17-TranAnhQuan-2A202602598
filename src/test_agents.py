from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Create isolated offline test settings without reading .env."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / 'data',
        state_dir=tmp_path,
        compact_threshold_tokens=50,
        compact_keep_messages=4,
        model=ProviderConfig('fake', 'offline-test', 0),
        judge_model=ProviderConfig('fake', 'offline-test', 0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(make_config(tmp_path).state_dir / 'profiles')
    content = '# User Profile\n- name: DũngCT\n- location: Đà Nẵng\n'
    path = store.write_text('dungct', content)
    assert path == tmp_path / 'profiles' / 'dungct' / 'User.md'
    assert store.read_text('dungct') == content
    assert store.edit_text('dungct', 'Đà Nẵng', 'Huế')
    assert store.read_text('dungct') == content.replace('Đà Nẵng', 'Huế', 1)
    assert not store.edit_text('dungct', 'missing', 'replacement')


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply('dungct', 'long-thread', f'Tin nhắn {index}: ' + 'Nội dung dài để kiểm tra compact. ' * 20)
    assert agent.compaction_count('long-thread') > 0
    context = agent.compact_memory.context('long-thread')
    assert context['summary']
    assert len(context['messages']) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply('dungct', 'thread-1', 'tên mình là DũngCT')
    advanced_answer = advanced.reply('dungct', 'thread-2', 'mình tên gì?')['response']
    baseline_answer = baseline.reply('dungct', 'thread-2', 'mình tên gì?')['response']
    assert 'DũngCT' in advanced_answer
    assert 'DũngCT' not in baseline_answer
    # A new agent must retrieve the fact from disk, without the old thread state.
    restarted = AdvancedAgent(config, force_offline=True)
    assert 'DũngCT' in restarted.reply('dungct', 'thread-3', 'mình tên gì?')['response']


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(30):
        message = f'Tin nhắn {index}: ' + 'Đây là dữ liệu hội thoại dài cho bài kiểm tra. ' * 20
        for agent in (advanced, baseline):
            agent.reply('dungct', 'long-thread', message)
    assert advanced.compaction_count('long-thread') > 0
    assert advanced.prompt_token_usage('long-thread') < baseline.prompt_token_usage('long-thread')
    assert baseline.compaction_count('long-thread') == 0


def test_corrections_and_noise(tmp_path):
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t", "Mình ở Đà Nẵng và đang làm backend engineer. Mình tên là Linh.")
    assert "backend engineer" in agent.profile_store.read_text("u")
    agent.reply("u", "t", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng. Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply("u", "t", "Mình đi họp tại Hà Nội. Hay là chuyển sang product manager? Nếu mình ở Cần Thơ thì sao? Bạn có biết tên mình là Mai không?")
    text = agent.profile_store.read_text("u")
    assert "- location: Huế" in text and "- profession: MLOps engineer" in text
    assert "Đà Nẵng" not in text and "backend engineer" not in text
    assert "Hà Nội" not in text and "product manager" not in text and "Mai" not in text
    assert text.count("- location:") == 1


def test_baseline_recalls_within_thread_only(tmp_path):
    agent = BaselineAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t", "Mình tên là Linh. Món ăn yêu thích là bánh mì.")
    assert "Linh" in agent.reply("u", "t", "Tên mình là gì?")["response"]
    assert "Linh" not in agent.reply("u", "new", "Tên mình là gì?")["response"]


def test_summary_merges_and_is_bounded():
    from memory_store import CompactMemoryManager
    memory = CompactMemoryManager(100, 2)
    memory.append("t", "user", "Mình tên là Linh.")
    for index in range(15):
        memory.append("t", "user", "Ghi chú kỹ thuật: " + "nội dung " * 100)
    context = memory.context("t")
    assert context["compactions"] > 1
    assert "name: Linh" in context["summary"]
    assert len(context["summary"]) <= 1600
    memory.append("t", "user", "Mình tên là Mai. " + "nội dung " * 100)
    assert "name: Mai" in memory.context("t")["summary"] or any("Mai" in item["content"] for item in memory.context("t")["messages"])


def test_user_profiles_are_isolated(tmp_path):
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Mình tên là Linh.")
    assert "Linh" not in agent.reply("u2", "t2", "Tên mình là gì?")["response"]


def test_live_pipeline_uses_model_and_usage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import agent_baseline
    import agent_advanced

    class FakeModel:
        def __init__(self):
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append(messages)
            return SimpleNamespace(content="Phản hồi model", usage_metadata={"input_tokens": 123, "output_tokens": 7})

    for module, agent_class in [(agent_baseline, BaselineAgent), (agent_advanced, AdvancedAgent)]:
        model = FakeModel()
        monkeypatch.setattr(module, "build_chat_model", lambda config: model)
        agent = agent_class(make_config(tmp_path))
        assert agent.reply("u", "t", "Mình tên là Linh.")["response"] == "Phản hồi model"
        assert agent.prompt_token_usage("t") == 123
        assert agent.token_usage("t") == 7
        if agent_class is AdvancedAgent:
            assert "name: Linh" in model.prompts[0][0]["content"]
        agent.reply("u", "t", "Bạn nhớ không?")
        assert any(item["content"] == "Phản hồi model" for item in model.prompts[-1])


def test_live_missing_key_is_explicit():
    import pytest
    from model_provider import build_chat_model
    with pytest.raises(ValueError, match="Missing API key"):
        build_chat_model(ProviderConfig("openai", "test", 0))


def test_live_failure_does_not_fake_reply(tmp_path, monkeypatch):
    import pytest
    import agent_baseline

    class BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("authentication failed")

    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: BrokenModel())
    agent = BaselineAgent(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="authentication failed"):
        agent.reply("u", "t", "Chào bạn")
    assert agent.token_usage("t") == 0


def test_indirect_questions_and_instructions_do_not_become_facts(tmp_path):
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t", "Mình tên là Linh. Đồ uống yêu thích là trà đào. Mình nuôi một bé mèo tên Mít.")
    agent.reply("u", "t", "Bạn nhớ giúp mình là thông tin nơi ở đã thay đổi. Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.")
    profile = agent.profile_store.read_text("u")
    assert "name: Linh" in profile and "drink: trà đào" in profile
    assert "pet: mèo tên Mít" in profile


def test_standard_dataset_recall(tmp_path):
    # Integration check against the unmodified public dataset; no expected facts
    # are passed to either agent or to the extractor.
    from dataclasses import replace
    from benchmark import load_conversations, run_agent_benchmark
    config = replace(make_config(tmp_path), compact_threshold_tokens=800)
    conversations = load_conversations(Path(__file__).resolve().parents[1] / "data" / "conversations.json")
    row = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)
    assert row.recall_score >= 0.9
