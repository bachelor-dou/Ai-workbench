import json

from hot_project.agent.history import Session
from hot_project.agent.loop import Agent
from hot_project.tests.test_agent import _LLM, _call
from hot_project.tools import local_tools
from hot_project.tools.spec import Ctx, Registry, Tool


def test_a_later_question_can_recall_evidence_beyond_the_original_result_limit():
    result = {"readme": "x" * 9000 + "WINDOWS_SUPPORTED"}
    client = _LLM(_call("evidence"), {"content": "已取得资料"})
    registry = Registry([
        Tool("evidence", "取得完整项目资料供后续问题使用", lambda ctx, args: result),
        *local_tools.TOOLS,
    ])
    agent = Agent(client=client, tools=registry, gh=object())
    agent.chat("先了解这个项目")
    observed = json.loads(next(m["content"] for m in agent.session.messages
                               if m["role"] == "tool"))
    ref = observed["ref"]
    client.replies.extend([
        _call("recall_tool_result", json.dumps({"ref": ref, "offset": 9000}), cid="c2"),
        {"content": "支持 Windows"},
    ])
    assert agent.chat("它支持 Windows 吗？") == "支持 Windows"
    recalled = json.loads(client.calls[-1][-1]["content"])
    assert "WINDOWS_SUPPORTED" in recalled["text"]
    assert recalled["next_offset"] is None


def test_tool_budgets_are_shared_within_a_turn_and_reset_for_the_next_question():
    def spend(ctx, args):
        ctx.turn_state["used"] = ctx.turn_state.get("used", 0) + 1
        return {"used": ctx.turn_state["used"]}

    client = _LLM(_call("spend"), _call("spend", cid="c2"), {"content": "完成"},
                  _call("spend", cid="c3"), {"content": "完成"})
    agent = Agent(client=client, tools=Registry([
        Tool("spend", "测试一个回合内共享而下回合恢复的预算", spend),
    ]), gh=object())
    agent.chat("第一问")
    assert json.loads(client.calls[-1][-1]["content"]) == {"used": 2}
    agent.chat("第二问")
    assert json.loads(client.calls[-1][-1]["content"]) == {"used": 1}


def test_compression_keeps_evidence_discoverable_even_when_the_summary_omits_its_ref():
    session = Session()
    session.user("之后要比较安装限制")
    session.assistant(None, _call("evidence")["tool_calls"])
    session.tool_result("c1", {"readme": "x" * 5000 + "NO_GPU_REQUIRED"})
    session.assistant("稍后比较")
    session.offload_old_results()
    ref = json.loads(session.messages[3]["content"])["ref"]
    for i in range(20):
        session.user(f"其他问题{i}")
        session.assistant("已回答")
    session.compress(lambda old: "用户还需要比较安装限制。")
    assert any(ref in (m.get("content") or "") for m in session.messages)
    result = Registry(local_tools.TOOLS).run(
        Ctx(state=session), "recall_tool_result", {"ref": ref, "offset": 4900})
    assert "NO_GPU_REQUIRED" in result["text"]


def test_evidence_eviction_is_explicit_and_does_not_remove_the_latest_result():
    session = Session()
    refs = []
    for i in range(33):
        session.tool_result(str(i), {"data": "x" * 9000, "fact": f"fact-{i}"})
        refs.append(json.loads(session.messages[-1]["content"])["ref"])
    registry = Registry(local_tools.TOOLS)
    ctx = Ctx(state=session)
    assert "已过期" in registry.run(ctx, "recall_tool_result", {"ref": refs[0]})["error"]
    assert "fact-32" in registry.run(
        ctx, "recall_tool_result", {"ref": refs[-1], "offset": 9000})["text"]


def test_the_summary_model_receives_tool_evidence_and_user_constraints():
    class Summarizer(_LLM):
        def text(self, prompt, **kwargs):
            self.prompt = prompt
            return "保留用户只用 CPU 的约束,项目支持离线部署。"

    client = Summarizer({"content": "继续比较"})
    agent = Agent(client=client, tools=Registry(local_tools.TOOLS), gh=object())
    agent.session.user("只能用 CPU,不要推荐云服务")
    agent.session.assistant(None, _call("evidence")["tool_calls"])
    agent.session.tool_result("c1", {"padding": "x" * 9000,
                                     "offline": "SUPPORTED_OFFLINE", "source": "README"})
    agent.session.assistant("资料已取得")
    for i in range(20):
        agent.session.user(f"其他问题{i}")
        agent.session.assistant("已回答")
    agent.chat("继续比较")
    assert "SUPPORTED_OFFLINE" in client.prompt
    assert "只能用 CPU,不要推荐云服务" in client.prompt


def test_failed_memory_extraction_preserves_history_and_can_be_retried():
    class UnavailableSummary(_LLM):
        summary_reply = ""

        def text(self, prompt, **kwargs):
            return self.summary_reply

    client = UnavailableSummary({"content": "按 CPU 约束继续"})
    agent = Agent(client=client, tools=Registry(local_tools.TOOLS), gh=object())
    agent.session.user("必须只使用 CPU")
    agent.session.assistant("记住了")
    for i in range(20):
        agent.session.user(f"问题{i}")
        agent.session.assistant("回答")
    before = list(agent.session.messages)
    assert "压缩失败" in agent.chat("继续")
    assert agent.session.messages == before
    assert client.calls == []
    client.summary_reply = "用户要求必须只使用 CPU。"
    assert agent.chat("继续") == "按 CPU 约束继续"
    assert any("必须只使用 CPU" in (m.get("content") or "") for m in client.calls[-1])


def test_a_few_large_messages_trigger_compression_before_the_next_model_call():
    client = _LLM({"content": "继续"})
    agent = Agent(client=client, tools=Registry(local_tools.TOOLS), gh=object())
    for i in range(4):
        agent.session.user(f"问题{i}")
        agent.session.assistant("回答" * 7000)
    agent.chat("继续")
    assert client.text_kwargs
    assert sum(len(m.get("content") or "") for m in client.calls[-1]) < 48000


def test_tool_evidence_is_compacted_within_a_long_turn_before_sending_more_context():
    class BoundedContext(_LLM):
        def chat(self, messages, **kwargs):
            assert sum(len(m.get("content") or "") for m in messages) < 48000
            return super().chat(messages, **kwargs)

    client = BoundedContext(*[_call("evidence", cid=f"c{i}") for i in range(10)],
                            {"content": "已完成比较"})
    agent = Agent(client=client, tools=Registry([
        Tool("evidence", "获取一份有后续查询价值的较长证据", lambda ctx, args: {"data": "x" * 5000}),
        *local_tools.TOOLS,
    ]), gh=object())
    assert agent.chat("比较这些资料") == "已完成比较"
    assert all(sum(len(m.get("content") or "") for m in sent) < 48000
               for sent in client.calls)


def test_recall_can_find_evidence_without_guessing_its_character_offset():
    session = Session()
    session.tool_result("c1", {"padding": "x" * 9000, "requirement": "Requires CUDA"})
    ref = json.loads(session.messages[-1]["content"])["ref"]
    out = Registry(local_tools.TOOLS).run(
        Ctx(state=session), "recall_tool_result", {"ref": ref, "keyword": "cuda"})
    assert "Requires CUDA" in out["text"]


def test_recalled_pages_round_trip_escaped_text_without_overflowing_context():
    expected = {"text": '"\\\n' * 2500 + "END_OF_EVIDENCE"}
    session = Session()
    session.tool_result("c1", expected)
    ref = json.loads(session.messages[-1]["content"])["ref"]
    registry = Registry(local_tools.TOOLS)
    chunks, offset = [], 0
    for _ in range(30):
        page = registry.run(Ctx(state=session), "recall_tool_result", {"ref": ref, "offset": offset})
        assert len(json.dumps(page, ensure_ascii=False)) <= 6000
        chunks.append(page["text"])
        following = page["next_offset"]
        if following is None:
            break
        assert following > offset
        offset = following
    else:
        raise AssertionError("分页没有结束")
    assert json.loads("".join(chunks)) == expected


def test_retained_evidence_is_also_bounded_by_total_character_count():
    session = Session()
    refs = []
    for i in range(3):
        session.tool_result(str(i), {"text": "x" * 400000, "end": f"end-{i}"})
        refs.append(json.loads(session.messages[-1]["content"])["ref"])
    registry = Registry(local_tools.TOOLS)
    ctx = Ctx(state=session)
    assert "error" in registry.run(ctx, "recall_tool_result", {"ref": refs[0]})
    assert "end-2" in registry.run(
        ctx, "recall_tool_result", {"ref": refs[-1], "keyword": "end-2"})["text"]


def test_a_result_too_large_to_retain_is_explicitly_unavailable():
    session = Session()
    session.tool_result("c1", {"text": "x" * 1000001})
    view = json.loads(session.messages[-1]["content"])
    assert view["unavailable"] is True
    assert "未保存全文" in view["note"]
    assert "ref" not in view


def test_summary_exceptions_do_not_discard_the_original_conversation():
    session = Session()
    for i in range(20):
        session.user(f"问题{i}")
        session.assistant("回答")
    before = list(session.messages)

    def unavailable(old):
        raise RuntimeError("summary backend unavailable")

    assert session.compress(unavailable) is False
    assert session.messages == before


def test_compression_preserves_complete_multi_tool_call_groups():
    session = Session()
    for i in range(12):
        session.user(f"比较{i}")
        session.assistant(None, [
            _call("first", cid=f"a{i}")["tool_calls"][0],
            _call("second", cid=f"b{i}")["tool_calls"][0],
        ])
        session.tool_result(f"a{i}", {"value": 1})
        session.tool_result(f"b{i}", {"value": 2})
        session.assistant("比较完成")
    assert session.compress(lambda old: "用户在比较项目") is True
    pending = set()
    for message in session.messages:
        if message.get("tool_calls"):
            assert not pending
            pending = {call["id"] for call in message["tool_calls"]}
        elif message["role"] == "tool":
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
    assert not pending


def test_offloading_older_messages_does_not_evict_newer_evidence():
    session = Session()
    session.user("取得这些资料")
    session.assistant(None, [_call("evidence", cid=f"c{i}")["tool_calls"][0]
                             for i in range(33)])
    for i in range(32):
        session.tool_result(f"c{i}", {"data": "x" * 2000})
    session.tool_result("c32", {"data": "y" * 9000, "fact": "LATEST_FACT"})
    ref = json.loads(session.messages[-1]["content"])["ref"]
    session.assistant("资料已取得")
    session.offload_old_results()
    out = Registry(local_tools.TOOLS).run(
        Ctx(state=session), "recall_tool_result", {"ref": ref, "keyword": "LATEST_FACT"})
    assert "LATEST_FACT" in out["text"]


def test_recalling_pages_does_not_evict_the_original_evidence():
    session = Session()
    session.tool_result("c1", {"data": "x" * 997000, "fact": "END_FACT"})
    ref = json.loads(session.messages[-1]["content"])["ref"]
    registry = Registry(local_tools.TOOLS)
    ctx = Ctx(state=session)
    first = registry.run(ctx, "recall_tool_result", {"ref": ref})
    session.tool_result("c2", first)
    session.offload_old_results()
    second = registry.run(ctx, "recall_tool_result", {"ref": ref, "offset": first["next_offset"]})
    assert second["offset"] == first["next_offset"]
    assert "END_FACT" in registry.run(
        ctx, "recall_tool_result", {"ref": ref, "keyword": "END_FACT"})["text"]
