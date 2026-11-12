"""ReAct 循环与对话历史。

历史那几条规矩(孤儿 tool 消息、大结果卸载、system 前缀稳定)出问题都是静默的:
400 会被当成「模型调用失败」,token 浪费根本没人看得见。所以这里测得比循环本身还细。
"""

import json

import pytest

from hot_project.agent import history
from hot_project.agent.history import Session
from hot_project.agent.loop import Agent
from hot_project.agent.prompts import SYSTEM_PROMPT
from hot_project.infra import llm as llm_module
from hot_project.tools.spec import Param, Registry, Tool


# ── 对话历史 ───────────────────────────────────────────────────────

def _conversation(rounds: int) -> list[dict]:
    out = []
    for i in range(rounds):
        out += [{"role": "user", "content": f"问题{i}"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": f"c{i}", "function": {"name": "t"}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
                {"role": "assistant", "content": f"回答{i}"}]
    return out


def test_a_fresh_session_starts_with_the_system_prompt():
    assert Session().messages == [{"role": "system", "content": SYSTEM_PROMPT}]


def test_compression_never_leaves_an_orphan_tool_message():
    """tool 消息必须跟在带 tool_calls 的 assistant 之后,否则接口直接 400。"""
    session = Session()
    session.messages += _conversation(12)
    session.compress(lambda old: "摘要")
    assert session.messages[0]["role"] == "system"
    for i, message in enumerate(session.messages):
        if message.get("role") == "tool":
            assert session.messages[i - 1].get("tool_calls"), f"第 {i} 条是孤儿"


def test_compression_prefers_to_start_the_kept_slice_at_a_user_message():
    old, recent = history.split_at_safe_boundary(_conversation(6), keep=10)
    assert recent[0]["role"] == "user"
    assert len(old) + len(recent) == 24


def test_a_tool_heavy_round_still_produces_a_legal_slice():
    """上一轮工具调太多、保留段里一条 user 都没有 —— 兜底只剥掉开头的孤儿。"""
    messages = ([{"role": "user", "content": "问"}]
                + [{"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c", "function": {"name": "t"}}]}]
                + [{"role": "tool", "tool_call_id": "c", "content": "{}"}] * 15)
    _, recent = history.split_at_safe_boundary(messages, keep=10)
    assert recent == [] or recent[0]["role"] != "tool"


def test_the_system_message_stays_byte_identical_after_compression():
    """插一个字都会让各家的前缀缓存全部落空。"""
    session = Session()
    session.messages += _conversation(12)
    session.compress(lambda old: "摘要")
    assert session.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_the_summary_is_its_own_message_not_glued_into_the_system_prompt():
    session = Session()
    session.messages += _conversation(12)
    session.compress(lambda old: "这是摘要")
    assert "这是摘要" not in session.messages[0]["content"]
    assert "这是摘要" in session.messages[1]["content"]


def test_a_failed_summary_keeps_the_previous_one_rather_than_losing_it():
    session = Session()
    session.summary = "旧摘要"
    session.messages += _conversation(12)
    before = list(session.messages)
    assert session.compress(lambda old: "") is False
    assert session.summary == "旧摘要"
    assert session.messages == before


def test_a_short_conversation_is_left_alone():
    session = Session()
    session.messages += _conversation(2)
    before = list(session.messages)
    session.compress(lambda old: pytest.fail("不该总结"))
    assert session.messages == before


def test_a_big_tool_result_becomes_a_stub_the_model_can_recall():
    session = Session()
    session.tool_result("c1", {"data": "x" * 5000})
    assert session.offload_old_results() == 1
    stub = json.loads(session.messages[-1]["content"])
    assert stub["offloaded"] is True
    assert session.tool_state["offloaded"][stub["ref"]].startswith('{"data"')


def test_a_stub_is_not_offloaded_a_second_time():
    session = Session()
    session.tool_result("c1", {"data": "x" * 5000})
    session.offload_old_results()
    assert session.offload_old_results() == 0


def test_a_small_result_is_left_intact():
    session = Session()
    session.tool_result("c1", {"ok": True})
    assert session.offload_old_results() == 0
    assert session.messages[-1]["content"] == '{"ok": true}'


def test_an_enormous_result_is_replaced_with_a_recallable_stub():
    session = Session()
    session.tool_result("c1", {"blob": "x" * 50000})
    text = session.messages[-1]["content"]
    assert len(text) < history.RESULT_MAX_CHARS
    assert json.loads(text)["offloaded"] is True


def test_results_that_are_not_json_serializable_do_not_crash_the_turn():
    """工具返回里混进 datetime / Path 是常事,不能让整轮对话挂在序列化上。"""
    from datetime import date
    session = Session()
    session.tool_result("c1", {"when": date(2026, 7, 30)})
    assert json.loads(session.messages[-1]["content"])["when"] == "2026-07-30"


# ── ReAct 循环 ────────────────────────────────────────────────────

class _LLM:
    """按脚本回话的假模型。每个元素是一条 message。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.kwargs = []
        self.text_kwargs = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if not self.replies:
            return {"choices": [{"message": {"content": "没词了"}}]}
        return {"choices": [{"message": self.replies.pop(0)}]}

    def text(self, prompt, **kwargs):
        self.text_kwargs.append(kwargs)
        return "摘要"


def _call(name, args="{}", cid="c1"):
    return {"tool_calls": [{"id": cid, "function": {"name": name, "arguments": args}}]}


def _agent(llm, *tools):
    return Agent(client=llm, tools=Registry(tools), gh=object())


def test_the_thinking_level_reaches_the_model_call():
    """档位得真传到调用上,不传时是默认档(思考)。

    非法值不在这里兜 —— 那是 `protocol.level` 的活(见 test_llm)。Agent 只负责转交,
    同一件事兜两处,迟早有一处忘了改。"""
    llm = _LLM({"content": "好"})
    _agent(llm).chat("问", effort="max")
    assert llm.kwargs[0]["effort"] == "max"

    llm = _LLM({"content": "好"})
    _agent(llm).chat("问")
    assert llm.kwargs[0]["effort"] == llm_module.EFFORT_DEFAULT


def test_compressing_the_history_thinks_at_the_batch_depth():
    """压缩要判断哪些细节后面还用得上,不思考就容易丢掉关键前提(这里曾经是显式关掉的)。
    但它卡在用户发消息和模型回答之间,用对话那一档会让每次压缩都变成一次可感知的停顿。"""
    llm = _LLM()
    _agent(llm)._summarize([{"role": "user", "content": "问"}])
    assert llm.text_kwargs[0]["effort"] == llm_module.EFFORT_MEDIUM


def test_a_plain_answer_needs_no_tools():
    llm = _LLM({"content": "你好"})
    assert _agent(llm).chat("嗨") == "你好"
    assert len(llm.calls) == 1


def test_a_tool_result_is_fed_back_and_the_model_answers():
    llm = _LLM(_call("ping"), {"content": "结果是 42"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {"value": 42}))
    assert agent.chat("问") == "结果是 42"
    assert '"value": 42' in llm.calls[1][-1]["content"]


def test_an_unknown_tool_is_reported_back_instead_of_crashing():
    llm = _LLM(_call("nope"), {"content": "换一个"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {}))
    agent.chat("问")
    assert "没有这个工具" in llm.calls[1][-1]["content"]


def test_broken_json_arguments_come_back_as_a_retryable_error():
    llm = _LLM(_call("ping", "{不是 json"), {"content": "好"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {}))
    agent.chat("问")
    assert json.loads(llm.calls[1][-1]["content"])["retryable"] is True


def test_bad_parameters_are_caught_before_the_tool_runs():
    ran = []
    tool = Tool("ping", "测试用的工具" * 6, lambda ctx, a: ran.append(a) or {},
                (Param("n", "int", "", default=1, max=10),))
    llm = _LLM(_call("ping", '{"n": 999}'), {"content": "好"})
    _agent(llm, tool).chat("问")
    assert ran == []
    assert "invalid_arguments" in llm.calls[1][-1]["content"]


def test_a_crashing_tool_does_not_take_down_the_conversation():
    def boom(ctx, args):
        raise RuntimeError("炸了")

    llm = _LLM(_call("ping"), {"content": "抱歉"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, boom))
    assert agent.chat("问") == "抱歉"
    assert "炸了" in llm.calls[1][-1]["content"]


def test_even_a_crash_in_the_validator_still_leaves_a_reply_for_every_call():
    """每条 tool_calls 必须配一条 tool 回复,少一条这个会话就永久 400 到 TTL 过期。

    所以校验阶段崩了也得配对 —— 这里直接让校验本身抛,模拟"兜底漏了一处"的情形。
    """
    class Exploding(Param):
        def coerce(self, value):
            raise OverflowError("cannot convert float infinity to integer")

    tool = Tool("ping", "测试用的工具" * 6, lambda ctx, a: {},
                (Exploding("n", "int", "", default=1),))
    llm = _LLM(_call("ping", '{"n": 1}'), {"content": "好"})
    assert _agent(llm, tool).chat("问") == "好"
    replies = [m for m in llm.calls[1] if m.get("role") == "tool"]
    assert len(replies) == 1
    assert "异常" in replies[0]["content"]


def test_a_tool_asking_for_confirmation_short_circuits_the_turn():
    """确认文案必须原样回给用户,不经模型转述 —— 转述会漏参数、会改数字。"""
    tool = Tool("rank", "昂贵的榜单工具" * 6, lambda ctx, a: {"ran": True},
                (Param("confirm", "bool", "确认", default=False),),
                expensive=True, confirmation=lambda params: "将执行【综合热榜】")
    llm = _LLM(_call("rank"), {"content": "模型自己又说了一遍"})
    assert _agent(llm, tool).chat("出个榜") == "将执行【综合热榜】"
    assert len(llm.calls) == 1      # 没有第二次请求


def test_the_step_guard_forces_an_answer_instead_of_looping_forever():
    llm = _LLM(*([_call("ping")] * 30), {"content": "不该走到这"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {}))
    answer = agent.chat("问")
    assert len(llm.calls) == 16     # 15 步 + 最后那次不带工具的收口
    assert llm.calls[-1] is not None
    assert answer


def test_the_final_call_offers_no_tools_so_the_model_must_conclude():
    seen = []

    class _Watch(_LLM):
        def chat(self, messages, **kwargs):
            seen.append(kwargs.get("tools"))
            return super().chat(messages, **kwargs)

    llm = _Watch(*([_call("ping")] * 20))
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {}))
    agent.chat("问")
    assert seen[-1] is None and seen[0] is not None


def test_an_llm_outage_gives_the_user_something_actionable():
    class _Dead(_LLM):
        def chat(self, messages, **kwargs):
            return None

    answer = _agent(_Dead()).chat("问")
    assert "失败" in answer and "重试" in answer


def test_an_overlong_message_is_refused_before_any_llm_call():
    llm = _LLM({"content": "不该被调用"})
    assert "过长" in _agent(llm).chat("x" * 3000)
    assert llm.calls == []


def test_each_round_gets_a_fresh_delta_callback_that_resets_the_frontend():
    """模型在决定调工具那轮常先吐几句过渡话,不清掉会和最终回答粘在一起。"""
    resets = []

    class _Streaming(_LLM):
        def chat(self, messages, **kwargs):
            if cb := kwargs.get("on_delta"):
                cb("片段")
            return super().chat(messages, **kwargs)

    llm = _Streaming(_call("ping"), {"content": "答案"})
    agent = _agent(llm, Tool("ping", "测试用的工具" * 6, lambda ctx, a: {}))
    agent.chat("问", on_delta=lambda piece, reset: resets.append(reset))
    assert resets == [True, True]       # 两轮,每轮第一片都带 reset
