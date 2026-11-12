import pytest

from hot_project.agent.history import Session
from hot_project.agent.loop import Agent
from hot_project.tests.test_agent import _LLM, _call
from hot_project.tools.spec import Ctx, Param, Registry, Tool


def _registry(runs):
    return Registry([Tool(
        "ranking", "生成需要用户确认的榜单", lambda ctx, args: runs.append(args) or {"ok": True},
        (Param("keywords", "list_str", "关键词"),
         Param("top_n", "int", "数量", default=20, min=1, max=200),
         Param("confirm", "bool", "明确确认", default=False)),
        expensive=True,
    )])


def test_repeating_the_same_request_does_not_confirm_it():
    runs = []
    registry = _registry(runs)
    ctx = Ctx(state=Session())
    first = registry.run(ctx, "ranking", {"keywords": ["ocr"]})
    second = registry.run(ctx, "ranking", {"keywords": ["ocr"]})
    assert first["needs_confirmation"] is True
    assert second.get("needs_confirmation") is True
    assert runs == []


def test_confirmation_can_omit_required_parameters_and_uses_the_shown_values():
    runs = []
    registry = _registry(runs)
    ctx = Ctx(state=Session())
    registry.run(ctx, "ranking", {"keywords": ["ocr"], "top_n": 12})
    ctx.begin_turn("开始")
    result = registry.run(ctx, "ranking", {"confirm": True})
    assert result == {"ok": True}
    assert runs == [{"keywords": ["ocr"], "top_n": 12}]


def test_a_model_supplied_confirm_flag_cannot_override_the_users_message():
    runs = []
    agent = Agent(client=_LLM(
        _call("ranking", '{"keywords": ["ocr"]}'),
        _call("ranking", '{"keywords": ["ocr"], "confirm": true}', cid="c2"),
        {"content": "已执行"},
    ), tools=_registry(runs), gh=object())
    agent.chat("给我关键词榜")
    response = agent.chat("先不要开始，解释一下")
    assert runs == []
    assert "确认" in response


def test_a_batch_cannot_replace_the_confirmation_that_the_user_was_shown():
    runs = []
    first = _call("ranking", '{"keywords": ["ocr"], "top_n": 10}')
    second = _call("ranking", '{"keywords": ["other"], "top_n": 30}', cid="c2")
    client = _LLM(
        {"tool_calls": first["tool_calls"] + second["tool_calls"]},
        _call("ranking", '{"confirm": true}', cid="c3"),
        {"content": "完成"},
    )
    agent = Agent(client=client, tools=_registry(runs), gh=object())
    shown = agent.chat("给我榜单")
    assert "'top_n': 10" in shown
    assert runs == []
    assert agent.chat("开始") == "完成"
    assert runs == [{"keywords": ["ocr"], "top_n": 10}]
    replies = [m for m in agent.session.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["c1", "c2", "c3"]


@pytest.mark.parametrize("message", ["开始", "确认！", " GO. ", "yes", "确认执行"])
def test_an_explicit_user_reply_authorizes_the_shown_parameters_once(message):
    runs = []
    client = _LLM(
        _call("ranking", '{"keywords": ["ocr"], "top_n": 12}'),
        _call("ranking", '{"confirm": true, "top_n": 30}', cid="c2"),
        {"content": "完成"},
        _call("ranking", '{"keywords": ["ocr"], "confirm": true}', cid="c3"),
    )
    agent = Agent(client=client, tools=_registry(runs), gh=object())
    agent.chat("出榜")
    assert agent.chat(message) == "完成"
    assert runs == [{"keywords": ["ocr"], "top_n": 12}]
    assert "确认" in agent.chat("再看看")
    assert len(runs) == 1


def test_withdrawn_confirmation_cannot_be_revived_by_a_later_start():
    runs = []
    client = _LLM(
        _call("ranking", '{"keywords": ["ocr"]}'),
        {"content": "不执行"},
        _call("ranking", '{"keywords": ["ocr"], "confirm": true}', cid="c2"),
    )
    agent = Agent(client=client, tools=_registry(runs), gh=object())
    agent.chat("出榜")
    agent.chat("取消")
    assert "确认" in agent.chat("开始")
    assert runs == []


@pytest.mark.parametrize("extra", [{"top_n": 999}, {"unknown": True}, {"keywords": 123}])
def test_confirmation_does_not_bypass_parameter_validation(extra):
    runs = []
    registry = _registry(runs)
    ctx = Ctx(state=Session())
    registry.run(ctx, "ranking", {"keywords": ["ocr"]})
    ctx.begin_turn("开始")
    result = registry.run(ctx, "ranking", {"confirm": True, **extra})
    assert result["invalid_arguments"]
    assert runs == []


def test_start_cannot_authorize_an_offer_created_in_the_same_turn():
    runs = []
    ctx = Ctx(state=Session())
    registry = _registry(runs)
    ctx.begin_turn("开始")
    first = registry.run(ctx, "ranking", {"keywords": ["ocr"], "confirm": True})
    second = registry.run(ctx, "ranking", {"confirm": True})
    assert first["needs_confirmation"] is second["needs_confirmation"] is True
    assert runs == []
