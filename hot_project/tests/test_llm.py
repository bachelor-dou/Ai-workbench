"""LLM 层的守卫:目录归一化、后端参数白名单、回退顺序、流式拼装。

一次网络都不发 —— `requests.post` 整个被替换掉。这些逻辑历史上出过的问题
(给 azure 发 temperature 直接 400、硬切换悄悄回退到别家、流式重试导致前端文字重复)
全都不需要真实的 LLM 才能复现。
"""

import json

import pytest
import requests

from hot_project.infra.llm import protocol
from hot_project.infra.llm.client import LLMClient
from hot_project.infra.llm.api import Api, build


def api(sid="p1", backend="openai", key="k", lite=()):
    return Api(id=sid, label=sid, backend=backend, url=f"https://{sid}.test",
               model=f"{sid}-main", key=key, lite_models=list(lite))


class Recorder:
    """记下每次请求发到了哪、发了什么;按预设脚本回复。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers=None, json=None, timeout=None, stream=False):
        self.calls.append((url, json))
        reply = self.replies.pop(0) if self.replies else _ok("ok")
        if callable(reply):
            reply = reply()
        return reply

    @property
    def urls(self):
        return [url for url, _ in self.calls]


class _Resp:
    def __init__(self, status=200, body=None, lines=None, text=""):
        self.status_code = status
        self._body = body
        self._lines = lines or []
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _ok(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return _Resp(body={"choices": [{"message": msg}]})


def _sse(*chunks):
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return _Resp(lines=lines)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """退避重试的 sleep 直接跳过 —— 测的是重试逻辑,不是计时器。"""
    monkeypatch.setattr(protocol.time, "sleep", lambda _s: None)


@pytest.fixture
def post(monkeypatch):
    def install(*replies):
        rec = Recorder(*replies)
        monkeypatch.setattr(protocol.requests, "post", rec)
        return rec
    return install


# ── 目录归一化 ────────────────────────────────────────────────────

def test_key_comes_from_the_environment_not_the_catalog():
    """目录里只有 key_env,真值到这里才落地 —— 这是 LLM_MODELS 能安全序列化的前提。"""
    got = build([{"id": "a", "url": "u", "model": "m", "key_env": "SOME_KEY"}],
                lambda name: "secret" if name == "SOME_KEY" else "")
    assert got[0].key == "secret"
    assert "secret" not in json.dumps(got[0].public())


def test_a_disabled_platform_disappears_entirely():
    got = build([{"id": "a", "enabled": 0}, {"id": "b", "enabled": 1}], lambda _n: "k")
    assert [s.id for s in got] == ["b"]


def test_enabled_as_the_string_zero_still_means_off():
    """手写配置里 `"enabled": "0"` 很自然(旁边几行都带引号)。裸 bool() 会判成真,
    后果是关掉的平台照样被调用 —— 不报错,只是账单上多一笔。"""
    assert build([{"id": "a", "enabled": "0"}], lambda _n: "k") == []


def test_duplicate_ids_are_a_hard_error():
    """id 是整条选择链路的键;重复会让「选 A 却调到 B」静默发生。"""
    with pytest.raises(ValueError, match="id 重复"):
        build([{"id": "same"}, {"id": "same"}], lambda _n: "k")


def test_lite_models_split_and_dedupe():
    got = build([{"id": "a", "lite_model": " x , y , x "}], lambda _n: "k")
    assert got[0].lite_models == ["x", "y"]


def test_a_platform_without_a_key_is_not_usable():
    assert not api(key="").usable


# ── 后端参数白名单 ────────────────────────────────────────────────

def test_azure_gets_max_completion_tokens_and_no_temperature():
    """发了 azure 不认的参数不是被忽略,是整个请求 400。"""
    body = protocol.payload("azure", "m", [], max_tokens=100, temperature=0.5,
                            effort="high")
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "enable_thinking" not in body      # azure 只认 reasoning_effort
    assert "thinking_budget" not in body


def test_openai_gets_the_full_parameter_set():
    """这里只管「哪些参数该出现」。有意用 off:开了思考 max_tokens 要额外给量,
    那笔算术归下面的配额测试,混在一起两个意图都读不清。"""
    body = protocol.payload("openai", "m", [], max_tokens=100, temperature=0.5,
                            effort="off")
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5
    assert body["enable_thinking"] is False


# ── 思考档位 ──────────────────────────────────────────────────────

def test_every_backend_translates_every_level():
    """档位是网页直接传进来的键:任何一档在任何后端都必须译得出参数,否则用户选了
    「更深思考」却发出一个不带思考的请求,而且没人报错。"""
    for backend in ("azure", "openai"):
        for effort in protocol.EFFORTS:
            body = protocol.payload(backend, "m", [], effort=effort)
            assert "reasoning_effort" in body or "enable_thinking" in body, (backend, effort)


def test_the_openai_side_asks_for_a_budget_and_never_an_effort():
    """百炼的 thinking_budget 和 reasoning_effort 同时出现直接报错,而两家的 effort 取值
    还各不相同 —— 统一只发预算,这条路才对 qwen 和 glm 同时成立。"""
    for effort in protocol.EFFORTS:
        assert "reasoning_effort" not in protocol.payload("openai", "m", [], effort=effort)


def test_thinking_off_is_a_real_switch_not_a_missing_parameter():
    """`off` 必须显式发出关闭指令:漏发等于用平台默认,而两边的默认都不是「不思考」
    (azure 5.1 起默认 none,百炼 qwen3.5 以上默认开)。"""
    assert protocol.payload("azure", "m", [], effort="off")["reasoning_effort"] == "none"
    assert protocol.payload("openai", "m", [], effort="off")["enable_thinking"] is False


def test_the_batch_level_thinks_but_stays_shallower_than_a_conversation():
    """内部批量调用(项目介绍 / 批量浓缩 / 历史压缩)走的中间档。两个条件都得成立:真的
    在思考(这三处曾经是显式关掉,总结明显更泛),又明显浅于对话档 —— 一份报告要跑几十个
    仓库,一次压缩卡在用户和回答之间,深度翻上去就是几十倍的等待。"""
    assert protocol.payload("azure", "m", [], effort="medium")["reasoning_effort"] == "medium"
    mid = protocol.payload("openai", "m", [], effort="medium")
    assert mid["enable_thinking"] is True
    assert mid["thinking_budget"] < protocol.payload(
        "openai", "m", [], effort=protocol.EFFORT_HIGH)["thinking_budget"]


def test_the_deeper_level_is_offered_under_the_platform_s_own_name():
    """菜单上写平台自己的说法(azure 的最深档叫 xhigh),用户才对得上人家的文档。
    但要发回来的值仍然是我们的档位名 —— 把展示名当档位传进来会被 `level()` 当错字,
    静默退回默认档,用户点了最深却拿到默认深度。"""
    assert protocol.deeper_label("azure", "gpt-5.4") == "xhigh"
    assert protocol.deeper_label("azure", "gpt-5.5") == ""      # 对话不能思考,连名字都不给
    assert protocol.deeper_label("foundry", "gpt-5.6-terra") == ""
    assert protocol.deeper_label("openai") == protocol.EFFORT_MAX
    assert protocol.deeper_label("gemini") == ""            # 没有更深档就没有名字
    assert protocol.level("xhigh") == protocol.EFFORT_DEFAULT


def test_an_empty_or_bogus_level_falls_back_to_thinking():
    """空档位曾经等于「一个思考参数都不发」,而两边的平台默认都不是「思考」(azure 5.1 起
    默认 none)。于是漏传就是静默不思考:回答只是变差,没有任何报错。认不出的值一律落回
    默认档 —— 想关思考只有显式传 `off` 一条路。"""
    for given in ("", "hgih", "extreme"):
        assert protocol.payload("azure", "m", [], effort=given)["reasoning_effort"] == "high"
        assert protocol.payload("openai", "m", [], effort=given)["enable_thinking"] is True


def test_an_unknown_backend_sends_no_thinking_parameter_at_all():
    """白名单:没登记的后端一个思考参数都不发(乱发是 400,不是被忽略)。"""
    body = protocol.payload("gemini", "m", [], effort="max")
    assert "enable_thinking" not in body and "reasoning_effort" not in body
    assert protocol.deeper("gemini") == ""       # 也就不该给它那个开关


def test_thinking_is_never_allowed_to_eat_the_answer_s_tokens():
    """思考和正文抢同一个上限。实测过一次后果:摘要那 600 的上限被思维链一个人用光
    (正文 0 字、finish_reason=length),而空回复在上层就是「这家失败了」—— 摘要静默
    回退成旧的,没有任何报错。所以开了思考就要在正文之外额外给量。"""
    deep = protocol.payload("azure", "m", [], max_tokens=16384, effort="max")
    assert deep["max_completion_tokens"] == protocol.DEEP_MIN_TOKENS
    assert protocol.payload("azure", "m", [], max_tokens=16384,
                            effort="high")["max_completion_tokens"] == 16384
    mid = protocol.payload("openai", "m", [], max_tokens=600, effort="medium")
    assert mid["max_tokens"] == 600 + mid["thinking_budget"], "预算要叠在正文之上,不是抢它的"
    assert protocol.payload("openai", "m", [], max_tokens=600,
                            effort="off")["max_tokens"] == 600, "不思考就不该多给"
    assert protocol.payload("openai", "m", [], max_tokens=16384,
                            effort="max")["max_tokens"] == protocol.WIRE_MAX_TOKENS


def test_the_deeper_level_reaches_the_wire(post):
    """翻译对了没用,还得真发出去 —— `request` 忘了带档位的话上面几个测试全绿。"""
    rec = post(_ok("hi"))
    LLMClient([api("p1")]).chat([], effort=protocol.deeper("openai"))
    assert rec.calls[0][1]["thinking_budget"] == 32768


def test_azure_authenticates_with_a_header_of_its_own():
    assert protocol.headers("azure", "k") == {"api-key": "k", "Content-Type": "application/json"}
    assert protocol.headers("openai", "k")["Authorization"] == "Bearer k"


def test_foundry_sends_an_azure_body_with_a_bearer_key():
    """Foundry 项目端点是两个轴的组合:认证像 openai(Bearer 项目 key),请求体像 azure。
    弄反哪一边都不是被忽略 —— 前者 401,后者 400。"""
    assert "api-key" not in protocol.headers("foundry", "k")
    assert protocol.headers("foundry", "k")["Authorization"] == "Bearer k"
    body = protocol.payload("foundry", "m", [], max_tokens=100, temperature=0.5, effort="high")
    assert body["max_completion_tokens"] == 100
    assert "temperature" not in body and "max_tokens" not in body
    assert body["reasoning_effort"] == "high"
    assert protocol.deeper("foundry", "gpt-5.6-terra") == ""   # 对话带工具不能思考,不给选项


def test_newer_azure_models_with_tools_must_not_mention_thinking_at_all():
    """gpt-5.5 起(terra 是 5.6)chat/completions 不许「工具 + reasoning_effort」并存,
    整个请求 400,错误原文让去 /v1/responses;5.5 连显式 none 都拒,这个键必须整个缺席。
    对话每一步都带工具表,探活不带 —— 于是「模型测试能通、问答用不了」。
    gpt-5.4 实测没这堵墙,不许被殃及,不带工具的调用也照常思考。"""
    tool = [{"type": "function", "function": {"name": "t"}}]
    for backend, model in (("azure", "gpt-5.5"), ("foundry", "gpt-5.6-terra")):
        body = protocol.payload(backend, model, [], tools=tool, effort="high")
        assert "reasoning_effort" not in body, (backend, model)
        assert protocol.payload(backend, model, [],
                                effort="high")["reasoning_effort"] == "high"
    # gpt-6:默认档不再是 none,带工具时键缺席照样 400,必须显式发 none(实测)。
    body = protocol.payload("azure", "gpt-6-sol", [], tools=tool, effort="high")
    assert body["reasoning_effort"] == "none"
    assert protocol.payload("azure", "gpt-6-sol", [],
                            effort="high")["reasoning_effort"] == "high"
    assert protocol.payload("azure", "gpt-5.4", [], tools=tool,
                            effort="high")["reasoning_effort"] == "high"


# ── 回退顺序 ──────────────────────────────────────────────────────

def test_an_internal_call_falls_through_to_the_next_platform(post):
    rec = post(_Resp(status=500, text="boom"), _Resp(status=500, text="boom"),
               _Resp(status=500, text="boom"), _ok("from second"))
    client = LLMClient([api("p1"), api("p2")])
    assert client.text("hi") == "from second"
    assert rec.urls[-1] == "https://p2.test"


def test_a_hard_switch_never_falls_back(post):
    """用户明确选了 A,悄悄给出 B 的答案是另一回事 —— 宁可返回 None。"""
    rec = post(*[_Resp(status=500, text="boom")] * 3)
    client = LLMClient([api("p1"), api("p2")])
    assert client.chat([{"role": "user", "content": "hi"}], model_id="p1") is None
    assert set(rec.urls) == {"https://p1.test"}


def test_asking_for_a_platform_that_is_not_configured_fails_fast(post):
    rec = post()
    assert LLMClient([api("p1")]).chat([], model_id="nope") is None
    assert rec.calls == []


def test_an_empty_reply_counts_as_failure(post):
    """200 但正文为空是最难查的一类失败:不换平台的话,整批描述静默变空。"""
    rec = post(_ok(""), _ok(""), _ok(""), _ok("real"))
    assert LLMClient([api("p1"), api("p2")]).text("hi") == "real"
    assert rec.urls[-1] == "https://p2.test"


def test_a_reply_with_only_tool_calls_is_not_empty(post):
    """agent 那条路上模型只回 tool_calls、正文为空是正常的,不能当失败重试。"""
    post(_ok(None, tool_calls=[{"id": "1"}]))
    data = LLMClient([api("p1")]).chat([], tools=[{"type": "function"}])
    assert data["choices"][0]["message"]["tool_calls"]


def test_a_stream_of_only_tool_calls_is_not_empty_either(post):
    """流式下同理,而且这是 agent 用工具时的常态。当成空响应会重试三遍、重复调工具。"""
    rec = post(_sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "search", "arguments": "{}"}}]}}]}))
    data = LLMClient([api("p1")]).chat([], tools=[{"type": "function"}],
                                          on_delta=lambda _p: None)
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
    assert len(rec.calls) == 1


def test_lite_borrows_a_sub_model_from_another_platform(post):
    """主模型选 p1,但只有 p2 配了子模型 —— 便宜活儿该借 p2 干。"""
    rec = post(_ok("cheap"))
    client = LLMClient([api("p1"), api("p2", lite=["tiny"])])
    assert client.text("hi", lite=True, model_id="p1") == "cheap"
    assert rec.calls[0][1]["model"] == "tiny"


def test_lite_falls_back_to_the_main_model_when_nobody_has_one(post):
    """全都没配子模型时退回主模型:贵,但至少调得通。"""
    rec = post(_ok("x"))
    LLMClient([api("p1")]).text("hi", lite=True)
    assert rec.calls[0][1]["model"] == "p1-main"


def test_an_unconfigured_platform_is_skipped_without_a_request(post):
    rec = post(_ok("x"))
    LLMClient([api("p1", key=""), api("p2")]).text("hi")
    assert rec.urls == ["https://p2.test"]


# ── 流式 ──────────────────────────────────────────────────────────

def test_streaming_pieces_are_emitted_live_and_joined(post):
    post(_sse({"choices": [{"delta": {"content": "你"}}]},
              {"choices": [{"delta": {"content": "好"}}]}))
    seen = []
    data = LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert seen == ["你", "好"]
    assert data["choices"][0]["message"]["content"] == "你好"


def test_a_stream_without_a_charset_header_still_decodes_as_utf8(monkeypatch):
    """硅基流动的 SSE 头是裸的 text/event-stream(没写 charset),requests 按 RFC2616 的
    老默认猜 ISO-8859-1 —— 每个中文字被逐字节错解,整屏乱码。SSE 规范本身固定 UTF-8,
    解码必须钉死,不看响应头的脸色。

    这条有意用**真的** requests.Response(假响应对象绕过了出错的那层解码),
    并先断言前提成立:requests 对这种头确实会猜错。"""
    import io

    sse = ('data: {"choices":[{"delta":{"content":"中文测试"}}]}\n\n'
           "data: [DONE]\n\n").encode("utf-8")
    resp = requests.Response()
    resp.status_code = 200
    resp.raw = io.BytesIO(sse)
    resp.headers["Content-Type"] = "text/event-stream"      # 和硅基流动一致:没有 charset
    resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)  # 适配层就是这么定的
    assert resp.encoding == "ISO-8859-1", "前提变了:requests 不再猜错的话这条测试该退役"

    monkeypatch.setattr(protocol.requests, "post", lambda *a, **k: resp)
    seen = []
    data, _ = protocol._stream(api("p1"), "m", {"model": "m", "messages": []},
                               seen.append, timeout=5)
    assert "".join(seen) == "中文测试"
    assert data["choices"][0]["message"]["content"] == "中文测试"


def test_tool_call_fragments_merge_by_index():
    """OpenAI 流式约定:首片带 name,后续片只带 arguments 的一截,要按 index 拼。"""
    acc = {}
    protocol.merge_toolcall_fragment(acc, {"index": 0, "id": "c1",
                                       "function": {"name": "search", "arguments": '{"q"'}})
    protocol.merge_toolcall_fragment(acc, {"index": 0, "function": {"arguments": ':"x"}'}})
    assert acc[0]["function"] == {"name": "search", "arguments": '{"q":"x"}'}


def test_repeated_index_in_one_chunk_lands_in_the_same_slot():
    """vLLM 的推测解码会在同一 chunk 里重复 index;各成一项的话参数会被切两半。"""
    acc = {}
    for frag in ({"index": 0, "function": {"arguments": "ab"}},
                 {"index": 0, "function": {"arguments": "cd"}}):
        protocol.merge_toolcall_fragment(acc, frag)
    assert len(acc) == 1 and acc[0]["function"]["arguments"] == "abcd"


def test_a_leaked_toolcall_blob_never_reaches_the_user(post):
    """有些模型把本该走 tool_calls 的调用当文本吐在正文开头,这段不能给用户看见。"""
    leak = '{"tool_uses":[{"recipient_name":"functions.x","parameters":{}}]}真正的回答'
    post(_sse({"choices": [{"delta": {"content": leak}}]}))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == "真正的回答"


def test_an_answer_that_legitimately_starts_with_json_is_left_alone(post):
    """用户问「给我一段 JSON」时,回答本来就以 `{` 开头 —— 不能误剥。"""
    answer = '{"name": "demo", "version": 1}'
    post(_sse({"choices": [{"delta": {"content": answer}}]}))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == answer


def test_an_unclosed_opening_brace_is_still_released_at_the_end(post):
    """收流结束仍卡在闸门里(花括号始终没闭合):放行,不能把用户的内容吞了。"""
    post(_sse({"choices": [{"delta": {"content": '{"half": '}}]}))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert "".join(seen) == '{"half": '


def test_prose_is_emitted_without_waiting_for_the_gate(post):
    """绝大多数答案以散文开头。一律缓冲到能判定的话,每个回答的首字延迟都要买单。"""
    post(_sse({"choices": [{"delta": {"content": "这"}}]},
              {"choices": [{"delta": {"content": "是答案"}}]}))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert seen[0] == "这"          # 第一片就出去了,没等第二片


class _DiesMidStream(_Resp):
    """发出一片正文之后连接断掉 —— 前端已经看到「半」了。"""

    def iter_lines(self, decode_unicode=False):
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "半"}}]})}'
        raise protocol.requests.ConnectionError("连接断了")


def test_a_stream_that_already_emitted_is_never_retried(post):
    """重试会让前端看到重复的文字。宁可只返回已收到的部分。"""
    rec = post(_DiesMidStream(), _sse({"choices": [{"delta": {"content": "完整答案"}}]}))
    seen = []
    LLMClient([api("p1"), api("p2")]).chat([], on_delta=seen.append)
    assert seen == ["半"]
    assert len(rec.calls) == 1      # 没有第二次请求,也没换平台


def test_a_stream_that_died_before_emitting_can_still_retry(post):
    rec = post(_Resp(status=500, text="boom"), _sse({"choices": [{"delta": {"content": "好"}}]}))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert seen == ["好"]
    assert len(rec.calls) == 2


def test_malformed_sse_lines_are_skipped_not_fatal(post):
    post(_Resp(lines=["", ": comment", "event: ping", "data: {half",
                      f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}',
                      "data: [DONE]"]))
    seen = []
    LLMClient([api("p1")]).chat([], on_delta=seen.append)
    assert seen == ["ok"]


def test_the_connection_is_returned_on_every_path(post):
    """非 200 早退这条路上忘了 close,连接池会慢慢耗干 —— 症状出现在几小时之后。"""
    dead = _Resp(status=500, text="boom")
    post(dead, dead, dead)
    LLMClient([api("p1")]).chat([], on_delta=lambda _p: None)
    assert dead.closed
