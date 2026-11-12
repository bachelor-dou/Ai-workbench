"""线上协议:请求头、请求体、一次请求(含流式)。

各家「兼容 OpenAI」的程度不一样,差异集中在这个文件里,别处不该再出现 `if backend ==`。
已知差异:azure(gpt-5.x)用 max_completion_tokens,不认 temperature;思考怎么开也一家一套,
翻译收在 `_THINKING` 表里,上层只说档位(off/high/max),不碰参数名。

发了对方不认的参数不是被忽略,是整个请求 400 —— 所以这里是白名单:新参数默认不发给
未知后端,由加参数的人显式列进去。
"""

from __future__ import annotations

import json
import logging
import time

import requests

logger = logging.getLogger("hot_project")

AZURE = "azure"
FOUNDRY = "foundry"     # Foundry 项目端点:请求体和 azure 同一套,认证却是 Bearer 项目 key
RETRY_BACKOFF = (1.0, 2.0, 4.0)


def _family(backend: str) -> str:
    """请求体按哪家的规矩拼。认证不同不代表请求体不同 —— Foundry 项目端点走 Bearer,
    但发的仍是 gpt-5.x 那套(max_completion_tokens + reasoning_effort)。
    """
    return AZURE if backend == FOUNDRY else backend

# ── 思考档位 ────────────────────────────────────────────────────────
# 对话有两档:默认「高」,以及有的家还能更深。`medium` 不上菜单 —— 它是内部批量调用
# (项目介绍 / 批量浓缩 / 历史压缩)那一档:要它思考,但不值得为一句摘要等 high 的时间。
# 档位名一律不透传 —— 各家的取值不同义(百炼 GLM 的 high 是它的最低档,azure 的 high 是
# 次高档),翻译只在这张表里。
EFFORT_OFF = "off"
EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"
EFFORT_MAX = "max"
EFFORTS = (EFFORT_OFF, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX)
EFFORT_DEFAULT = EFFORT_HIGH    # 对话默认思考;内部批量调用降到 medium,只有预检显式关掉

_THINKING: dict[str, dict[str, dict]] = {
    # azure(gpt-5.x):只认 reasoning_effort,而且 gpt-5.1 起默认是 none —— 不显式发就
    # 根本不思考。思考 token 计入 max_completion_tokens,越深正文被截断的风险越大。
    # (`max` 只有 Responses API 支持,这条路最深就是 xhigh。)
    AZURE: {
        EFFORT_OFF: {"reasoning_effort": "none"},
        EFFORT_MEDIUM: {"reasoning_effort": "medium"},
        EFFORT_HIGH: {"reasoning_effort": "high"},
        EFFORT_MAX: {"reasoning_effort": "xhigh"},
    },
    # 兼容 OpenAI 的几家(硅基流动、百炼):开关 + 思考预算。这里刻意用预算而不是各家的
    # reasoning_effort —— 后者取值一家一套(百炼 qwen 只认 low/medium/xhigh、glm 只认
    # high/max),而预算对两边都通用,还顺带避开百炼「预算和 effort 同时发直接报错」。
    # 32768 是硅基流动的上限,在百炼那边换算过去正好是 xhigh。
    "openai": {
        EFFORT_OFF: {"enable_thinking": False},
        EFFORT_MEDIUM: {"enable_thinking": True, "thinking_budget": 4096},
        EFFORT_HIGH: {"enable_thinking": True, "thinking_budget": 16384},
        EFFORT_MAX: {"enable_thinking": True, "thinking_budget": 32768},
    },
}


DEEP_MIN_TOKENS = 32768     # azure 最深档的配额地板
WIRE_MAX_TOKENS = 32768     # 各家愿意接受的正文上限,再往上是 400(数值和上面撞巧合,含义无关)


def _ceiling(backend: str, effort: str, max_tokens: int | None) -> int | None:
    """思考 token 和正文抢同一个上限,开了思考就得额外给量。

    不给的后果实测过一次:摘要那 600 的上限,被思维链一个人用光(1493 字思考、正文 0 字、
    finish_reason=length),而空回复在上层就是「这家失败了」—— 摘要静默回退成旧的。

    `thinking_budget` 只是**上限**,不是配额:实测预算给 512,模型照样思考了 2000 多字。
    所以这里按声明的预算给正文让路,而不是指望模型自己收着。azure 不说思考会用多少,
    只能在最深档给一个够大的地板。
    """
    if max_tokens is None:
        return None
    family = _family(backend)
    if family == AZURE:
        return max(max_tokens, DEEP_MIN_TOKENS) if effort == EFFORT_MAX else max_tokens
    budget = _THINKING.get(family, {}).get(effort, {}).get("thinking_budget", 0)
    return max(max_tokens, min(max_tokens + budget, WIRE_MAX_TOKENS))


# ponytail: 按部署名前缀识别这堵墙 —— azure 的 model 字段是部署名,我们的部署恰好都按
# 模型名起名;部署名不这么起就识别不出,根治是实现 /v1/responses 方言。
NO_THINKING_WITH_TOOLS = ("gpt-5.5", "gpt-5.6", "gpt-6")
# gpt-6 的默认档不再是 none(实测):带工具时键缺席照样 400,必须显式 reasoning_effort=none;
# 5.5/5.6 相反,连显式 none 都拒,键必须整个缺席。
MUTE_IS_EXPLICIT_NONE = ("gpt-6",)


def tools_mute_thinking(backend: str, model: str) -> bool:
    """这个模型在 chat/completions 上是否「带工具就不许提 reasoning_effort」。

    gpt-5.5 起(terra 是 5.6)的墙:两者并存整个请求 400,错误原文让去 /v1/responses;
    5.5 连显式 none 都拒,所以带工具时这个键必须整个缺席。gpt-5.4 实测没有这堵墙。
    """
    return _family(backend) == AZURE and model.startswith(NO_THINKING_WITH_TOOLS)


def level(effort: str) -> str:
    """认不出的档位落回默认档,而不是「什么都不发」。

    这是整条链上唯一守住「不可能静默不思考」的地方:两边的平台默认都不是思考
    (azure 5.1 起是 none),所以漏传或写错一旦被当成「用平台默认」,结果就是回答变差
    而没有任何报错。想关思考只有显式传 `off` 一条路。
    """
    return effort if effort in EFFORTS else EFFORT_DEFAULT


def deeper(backend: str, model: str = "") -> str:
    """这个接入有没有比默认更深的一档:有就返回那个档位名,没有返回空串。

    网页据此决定要不要给这一组选项 —— 点了没反应的选项比没有选项更糟。
    被「工具 + 思考」那堵墙拦住的模型(5.5/5.6)直接没有:对话每一步都带工具,
    档位选得再深也发不出去,给选项就是骗人。
    """
    if tools_mute_thinking(backend, model):
        return ""
    return EFFORT_MAX if EFFORT_MAX in _THINKING.get(_family(backend), {}) else ""


def deeper_label(backend: str, model: str = "") -> str:
    """更深那一档在这家的原名(azure 是 xhigh),没有原名就用我们的档位名。

    纯展示用:菜单上写平台自己的说法,用户才对得上文档。要传的值仍然是 `deeper()`
    给的档位名,两者不能互换 —— 把 `xhigh` 当档位发进来会被 `level()` 当错字。
    """
    if not deeper(backend, model):
        return ""
    fragment = _THINKING[_family(backend)][EFFORT_MAX]
    return fragment.get("reasoning_effort") or EFFORT_MAX


def headers(backend: str, key: str) -> dict:
    """`api-key` 只有 Azure OpenAI 资源认;Foundry 项目端点要项目 key 走 Bearer
    (拿 api-key 发过去是 401,不是 403,报错信息还会指向「wrong API endpoint」误导人)。
    """
    if backend == AZURE:
        return {"api-key": key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def payload(
    backend: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    effort: str = "",
    tools: list[dict] | None = None,
) -> dict:
    body: dict = {"model": model, "messages": messages}
    effort = level(effort)
    # 撞上「工具 + 思考」那堵墙的模型:对话每一步都带工具表,探活不带 —— 于是
    # 「模型测试能通、问答用不了」。空档位在翻译表里查不到,自然一个思考参数都不发,
    # 这正是这堵墙要的(5.5 连显式 none 都拒;平台默认就是 none,效果一样)。
    if tools and tools_mute_thinking(backend, model):
        effort = EFFORT_OFF if model.startswith(MUTE_IS_EXPLICIT_NONE) else ""
    max_tokens = _ceiling(backend, effort, max_tokens)
    family = _family(backend)
    if family == AZURE:
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
    else:
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
    # 白名单:档位到这里一定认得(上面归一化过),但没登记的后端仍然一个参数都不发 ——
    # 乱发是 400,不是被忽略
    if fragment := _THINKING.get(family, {}).get(effort):
        body.update(fragment)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


# ── 工具调用泄漏 ─────────────────────────────────────────────────────

_LEAK_KEYS = ('"tool_uses"', '"recipient_name"', '"tool_calls"', '"parameters"')


def strip_leaked_toolcall(text: str) -> str | None:
    """剥掉正文开头那段疑似工具调用泄漏的 JSON。

    三态,流式靠它决定要不要继续缓冲:剥完的余文 / 原文(不含泄漏特征,不能动)/
    None(花括号未闭合,判不了)。
    """
    s = text.lstrip()
    if not s.startswith("{"):
        return text
    depth = 0
    in_str = esc = False
    end = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    blob = s[:end]
    return text if not any(k in blob for k in _LEAK_KEYS) else s[end:].lstrip()


def merge_toolcall_fragment(acc: dict, frag: dict) -> None:
    """把一个流式 tool_call 增量片段按 index 合并进累加器。

    首片带 id/type/name,后续片只带 arguments 的一截。同一 chunk 里重复的 index
    (vLLM 推测解码会这么发)必须落进同一槽位,不能各成一项。
    """
    slot = acc.setdefault(frag.get("index", 0),
                          {"id": "", "type": "function",
                           "function": {"name": "", "arguments": ""}})
    if frag.get("id"):
        slot["id"] = frag["id"]
    if frag.get("type"):
        slot["type"] = frag["type"]
    fn = frag.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


class _HeadGate:
    """正文开头的闸门:只拦「以 `{` 起头、疑似工具泄漏」的那一段。

    散文和 Markdown 零延迟放行,否则每个回答的首字延迟都要为这个罕见情况买单。
    """

    def __init__(self, emit) -> None:
        self._emit = emit
        self._open = False
        self._buf = ""

    def feed(self, piece: str) -> None:
        if self._open:
            self._emit(piece)
            return
        self._buf += piece
        if not self._buf.strip():
            return                      # 目前只有空白,判不了,继续等
        result = strip_leaked_toolcall(self._buf)
        if result is None:
            return                      # 开头是还没闭合的 JSON,继续缓冲
        self._release(result)           # 散文在这里原样放行,不多等一片

    def flush(self) -> None:
        """收流结束仍卡在闸门里(开头的 `{` 始终没闭合):尽力剥一次再放行,不丢用户内容。"""
        if self._open or not self._buf.strip():
            return
        result = strip_leaked_toolcall(self._buf)
        self._release(self._buf if result is None else result)

    def _release(self, text: str) -> None:
        self._open = True
        self._buf = ""
        self._emit(text)


def _stream(api, model: str, body: dict, on_delta, timeout: int) -> tuple[dict | None, bool]:
    """一次 SSE 请求,边收边把正文增量喂给 `on_delta`。不重试。

    返回 `(data, emitted)`。`emitted` 为真就**不能重试**,否则前端会看到重复的文字。
    """
    body = dict(body, stream=True)
    parts: list[str] = []
    tool_acc: dict = {}
    finish_reason = None
    emitted = False
    resp = None
    started = time.time()
    first_at = None     # 首字耗时:分辨「转半天不吐字」是模型在思考还是流式假死
    pieces = 0          # 增量条数:接近字符数 = 逐 token 流;远小于 = 被网关整块缓冲

    def emit(text: str) -> None:
        nonlocal first_at, pieces, emitted
        if not text:
            return
        if first_at is None:
            first_at = time.time() - started
        pieces += 1
        parts.append(text)
        emitted = True
        on_delta(text)

    gate = _HeadGate(emit)
    try:
        resp = requests.post(api.url, headers=headers(api.backend, api.key),
                             json=body, timeout=timeout, stream=True)
        if resp.status_code != 200:
            logger.warning("[LLM] %s 流式 HTTP %s: %s",
                           api.id, resp.status_code, resp.text[:200])
            return None, emitted
        # SSE 按规范固定 UTF-8,不看响应头的脸色:硅基流动的头是裸的 text/event-stream
        # (没写 charset),requests 按 RFC2616 老默认猜 ISO-8859-1,中文逐字节错解成整屏乱码
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue                        # 空行、注释行、SSE 的 event: 行
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue                        # 半个 JSON / 填充行
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if piece := delta.get("content"):
                gate.feed(piece)
            for frag in delta.get("tool_calls") or []:
                merge_toolcall_fragment(tool_acc, frag)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    except requests.RequestException as e:
        logger.warning("[LLM] %s 流式请求异常: %s", api.id, e)
        return None, emitted
    finally:
        if resp is not None:
            resp.close()        # 非 200 早退、[DONE] break、异常,各路径都要归还连接

    gate.flush()

    if chars := sum(len(p) for p in parts):
        logger.info("[LLM] %s 流式: 首字 %.1fs, 增量 %d 条, 正文 %d 字, 总 %.1fs",
                    api.id, first_at or 0.0, pieces, chars, time.time() - started)

    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    content = "".join(parts)
    if not (content.strip() or tool_calls):
        return None, emitted                    # 全空视为失败(没外发过就还能重试)
    message: dict = {"role": "assistant", "content": content or (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}, emitted


def _blocking(api, body: dict, timeout: int) -> dict | None:
    """一次普通请求。成功且回复非空才算成功。"""
    try:
        resp = requests.post(api.url, headers=headers(api.backend, api.key),
                             json=body, timeout=timeout)
        if resp.status_code != 200:
            logger.warning("[LLM] %s HTTP %s: %s", api.id, resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        if (msg.get("content") or "").strip() or msg.get("tool_calls"):
            return data
        logger.warning("[LLM] %s 回复为空。", api.id)
    except requests.RequestException as e:
        logger.warning("[LLM] %s 请求异常: %s", api.id, e)
    except ValueError as e:
        logger.warning("[LLM] %s 响应不是 JSON: %s", api.id, e)
    return None


def request(
    api,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    effort: str = "",
    timeout: int = 300,
    attempts: int | None = None,
    on_delta=None,
) -> dict | None:
    """向一个平台发请求,自带退避重试。失败返回 None,由上层决定要不要换平台。"""
    if not api.usable:
        logger.warning("[LLM] %s 没配 url/key,跳过。", api.id)
        return None

    body = payload(api.backend, model, messages,
                   max_tokens=max_tokens, temperature=temperature,
                   effort=effort, tools=tools)
    rounds = attempts or len(RETRY_BACKOFF)
    for i in range(rounds):
        if on_delta is not None:
            data, emitted = _stream(api, model, body, on_delta, timeout)
            if data is not None:
                return data
            if emitted:
                return None                     # 已经外发过,不能重试
        else:
            if (data := _blocking(api, body, timeout)) is not None:
                return data
        if i < rounds - 1:
            time.sleep(RETRY_BACKOFF[min(i, len(RETRY_BACKOFF) - 1)])
    return None
