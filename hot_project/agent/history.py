"""会话状态与对话历史。三条规矩,违反了都是静默出错:

1. `role="tool"` 必须紧跟带 `tool_calls` 的 assistant 消息。压缩切在中间会留下孤儿,
   OpenAI 兼容接口一律 400。
2. 超过阈值的大结果换成小存根,容量内的完整内容留在本地,按需召回;过期或超限明确提示。
3. system 消息必须逐字节不变,摘要作为它之后的独立消息插入 —— 拼进 system 会让每次压缩
   之后的前缀缓存全部落空。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger("hot_project")

MAX_MESSAGES = 35
# ponytail: 字符数仅作保守护栏；需要按模型精确利用窗口时再换成 token 计量。
MAX_CONTEXT_CHARS = 48000
KEEP_RECENT = 10

OFFLOAD_THRESHOLD = 1200
DIGEST_CHARS = 200

RESULT_MAX_CHARS = 8000
OFFLOAD_MAX_RESULTS = 32
OFFLOAD_MAX_CHARS = 1_000_000


@dataclass
class Session:
    """一次会话。工具通过 `Ctx.state` 拿到它。

    没有 db 字段:要读库的工具自己读,写库走事务。
    """

    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    tool_state: dict = field(default_factory=dict)
    pending_confirmation_signature: str | None = None
    _memory_message: dict | None = field(default=None, repr=False)
    _result_stubs: dict[int, str | None] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # ── 追加 ────────────────────────────────────────────────────

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def assistant(self, text: str | None, tool_calls: list | None = None) -> None:
        message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def tool_result(self, call_id: str, result: dict) -> None:
        content = json.dumps(result, ensure_ascii=False, default=str)
        recalled = len(content) <= RESULT_MAX_CHARS and self._is_recalled_page(result)
        stub = self._offload(content) if len(content) > OFFLOAD_THRESHOLD and not recalled else None
        message = {"role": "tool", "tool_call_id": call_id,
                   "content": stub if len(content) > RESULT_MAX_CHARS else content}
        if recalled or (stub is not None and len(content) <= RESULT_MAX_CHARS):
            self._result_stubs[id(message)] = stub
        self.messages.append(message)

    def _is_recalled_page(self, result: dict) -> bool:
        if set(result) != {"ref", "text", "offset", "next_offset", "total_chars"}:
            return False
        ref, offset, text = result["ref"], result["offset"], result["text"]
        if not isinstance(ref, str) or type(offset) is not int or not isinstance(text, str):
            return False
        raw = self.tool_state.get("offloaded", {}).get(ref)
        if raw is None or not 0 <= offset <= len(raw):
            return False
        end = offset + len(text)
        return (raw[offset:end] == text and result["total_chars"] == len(raw)
                and result["next_offset"] == (end if end < len(raw) else None))

    def _offload(self, content: str) -> str:
        if len(content) > OFFLOAD_MAX_CHARS:
            return json.dumps({
                "truncated": True, "unavailable": True, "digest": content[:DIGEST_CHARS],
                "note": "结果超过会话留存上限,未保存全文;请缩小查询重新取证。",
            }, ensure_ascii=False)
        store = self.tool_state.setdefault("offloaded", {})
        while store and (len(store) >= OFFLOAD_MAX_RESULTS
                         or sum(map(len, store.values())) + len(content) > OFFLOAD_MAX_CHARS):
            del store[next(iter(store))]
        seq = self.tool_state.get("offload_seq", 0) + 1
        self.tool_state["offload_seq"] = seq
        ref = f"tr{seq}"
        store[ref] = content
        return json.dumps({
            "offloaded": True, "ref": ref, "digest": content[:DIGEST_CHARS],
            "note": "完整结果已暂存;用 recall_tool_result 按需读取,支持 offset 续读。",
        }, ensure_ascii=False)

    # ── 卸载 ────────────────────────────────────────────────────

    def offload_old_results(self) -> int:
        """把历史里的大结果换成存根。返回卸载了几条。

        正常在新回合开头卸载;回合内触及上下文护栏时也允许改为可召回存根。
        """
        done = 0
        for message in self.messages:
            content = message.get("content") or ""
            if message.get("role") != "tool" or len(content) <= OFFLOAD_THRESHOLD:
                continue
            if id(message) in self._result_stubs and self._result_stubs[id(message)] is None:
                continue      # 召回页已由原始证据覆盖,重复存储会驱逐仍在续读的原文。
            message["content"] = self._result_stubs.pop(id(message), None) or self._offload(content)
            done += 1
        if done:
            logger.info("[Agent] 卸载了 %d 条大结果。", done)
        return done

    # ── 压缩 ────────────────────────────────────────────────────

    def needs_compress(self) -> bool:
        return len(self.messages) > MAX_MESSAGES or _size(self.messages) > MAX_CONTEXT_CHARS

    def compress(self, summarize) -> bool:
        """失败保留原历史并返回 False,调用方不能假装已提取出新的记忆。"""
        if not self.needs_compress():
            return True
        self.offload_old_results()
        if not self.needs_compress():
            return True
        system = next((m for m in self.messages if m.get("role") == "system"),
                      {"role": "system", "content": SYSTEM_PROMPT})
        rest = [m for m in self.messages
                if m.get("role") != "system" and m is not self._memory_message]
        keep = min(KEEP_RECENT, len(rest))
        while keep and (keep == len(rest) or _size(rest[-keep:]) > MAX_CONTEXT_CHARS // 2):
            keep -= 1
        old, recent = split_at_safe_boundary(rest, keep) if keep else (rest, [])
        if not old:
            return True

        try:
            text = summarize(old)
        except Exception:
            logger.exception("[Agent] 历史摘要失败,保留原始对话。")
            return False
        if not text or not text.strip():
            logger.warning("[Agent] 历史摘要为空,保留原始对话。")
            return False
        self.summary = text.strip()

        rebuilt = [system]
        catalog = [{"ref": ref, "digest": raw[:DIGEST_CHARS]}
                   for ref, raw in self.tool_state.get("offloaded", {}).items()]
        memory = f"[对话历史摘要]\n{self.summary}"
        if catalog:
            memory += "\n[可召回工具证据]\n" + json.dumps(catalog, ensure_ascii=False)
        self._memory_message = {"role": "user", "content": memory}
        rebuilt.append(self._memory_message)
        self.messages = rebuilt + recent
        remaining = {id(message) for message in recent}
        self._result_stubs = {key: value for key, value in self._result_stubs.items() if key in remaining}
        logger.info("[Agent] 历史已压缩:%d 条 → 摘要,保留 %d 条。", len(old), len(recent))
        return True


def _size(messages: list[dict]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)


def split_at_safe_boundary(messages: list[dict], keep: int) -> tuple[list[dict], list[dict]]:
    """在不产生孤儿 tool 消息的位置切开。返回 `(丢弃的, 保留的)`。

    线优先推到保留段最早的 user 消息;没有 user 就只剥掉开头连续的 tool 消息。
    """
    old, recent = messages[:-keep], messages[-keep:]
    boundary = next((i for i, m in enumerate(recent) if m.get("role") == "user"), None)
    if boundary is None:
        boundary = 0
        while boundary < len(recent) and recent[boundary].get("role") == "tool":
            boundary += 1
    return old + recent[:boundary], recent[boundary:]
