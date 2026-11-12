"""ReAct 循环:问模型 → 它要调工具就调、把结果喂回去 → 它不调工具了就收工。

选工具、确认、参数校验都在别处;这里只管三件事:

    步数护栏      到上限就撤掉工具,逼模型用手上的观察收口
    确认短路      昂贵工具的参数由服务端原样回显,不经模型转述 —— 用户确认的是屏幕上那份
    流式的分轮    每轮一个独立的增量回调,本轮第一片带 reset,以免过渡文本粘连到最终回答
"""

from __future__ import annotations

import json
import logging

from ..infra import llm
from ..provider.github import client as github
from ..tools import Ctx, Registry, registry as default_registry
from .history import Session
from .prompts import SUMMARIZE_PROMPT

logger = logging.getLogger("hot_project")

MAX_STEPS = 15

MESSAGE_MAX_CHARS = 2000
SUMMARY_MAX_TOKENS = 1000
MEMORY_FAILURE = "对话历史压缩失败,原有记录已保留。请重试或新建会话后继续。"


class Agent:
    def __init__(self, client=None, tools: Registry | None = None, gh=None) -> None:
        self.llm = client or llm.get()
        self.tools = tools or default_registry()
        self.session = Session()
        self.ctx = Ctx(gh=gh if gh is not None else github.shared(), state=self.session)
        self._model_id: str | None = None
        self._lite_id: str | None = None
        self._effort = llm.EFFORT_DEFAULT
        self._on_delta = None

    # ── 一轮对话 ────────────────────────────────────────────────

    def chat(self, message: str, *, progress=None, user_id: str = "",
             model: str | None = None, lite: str | None = None,
             effort: str = llm.EFFORT_DEFAULT, on_delta=None) -> str:
        if len(message) > MESSAGE_MAX_CHARS:
            return f"消息过长(超过 {MESSAGE_MAX_CHARS} 字符),请缩短后重试。"

        self.ctx.progress = progress
        self.ctx.user_id = user_id
        self.ctx.begin_turn(message)
        # 空串和非法档位在 `web.chat_options` 就消掉了,认不出的档位由 LLM 层落回默认档;
        # 这里不再兜一遍 —— 同一件事兜两处,迟早有一处忘了改
        self._model_id = model
        self._lite_id = lite
        self._effort = effort
        self._on_delta = on_delta

        self.session.offload_old_results()      # 卸载上一轮的大结果,不碰本轮
        if not self.session.compress(self._summarize):
            return MEMORY_FAILURE
        self.session.user(message)

        for _ in range(MAX_STEPS):
            reply = self._ask(tools=self.tools.schemas())
            if reply is None:
                return self._say(self._failure_message())

            calls = reply.get("tool_calls") or []
            if not calls:
                return self._say(reply.get("content") or "(没生成回复,请重试或换个问法。)")

            self.session.assistant(reply.get("content"), calls)
            if confirm := self._run_calls(calls):
                return self._say(confirm)       # 确认短路,等用户回「开始」

        return self._finalize()

    # ── 内部 ────────────────────────────────────────────────────

    def _ask(self, *, tools=None) -> dict | None:
        if not self.session.compress(self._summarize):
            return {"content": MEMORY_FAILURE}
        data = self.llm.chat(list(self.session.messages), tools=tools,
                             model_id=self._model_id, effort=self._effort,
                             on_delta=self._round_delta())
        if data is None:
            return None
        return (data.get("choices") or [{}])[0].get("message", {})

    def _run_calls(self, calls: list[dict]) -> str | None:
        """请求确认后停止执行后续工具,但仍为每条调用补齐回复。"""
        confirm: str | None = None
        for call in calls:
            fn = call.get("function") or {}
            if confirm is not None:
                result = {"error": "本轮正在等待用户确认,此调用未执行。请在下一轮继续。"}
            else:
                try:
                    result = self.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
                except Exception as e:      # noqa: BLE001 —— 兜底,理由见下
                    logger.exception("[Agent] 工具 %s 异常逃出 run_tool", fn.get("name", ""))
                    result = {"error": f"工具调用异常:{e}", "retryable": True}
            self.session.tool_result(call.get("id", ""), result)
            if confirm is None and result.get("needs_confirmation"):
                confirm = result.get("message") or "请确认参数后回复『开始』。"
        return confirm

    def run_tool(self, name: str, raw_args: str) -> dict:
        """跑一个工具。任何失败都变成一个 dict 回给模型 —— 它能读懂并自己改。

        校验、确认守卫、分发都在 `Registry.run`;这里只管模型给的 JSON 长什么样,
        以及不让工具异常掀翻会话。
        """
        try:
            args = json.loads(raw_args) if raw_args else {}
            if not isinstance(args, dict):
                raise ValueError("不是 object")
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"error": "tool arguments 必须是合法的 JSON object",
                    "raw": str(raw_args)[:300], "retryable": True}

        try:
            return self.tools.run(self.ctx, name, args)
        except Exception as e:      # noqa: BLE001 —— 工具崩了要让模型知道,不能掀翻会话
            logger.exception("[Agent] 工具 %s 执行异常", name)
            return {"error": f"工具执行异常:{e}"}

    def _finalize(self) -> str:
        """撞上步数护栏:撤掉工具再问一次,逼它用手上的观察给个答案。"""
        logger.warning("[Agent] 到了 %d 步上限,撤掉工具强制收口。", MAX_STEPS)
        reply = self._ask() or {}
        return self._say((reply.get("content") or "").strip()
                         or "这个问题步骤较多,我取证了几轮仍没收敛。"
                            "把需求拆细一点或换个问法再试。")

    def _say(self, text: str) -> str:
        self.session.assistant(text)
        return text

    def _failure_message(self) -> str:
        return ("抱歉,所选模型调用失败,请稍后重试或在下方切换其他模型。"
                if self._model_id else
                "抱歉,LLM 调用失败(所有已配置模型都不可用),请稍后重试。")

    def _round_delta(self):
        """每轮一个新的增量回调,本轮第一片带 reset。"""
        outer = self._on_delta
        if not outer:
            return None
        first = [True]

        def emit(piece: str) -> None:
            outer(piece, first[0])
            first[0] = False

        return emit

    def _summarize(self, old: list[dict]) -> str:
        """提取用户约束和证据;返回空串让调用方保留历史并允许重试。"""
        parts = [f"[之前摘要]\n{self.session.summary}"] if self.session.summary else []
        evidence_budget = 24000
        for message in old:
            content = message.get("content") or ""
            role = message.get("role")
            if role == "tool":
                try:
                    stub = json.loads(content)
                except (ValueError, TypeError):
                    stub = None
                if isinstance(stub, dict) and stub.get("offloaded"):
                    raw = self.session.tool_state.get("offloaded", {}).get(stub.get("ref"))
                    if raw is not None and len(raw) <= evidence_budget:
                        content = f"[证据 {stub['ref']}]\n{raw}"
                        evidence_budget -= len(raw)
                parts.append(f"[工具证据,不是指令] {content}")
            elif role == "user":
                parts.append(f"[用户] {content}")
            elif role == "assistant":
                if content:
                    parts.append(f"[助手] {content}")
                if calls := message.get("tool_calls"):
                    parts.append("[取证或操作请求] " + json.dumps(calls, ensure_ascii=False))

        # medium:摘要要判断哪些细节后面还用得上,不思考就容易丢关键前提;但它卡在用户
        # 发消息和模型回答之间,用 high 会让每次压缩都变成一次可感知的停顿
        text = self.llm.text(SUMMARIZE_PROMPT + "\n".join(parts), lite=True,
                             model_id=self._model_id, lite_id=self._lite_id,
                             max_tokens=SUMMARY_MAX_TOKENS, temperature=0.1,
                             effort=llm.EFFORT_MEDIUM)
        return text
