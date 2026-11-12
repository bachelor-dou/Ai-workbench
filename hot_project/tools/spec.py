"""工具契约:一个参数只声明一次。

一个 `Param` 同时长出两样东西 —— `json_schema()` 给模型看,`coerce()` 做校验。分成两份写
的话它们会漂移,而漂移是**静默**的:校验那份认得 `"all"`、模型那份的 enum 里没有,于是
模型永远请求不到它,谁都不报错。

只有严格校验,没有「宽松模式」:静默把越界值裁到边界会让模型永远学不会自己传错了。
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hot_project")

REQUIRED = object()     # default 取这个值 = 必填


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                       # int / str / bool / enum / list_str
    desc: str
    default: Any = REQUIRED
    min: int | None = None
    max: int | None = None
    choices: tuple[str, ...] = ()

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def json_schema(self) -> dict:
        """给模型看的那份。"""
        types = {"int": "integer", "bool": "boolean", "list_str": "array", "enum": "string"}
        out: dict[str, Any] = {"type": types.get(self.kind, "string"),
                               "description": self.desc}
        if self.kind == "list_str":
            out["items"] = {"type": "string"}
        if self.choices:
            out["enum"] = list(self.choices)
        return out

    def coerce(self, value: Any) -> tuple[Any, str | None]:
        """校验一个值。返回 `(值, 出错原因)`,原因为 None 表示通过。

        不做静默纠偏:越界就报错,让模型看到 `must_be_lte_200` 并自己改。
        """
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, "expected_integer"
            if isinstance(value, float) and not math.isfinite(value):
                return None, "expected_integer"
            number = int(value)
            if self.min is not None and number < self.min:
                return None, f"must_be_gte_{self.min}"
            if self.max is not None and number > self.max:
                return None, f"must_be_lte_{self.max}"
            return number, None
        if self.kind == "bool":
            return (value, None) if isinstance(value, bool) else (None, "expected_boolean")
        if self.kind == "enum":
            return ((value, None) if value in self.choices
                    else (None, f"must_be_one_of_{list(self.choices)}"))
        if self.kind == "list_str":
            if not isinstance(value, list) or any(not isinstance(i, str) for i in value):
                return None, "expected_array_of_strings"
            return value, None
        return (value, None) if isinstance(value, str) else (None, "expected_string")


@dataclass(frozen=True)
class Tool:
    name: str
    desc: str
    run: Callable[[Any, dict], dict]        # (ctx, args) -> dict
    params: tuple[Param, ...] = ()
    expensive: bool = False                 # 昂贵工具执行前要用户确认,守卫在 Registry.run
    confirmation: Callable[[dict], str] | None = None   # 昂贵工具的参数回显文案

    def __post_init__(self) -> None:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.name} 的参数重名:{dupes}")

    def schema(self) -> dict:
        """LLM function-calling schema。"""
        fn: dict[str, Any] = {
            "name": self.name,
            "description": self.desc,
            "parameters": {"type": "object",
                           "properties": {p.name: p.json_schema() for p in self.params}},
        }
        if required := [p.name for p in self.params if p.required]:
            fn["parameters"]["required"] = required
        return {"type": "function", "function": fn}

    def validate(self, args: dict) -> tuple[dict, list[dict]]:
        """校验模型给的参数。返回 `(干净参数, 错误列表)`,错误为空才能执行。"""
        clean: dict[str, Any] = {}
        errors: list[dict] = []
        for param in self.params:
            if param.name not in args:
                if param.required:
                    errors.append({"param": param.name, "reason": "missing_required"})
                elif param.default is not None:
                    clean[param.name] = param.default
                continue
            value, reason = param.coerce(args[param.name])
            if reason:
                errors.append({"param": param.name, "reason": reason,
                               "received": args[param.name]})
            else:
                clean[param.name] = value

        known = {p.name for p in self.params}
        errors += [{"param": name, "reason": "unknown_parameter", "received": args[name]}
                   for name in sorted(set(args) - known)]
        return clean, errors


@dataclass
class Ctx:
    """一次工具调用能看到的全部外部世界。

    **刻意没有 `db` 字段**:挂上去供人就地改,「谁改了什么」就没人说得清了。要读的自己
    `universe.load()`,要写的走 `universe.write_*`,每次写都是一个事务。
    """

    gh: Any = None                      # provider.github 的同步客户端;None = 不联网的场景
    state: Any = None                   # 会话状态,只有 agent 路径有
    user_id: str = ""
    progress: Callable[[int, str], None] | None = None
    turn_state: dict = field(default_factory=dict)
    confirmation_signature: str | None = field(default=None, init=False)

    def begin_turn(self, message: str) -> None:
        """只有本轮用户的明确确认,才能授权上一轮展示过的参数。"""
        self.turn_state.clear()
        self.confirmation_signature = None
        if self.state is None:
            return
        if message.strip().rstrip("。.!！").casefold() in {
            "开始", "开始吧", "开始执行", "确认", "确认执行", "go", "yes",
        }:
            self.confirmation_signature = self.state.pending_confirmation_signature
        else:
            self.state.pending_confirmation_signature = None
            self.state.tool_state.pop(_PENDING, None)


_PENDING = "pending_confirmation"       # tool_state 里的待确认槽,会话全局只有一格


def _signature(name: str, params: dict) -> str:
    return json.dumps({"tool": name, **params}, ensure_ascii=False,
                      sort_keys=True, default=str)


class Registry:
    """工具表。名字 → 工具,分发和执行策略只此一处。

    `run` 是唯一入口:校验 → (昂贵工具)确认守卫 → 执行。守卫放在这而不是各 handler 里,
    `expensive=True` 才是真约束 —— 标了就一定先回显参数等用户确认,忘写状态机这种事不存在。
    """

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名:{tool.name}")
        if tool.expensive and not any(p.name == "confirm" and p.kind == "bool"
                                      for p in tool.params):
            raise ValueError(f"昂贵工具 {tool.name} 没声明 confirm 参数,模型将永远无法确认执行")
        self._tools[tool.name] = tool

    def run(self, ctx: Ctx, name: str, args: dict) -> dict:
        """跑一个工具。任何失败都变成一个 dict 回给调用方(模型能读懂并自己改)。"""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"没有这个工具:{name}", "available": self.names()}
        if tool.expensive and args.get("confirm") is True and ctx.state is not None:
            stored = ctx.state.tool_state.get(_PENDING) or {}
            if ctx.state.pending_confirmation_signature and stored.get("tool") == name:
                args = {**stored["params"], **args}
        clean, errors = tool.validate(args)
        if errors:
            return {"error": "参数校验失败,请按 invalid_arguments 修正后重试。",
                    "invalid_arguments": errors, "retryable": True}
        logger.info("[Tools] 调用 %s(%s)", name, clean)
        if tool.expensive:
            return self._run_confirmed(ctx, tool, clean)
        return tool.run(ctx, clean)

    def _run_confirmed(self, ctx: Ctx, tool: Tool, params: dict) -> dict:
        """昂贵工具的确认守卫:首次调用只回显参数,等用户回「开始」。

        回显和执行必须是同一份参数:参数存进会话,`confirm=true` 复调时用存下的那份 ——
        模型复述参数时会漂移(少个 `min_star`、把 top_n 从 20 写成 10),而用户确认的是
        屏幕上那份。确认必须同时匹配本轮用户授权与工具,换工具或重复调用都不能开门。
        """
        confirm = bool(params.pop("confirm", False))
        signature = _signature(tool.name, params)
        state = ctx.state

        pending = state.pending_confirmation_signature if state else None
        stored = (state.tool_state.get(_PENDING) or {}) if state else {}
        if not (confirm and pending and ctx.confirmation_signature == pending
                and stored.get("tool") == tool.name):
            ctx.confirmation_signature = None
            if state is not None:
                state.pending_confirmation_signature = signature
                state.tool_state[_PENDING] = {"tool": tool.name, "params": params}
            message = (tool.confirmation(params) if tool.confirmation
                       else f"将执行【{tool.name}】(昂贵操作),参数:{params}。"
                            "确认无误请回复『开始』。")
            return {"needs_confirmation": True, "tool": tool.name,
                    "params": params, "message": message}

        params = stored["params"]
        ctx.confirmation_signature = None
        if state is not None:
            state.pending_confirmation_signature = None
            state.tool_state.pop(_PENDING, None)
        return tool.run(ctx, params)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
