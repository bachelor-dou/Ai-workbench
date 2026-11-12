"""GitHub 出站请求 —— 发一次,把结果或失败原因还回去。**不重试。**

重试只能由任务池做:多层各重试一遍会把一次限流放大成几十次请求,而限流恰恰是请求太多引起的。

每个函数都要一张租约,由调用方借好递进来 —— 本模块不认识 token 池。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from ...infra.exceptions import RateLimitError, RetryableError, TokenInvalidError
from .tokens import Lease

logger = logging.getLogger("hot_project")

SEARCH_URL = "https://api.github.com/search/repositories"
GRAPHQL_URL = "https://api.github.com/graphql"

BATCH_SIZE = 100

DEFAULT_COOLDOWN = 60.0


def build_client(timeout: float = 90.0) -> httpx.AsyncClient:
    """建一个可跨请求复用的连接池。"""
    return httpx.AsyncClient(
        timeout=timeout,
        # 改过名的仓库,REST 会 301 到 /repositories/{id};不跟随就抓不到 readme/commits。
        # 同源重定向,httpx 会保留 Authorization 头。
        follow_redirects=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
    )


def _reset_at(headers) -> float:
    """从响应头推算限流何时解除。

    `Retry-After` 优先:二级限流时两个头同时在且不一致(实测 60s vs 38s,后者是**主**限额
    那一分钟的窗口),先读 reset 会提前重试、撞进没结束的罚时,GitHub 明说这可能导致封号。
    """
    for key, absolute in (("Retry-After", False), ("X-RateLimit-Reset", True)):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        when = value if absolute else time.time() + value
        if when > time.time():
            return when
    return time.time() + DEFAULT_COOLDOWN


def _limit_reason(resp: httpx.Response) -> str:
    """403/429 是主限额耗尽还是二级限流 —— 两者处置不同,先把证据记进日志。

    主限额看 `x-ratelimit-remaining: 0`;二级限流 GitHub 在正文里明说,且不保证 reset 头适用。
    """
    kind = ("二级限流" if "secondary rate limit" in resp.text[:500].lower()
            else "主限额耗尽" if resp.headers.get("x-ratelimit-remaining") == "0"
            else "未分类")
    return (f"HTTP {resp.status_code} {kind} "
            f"remaining={resp.headers.get('x-ratelimit-remaining', '-')} "
            f"retry-after={resp.headers.get('retry-after', '-')}")


def _classify(resp: httpx.Response) -> None:
    """HTTP 状态 → 异常。**全项目只有这一处做这个翻译。**

    分类决定任务池怎么处置:401 记 strike、403/429 冷却 token、5xx 退避重试、4xx 当场失败。
    """
    code = resp.status_code
    if code == 200:
        return
    if code == 401:
        error = TokenInvalidError(f"HTTP 401: {resp.text[:200]}")
    elif code in (403, 429):
        error = RateLimitError(_reset_at(resp.headers), _limit_reason(resp))
    elif code >= 500:
        error = RetryableError(f"HTTP {code}: {resp.text[:200]}")
    else:
        error = RuntimeError(f"HTTP {code}: {resp.text[:200]}")
    error.http_status = code      # 保留结构化状态;上层不必从异常文案猜 HTTP 码。
    raise error


async def _send(coro) -> httpx.Response:
    """把网络层异常翻译成 `RetryableError`,让任务池按瞬时故障处理。"""
    try:
        return await coro
    except httpx.HTTPError as e:
        raise RetryableError(f"{type(e).__name__}: {e}") from e


class InvalidSearchResponse(RetryableError):
    """HTTP 成功但搜索数据不可信,不能当作零匹配。"""

    http_status = 200


async def search_page(
    client: httpx.AsyncClient,
    lease: Lease,
    query: str,
    *,
    page: int = 1,
    per_page: int = 100,
    sort: str = "stars",
    order: str = "desc",
    with_metadata: bool = False,
    empty_on_422: bool = True,
) -> list[dict[str, Any]] | dict[str, Any]:
    """搜一页;交互搜索保留计数和不完整标记,旧批量调用仍返回列表。

    列表形式的仓库定位也应关闭 empty_on_422;只有批量翻页保留旧的 422 终止语义。
    `query` 是**完整的** Search 语法串(含 `stars:>=N` 之类的限定),本函数不替调用方拼。
    """
    params = {"q": query, "per_page": per_page, "page": page}
    if sort != "relevance":
        params.update(sort=sort, order=order)
    resp = await _send(client.get(
        SEARCH_URL, headers=lease.rest_headers, params=params,
    ))
    if resp.status_code == 422 and empty_on_422 and not with_metadata:
        logger.debug("搜索 422,当作没有更多:q=%r page=%d", query, page)
        return []
    _classify(resp)
    try:
        data = _body(resp)
    except RetryableError as e:
        raise InvalidSearchResponse("搜索响应不是合法 JSON") from e
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise InvalidSearchResponse("搜索响应缺少有效 items 数组")
    if with_metadata:
        total, incomplete = data.get("total_count"), data.get("incomplete_results")
        if (type(total) is not int or not len(items) <= total < 2 ** 63
                or type(incomplete) is not bool):
            raise InvalidSearchResponse("搜索响应缺少有效计数或完整性标记")
        return {"items": items, "total_count": total, "incomplete_results": incomplete}
    return items


async def search_count(client: httpx.AsyncClient, lease: Lease, query: str) -> int:
    """只问命中多少条。用于星段拆分:超过 1000 就得把区间切开。"""
    resp = await _send(client.get(
        SEARCH_URL, headers=lease.rest_headers, params={"q": query, "per_page": 1},
    ))
    if resp.status_code == 422:
        return 0
    _classify(resp)
    return int(_body(resp).get("total_count", 0))


def _body(resp: httpx.Response) -> dict:
    """解响应体。解不开算**可重试**,不是永久失败。

    截断的响应或网关 HTML 错误页会让 `resp.json()` 抛 ValueError,不包住会落进任务池的兜底
    `except Exception` 被当成代码 bug 而不重试。
    """
    try:
        return resp.json()
    except ValueError as e:
        raise RetryableError(f"响应不是合法 JSON({e}):{resp.text[:200]}") from e


def _star_query(names: list[str]) -> str:
    """把一批 owner/repo 拼成别名查询。

    owner 与 name 必须 `json.dumps` 转义(引号会把查询拼坏);别名用序号 —— GraphQL 别名不许有 `-`、`.`。
    """
    parts = []
    for i, full_name in enumerate(names):
        owner, _, repo = full_name.partition("/")
        parts.append(
            f"r{i}: repository(owner:{json.dumps(owner)}, name:{json.dumps(repo)})"
            " { stargazerCount nameWithOwner databaseId }"
        )
    return "query{" + "\n".join(parts) + "}"


async def fetch_stars(
    client: httpx.AsyncClient, lease: Lease, names: list[str],
    *, renames: dict[str, tuple[str, int | None]] | None = None,
    ids: dict[str, int] | None = None,
) -> dict[str, int] | None:
    """批量取 star。

    键缺失 = **GitHub 确认查不到**(已删/改名/转私有),是淘汰判定的依据,不能混进「没问到」。
    返回 None = 整批全 null 的退化响应,应对半拆开重来,**绝不可当成「都没了」**。

    `ids` 给了就填 {请求名: databaseId};`renames` 给了就记改名(请求名 ≠ 规范
    `nameWithOwner` 时填 {旧名: (规范名, databaseId)})。star 仍按**请求名**回填,
    身份信息来自同一次请求,不额外发包。
    """
    resp = await _send(client.post(
        GRAPHQL_URL, headers=lease.graphql_headers, json={"query": _star_query(names)},
    ))
    _classify(resp)

    payload = _body(resp)
    data = payload.get("data")
    if not isinstance(data, dict):
        errors = str(payload.get("errors", ""))[:200]
        if "RATE_LIMITED" in errors:
            raise RateLimitError(time.time() + DEFAULT_COOLDOWN)
        raise RetryableError(f"GraphQL 响应没有 data:{errors}")

    stars: dict[str, int] = {}
    for i, full_name in enumerate(names):
        node = data.get(f"r{i}")
        if isinstance(node, dict) and isinstance(node.get("stargazerCount"), int):
            stars[full_name] = node["stargazerCount"]
            if ids is not None and isinstance(node.get("databaseId"), int):
                ids[full_name] = node["databaseId"]
            if renames is not None:
                canon = node.get("nameWithOwner")
                if isinstance(canon, str) and canon and canon != full_name:
                    renames[full_name] = (canon, node.get("databaseId"))

    if not stars:
        if len(names) > 1:
            return None
        types = {e.get("type") for e in payload.get("errors") or [] if isinstance(e, dict)}
        if types != {"NOT_FOUND"}:
            raise RetryableError(
                f"{names[0]} 返回 null 但 errors 里没有 NOT_FOUND({types or '无 errors'})"
                " —— 按「没问到」处理,不当作已删除"
            )
    return stars
