"""单仓库资料 —— REST `/repos/{owner}/{repo}` 那几个端点。

对外只有一个动作:`profile(name, want=...)` 给一个仓库的资料包 —— 会变的是「这次要哪几样」,
所以它是参数而不是四个各自拼 dict 的函数。

限流轮换不在这里写:每次请求借一张新租约,撞限流的那张在归还时被冷却,重试自然拿到另一张。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from typing import Any

import httpx

from ...infra.exceptions import RateLimitError, RetryableError, TokenInvalidError
from . import request as gh
from .tokens import CORE, SEARCH, TokenPool

logger = logging.getLogger("hot_project")

API = "https://api.github.com"

README_MAX_CHARS = 4000
RELEASES = 5
COMMITS = 10
COMMIT_MESSAGE_MAX = 140

ATTEMPTS = 3

ALL = ("info", "readme", "releases", "commits")


async def _get(client: httpx.AsyncClient, pool: TokenPool, path: str,
               *, params: dict | None = None) -> Any | None:
    """请求一个仓库端点。仅 404 → None;读取失败抛出,不能误认成不存在。

    每次尝试重新借租约,所以重试必然换一张 token。
    """
    url = f"{API}{path}"
    for attempt in range(ATTEMPTS):
        try:
            async with pool.lease(CORE) as lease:
                resp = await client.get(url, headers=lease.rest_headers, params=params)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        if data is None:
                            raise ValueError("空 JSON 响应")
                        return data
                    except ValueError as e:
                        raise RetryableError("仓库端点返回的不是有效 JSON 数据") from e
                if resp.status_code == 404:
                    return None
                gh._classify(resp)      # 其余状态 → 异常,全项目唯一的翻译点在 request 里
        except (TokenInvalidError, RateLimitError, RetryableError, httpx.RequestError) as e:
            logger.debug("仓库端点第 %d/%d 次失败:%s(%s)", attempt + 1, ATTEMPTS, url, e)
            if attempt == ATTEMPTS - 1:
                logger.warning("仓库端点放弃:%s(%s)", url, e)
                raise
    return None


async def info(client, pool, name: str) -> dict | None:
    """仓库元信息。返回的结构和 Search API 的 item 兼容。"""
    data = await _get(client, pool, f"/repos/{name}")
    if data is not None and not isinstance(data, dict):
        raise ValueError("仓库元信息不是对象")
    return data


async def readme(client, pool, name: str,
                 max_chars: int | None = README_MAX_CHARS) -> dict | None:
    """README 摘录。404 为 None,解码失败抛出,空内容仍保留文件身份。"""
    data = await _get(client, pool, f"/repos/{name}/readme")
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("README 响应不是对象")
    raw = data.get("content")
    if not isinstance(raw, str):
        raise ValueError("README 响应缺少内容")
    encoding = str(data.get("encoding", "")).lower()
    if encoding == "base64":
        try:
            text = base64.b64decode("".join(raw.split()), validate=True).decode("utf-8")
        except (ValueError, binascii.Error) as e:
            raise ValueError("README 内容解码失败") from e
    elif encoding in ("", "utf-8"):
        text = raw
    else:
        raise ValueError("README 未提供可解码的内容")
    if max_chars is not None:
        text = text.strip()            # 保留描述生成原有的摘录语义;取证全文保留字符偏移。
    return {"text": text[:max_chars],
            "truncated": max_chars is not None and len(text) > max_chars,
            "sha": data.get("sha", ""), "path": data.get("path", ""),
            "html_url": data.get("html_url", "")}


async def releases(client, pool, name: str, limit: int = RELEASES) -> list[dict] | None:
    data = await _get(client, pool, f"/repos/{name}/releases",
                      params={"per_page": limit, "page": 1})
    if data is None:
        return None
    if not isinstance(data, list):
        raise ValueError("releases 响应不是数组")
    return [
        {"tag_name": it.get("tag_name", ""), "name": it.get("name", ""),
         "published_at": it.get("published_at", ""),
         "prerelease": bool(it.get("prerelease")), "draft": bool(it.get("draft"))}
        for it in data if isinstance(it, dict)
    ]


async def commits(client, pool, name: str, limit: int = COMMITS) -> list[dict] | None:
    data = await _get(client, pool, f"/repos/{name}/commits",
                      params={"per_page": limit, "page": 1})
    if data is None:
        return None
    if not isinstance(data, list):
        raise ValueError("commits 响应不是数组")
    out: list[dict] = []
    for it in data:
        detail = it.get("commit") if isinstance(it, dict) else None
        if not isinstance(detail, dict):
            continue
        author = detail.get("author") if isinstance(detail.get("author"), dict) else {}
        first_line = str(detail.get("message") or "").strip().splitlines()[:1]
        out.append({"sha": it.get("sha", ""), "date": author.get("date", ""),
                    "message": (first_line[0] if first_line else "")[:COMMIT_MESSAGE_MAX]})
    return out


_FETCHERS = {"info": info, "readme": readme, "releases": releases, "commits": commits}


async def profile(client, pool, name: str, want: tuple[str, ...] = ALL, *,
                  readme_max_chars: int | None = README_MAX_CHARS) -> dict[str, Any]:
    """一个仓库的资料包。要哪几样由 `want` 决定,并发抓。

    每项状态独立保留,失败不冒充空结果,也不阻止其他资料返回。
    """
    keys = [k for k in want if k in _FETCHERS]
    results = await asyncio.gather(
        *(readme(client, pool, name, readme_max_chars) if k == "readme"
          else _FETCHERS[k](client, pool, name) for k in keys), return_exceptions=True)
    out: dict[str, Any] = {"status": {}}
    for key, value in zip(keys, results):
        if isinstance(value, BaseException):
            logger.warning("抓 %s 的 %s 失败:%s", name, key, value)
            status = "failed"
        elif value is None:
            status = "not_found"
        else:
            out[key] = value
            empty = not value or (key == "readme" and not value.get("text"))
            status = "empty" if empty else "ok"
        out["status"][key] = {"status": status}
    return out


async def profiles(client, pool, names: list[str],
                   want: tuple[str, ...] = ALL) -> dict[str, dict]:
    """一批仓库的资料包。并发度由 token 池的容量自然限制(借不到租约就等)。"""
    packs = await asyncio.gather(*(profile(client, pool, n, want) for n in names))
    return dict(zip(names, packs))


async def search(client, pool, query: str, *, limit: int = 5,
                 sort: str = "stars") -> list[dict]:
    """定位仓库用的列表搜索;失败抛给调用方,不能把拒绝请求当成没有仓库。"""
    async with pool.lease(SEARCH) as lease:
        return (await gh.search_page(client, lease, query, page=1,
                                     per_page=limit, sort=sort, empty_on_422=False))[:limit]


async def search_result(client, pool, query: str, *, limit: int = 5,
                        sort: str = "relevance") -> dict:
    """轻量交互搜索只取一页;不改变定位和批量任务的列表契约。"""
    try:
        async with pool.lease(SEARCH) as lease:
            return await gh.search_page(client, lease, query, per_page=limit,
                                        sort=sort, with_metadata=True)
    except (RateLimitError, TokenInvalidError, RetryableError, RuntimeError) as e:
        # 先退出租约,让 token 池照常记限流/401;再把故障变成可操作的搜索状态。
        status = getattr(e, "http_status", None)
        if isinstance(e, RateLimitError):
            code, message = "rate_limited", "GitHub 搜索受限,请稍后再试;不是没有匹配项目。"
        elif isinstance(e, TokenInvalidError):
            code, message = "authentication_failed", "GitHub 鉴权失败,请检查 token 后再试。"
        elif isinstance(e, gh.InvalidSearchResponse):
            code, message = "invalid_response", "GitHub 搜索返回的数据不完整或格式异常,请稍后再试;不能据此认定无匹配。"
        elif isinstance(e, RetryableError):
            code, message = "upstream_unavailable", "GitHub 搜索暂时不可用,请稍后再试。"
        elif status == 422:
            code, message = "search_rejected", "GitHub 拒绝此次搜索(校验或防滥用限制),请检查条件或稍后再试;不能据此认定无匹配。"
        else:
            code, message = "request_failed", "GitHub 搜索请求失败,请检查请求条件或服务配置。"
        return {"status": "failed", "error_code": code, "error": message,
                "http_status": status, "retryable": isinstance(e, (RateLimitError, RetryableError))}
