"""项目介绍生成 —— 抓素材 + 写提示词 + 调 LLM。

报告条目、单项目查询、收藏时的一句话都走这一份,只差 `level`。

**提示词在工具层而不是 infra/llm**:「用中文、四段式、不许编造」是产品决定,换个项目就得
重写;而「怎么把消息发出去、某家挂了换哪家」换个项目照样能用。
"""

from __future__ import annotations

import logging
import re

from ..infra import llm

logger = logging.getLogger("hot_project")

STANDARD = "standard"
DETAILED = "detailed"

SECTIONS = ("项目定位与用途", "解决的问题", "使用场景", "技术架构与特性")

# 旧版报告里出现过、现在不再要求的段名。识别(解析/DB 同步)要认,生成不再写。
LEGACY_SECTIONS = ("核心依赖与生态", "已知局限或注意事项")


def has_all_sections(text: str) -> bool:
    """正文是否带齐四段标题。缺一段就不能当缓存 —— 对话曾经写过三段的会把报告第 4 段变成模板句。"""
    body = (text or "").strip()
    if not body:
        return False
    return all(f"{title}:" in body or f"{title}：" in body for title in SECTIONS)

README_IN_PROMPT = 3200     # 塞进提示词的 README 上限
COMMIT_IN_PROMPT = 60       # 每条提交摘要的上限
_MAX_TOKENS = {STANDARD: 1536, DETAILED: 2048}

_RULES = (
    "请基于以下已提供信息,用中文总结这个 GitHub 开源项目。\n"
    "输出要求:\n"
    "1. 只能基于下方明确提供的信息,不要把项目地址或 README 链接当作已读取内容。\n"
    "2. 不要补充未在输入中出现、且无法确认的外部知识;信息不足时使用保守表述。\n"
    "3. 如果输入中包含 README 摘录、发布记录或提交记录,可以引用;若缺失,请明确说明信息不足。\n"
)

_FORMAT = {
    DETAILED: (
        "4. 必须严格输出以下四个字段,每个字段单独成段,字段间用换行分隔:\n"
        "项目定位与用途:...(100-200字,说明是什么、做什么,简要介绍核心定位)\n"
        "解决的问题:...(100-200字,聚焦核心痛点,说明为什么需要这个项目)\n"
        "使用场景:...(100-200字,列举典型应用场景和目标用户)\n"
        "技术架构与特性:...(100-200字,关键技术栈、架构特点和核心特性)\n"
        "5. 总长度控制在 400-800 字,信息详实但不冗余。\n"
        "6. 不要使用列表、不要加 Markdown 标题、字段名后必须换行。\n\n"
    ),
    STANDARD: (
        "4. 必须严格输出以下三个字段,字段名保持原样:\n"
        "项目定位与用途:...\n解决的问题:...\n使用场景:...\n"
        "5. 每个字段建议 80-160 字,总长度控制在 260-520 字。\n"
        "6. 不要使用列表、不要加 Markdown 标题、不要输出字段以外的说明。\n\n"
    ),
}


def _clip(text: str, limit: int) -> str:
    clean = (text or "").strip()
    return clean if len(clean) <= limit else clean[:limit] + "..."


def _releases_line(items: list[dict]) -> str:
    parts = []
    for item in items[:5]:
        tag = str(item.get("tag_name") or item.get("name") or "").strip()
        if not tag:
            continue
        state = "/".join(s for s, on in (("prerelease", item.get("prerelease")),
                                         ("draft", item.get("draft"))) if on)
        date = str(item.get("published_at") or "")[:10]
        parts.append(f"{tag}{f'({state})' if state else ''}{f'@{date}' if date else ''}")
    return "; ".join(parts)


def _commits_line(items: list[dict]) -> str:
    parts = []
    for item in items[:8]:
        date = str(item.get("date") or "")[:10]
        message = _clip(str(item.get("message") or ""), COMMIT_IN_PROMPT)
        if bit := ":".join(p for p in (date, message) if p):
            parts.append(bit)
    return "; ".join(parts)


def build_prompt(name: str, facts: dict, level: str = STANDARD) -> str:
    """把手上的素材拼成提示词。

    `facts` 可来自 DB(gh_desc / topics)或实时抓取(readme / releases / commits),字段名一致,可混。
    """
    lines = [f"项目名称: {name}", f"项目地址: https://github.com/{name}"]
    if desc := (facts.get("gh_desc") or facts.get("short_desc") or ""):
        lines.append(f"官方简介: {desc}")
    if topics := facts.get("topics"):
        lines.append(f"标签: {', '.join(topics)}")
    if excerpt := facts.get("readme_excerpt"):
        lines.append(f"README摘录(已读取文本,可能截断): {_clip(str(excerpt), README_IN_PROMPT)}")
    if line := _releases_line(facts.get("recent_releases") or []):
        lines.append(f"近期发布节奏: {line}")
    if line := _commits_line(facts.get("recent_commits") or []):
        lines.append(f"近期提交线索: {line}")

    return _RULES + _FORMAT.get(level, _FORMAT[STANDARD]) + "\n".join(lines) + "\n"


def merge_profile(facts: dict, pack: dict) -> dict:
    """把实时抓来的资料包并进已有的事实。原地不改,返回新字典。

    `profile` 的键和提示词认的键不同名(readme / readme_excerpt),翻译只在这一处。
    """
    merged = dict(facts)
    if text := (pack.get("readme") or {}).get("text"):
        merged["readme_excerpt"] = text
    if items := pack.get("releases"):
        merged["recent_releases"] = items
    if items := pack.get("commits"):
        merged["recent_commits"] = items
    if info := pack.get("info"):
        merged.setdefault("gh_desc", info.get("description") or "")
        merged.setdefault("language", info.get("language") or "")
        merged.setdefault("topics", info.get("topics") or [])
    return merged


def describe(name: str, facts: dict, level: str = STANDARD,
             *, client: llm.LLMClient | None = None) -> str:
    """生成一段中文介绍。LLM 没配、或全部平台失败 → 空串。

    空串是有意的:抛异常会让整份报告因为一个仓库的描述失败而作废。

    `client` 不传就用进程共享的那份;测试传 stub 进来,从接口测 prompt 和解析,
    不必 monkeypatch 模块属性。
    """
    client = client or llm.get()
    if not client.configured():
        logger.warning("LLM 未配置,跳过描述生成。")
        return ""
    # medium 而不是 off:一句介绍也要先读懂一堆事实,不思考的版本明显更泛;也不用 high ——
    # 一份报告要跑几十个仓库,那点质量差不值几十倍的等待
    text = client.text(build_prompt(name, facts, level), lite=True,
                       max_tokens=_MAX_TOKENS.get(level, _MAX_TOKENS[STANDARD]),
                       temperature=0.2, effort=llm.EFFORT_MEDIUM)
    if not text:
        logger.warning("描述生成失败(所有平台都失败):%s", name)
    return text


_NUMBERED = re.compile(r"(\d+)\.\s*(.+)")
CONDENSE_MIN_PARSED = 0.5       # 解析出的条数低于这个比例就整批回退

def condense(repos: list[dict], max_chars: int = 70,
             *, client: llm.LLMClient | None = None) -> list[str]:
    """把一批项目的英文简介批量浓缩成中文短句。返回和输入等长。

    整批一次请求,代价是解析靠序号对齐;解析不出一半以上就整批回退截断原文。
    `client` 同 `describe`:默认进程共享,测试注入 stub。
    """
    if not repos:
        return []
    fallback = [(r.get("description") or "")[:max_chars] for r in repos]

    client = client or llm.get()
    if not client.configured():
        return fallback

    listing = "\n".join(
        f"{i + 1}. {r['full_name']}: {(r.get('description') or '').strip() or '(无描述)'}"
        for i, r in enumerate(repos)
    )
    text = client.text(
        f"请将以下 {len(repos)} 个 GitHub 项目的描述各浓缩为不超过{max_chars}字的中文简介。\n"
        f"要求:保留核心功能和用途,去掉修饰语,每行格式为「序号. 浓缩描述」,不要项目名。\n\n"
        f"{listing}\n",
        lite=True, max_tokens=2048, temperature=0.1, effort=llm.EFFORT_MEDIUM,
    )
    if not text:
        logger.warning("批量浓缩失败,回退截断原文。")
        return fallback

    out = [""] * len(repos)
    for line in text.splitlines():
        if m := _NUMBERED.match(line.strip()):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(repos):
                out[idx] = m.group(2).strip()[:max_chars]

    parsed = sum(1 for item in out if item)
    if parsed < len(repos) * CONDENSE_MIN_PARSED:
        logger.warning("批量浓缩只解析出 %d/%d 条,整批回退截断。", parsed, len(repos))
        return fallback
    logger.info("批量浓缩完成:解析 %d/%d 条,其余回退截断。", parsed, len(repos))
    return [item or fallback[i] for i, item in enumerate(out)]
