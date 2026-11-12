"""周报生成 —— Top N 项目 → 一份 Markdown。

写出来的格式由 `infra.data_access.reports` 解析读回。标题这类有格式含义的行必须经
`reports.heading` / `trend_heading` / `appendix_mark` 写,它们和解析正则同住一处、
成对维护;守卫在 `test_report.py`。

描述按优先级取:DB 里没过期的缓存 → LLM 现生成 → 从元数据兜底拼一段。第三档不是摆设 ——
一份「只有 star 数没有介绍」的报告仍然有用,而一次异常会让跑了两小时的流水线颗粒无收。

文件名 `2026-07-30.md` / `_NEW.md` / `_KEY_向量库.md`(关键词榜必须带方向,否则同日两个
方向互相覆盖)。窗口偏离默认值时追加 `_10d`;新项目榜的增长窗口偏离时追加 `_win7d`。
"""

from __future__ import annotations

import logging
import re

from .. import config
from ..common.timeutil import age_days, format_day, utc_today
from ..infra.data_access import reports, universe
from . import describe

logger = logging.getLogger("hot_project")

_MODE = {
    "comprehensive": ("", "GitHub 热门项目"),
    "hot_new": ("_NEW", "GitHub 新项目热度榜"),
    "keyword": ("_KEY", "GitHub 热门项目"),
}

TOPIC_MAX = 6           # 标题里的方向名长度上限
TOPICS_SHOWN = 6        # 每个项目展示多少个标签


def filename(mode: str, *, growth_days: int, created_days: int | None,
             topic: str | None) -> str:
    suffix, _ = _MODE.get(mode, _MODE["comprehensive"])
    if mode in ("comprehensive", "keyword") and growth_days != config.GROWTH_CALC_DAYS:
        suffix += f"_{growth_days}d"
    if mode == "hot_new":
        if created_days is not None and created_days != config.DAYS_SINCE_CREATED:
            suffix += f"_{created_days}d"
        if growth_days != config.GROWTH_CALC_DAYS:
            suffix += f"_win{growth_days}d"
    if topic and mode == "keyword" and (slug := reports.safe_slug(topic, TOPIC_MAX)):
        suffix += f"_{slug}"
    return f"{format_day(utc_today())}{suffix}.md"


def _title(mode: str, *, growth_days: int, created_days: int | None,
           topic: str | None) -> str:
    _, prefix = _MODE.get(mode, _MODE["comprehensive"])
    if mode in ("comprehensive", "keyword") and growth_days != config.GROWTH_CALC_DAYS:
        prefix += f"(近{growth_days}天增长)"
    if mode == "hot_new" and created_days is not None:
        prefix += f"(近{created_days}天)"
    if topic:
        prefix += f"|方向:{topic}"
    return prefix


# ── 描述缺失时的兜底 ────────────────────────────────────────────────

def _blocks(text: str) -> list[str]:
    """把一段自由文本切成两块,用于从 LLM 的散文里凑出「定位」和「问题」两段。"""
    paragraphs = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    if len(paragraphs) >= 2:
        return paragraphs[:2]
    flat = " ".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if not flat:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[。!?.!?])\s+", flat) if s.strip()]
    if len(sentences) <= 1:
        return [flat]
    half = max(1, len(sentences) // 2)
    return [" ".join(sentences[:half]).strip(), " ".join(sentences[half:]).strip()]


def _extract(text: str) -> dict[str, str]:
    """从 LLM 输出里按字段名切出四段。中英文冒号都认,两种它都会写。

    遇到没要求的字段名(模型爱加「核心依赖与生态」)要**停止**追加,否则几百字会粘在最后一段尾巴上。
    """
    found: dict[str, str] = {}
    current = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = next((s for s in SECTION_NAMES
                        if line.startswith(f"{s}:") or line.startswith(f"{s}：")), "")
        if heading:
            body = re.split(r"[:：]", line, maxsplit=1)[1].strip()
            current = heading if heading in describe.SECTIONS else ""
            if current:
                found[current] = body
            continue
        if current:
            found[current] = f"{found.get(current, '')} {line}".strip()
    return {k: v for k, v in found.items() if v}


SECTION_NAMES = describe.SECTIONS + describe.LEGACY_SECTIONS


def _fallback(name: str, saved: dict, prose: str, new_window: int) -> dict[str, str]:
    """LLM 没给结构化字段时,从元数据凑四段。信息不足就直说,不编。"""
    desc = (saved.get("gh_desc") or "").strip()
    language = (saved.get("language") or "").strip()
    topics = [t for t in saved.get("topics") or [] if t][:4]
    chunks = _blocks(prose)

    positioning = " ".join(p for p in (desc, chunks[0] if chunks else "") if p).strip()

    problem = chunks[1] if len(chunks) >= 2 else " ".join(filter(None, [
        f"从当前简介看,它主要围绕“{desc}”这一方向提供能力。" if desc else "",
        f"从标签信号看,它重点瞄准 {', '.join(topics)} 相关问题。" if topics else "",
        f"实现侧主要落在 {language} 技术栈中。" if language else "",
    ])) or "当前信息不足以明确拆解其具体痛点。"

    age = age_days(saved.get("created_at", ""))
    usage = " ".join(filter(None, [
        f"适合关注 {language} 生态的开发者、研究者或技术团队先行评估。" if language else "",
        f"如果你的需求和 {', '.join(topics)} 相关,这个项目更值得进入候选清单。" if topics else "",
        f"由于其仍在近 {new_window} 天的新项目窗口内,也适合用来跟踪早期工具形态。"
        if age is not None and age <= new_window else "",
    ])) or "当前更适合把它当作方向性线索,再结合仓库主页和 README 判断。"

    tech = " ".join(filter(None, [
        f"主要使用 {language} 实现。" if language else "",
        f"技术关键词包括 {', '.join(topics)}。" if topics else "",
    ])) or "暂无详细技术架构信息,可查看仓库 README 或源码了解实现细节。"

    return dict(zip(describe.SECTIONS, [
        positioning or "暂无结构化摘要,可直接查看仓库主页与 README。",
        problem, usage, tech,
    ]))


def _sections(name: str, saved: dict, prose: str, new_window: int) -> list[tuple[str, str]]:
    structured = _extract(prose)
    backup = _fallback(name, saved, prose, new_window)
    return [(title, structured.get(title) or backup[title]) for title in describe.SECTIONS]


# ── 描述的取得 ──────────────────────────────────────────────────────

def _needs_llm(saved: dict) -> bool:
    """没描述、缺段(对话渠道写过三段版)、或描述过期了就要重生成。

    没有时间戳的旧数据**不**算过期,否则全库描述会一次性失效、一轮重刷几百个。
    """
    if not describe.has_all_sections(saved.get("desc", "")):
        return True
    stamp = saved.get("desc_updated_at") or ""
    age = age_days(stamp)
    return age is not None and age >= config.DESC_REFRESH_DAYS


def descriptions(ranked: list[tuple[str, dict]], saved_by_name: dict[str, dict],
                 gh=None, progress=None,
                 force: bool = False) -> tuple[dict[str, str], dict[str, dict]]:
    """给每个项目备好描述。返回 `(名字 → 描述, 要写回 DB 的条目)`。

    不碰库,写回交给调用方。素材先批量抓完再逐个调 LLM(抓取能并发、LLM 只能串行)。
    `force` 连没过期的也重生成。
    """
    today = format_day(utc_today())
    ready: dict[str, str] = {}
    writeback: dict[str, dict] = {}
    todo: list[str] = []

    for name, _ in ranked:
        saved = saved_by_name.get(name, {})
        if force or _needs_llm(saved):
            todo.append(name)
            continue
        ready[name] = saved["desc"]
        if not saved.get("desc_updated_at"):
            writeback[name] = {"desc": saved["desc"], "desc_updated_at": today}

    if not todo:
        return ready, writeback

    logger.info("报告生成:需要写介绍的 %d 个,先批量抓素材。", len(todo))
    packs = gh.profiles(todo, want=("readme", "commits")) if gh is not None else {}

    for done, name in enumerate(todo, 1):
        if progress is not None:
            progress((done - 1) / len(todo), f"生成报告 {done}/{len(todo)}")
        logger.info("[%d/%d] 写介绍:%s", done, len(todo), name)
        facts = describe.merge_profile(saved_by_name.get(name, {}), packs.get(name, {}))
        if text := describe.describe(name, facts, describe.DETAILED):
            ready[name] = text
            writeback[name] = {"desc": text, "desc_updated_at": today}
        else:
            ready.setdefault(name, "")
    logger.info("描述生成完成:%d/%d 成功。", sum(1 for n in todo if ready.get(n)), len(todo))
    return ready, writeback


def regenerate(name: str, gh) -> str:
    """重写一个仓库的介绍并落库,返回新正文;失败返回空串,库里那份不动。

    报告页手动刷新按钮用。走 `descriptions` 同一条路,只是强制越过缓存。
    """
    ready, writeback = descriptions([(name, {})], universe.load(), gh=gh, force=True)
    if writeback:
        universe.write_descriptions(writeback)
    return (ready.get(name) or "").strip()


# ── 组装 ────────────────────────────────────────────────────────────

def render(ranked: list[tuple[str, dict]], saved_by_name: dict[str, dict],
           descs: dict[str, str], *, mode: str, growth_days: int,
           growth_threshold: int, min_star: int, created_days: int | None,
           topic: str | None) -> str:
    """拼出整份 Markdown。纯字符串处理,不发请求不写盘 —— 所以能直接测。"""
    today = format_day(utc_today())
    new_window = created_days if created_days is not None else config.DAYS_SINCE_CREATED
    topic = (topic or "").strip()[:TOPIC_MAX] or None

    summary = [f"共 {len(ranked)} 个项目", f"增长统计窗口: {growth_days} 天",
               f"增长阈值: >={growth_threshold} stars", f"最低 star: >={min_star}",
               f"报告生成: {today}"]
    if mode == "hot_new":
        summary.insert(1, f"新项目创建窗口: <= {new_window} 天")

    out = [f"# {_title(mode, growth_days=growth_days, created_days=created_days, topic=topic)}"
           f" — {today}\n", f"> {' | '.join(summary)}\n"]

    for idx, (name, info) in enumerate(ranked, 1):
        saved = saved_by_name.get(name, {})
        created = (saved.get("created_at") or "")[:10]
        age = age_days(saved.get("created_at", ""))
        topics = [t for t in saved.get("topics") or [] if t][:TOPICS_SHOWN]

        out += [reports.heading(idx, name), "", f"链接: https://github.com/{name}", "",
                f"- 创建时间: {created or '未知'}"]
        if age is not None and age <= new_window:
            out.append(f"- 项目状态: NEW({new_window}天内)")
        if language := (saved.get("language") or "").strip():
            out.append(f"- 主语言: {language}")
        out.append(f"- 总 Star: {info['star']:,}")
        out.append(f"- 近{growth_days}天增长: +{info['growth']:,}")
        if topics:
            out.append(f"- 主题标签: {', '.join(topics)}")
        out.append("")

        for title, content in _sections(name, saved, descs.get(name, ""), new_window):
            out += [f"### {title}", "", *_paragraphs(content), ""]

        if idx != len(ranked):
            out += ["---", ""]

    return "\n".join(out).rstrip() + "\n"


def _paragraphs(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    return blocks or [(text or "").strip() or "暂无补充信息,可进入仓库查看 README。"]


def generate(ranked: list[tuple[str, dict]], *, mode: str = "comprehensive",
             growth_days: int, growth_threshold: int, min_star: int,
             created_days: int | None = None,
             topic: str | None = None, gh=None, progress=None) -> str:
    """生成并落盘,返回文件路径(写失败返回空串)。

    门槛三件套必填,理由同 `ranking.run`:策略由入口定,这里不替调用方翻 config。
    """
    saved_by_name = universe.load()
    descs, writeback = descriptions(ranked, saved_by_name, gh=gh, progress=progress)
    if writeback:
        universe.write_descriptions(writeback)

    text = render(ranked, saved_by_name, descs, mode=mode, growth_days=growth_days,
                  growth_threshold=growth_threshold, min_star=min_star,
                  created_days=created_days, topic=topic)
    name = filename(mode, growth_days=growth_days, created_days=created_days, topic=topic)
    path = reports.save(name, text)
    return str(path) if path else ""


# ── Trending 对照附栏 ────────────────────────────────────────────────
# 附栏标题与周期文案在 reports.py,和解析正则成对;这里只负责内容。


def _positions(ranked: list[tuple[str, dict]],
               saved_by_name: dict[str, dict]) -> tuple[dict[str, int], dict[int, int]]:
    """名字 → 榜内排名,以及 databaseId → 排名。"""
    by_name: dict[str, int] = {}
    by_id: dict[int, int] = {}
    for pos, (name, info) in enumerate(ranked, 1):
        by_name[name] = pos
        rid = info.get("id") or (saved_by_name.get(name) or {}).get("id")
        if isinstance(rid, int):
            by_id[rid] = pos
    return by_name, by_id


def _position_of(name: str, saved_by_name: dict[str, dict],
                 by_name: dict[str, int], by_id: dict[int, int]) -> int | None:
    rid = (saved_by_name.get(name) or {}).get("id")
    return by_name.get(name) or (by_id.get(rid) if isinstance(rid, int) else None)


def render_trending(rows: list[dict], ranked: list[tuple[str, dict]],
                    saved_by_name: dict[str, dict], descs: dict[str, str],
                    period: str = "weekly") -> str:
    """Trending 某个周期的榜单 → 对照附栏 Markdown。纯字符串处理。

    条目标题用 `T1.` 而非纯数字排名:报告解析器只认 `## 数字. owner/repo`,附栏对它
    不可见,不会混进「上新/移出」、出场次数、star 趋势的统计。已上榜的一行指回正文排名;
    没上榜的按正文同款格式补全,增长数字是 Trending 页面的口径,和我们的窗口增量不同源,
    字段名里写明是哪个窗口,不冒充。

    周期写进标题的括号里(`(weekly)` / `(monthly)`),解析端靠它把两段附栏分开归位。
    """
    label, gained = reports.PERIOD_TEXT.get(period, reports.PERIOD_TEXT["weekly"])
    by_name, by_id = _positions(ranked, saved_by_name)
    hits = sum(1 for r in rows
               if _position_of(r["full_name"], saved_by_name, by_name, by_id))

    out = ["---", "", reports.appendix_mark(period), "",
           f"> Trending {label} {len(rows)} 个 | 已在本期榜单 {hits} 个 | "
           f"附栏补全 {len(rows) - hits} 个 | 抓取: {format_day(utc_today())}", ""]

    for i, row in enumerate(rows, 1):
        name = row["full_name"]
        out += [reports.trend_heading(i, name), ""]
        if pos := _position_of(name, saved_by_name, by_name, by_id):
            out += [f"已入选本期榜单 → 见正文 #{pos}。", ""]
            continue

        saved = saved_by_name.get(name, {})
        created = (saved.get("created_at") or "")[:10]
        out += [f"链接: https://github.com/{name}", "",
                f"- 创建时间: {created or '未知'}"]
        if language := (row.get("language") or saved.get("language") or "").strip():
            out.append(f"- 主语言: {language}")
        out.append(f"- 总 Star: {row.get('star', 0):,}")
        out.append(f"- {gained}(Trending 口径): +{row.get('stars_today', 0):,}")
        if topics := [t for t in saved.get("topics") or [] if t][:TOPICS_SHOWN]:
            out.append(f"- 主题标签: {', '.join(topics)}")
        out.append("")
        for title, content in _sections(name, saved, descs.get(name, ""),
                                        config.DAYS_SINCE_CREATED):
            out += [f"### {title}", "", *_paragraphs(content), ""]

    return "\n".join(out).rstrip() + "\n"


def append_trending(path: str, rows: list[dict], ranked: list[tuple[str, dict]],
                    gh=None, progress=None, period: str = "weekly") -> bool:
    """把某个周期的 Trending 对照附栏接到已生成的报告尾部。

    幂等**按周期算**:周榜已在不挡月榜,重跑同一个周期才跳过。

    没上榜的条目走和正文同一条 LLM 介绍管线;不在 DB 里的仓库拿 Trending 自带的简介
    当素材兜底,写回时 `write_descriptions` 自己会跳过它们。
    """
    name = path.rsplit("/", 1)[-1]
    text = reports.read(name)
    if text is None:
        return False
    if reports.appendix_mark(period) in text:
        return True

    saved_by_name = universe.load()
    by_name, by_id = _positions(ranked, saved_by_name)
    enriched = dict(saved_by_name)
    for row in rows:
        enriched.setdefault(row["full_name"], {
            "gh_desc": row.get("description", ""),
            "language": row.get("language", ""),
        })

    offlist = [r["full_name"] for r in rows
               if not _position_of(r["full_name"], saved_by_name, by_name, by_id)]
    descs, writeback = descriptions([(n, {}) for n in offlist], enriched,
                                    gh=gh, progress=progress)
    if writeback:
        universe.write_descriptions(writeback)

    appendix = render_trending(rows, ranked, enriched, descs, period)
    return reports.save(name, text.rstrip() + "\n\n" + appendix) is not None
