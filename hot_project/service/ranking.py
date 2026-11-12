"""榜单流水线 —— 综合 / 新项目 / 关键词三榜共用。

    候选名单 → 实时 star → 增长 → 阈值 → 爆发探针 → 打分 → 报告

**当前 star 一律实时取,不读当天的快照。** 读当天快照会把两个定时任务焊死在一起:榜单的
数字锚死在每日采集那一刻,采集到出榜之间涨的 star 凭空消失,而且当天采集没跑成榜单就出不来。
快照只负责一件事 —— 给出窗口内**最早**那天的基线。

两条不变量:

- **达标才进内存。** 增长低于阈值的当场丢:它既出不了榜,也不参与探针和排序。7.8 万个候选
  收到几百个,省下的是几十 MB 和之后每一步的全表遍历。
- **算不出增长的不参与。** 窗口内没有任何快照测到过它、创建时间又不在窗口内 → 这轮它不
  出现。记 0 的话它会以「一点没涨」的身份进排名,永远出不了榜,也没人看得出原因。
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

from .. import config
from ..common.timeutil import age_days
from ..infra.data_access import snapshots, universe
from . import growth as growth_calc
from . import report as report_tool

logger = logging.getLogger("hot_project")

MODES = ("comprehensive", "hot_new", "keyword")


class Funnel(dict):
    """每一层剩下多少。出榜数少于预期时,唯一能回答「卡在哪一层」的东西。"""

    def line(self) -> str:
        return " → ".join(f"{k} {v}" for k, v in self.items())


def _emit(progress, percent: int, label: str) -> None:
    """回传进度。回调自己抛异常绝不能影响流水线 —— 那只是给前端画进度条的。"""
    if progress is None:
        return
    try:
        progress(percent, label)
    except Exception:       # noqa: BLE001
        logger.debug("进度回调异常已忽略", exc_info=True)


def current_stars(gh, names: list[str]) -> dict[str, int]:
    """实时取这批仓库此刻的 star。批量走任务池 + token 池,和每日采集同一条路。

    只取 `stars`:「查不到」和「没问到」在这里都只意味着它这轮排不了名,两者的区别只对淘汰判定有意义。
    """
    if gh is None or not getattr(gh, "usable", False):
        logger.error("没有可用的 GitHub token,取不到当前 star,本轮无从算起。")
        return {}
    harvest = gh.stars(names)
    logger.info("实时取 star:请求 %d 个,取到 %d,GitHub 查不到 %d,没问到 %d。",
                len(names), len(harvest.stars), len(harvest.missing), len(harvest.failed))
    return harvest.stars


def qualify(stars: dict[str, int], meta: dict[str, dict], base: snapshots.Baseline,
            *, min_star: int, threshold: int) -> tuple[dict[str, dict], int]:
    """边算边筛:算一个增长,达标才留。返回 `(候选池, 缺基线算不出的个数)`。

    低于阈值的当场丢 —— 先建全量表再过滤的话,那张表七万多条、几十 MB,建出来只为下一行扔掉。
    """
    pool: dict[str, dict] = {}
    unresolved = 0
    for name, star in stars.items():
        if star < min_star:
            continue
        info = meta.get(name, {})
        created = info.get("created_at", "")
        base_star, base_days = base.get(name, info.get("id"))
        result = growth_calc.resolve(star, base_star, base_days,
                                     age_days(created), base.span)
        if result is None:
            unresolved += 1
            continue
        if result.value < threshold:
            continue
        pool[name] = {"star": star, "growth": result.value,
                      "window_days": result.window_days, "created_at": created,
                      "id": info.get("id")}
    return pool, unresolved


def recent(pool: dict[str, dict], days: int) -> int:
    """爆发探针:把最近几天的增长写回每个候选。返回名义天数。

    缺快照时什么都不写 —— 探针不加成即可,绝不能让出榜失败。
    """
    if not pool:
        return days
    base = snapshots.earliest_in_window(days)
    if base.oldest is None:
        logger.info("最近 %d 天内没有快照,本轮跳过爆发加成。", days)
        return days

    hit = 0
    for name, info in pool.items():
        anchor, anchor_days = base.get(name, info.get("id"))
        if anchor is None or info["star"] < anchor:
            continue
        info["recent_growth"] = info["star"] - anchor
        info["recent_days"] = anchor_days if anchor_days is not None else base.span
        hit += 1
    logger.info("爆发探针基线 %s(%d 天):%d/%d 个候选可算。",
                base.oldest, base.span, hit, len(pool))
    return base.span


# ── 打分公式 —— 纯算术,不许读 DB,补 created_at 之类是上面流水线的事 ──
#
#     base  = log(1+增长) × 1000 + log2(1+增长率) × 1200
#     score = base × 爆发加成
#
# log(1+增长) 衡量绝对热度;log2(1+增长率) 衡量爆发力(增长率 = 增长 / 总 star,
# 让小项目的暴涨也能上榜)。`1000 / 1200` 配平的是两项在**真候选池**(过了增长阈值那
# 几十个,不是全库)里的离散度,目标热度 : 增长率 ≈ 6.5 : 3.5;改偏好就动这对系数,按
# `w2 = 1000 × (3.5/6.5) × σ(log(1+增长)) / σ(log2(1+增长率))` 重算。
# σ 随窗口漂:实测 2 天窗口要 1231、7 天要 1112,1200 是折中。
# 新项目榜(hot_new)不用这套公式,直接按增长量降序。

COMPREHENSIVE = "comprehensive"
HOT_NEW = "hot_new"


class Weights(NamedTuple):
    """爆发加成的三个旋钮。默认值由 config 提供,调用方传进来。"""

    window_days: int        # 全局名义窗口(最早那份快照的跨度),逐仓没给天数时的退路
    recent_days: int        # 探针的全局名义天数,同上
    alpha: float            # 加成强度
    cap: float              # 加速比的封顶,防止一个异常值把榜单掀翻


def burst_boost(growth: int, recent_growth: int | None, w: Weights, *,
                window_days: int | None = None, recent_days: int | None = None) -> float:
    """最近爆发加成:加速比 = 最近速率 / 整窗平均速率,大于 1 给加成,小于 1 不扣分。

    `recent_growth` 缺失返回 1.0(探针失效该退化成纯基础分)。**两个分母必须是各自基线的
    实际天数**,否则速率虚高造出假爆发:基线只有 3 天的仓库按 7 天摊就是 2.3 倍假加速。
    """
    if recent_growth is None or recent_growth < 0 or growth <= 0:
        return 1.0
    span = window_days or w.window_days
    recent_span = recent_days or w.recent_days
    if span <= 0 or recent_span <= 0:
        return 1.0
    avg_rate = growth / span
    if avg_rate <= 0:
        return 1.0
    acceleration = (recent_growth / recent_span) / avg_rate
    return 1.0 + w.alpha * min(max(acceleration - 1.0, 0.0), w.cap)


def score(item: dict, mode: str, w: Weights) -> float:
    """一个候选的分数。`item` 要有 `growth` 和 `star`;`recent_growth` 与两个天数可选。"""
    growth = item["growth"]
    star = item["star"]
    if mode != COMPREHENSIVE:
        return float(growth)
    if star <= 0:
        return float(growth)        # 0 星仓库算不出增长率,退化成纯增长量

    rate = max(growth / star, 0.0)
    base = math.log1p(max(growth, 0)) * 1000 + math.log2(1 + rate) * 1200
    return base * burst_boost(growth, item.get("recent_growth"), w,
                              window_days=item.get("window_days"),
                              recent_days=item.get("recent_days"))


def rank(candidates: dict[str, dict], mode: str, w: Weights) -> list[tuple[str, dict]]:
    """按分数降序排全部候选,同时把 `_score` 写回每个候选(日志和调试要看)。

    不截断 Top N:同一份排序结果常要按不同 top_n 复用。
    """
    for info in candidates.values():
        info["_score"] = score(info, mode, w)
    return sorted(candidates.items(), key=lambda kv: kv[1]["_score"], reverse=True)


def run(*, mode: str = "comprehensive", min_star: int, growth_threshold: int,
        growth_days: int,
        created_days: int | None = None, top_n: int | None = None,
        topic: str | None = None, do_report: bool = True,
        gh=None, progress=None, pool: dict[str, dict] | None = None) -> dict:
    """跑一轮榜单。返回排名结果 + 漏斗。

    门槛三件套(`min_star`/`growth_threshold`/`growth_days`)必填:它们是策略,由入口
    (cron 参数、工具 schema 的默认值)决定,这里不替调用方翻 config —— 签名即契约。

    `pool` 给了就只在那批里排,否则排 DB 全库。它只提供名单和 `created_at`,star 一律现取。
    """
    _emit(progress, 5, "读取候选名单…")
    meta = universe.load() if pool is None else pool
    names = sorted(n for n in meta if n.lower() not in config.EXCLUDED_REPOS)
    if dropped := len(meta) - len(names):
        logger.info("黑名单 EXCLUDED_REPOS 剔除 %d 个。", dropped)
    collected = len(names)

    _emit(progress, 15, "实时取当前 star…")
    stars = current_stars(gh, names)

    _emit(progress, 50, "计算增长,筛选达标候选…")
    base = snapshots.earliest_in_window(growth_days)
    window, unresolved = growth_days, 0
    qualified: dict[str, dict] = {}
    if base.oldest is None:
        logger.error("最近 %d 天内一份快照都没有,算不了增长(每日任务挂了?)。", growth_days)
    else:
        window = base.span
        if window != growth_days:
            logger.warning("窗口内最早的快照是 %s:实际窗口 %d 天而非请求的 %d 天,全程按实际算。",
                           base.oldest, window, growth_days)
        qualified, unresolved = qualify(stars, meta, base,
                                        min_star=min_star, threshold=growth_threshold)
    logger.info("达标候选(增长 >= %d):%d 个;另有 %d 个缺基线算不出增长。",
                growth_threshold, len(qualified), unresolved)

    if mode == "hot_new":
        window_created = created_days if created_days is not None else config.DAYS_SINCE_CREATED
        qualified = {
            n: i for n, i in qualified.items()
            if (age := age_days(i.get("created_at", ""))) is not None and age <= window_created
        }
        logger.info("新项目窗口(<= %d 天):剩 %d 个。", window_created, len(qualified))

    _emit(progress, 60, "爆发探针…")
    recent_window = config.RECENT_GROWTH_DAYS
    boosted = 0
    if mode != "hot_new" and qualified:
        recent_window = recent(qualified, config.RECENT_GROWTH_DAYS)
        boosted = sum(1 for i in qualified.values()
                      if (rg := i.get("recent_growth")) is not None and i["growth"] > 0
                      and rg / i["recent_days"] > i["growth"] / i["window_days"])
        logger.info("爆发加成生效 %d 个。", boosted)

    _emit(progress, 70, "排序…")
    weights = Weights(window_days=window, recent_days=recent_window,
                      alpha=config.BURST_ALPHA, cap=config.BURST_CAP)
    ordered = rank(qualified, HOT_NEW if mode == "hot_new" else COMPREHENSIVE, weights)
    ranked = ordered[:top_n] if top_n else ordered

    if top_n and len(ranked) < top_n:
        logger.warning("达标候选不够:要 %d 个,只有 %d 个(名单 %d,取到 star %d,达标 %d)。",
                       top_n, len(ranked), collected, len(stars), len(qualified))

    funnel = Funnel(名单=collected, 取到star=len(stars), 达标=len(qualified),
                    爆发加成=boosted, 出榜=len(ranked))
    logger.info("漏斗:%s", funnel.line())

    result = {"mode": mode, "ranked": ranked, "growth_days": window,
              "recent_days": recent_window, "funnel": dict(funnel)}

    if do_report:
        _emit(progress, 75, "生成报告…")
        result["report_path"] = report_tool.generate(
            ranked, mode=mode, growth_days=window, growth_threshold=growth_threshold,
            min_star=min_star, created_days=created_days, topic=topic, gh=gh,
            progress=lambda frac, label: _emit(progress, 75 + int(24 * frac), label),
        )
    _emit(progress, 100, "完成")
    return result
