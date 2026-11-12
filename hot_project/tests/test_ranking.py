"""打分公式与榜单流水线。

流水线现在会实时取 star,所以这里要假一个 GitHub 门面出来;快照仍然只出基线,换成内存字典。
"""

from datetime import timedelta

import pytest

from hot_project import config
from hot_project.common.timeutil import utc_today
from hot_project.infra.data_access import snapshots
from hot_project.provider.github import collect
from hot_project.service import ranking

W = ranking.Weights(window_days=7, recent_days=3, alpha=0.5, cap=2.0)


# ── 打分公式 ───────────────────────────────────────────────────────

def test_more_growth_scores_higher():
    small = ranking.score({"growth": 100, "star": 10000}, ranking.COMPREHENSIVE, W)
    big = ranking.score({"growth": 5000, "star": 10000}, ranking.COMPREHENSIVE, W)
    assert big > small


def test_a_small_project_growing_fast_can_beat_a_giant_growing_slowly():
    """增长率那一项存在的全部理由。没有它,榜单永远是那几个巨无霸。"""
    nimble = ranking.score({"growth": 900, "star": 1200}, ranking.COMPREHENSIVE, W)
    giant = ranking.score({"growth": 1000, "star": 200000}, ranking.COMPREHENSIVE, W)
    assert nimble > giant


def test_an_absurd_growth_rate_buys_much_less_than_it_looks():
    """100 星涨到 300 的「三倍」和 10000 星涨 2 万不是一回事,两处 log 负责压平。"""
    plain = {"growth": 400, "star": 800}          # 率 0.5
    absurd = {"growth": 1200, "star": 800}        # 率 1.5
    ratio = (ranking.score(absurd, ranking.COMPREHENSIVE, W)
             / ranking.score(plain, ranking.COMPREHENSIVE, W))
    assert ratio == pytest.approx(1.30, rel=0.02)


def test_hot_new_ranks_purely_by_growth():
    """新项目榜比的就是谁涨得多。叠增长率会让三天涨 200 压过三十天涨 3000。"""
    assert ranking.score({"growth": 3000, "star": 50000}, ranking.HOT_NEW, W) == 3000.0
    assert ranking.score({"growth": 200, "star": 210}, ranking.HOT_NEW, W) == 200.0


# ── 爆发加成 ───────────────────────────────────────────────────────

def test_speeding_up_earns_a_boost():
    assert ranking.burst_boost(700, 600, W) == pytest.approx(1.5)


def test_slowing_down_is_not_punished():
    """「前期猛、最近平」本来就该靠基础分排;再罚一次是对同一件事收两遍税。"""
    assert ranking.burst_boost(700, 30, W) == 1.0


def test_one_freak_number_cannot_flip_the_board():
    huge = ranking.burst_boost(7, 100000, W)
    assert huge == pytest.approx(1.0 + W.alpha * W.cap)


def test_no_probe_data_means_no_boost_not_a_penalty():
    """缺快照的日子探针整个失效。全场一起挨罚的话顺序其实没变,
    只是分数集体缩水,看日志的人会以为打分坏了。"""
    assert ranking.burst_boost(700, None, W) == 1.0


def test_the_boost_uses_the_real_span_on_both_sides():
    """主窗口修正了、探针没修正的话,5 天的增量除以名义的 3 天,凭空造出一场爆发。"""
    honest = ranking.burst_boost(700, 500, W._replace(recent_days=5))
    inflated = ranking.burst_boost(700, 500, W._replace(recent_days=3))
    assert inflated > honest


def test_each_repo_uses_its_own_baseline_span_not_the_global_one():
    """晚进库的仓库基线只有 3 天。拿全局 7 天当分母,它的整窗均速被低估一半多,
    加速比虚高,凭空多拿一档加成 —— 逐仓天数就是为了堵这个。"""
    item = {"growth": 700, "star": 9000, "recent_growth": 600,
            "window_days": 3, "recent_days": 3}
    honest = ranking.score(item, ranking.COMPREHENSIVE, W)
    inflated = ranking.score({**item, "window_days": 7}, ranking.COMPREHENSIVE, W)
    assert inflated > honest


def test_ranking_writes_the_score_back_for_the_logs():
    ordered = ranking.rank({"a/x": {"growth": 10, "star": 100}}, ranking.COMPREHENSIVE, W)
    assert ordered[0][1]["_score"] > 0


# ── 流水线 ────────────────────────────────────────────────────────

class _GH:
    """假门面:只回答「这批仓库此刻多少星」。"""

    def __init__(self, live: dict[str, int], usable: bool = True) -> None:
        self.live, self.usable = live, usable
        self.asked: list[str] = []

    def stars(self, names):
        self.asked = list(names)
        return collect.Harvest(stars={n: s for n, s in self.live.items() if n in set(names)})


def _baseline(stars, span, spans=None):
    """构造一份「窗口内最早」的基线。`spans` 只在要逐仓不同天数时给。"""
    return snapshots.Baseline(stars, spans or {n: span for n in stars},
                              utc_today() - timedelta(days=span), span)


@pytest.fixture
def world(monkeypatch):
    """把「窗口内最早的快照」和「DB」换成内存里的东西。"""
    state: dict = {"baselines": {}, "db": {}}

    def earliest_in_window(days, today=None):
        return state["baselines"].get(days) or snapshots.Baseline({}, {}, None, days)

    monkeypatch.setattr(snapshots, "earliest_in_window", earliest_in_window)
    monkeypatch.setattr(ranking.universe, "load", lambda: state["db"])
    return state


def _created(days_ago: int) -> str:
    return (utc_today() - timedelta(days=days_ago)).isoformat()


def test_the_current_star_is_asked_for_live_never_read_from_a_snapshot(world):
    """被减数必须是此刻的真值。读当天快照的话,采集到出榜之间涨的 star 全部消失,
    而且当天采集没跑成就出不了榜 —— 两个定时任务被焊死在一起。"""
    world["db"] = {"a/x": {"created_at": _created(500)}}
    world["baselines"][7] = _baseline({"a/x": 1000}, 7)
    gh = _GH({"a/x": 3000})

    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=gh)
    assert gh.asked == ["a/x"]
    assert out["ranked"][0][1]["growth"] == 2000        # 3000(实时) − 1000(基线)


def test_a_blacklisted_repo_is_dropped_before_anything_is_fetched(world, monkeypatch):
    """黑名单在取名单那一步就剔,不为它花 API 配额;大小写要认。"""
    monkeypatch.setattr(config, "EXCLUDED_REPOS", frozenset({"a/x"}))
    world["db"] = {"A/X": {"created_at": _created(500)}, "b/y": {"created_at": _created(500)}}
    world["baselines"][7] = _baseline({"A/X": 1000, "b/y": 1000}, 7)
    gh = _GH({"A/X": 3000, "b/y": 3000})

    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=gh)
    assert gh.asked == ["b/y"]
    assert [n for n, _ in out["ranked"]] == ["b/y"]


def test_no_github_token_means_no_ranking_rather_than_a_wrong_one(world):
    world["db"] = {"a/x": {"created_at": _created(500)}}
    world["baselines"][7] = _baseline({"a/x": 1000}, 7)
    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=_GH({"a/x": 3000}, usable=False))
    assert out["ranked"] == []


def test_no_baseline_snapshot_at_all_means_no_ranking(world):
    world["db"] = {"a/x": {"created_at": _created(500)}}
    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=_GH({"a/x": 3000}))
    assert out["ranked"] == []


def test_below_threshold_candidates_never_enter_the_pool(world):
    """低于阈值的当场丢。留下来只会在之后每一步被重新遍历一遍,7.8 万条几十 MB。"""
    base = _baseline({"a/hot": 1000, "b/flat": 1000}, 7)
    pool, unresolved = ranking.qualify(
        {"a/hot": 3000, "b/flat": 1010},
        {"a/hot": {"created_at": _created(500)}, "b/flat": {"created_at": _created(500)}},
        base, min_star=500, threshold=1000)
    assert set(pool) == {"a/hot"}
    assert unresolved == 0


def test_a_repo_below_min_star_is_filtered_on_the_live_value(world):
    base = _baseline({"a/x": 10}, 7)
    pool, _ = ranking.qualify({"a/x": 400}, {"a/x": {"created_at": _created(500)}},
                              base, min_star=500, threshold=100)
    assert pool == {}


def test_a_repo_with_no_baseline_and_no_creation_date_is_counted_but_dropped(world):
    """算不出增长 ≠ 涨了 0:它不进池,但要有个数报出来,否则「榜怎么空了」无从查起。"""
    base = _baseline({"a/known": 400}, 7)
    pool, unresolved = ranking.qualify(
        {"a/known": 1500, "b/mystery": 9000},
        {"a/known": {"created_at": _created(900)}, "b/mystery": {"created_at": ""}},
        base, min_star=500, threshold=1000)
    assert set(pool) == {"a/known"} and unresolved == 1


def test_a_repo_missing_from_the_oldest_snapshot_still_ranks_off_a_later_one(world):
    """晚进库的仓库在窗口第一天的快照里没有,但三天前那份有。按 3 天的实测增长排名,
    比整个丢掉强 —— 旧实现在这里把它记成未决,一个刚爆火的项目就此永远上不了榜。"""
    base = _baseline({"a/old": 1000, "b/late": 2000}, 7,
                     spans={"a/old": 7, "b/late": 3})
    pool, unresolved = ranking.qualify(
        {"a/old": 1500, "b/late": 5000},
        {n: {"created_at": _created(500)} for n in ("a/old", "b/late")},
        base, min_star=500, threshold=1000)
    assert unresolved == 0
    assert pool["b/late"]["growth"] == 3000
    assert pool["b/late"]["window_days"] == 3       # 逐仓天数,不是全局的 7


def test_the_probe_writes_back_each_repos_own_span(world):
    world["baselines"][config.RECENT_GROWTH_DAYS] = _baseline({"a/x": 100}, 5)
    pool = {"a/x": {"star": 400, "growth": 300, "window_days": 7}}
    span = ranking.recent(pool, config.RECENT_GROWTH_DAYS)
    assert span == 5
    assert pool["a/x"]["recent_growth"] == 300 and pool["a/x"]["recent_days"] == 5


def test_a_missing_probe_baseline_never_fails_the_ranking(world):
    pool = {"a/x": {"star": 400, "growth": 300, "window_days": 7}}
    assert ranking.recent(pool, 3) == 3
    assert "recent_growth" not in pool["a/x"]


def test_a_repo_that_lost_stars_is_skipped_by_the_probe(world):
    """探针只做加成。负的最近增长喂进加速比是没有意义的输入。"""
    world["baselines"][3] = _baseline({"a/x": 500}, 3)
    pool = {"a/x": {"star": 400, "growth": 300, "window_days": 7}}
    ranking.recent(pool, 3)
    assert "recent_growth" not in pool["a/x"]


def test_the_window_follows_the_oldest_snapshot_not_the_request(world):
    """请求 7 天但最早只有 T−5 的快照:全程按 5 天算,报告标题也写 5 天。
    两套值并存过一次,后果是新仓库被误判出局。"""
    world["db"] = {"a/new": {"created_at": _created(6)}}
    world["baselines"][7] = _baseline({}, 5)
    out = ranking.run(min_star=500, growth_threshold=500, growth_days=7,
                      do_report=False, gh=_GH({"a/new": 600}))
    assert out["growth_days"] == 5
    assert out["ranked"] == []          # 6 天大 > 5 天窗口,且没基线 → 算不出来


def test_a_full_run_ranks_and_reports_the_funnel(world):
    world["db"] = {n: {"created_at": _created(500)}
                   for n in ("a/hot", "b/flat", "c/tiny", "d/mystery")}
    world["baselines"][7] = _baseline({"a/hot": 3000, "b/flat": 2990, "c/tiny": 50}, 7)
    world["baselines"][config.RECENT_GROWTH_DAYS] = _baseline({"a/hot": 3500}, 3)
    gh = _GH({"a/hot": 5000, "b/flat": 3000, "c/tiny": 100, "d/mystery": 8000})

    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=gh)

    assert [n for n, _ in out["ranked"]] == ["a/hot"]       # b 只涨 10,c 星太低
    assert out["funnel"] == {"名单": 4, "取到star": 4, "达标": 1,
                             "爆发加成": 1, "出榜": 1}
    assert out["growth_days"] == 7


def test_a_renamed_repo_keeps_its_history_through_the_id(world):
    """基线按 databaseId 归键:仓库改了名,快照里旧名条目折到同一个 id 下,
    增长不被重置成改名后那几天。"""
    world["db"] = {"new/n": {"created_at": _created(500), "id": 42}}
    world["baselines"][7] = _baseline({42: 1000}, 7)      # 旧名时代的历史,挂在 id 下
    gh = _GH({"new/n": 4000})

    out = ranking.run(min_star=500, growth_threshold=1000, growth_days=7,
                      do_report=False, gh=gh)

    assert [n for n, _ in out["ranked"]] == ["new/n"]
    assert out["ranked"][0][1]["growth"] == 3000              # 4000 − 1000,历史没丢
    assert gh.asked == ["new/n"]


def test_hot_new_keeps_only_recently_created_repos(world):
    world["db"] = {"a/old": {"created_at": _created(900)},
                   "b/fresh": {"created_at": _created(3)}}
    world["baselines"][7] = _baseline({}, 7)        # 都没基线 → 只能靠创建时间判定
    out = ranking.run(mode="hot_new", min_star=500, growth_threshold=1000,
                      growth_days=7, created_days=30, do_report=False,
                      gh=_GH({"a/old": 5000, "b/fresh": 4000}))
    assert [n for n, _ in out["ranked"]] == ["b/fresh"]
    assert out["ranked"][0][1]["growth"] == 4000        # 有尾无头:全部 star 都是增长
