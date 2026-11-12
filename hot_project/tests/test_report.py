"""报告的生成 ↔ 解析往返,以及报告目录的读写。

往返是这里唯一真正重要的测试:`service/report.py` 写、`data_access/reports.py` 读,
两个文件谁改了格式而另一个没跟上,Web 页面和 `star_trend` 会静默变成空 —— 不报错,
只是报告"不是结构化榜单"。单看任何一边的测试都发现不了。
"""

from types import SimpleNamespace

import pytest

from hot_project import config
from hot_project import cron_weekly_report as weekly
from hot_project.common.timeutil import format_day, utc_today
from hot_project.infra.data_access import reports
from hot_project.service import describe, report
from hot_project.web import render, view_model


@pytest.fixture(autouse=True)
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    return tmp_path


RANKED = [
    ("openai/whisper", {"star": 82000, "growth": 1500}),
    ("langchain-ai/langchain", {"star": 95000, "growth": 900, "recent_growth": 400}),
]
SAVED = {
    "openai/whisper": {"created_at": "2022-09-16T00:00:00Z", "language": "Python",
                       "topics": ["speech", "asr"], "gh_desc": "Robust speech recognition"},
    "langchain-ai/langchain": {"created_at": "2022-10-17T00:00:00Z", "language": "Python",
                               "topics": ["llm"], "gh_desc": "Build LLM apps"},
}


def _render(descs=None, **kwargs):
    opts = dict(mode="comprehensive", growth_days=7, growth_threshold=1000,
                min_star=500, created_days=None, topic=None)
    opts.update(kwargs)
    return report.render(RANKED, SAVED, descs or {}, **opts)


# ── 往返 ──────────────────────────────────────────────────────────

def test_a_generated_report_parses_back_into_the_same_facts():
    parsed = reports.parse(_render())
    assert parsed is not None
    assert [e.repo for e in parsed.entries] == [n for n, _ in RANKED]
    first = parsed.entries[0]
    assert first.rank == 1
    assert first.link == "https://github.com/openai/whisper"
    assert first.metadata["总 Star"] == "82,000"
    assert first.metadata["创建时间"] == "2022-09-16"
    assert first.metadata["主语言"] == "Python"


def test_the_growth_field_is_found_no_matter_the_window():
    """字段名里带着天数(近7天增长 / 近10天增长),写死键名的那版一改窗口就静默返回空。"""
    for days in (3, 7, 30):
        parsed = reports.parse(_render(growth_days=days))
        assert f"近{days}天增长" in parsed.entries[0].metadata
        assert reports.growth_of(parsed.entries[0].metadata) == "+1,500"


def test_numbers_survive_the_round_trip():
    parsed = reports.parse(_render())
    assert reports.number_of(parsed.entries[0].metadata["总 Star"]) == 82000
    assert reports.number_of(
        reports.growth_of(parsed.entries[0].metadata)) == 1500


def test_all_four_sections_come_back():
    parsed = reports.parse(_render())
    titles = [s["title"] for s in parsed.entries[0].sections]
    assert titles == list(report.describe.SECTIONS)
    assert all(s["content"] for s in parsed.entries[0].sections)


def test_a_report_without_the_required_metadata_is_not_a_ranking():
    """随手记的 md 不该被当报告解析,否则 Web 端会渲染出一堆空条目。"""
    assert reports.parse("# 随手记\n\n## 1. owner/repo\n\n一些说明\n") is None


def test_an_empty_document_is_not_a_ranking():
    assert reports.parse("") is None


# ── 解析细节 ───────────────────────────────────────────────────────

def test_the_llm_writing_an_extra_field_does_not_glue_onto_the_previous_one():
    """模型常多写「核心依赖与生态」。不认它的话,后面几百字会全粘在上一段尾巴上。"""
    prose = ("项目定位与用途:一个语音识别模型。\n"
             "解决的问题:多语言转写。\n"
             "使用场景:字幕生成。\n"
             "技术架构与特性:Transformer。\n"
             "核心依赖与生态:PyTorch、numpy、ffmpeg,以及一大段没人要的展开说明。\n")
    sections = report._extract(prose)
    assert sections["技术架构与特性"] == "Transformer。"
    assert "PyTorch" not in sections["技术架构与特性"]


def test_both_kinds_of_colon_are_accepted():
    """模型中英文冒号混着写,只认一种会让一半的字段解析不出来。"""
    assert report._extract("项目定位与用途: 半角冒号\n")["项目定位与用途"] == "半角冒号"
    assert report._extract("项目定位与用途：全角冒号\n")["项目定位与用途"] == "全角冒号"


def test_the_refresh_button_regenerates_even_when_the_cache_is_not_stale(monkeypatch):
    """报告页那个刷新按钮走的就是这条路,靠 `force` 越过缓存。

    `force` 要是不起作用,表现是按钮转一圈、内容一字没变 —— 看着像模型又写了个差不多的,
    没人会当成 bug 报上来。
    """
    old = "".join(f"{t}:旧介绍。\n" for t in describe.SECTIONS)
    fresh = {"openai/whisper": {"desc": old, "desc_updated_at": format_day(utc_today())}}
    monkeypatch.setattr(report.describe, "describe", lambda *a, **k: "新介绍")
    gh = SimpleNamespace(profiles=lambda names, want=(): {n: {} for n in names})

    ready, writeback = report.descriptions([("openai/whisper", {})], fresh, gh=gh)
    assert ready["openai/whisper"] == old and not writeback

    ready, writeback = report.descriptions([("openai/whisper", {})], fresh, gh=gh, force=True)
    assert ready["openai/whisper"] == "新介绍"
    assert writeback["openai/whisper"]["desc"] == "新介绍"


def test_a_fresh_but_incomplete_cache_is_regenerated_not_served():
    """对话渠道写过三段版、老库存过一句话版:缺段的缓存拿来用,报告第 4 段就只剩模板句。"""
    today = format_day(utc_today())
    three = "".join(f"{t}:三段版。\n" for t in describe.SECTIONS[:3])
    assert report._needs_llm({"desc": three, "desc_updated_at": today})
    assert report._needs_llm({"desc": "一句话的老介绍", "desc_updated_at": today})
    four = "".join(f"{t}:齐的。\n" for t in describe.SECTIONS)
    assert not report._needs_llm({"desc": four, "desc_updated_at": today})


def test_the_refreshed_sections_are_split_the_same_way_the_page_renders_them():
    """刷新按钮回传的分段必须和整页渲染逐字一致,否则点完刷新与刷新页面看到的排版不同。"""
    desc = ("项目定位与用途:一个语音识别模型。\n续行不能丢。\n"
            "解决的问题:多语言转写。\n"
            "核心依赖与生态:模型多写的字段,报告页不展示它。\n")
    payload = view_model.section_payload(desc)

    assert [s["title"] for s in payload] == ["项目定位与用途", "解决的问题"]
    assert payload[0]["paragraphs"] == ["一个语音识别模型。 续行不能丢。"]


def test_a_missing_description_falls_back_to_metadata_not_a_blank_section():
    """LLM 全挂时报告照样要能出:只有 star 数没有介绍的报告仍然有用。"""
    parsed = reports.parse(_render(descs={}))
    for section in parsed.entries[0].sections:
        assert section["content"].strip()


def test_finding_a_repo_in_a_report_accepts_a_partial_name():
    parsed = reports.parse(_render())
    assert parsed.find("langchain").repo == "langchain-ai/langchain"
    assert parsed.find("LANGCHAIN-AI/LANGCHAIN").repo == "langchain-ai/langchain"
    assert parsed.find("nope/nope") is None


# ── 文件名 ────────────────────────────────────────────────────────

def test_each_mode_gets_its_own_filename_so_they_do_not_overwrite():
    names = {
        report.filename(mode, growth_days=config.GROWTH_CALC_DAYS,
                        created_days=None, topic=None)
        for mode in ("comprehensive", "hot_new")
    }
    assert len(names) == 2


def test_two_keyword_reports_on_the_same_day_do_not_overwrite_each_other():
    a = report.filename("keyword", growth_days=config.GROWTH_CALC_DAYS,
                        created_days=None, topic="向量库")
    b = report.filename("keyword", growth_days=config.GROWTH_CALC_DAYS,
                        created_days=None, topic="OCR")
    assert a != b and "向量库" in a and "OCR" in b


def test_a_non_default_window_is_marked_in_the_filename():
    name = report.filename("comprehensive", growth_days=config.GROWTH_CALC_DAYS + 3,
                           created_days=None, topic=None)
    assert f"_{config.GROWTH_CALC_DAYS + 3}d" in name


def test_a_topic_can_never_escape_the_report_directory():
    """方向名来自用户输入,而写报告这一步是有权限建目录的。"""
    name = report.filename("keyword", growth_days=config.GROWTH_CALC_DAYS,
                           created_days=None, topic="../../etc")
    assert "/" not in name and ".." not in name


# ── 报告目录 ───────────────────────────────────────────────────────

def _write(dir_, name, body="# 标题\n\n> 摘要\n\n## 1. a/b\n\n链接: x\n\n- 创建时间: 2024-01-01\n- 总 Star: 10\n"):
    (dir_ / name).write_text(body, encoding="utf-8")


def test_listing_is_newest_first_and_carries_the_title(report_dir):
    import os, time
    _write(report_dir, "2026-07-01.md")
    time.sleep(0.01)
    _write(report_dir, "2026-07-02.md")
    items = reports.listing()
    assert [i.name for i in items] == ["2026-07-02.md", "2026-07-01.md"]
    assert items[0].title == "标题"
    assert str(items[0].day) == "2026-07-02"


def test_latest_resolves_to_the_newest_report(report_dir):
    _write(report_dir, "2026-07-01.md")
    assert reports.resolve_name("最新") == "2026-07-01.md"
    assert reports.resolve_name("latest") == "2026-07-01.md"


def test_the_md_suffix_is_optional(report_dir):
    _write(report_dir, "2026-07-01.md")
    assert reports.resolve_name("2026-07-01") == "2026-07-01.md"


def test_a_report_name_can_never_escape_the_directory(report_dir):
    """名字来自工具参数,工具参数来自模型,模型的输入来自用户。"""
    for evil in ("../../etc/passwd", "..", "sub/dir.md", r"a\b.md"):
        assert reports.resolve_name(evil) is None
    listed = [reports.Listed("../x.md", "", None)]
    assert reports.resolve_name("../x.md", listed) is None


def test_an_unknown_report_name_resolves_to_nothing(report_dir):
    _write(report_dir, "2026-07-01.md")
    assert reports.resolve_name("2020-01-01") is None


def test_an_empty_directory_lists_nothing_instead_of_exploding(report_dir):
    assert reports.listing() == []
    assert reports.resolve_name("最新") is None


def test_several_reports_on_one_day_contribute_a_single_point(report_dir):
    """时间序列一天只能有一个点,混进来会让同一周出现两个矛盾的 star 值。"""
    _write(report_dir, "2026-07-01.md")
    _write(report_dir, "2026-07-01_NEW.md")
    _write(report_dir, "2026-07-08.md")
    assert [str(item.day) for item, _ in reports.load_all()] == ["2026-07-01", "2026-07-08"]


def test_saving_then_loading_gives_the_report_back(report_dir):
    path = reports.save("2026-07-30.md", _render())
    assert path is not None
    loaded = reports.load("2026-07-30.md")
    assert loaded is not None and len(loaded.entries) == 2


# ── 上一期配对 ─────────────────────────────────────────────────────

def test_the_previous_issue_is_picked_by_date_not_by_file_mtime(report_dir):
    """CI 每次全新 checkout,所有报告的 mtime 几乎相同、先后随机。

    按 `listing()` 的顺序取第一条就等于随机挑一期当"上一期",推送里的
    「上新 N · 移出 M」会全错,而收到推送的人没法看出来。这里把 mtime 反着设 ——
    让最该被选中的那期看起来"最旧" —— 按 mtime 挑就必然选错。
    """
    import os
    for name in ("2026-06-01.md", "2026-07-01.md", "2026-07-20.md", "2026-07-27.md"):
        _write(report_dir, name)
    for offset, name in enumerate(("2026-07-27.md", "2026-07-20.md",
                                   "2026-07-01.md", "2026-06-01.md")):
        stamp = 1_000_000 + offset * 1000
        os.utime(report_dir / name, (stamp, stamp))

    found = weekly.previous_report("2026-07-30.md")
    assert found is not None
    assert found[0] == "2026-07-27.md"


def test_the_previous_issue_must_be_the_same_kind_of_ranking(report_dir):
    """综合榜要和综合榜比。拿 _NEW 那份当上期,「上新/移出」全是噪音。"""
    _write(report_dir, "2026-07-20.md")
    _write(report_dir, "2026-07-27_NEW.md")

    assert weekly.previous_report("2026-07-30.md")[0] == "2026-07-20.md"
    assert weekly.previous_report("2026-07-30_NEW.md")[0] == "2026-07-27_NEW.md"


def test_the_first_issue_ever_has_no_previous(report_dir):
    _write(report_dir, "2026-07-30.md")
    assert weekly.previous_report("2026-07-30.md") is None


# ── Trending 对照附栏 ──────────────────────────────────────────────

TRENDING_ROWS = [
    {"full_name": "openai/whisper", "star": 82500, "stars_today": 900,
     "language": "Python", "description": "Robust speech recognition"},
    {"full_name": "uber/adr", "star": 828, "stars_today": 148,
     "language": "Python", "description": "Agent security"},
]


def test_the_trending_appendix_is_invisible_to_the_report_parser():
    """附栏条目要是被解析成正文条目,「上新/移出」、出场次数、star 趋势全被污染。"""
    text = _render() + "\n" + report.render_trending(TRENDING_ROWS, RANKED, SAVED, {})
    parsed = reports.parse(text)
    assert [e.repo for e in parsed.entries] == [n for n, _ in RANKED]


def _with_appendices() -> str:
    """正文 + 周榜附栏 + 月榜附栏,和 cron 实际写出来的报告同一个形状。"""
    return "\n".join([
        _render(),
        report.render_trending(TRENDING_ROWS, RANKED, SAVED, {}, "weekly"),
        report.render_trending(TRENDING_ROWS[:1], RANKED, SAVED, {}, "monthly"),
    ])


def test_each_period_is_parsed_into_its_own_bucket():
    """周榜和月榜的条目都从 T1 起,靠附栏标题分段;串了段就会张冠李戴。"""
    parsed = reports.parse(_with_appendices())
    assert [e.repo for e in parsed.entries] == [n for n, _ in RANKED]   # 正文不受影响
    assert list(parsed.trending) == ["weekly", "monthly"]
    assert len(parsed.trending["weekly"]) == 2
    assert [e.repo for e in parsed.trending["monthly"]] == ["openai/whisper"]


def test_both_trending_appendices_reach_the_rendered_page():
    """附栏进得了 Markdown、上不了网页,是这功能上一次的坏法 —— 生成端改了前端没跟。

    最后一条断言守的是 `report.js` 的配对:面板和侧栏项按下标配对,数量对不上整页错位。
    """
    html = render.report_html("2026-07-30.md", _with_appendices())

    assert "GitHub Trending 周榜对照" in html
    assert "GitHub Trending 月榜对照" in html
    assert html.count(f'id="{view_model.TREND_ANCHOR}-') == 2
    assert 'href="#repo-1-openai-whisper"' in html      # 已上榜的一行跳回正文
    assert "uber/adr" in html                           # 没上榜的整卡补全
    assert "同时在 GitHub Trending 周榜、月榜上" in html   # 两个榜都在的挂一个角标,列全
    assert html.count('<section class="repo-detail') == html.count('<a class="repo-nav__item')


def test_writers_and_parsers_agree():
    """写函数和解析正则成对同源的守卫:任何一边单独改格式,这里立刻红。

    附栏曾因写端加了 `## T1.` 而解析端不认识,整段在页面上静默消失 —— 就防这个。
    """
    assert reports._HEADING.match(reports.heading(3, "owner/repo"))
    assert reports._TREND_HEADING.match(reports.trend_heading(1, "owner/repo"))
    for period in reports.PERIOD_TEXT:
        m = reports._APPENDIX.match(reports.appendix_mark(period))
        assert m and m.group("period") == period


def test_a_listed_trending_repo_links_back_and_an_unlisted_one_is_rendered_in_full():
    appendix = report.render_trending(TRENDING_ROWS, RANKED, SAVED, {})
    assert "## T1. openai/whisper" in appendix
    assert "见正文 #1" in appendix
    assert "## T2. uber/adr" in appendix
    assert "链接: https://github.com/uber/adr" in appendix
    assert "- 本周新增(Trending 口径): +148" in appendix
    assert "近7天增长" not in appendix          # 不冒充我们的窗口口径


def test_a_renamed_trending_repo_is_matched_through_its_id():
    """Trending 上挂新名、榜内还是旧名时,靠 databaseId 也要能对上,不重复补全。"""
    ranked = [("old/name", {"star": 100, "growth": 50, "id": 7})]
    saved = {"new/name": {"id": 7}}
    rows = [{"full_name": "new/name", "star": 100, "stars_today": 10,
             "language": "", "description": ""}]
    appendix = report.render_trending(rows, ranked, saved, {})
    assert "见正文 #1" in appendix


def test_append_trending_only_appends_once(report_dir, monkeypatch):
    monkeypatch.setattr(report.universe, "load", lambda: SAVED)
    monkeypatch.setattr(report, "descriptions", lambda *a, **k: ({}, {}))
    path = str(reports.save("2026-07-30.md", _render()))

    assert report.append_trending(path, TRENDING_ROWS, RANKED)
    once = reports.read("2026-07-30.md")
    assert reports.appendix_mark("weekly") in once
    assert report.append_trending(path, TRENDING_ROWS, RANKED)
    assert reports.read("2026-07-30.md") == once
