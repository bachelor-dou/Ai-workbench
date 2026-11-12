"""工具契约:给模型的 schema 和运行时校验必须是同一份声明长出来的。

旧包这两份是分开写的,漂移过一次(`fetch_trending` 的 `all` 只写进了校验那份,模型
永远请求不到)。这里的第一组测试就是为了让那种漂移在类型上不可能发生。
"""

import json

import pytest

from hot_project import tools
from hot_project.service import favorites as favorite_service
from hot_project.tools import rank_tools, repo_tools
from hot_project.tools.spec import Ctx, Param, Registry, Tool


def _noop(ctx, args):
    return {}


# ── 一份声明,两处使用 ────────────────────────────────────────────

def test_every_declared_param_shows_up_in_the_model_schema():
    """模型看不见的参数等于不存在 —— 它永远不会传。"""
    for tool in (tools.registry().get(n) for n in tools.registry().names()):
        properties = tool.schema()["function"]["parameters"]["properties"]
        assert {p.name for p in tool.params} == set(properties), tool.name


def test_an_enum_offers_the_model_exactly_the_values_it_will_accept():
    tool = tools.registry().get("fetch_trending")
    offered = tool.schema()["function"]["parameters"]["properties"]["trending_range"]["enum"]
    for value in offered:
        assert tool.validate({"trending_range": value})[1] == []
    assert tool.validate({"trending_range": "yearly"})[1][0]["reason"].startswith("must_be_one_of")


def test_required_params_are_marked_as_required_for_the_model():
    schema = tools.registry().get("repo_growth").schema()
    assert schema["function"]["parameters"]["required"] == ["repo"]


def test_a_tool_with_no_params_has_no_required_list():
    schema = tools.registry().get("get_keyword_catalog").schema()
    assert "required" not in schema["function"]["parameters"]


def test_duplicate_param_names_are_caught_at_construction():
    """重名在 schema 里会静默合并,校验时却两条都跑 —— 模型看到的和生效的不是一条。"""
    with pytest.raises(ValueError, match="重名"):
        Tool("x", "", _noop, (Param("n", "int", "", default=1),
                              Param("n", "int", "", default=2)))


def test_every_registered_tool_has_a_description_the_model_can_act_on():
    for name in tools.registry().names():
        assert len(tools.registry().get(name).desc) > 30, name


def test_registering_the_same_name_twice_is_an_error():
    with pytest.raises(ValueError, match="重名"):
        Registry([Tool("a", "d", _noop), Tool("a", "d", _noop)])


# ── 校验 ──────────────────────────────────────────────────────────

def test_defaults_are_filled_in_when_the_model_omits_them():
    clean, errors = tools.registry().get("search_repos").validate({"query": "x"})
    assert errors == []
    assert clean["top_n"] == repo_tools.SEARCH_DEFAULT_N


def test_a_missing_required_param_is_reported_not_silently_defaulted():
    _, errors = tools.registry().get("repo_growth").validate({})
    assert errors == [{"param": "repo", "reason": "missing_required"}]


def test_out_of_range_is_rejected_rather_than_clamped():
    """静默裁到边界的话,模型永远学不会自己传错了。"""
    _, errors = tools.registry().get("search_repos").validate(
        {"query": "x", "top_n": 999})
    assert errors[0]["reason"] == f"must_be_lte_{repo_tools.SEARCH_MAX_N}"


def test_a_hallucinated_param_is_rejected_not_swallowed():
    _, errors = tools.registry().get("get_db_info").validate({"limit": 10})
    assert errors == [{"param": "limit", "reason": "unknown_parameter", "received": 10}]


def test_true_is_not_an_integer():
    """bool 是 int 的子类。不拦的话 top_n=true 会静默变成 top_n=1。"""
    _, errors = tools.registry().get("search_repos").validate(
        {"query": "x", "top_n": True})
    assert errors[0]["reason"] == "expected_integer"


@pytest.mark.parametrize("raw", ["1e400", "-1e400", "NaN"])
def test_a_json_legal_infinity_is_an_error_not_a_crash(raw):
    """`json.loads` 认这三种写法,`int()` 对它们抛的却是 OverflowError/ValueError。

    抛出去比传错值更糟:异常逃出这一轮,tool_calls 就配不上 tool 回复,该会话之后
    每次请求都被接口 400。这里必须返回错误码,让模型看见自己传错了。
    """
    _, errors = tools.registry().get("search_repos").validate(
        {"query": "x", "top_n": json.loads(raw)})
    assert errors[0]["reason"] == "expected_integer"


def test_a_list_of_non_strings_is_rejected():
    _, errors = tools.registry().get("keyword_ranking").validate({"keywords": [1, 2]})
    assert any(e["reason"] == "expected_array_of_strings" for e in errors)


def test_a_none_default_does_not_become_a_literal_parameter():
    """可选参数缺省时不该出现在参数里,否则下游要在每处判 None。"""
    clean, _ = tools.registry().get("analyze_report").validate({})
    assert clean == {}


# ── 确认守卫(在 Registry.run,不在各 handler)──────────────────────

class _State:
    def __init__(self):
        self.pending_confirmation_signature = None
        self.tool_state = {}


def test_an_expensive_tool_asks_before_it_runs():
    ctx = Ctx(state=_State())
    out = tools.registry().run(ctx, "comprehensive_ranking", {"top_n": 20})
    assert out["needs_confirmation"] is True
    assert "Top 20" in out["message"]


def test_an_expensive_tool_must_declare_a_confirm_param():
    """守卫靠 `confirm=true` 放行;昂贵工具漏声明它,模型就永远确认不了。建表时就炸。"""
    with pytest.raises(ValueError, match="confirm"):
        Registry([Tool("slow", "很贵的工具" * 6, _noop, expensive=True)])


def test_the_second_call_runs_the_parameters_that_were_shown(monkeypatch):
    """模型复述参数时会漂移,而用户确认的是屏幕上那份。"""
    seen = {}
    monkeypatch.setattr(rank_tools, "_run", lambda ctx, mode, params: seen.update(params) or {})

    registry = tools.registry()
    ctx = Ctx(state=_State())
    registry.run(ctx, "comprehensive_ranking", {"top_n": 20, "min_star": 500})
    ctx.begin_turn("开始")
    registry.run(ctx, "comprehensive_ranking", {"top_n": 5, "confirm": True})  # 20 被记成了 5

    assert seen["top_n"] == 20 and seen["min_star"] == 500


def test_confirming_clears_the_pending_state_so_the_next_run_asks_again(monkeypatch):
    monkeypatch.setattr(rank_tools, "_run", lambda ctx, mode, params: {})
    registry = tools.registry()
    ctx = Ctx(state=_State())
    registry.run(ctx, "comprehensive_ranking", {"top_n": 20})
    ctx.begin_turn("开始")
    registry.run(ctx, "comprehensive_ranking", {"confirm": True})
    assert registry.run(ctx, "comprehensive_ranking",
                        {"top_n": 30})["needs_confirmation"] is True


def test_confirming_one_ranking_does_not_authorize_a_different_one(monkeypatch):
    """确认必须认「是哪个工具」,不能只看"有没有待确认的东西"。

    待确认槽是会话全局的一个格子。只看它非空的话:先请求关键词榜(参数回显给用户看),
    再拿 `confirm=true` 去调综合榜 —— 门就开了,而且跑的是模型这次传的参数,不是屏幕上
    那份。用户为一次自己从没见过的昂贵执行付账,而"回显即执行"是这套机制的全部意义。
    """
    ran = []
    monkeypatch.setattr(rank_tools, "_run",
                        lambda ctx, mode, params: ran.append((mode, params)) or {})
    ctx = Ctx(state=_State())
    registry = tools.registry()

    shown = registry.run(ctx, "keyword_ranking", {"keywords": ["vector db"]})
    assert shown["needs_confirmation"] is True
    ctx.begin_turn("开始")

    other = registry.run(ctx, "comprehensive_ranking", {"top_n": 200, "confirm": True})

    assert ran == [], f"换个工具带 confirm 就绕过了确认:{ran}"
    assert other["needs_confirmation"] is True, "综合榜应当自己再回显一次"


def test_an_empty_keyword_list_never_degrades_into_a_whole_database_ranking(monkeypatch):
    """`keywords=[]` 能过校验(`any([])` 是假),此时候选池必须是空的,不能是"没有池"。

    "没有池"在 `ranking.run` 里的含义是"排全库" —— 于是用户要的关键词榜变成一份综合榜,
    标题还写着关键词,几分钟的模型调用全花在不相关的项目上,一处都不报错。
    """
    pool = rank_tools._keyword_pool(Ctx(), {"keywords": [], "min_star": 500})
    assert pool == {}, "返回 None 会被当成「调用方没给池」,进而去排全库"


def test_the_confirmation_text_shows_every_parameter_that_will_take_effect():
    """展示即执行。回显漏一个参数,用户确认的就不是实际要跑的东西。"""
    text = rank_tools.confirmation("keyword", {
        "min_star": 500, "growth_days": 7, "growth_threshold": 0,
        "keywords": ["vector db"], "topic": "向量库", "top_n": 20,
        "generate_report": True})
    for expected in ("向量库", "Top 20", "500", "7天", "不过滤增长", "生成报告文件"):
        assert expected in text


# ── 仓库名消歧 ─────────────────────────────────────────────────────

class _FakeGH:
    usable = True

    def __init__(self, known=(), found=()):
        self.known = set(known)
        self.found = list(found)

    def info(self, name):
        return {"full_name": name} if name in self.known else None

    def search(self, query, limit=5):
        return self.found[:limit]


def _repo(full_name):
    return {"full_name": full_name, "stargazers_count": 10, "description": ""}


# ── add_favorite:和网页 ☆ 走同一条写路径 ────────────────────────

def test_agent_re_add_does_not_overwrite_a_hand_written_desc(tmp_path, monkeypatch):
    """收藏的不变式(概要只给新收藏生成、不覆盖用户手写的那句)只在 service.favorites
    一份 —— agent 的 add_favorite 曾绕过它直写库,重复收藏会把手写概要冲掉。"""
    from hot_project import config
    from hot_project.infra.data_access import favorites as store
    from hot_project.infra.data_access import universe

    monkeypatch.setattr(config, "FAVORITES_PATH", tmp_path / "favorites.json")
    monkeypatch.setattr(universe, "load", lambda: {"a/b": {"gh_desc": "x"}})
    monkeypatch.setattr(favorite_service, "_auto_short", lambda repo: "自动生成的概要")
    monkeypatch.setattr(favorite_service.reports, "appearance_counts", lambda: ({}, 0))

    favorite_service.update("user123", "a/b", "add", short_desc="用户手写的一句")

    got = repo_tools.add_favorite(
        Ctx(gh=_FakeGH(known={"a/b"}), user_id="user123"), {"repo": "a/b"})

    assert got["ok"] and got["short_desc"] == "用户手写的一句"
    assert store.get("user123")[0]["short_desc"] == "用户手写的一句"


def test_an_exact_name_that_exists_is_used_as_is():
    ctx = Ctx(gh=_FakeGH(known={"a/b"}))
    assert repo_tools.resolve(ctx, "a/b") == ("a/b", None)


def test_a_bare_project_name_with_one_hit_needs_no_asking():
    ctx = Ctx(gh=_FakeGH(found=[_repo("langchain-ai/langchain")]))
    assert repo_tools.resolve(ctx, "langchain")[0] == "langchain-ai/langchain"


def test_a_wrong_owner_still_finds_the_project():
    """用户常记错 owner。拿仓库名再搜一遍比直接报错有用。"""
    ctx = Ctx(gh=_FakeGH(found=[_repo("langchain-ai/langchain")]))
    assert repo_tools.resolve(ctx, "wrong/langchain")[0] == "langchain-ai/langchain"


def test_one_exact_name_match_wins_over_the_other_candidates():
    ctx = Ctx(gh=_FakeGH(found=[_repo("someone/whisper-cpp"), _repo("openai/whisper"),
                                _repo("other/whisper-ui")]))
    assert repo_tools.resolve(ctx, "whisper")[0] == "openai/whisper"


def test_genuine_ambiguity_goes_back_to_the_user_rather_than_guessing():
    """猜错一个仓库比多问一句代价大得多。"""
    ctx = Ctx(gh=_FakeGH(found=[_repo("a/agent-x"), _repo("b/agent-y")]))
    name, payload = repo_tools.resolve(ctx, "agent")
    assert name is None and payload["disambiguation"] is True
    assert len(payload["candidates"]) == 2


def test_nothing_found_says_so_instead_of_returning_an_empty_result():
    name, payload = repo_tools.resolve(Ctx(gh=_FakeGH()), "zzzz")
    assert name is None and "没找到" in payload["error"]


def test_an_empty_repo_argument_is_an_error_not_a_search():
    name, payload = repo_tools.resolve(Ctx(gh=_FakeGH()), "  ")
    assert name is None and "缺少" in payload["error"]


def test_without_tokens_the_tool_says_so_instead_of_crashing():
    class _NoTokens(_FakeGH):
        usable = False
    name, payload = repo_tools.resolve(Ctx(gh=_NoTokens()), "a/b")
    assert name is None and "token" in payload["error"]


# ── 收藏卡片上那句概要 ──────────────────────────────────────────────

class _ReadmeGH(_FakeGH):
    def profile(self, name, want=()):
        return {"readme": {"text": "This project streams logs from edge devices."}}


@pytest.fixture
def _echo_condense(monkeypatch):
    """把浓缩换成回声,好看清到底喂了哪一档素材进去。"""
    monkeypatch.setattr(favorite_service.describe, "condense",
                        lambda repos, max_chars=70: [r["description"] for r in repos])


@pytest.mark.parametrize("saved, gh, expected", [
    ({"gh_desc": "官方简介", "desc": "四段介绍"}, None, "官方简介"),
    ({"desc": "四段介绍"}, None, "四段介绍"),
    ({}, _ReadmeGH(), "This project streams logs from edge devices."),
    ({}, None, ""),
])
def test_the_summary_falls_back_until_it_finds_something_to_say(
        saved, gh, expected, _echo_condense):
    """作者没填简介的仓库占了全库六成半,只认 gh_desc 的话收藏卡片就是一片空白。

    退让顺序也是有讲究的:库里那份四段介绍不用联网、还是现成的中文,所以排在 README 前面。
    """
    assert favorite_service.short_desc("a/b", saved, gh) == expected


def test_no_token_means_no_readme_lookup_rather_than_a_crash(_echo_condense):
    class _NoTokens(_ReadmeGH):
        usable = False
    assert favorite_service.short_desc("a/b", {}, _NoTokens()) == ""
