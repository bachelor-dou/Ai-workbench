"""轻量搜索的公开边界:Registry + 真实 GitHub 门面,仅替换 HTTP。"""

import base64
import json

import httpx
import pytest

from hot_project import tools
from hot_project.agent.loop import Agent
from hot_project.provider.github import request
from hot_project.provider.github.client import GitHub
from hot_project.provider.github.tokens import TokenPool
from hot_project.tools.spec import Ctx
from hot_project.tests.test_agent import _LLM, _call


@pytest.fixture
def github(monkeypatch):
    def make(handler):
        monkeypatch.setattr(request, "build_client", lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True))
        return GitHub(TokenPool([f"test-{i}" for i in range(100)]))
    return make


def test_search_defaults_to_relevance_and_stars_is_an_explicit_wire_choice(github):
    requests = []

    def respond(req):
        requests.append(req)
        return httpx.Response(200, json={"total_count": 1, "incomplete_results": False,
                                        "items": [{"full_name": "a/project"}]})

    ctx = Ctx(gh=github(respond))
    registry = tools.registry()
    default = registry.run(ctx, "search_repos", {"query": "remote agent"})
    popular = registry.run(ctx, "search_repos", {
        "query": "remote agent", "sort": "stars", "top_n": 20, "min_star": 100})

    assert default["count"] == popular["count"] == 1
    assert default["truncated"] is default["query_truncated"] is False
    assert not registry.get("search_repos").expensive
    assert dict(requests[0].url.params) == {
        "q": "remote agent in:name,description,readme", "per_page": "5", "page": "1"}
    assert dict(requests[1].url.params) == {
        "q": "remote agent in:name,description,readme stars:>=100",
        "sort": "stars", "order": "desc", "per_page": "20", "page": "1"}
    schema = registry.get("search_repos").schema()["function"]["parameters"]
    assert schema["properties"]["sort"]["enum"] == ["relevance", "stars"]


@pytest.mark.parametrize("total, incomplete, items, status", [
    (37, False, [{"full_name": "a/project"}], "ok"),
    (37, True, [{"full_name": "a/project"}], "partial"),
    (0, False, [], "empty"),
    (0, True, [], "partial"),
    (37, False, [], "partial"),
])
def test_search_retains_totals_and_never_calls_incomplete_results_no_matches(
        github, total, incomplete, items, status):
    gh = github(lambda req: httpx.Response(200, json={
        "total_count": total, "incomplete_results": incomplete, "items": items}))

    out = tools.registry().run(Ctx(gh=gh), "search_repos", {"query": "remote agent"})

    assert out["status"] == status
    assert out["total_count"] == total
    assert out["incomplete_results"] is incomplete
    assert out["count"] == out["returned_count"] == len(items)
    assert len(out["results"]) == len(items)
    assert "error" not in out


@pytest.mark.parametrize("status, body, code, retryable", [
    (422, "Validation Failed", "search_rejected", False),
    (422, "You have triggered an abuse detection mechanism", "search_rejected", False),
    (503, "Service Unavailable", "upstream_unavailable", True),
    (429, "rate limit exceeded", "rate_limited", True),
    (403, "secondary rate limit", "rate_limited", True),
    (401, "Bad credentials", "authentication_failed", False),
])
def test_search_failures_are_actionable_not_empty_matches(github, status, body, code, retryable):
    requests = []

    def respond(req):
        requests.append(req)
        return httpx.Response(status, json={"message": body})

    out = tools.registry().run(Ctx(gh=github(respond)), "search_repos", {"query": "remote"})

    assert out["status"] == "failed"
    assert out["error_code"] == code
    assert out["http_status"] == status
    assert out["retryable"] is retryable
    assert out["total_count"] is out["incomplete_results"] is None
    assert out["count"] == out["returned_count"] == 0
    assert out["results"] == []
    assert out["error"]
    assert "本回合同参数" in out["error"] and "下一回合" in out["error"]
    assert body not in out["error"]  # 上游文本是不可信数据,不用它冒充操作指示。
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [
    None, [], "not an object", {},
    {"items": [], "incomplete_results": False},
    {"items": [], "total_count": 0},
    {"items": {}, "total_count": 0, "incomplete_results": False},
    {"items": [None], "total_count": 1, "incomplete_results": False},
    {"items": [], "total_count": True, "incomplete_results": False},
    {"items": [], "total_count": -1, "incomplete_results": False},
    {"items": [], "total_count": "0", "incomplete_results": False},
    {"items": [], "total_count": 0, "incomplete_results": "false"},
    {"items": [{"full_name": "a/project"}], "total_count": 0, "incomplete_results": False},
])
def test_invalid_search_payloads_are_failures_not_empty_matches(github, payload):
    gh = github(lambda req: httpx.Response(200, content=json.dumps(payload)))

    out = tools.registry().run(Ctx(gh=gh), "search_repos", {"query": "remote"})

    assert out["status"] == "failed"
    assert out["error_code"] == "invalid_response"
    assert out["http_status"] == 200
    assert out["total_count"] is None
    assert out["results"] == []


def test_only_three_distinct_attempts_run_and_failed_or_empty_results_are_reused(github):
    requests = []

    def respond(req):
        requests.append(req)
        query = req.url.params["q"]
        if query.startswith("unavailable"):
            return httpx.Response(503)
        if query.startswith("rejected"):
            return httpx.Response(422)
        return httpx.Response(200, json={"items": [], "total_count": 0,
                                        "incomplete_results": False})

    ctx = Ctx(gh=github(respond))
    registry = tools.registry()
    first = registry.run(ctx, "search_repos", {"query": "unavailable"})
    duplicate = registry.run(ctx, "search_repos", {
        "query": "  unavailable in:name,description,readme  ",
        "sort": "relevance", "top_n": 5, "min_star": 0})
    assert duplicate == first
    assert first["remaining_searches"] == 2
    assert len(requests) == 1
    rejected = registry.run(ctx, "search_repos", {"query": "rejected"})
    assert rejected["remaining_searches"] == 1
    empty = registry.run(ctx, "search_repos", {"query": "empty"})
    assert empty["remaining_searches"] == 0
    blocked = registry.run(ctx, "search_repos", {"query": "fourth"})
    assert blocked["status"] == "budget_exhausted"
    assert blocked["error_code"] == "search_budget_exhausted"
    assert blocked["remaining_searches"] == 0
    assert len(requests) == 3
    for query, stored in (("unavailable", first), ("rejected", rejected), ("empty", empty)):
        assert registry.run(ctx, "search_repos", {"query": query}) == {**stored, "remaining_searches": 0}
    assert len(requests) == 3


@pytest.mark.parametrize("stars, expected", [(0, 0), (None, None), ("unknown", None),
                                            (True, None), (-1, None)])
def test_candidates_keep_canonical_identity_public_sources_and_unknown_counts(github, stars, expected):
    gh = github(lambda req: httpx.Response(200, json={
        "total_count": 2, "incomplete_results": False, "items": [
            {"full_name": "canonical/project", "stargazers_count": stars,
             "html_url": "https://github.com/canonical/project?tab=readme-ov-file", "description": "远程工具",
             "topics": ["mobile", "agent"], "archived": False, "pushed_at": "2026-11-01T00:00:00Z"},
            {"full_name": "other/project", "html_url": "https://evil.example/redirect"}]}))

    out = tools.registry().run(Ctx(gh=gh), "search_repos", {"query": "remote"})

    first, second = out["results"]
    assert first["repo"] == "canonical/project"
    assert first["url"] == "https://github.com/canonical/project?tab=readme-ov-file"
    assert first["star"] == expected
    assert first["description"] == "远程工具"
    assert first["topics"] == ["mobile", "agent"]
    assert first["archived"] is False
    assert first["pushed_at"] == "2026-11-01T00:00:00Z"
    assert second["url"] == "https://github.com/other/project"
    assert second["star"] is second["archived"] is None


@pytest.mark.parametrize("name", [None, 42, "", "missing-owner", "a/../b", "https://github.com/a/b"])
def test_candidates_without_a_valid_repo_identity_are_rejected(github, name):
    gh = github(lambda req: httpx.Response(200, json={"total_count": 1,
        "incomplete_results": False, "items": [{"full_name": name}]}))

    out = tools.registry().run(Ctx(gh=gh), "search_repos", {"query": "remote"})

    assert out["status"] == "failed"
    assert out["error_code"] == "invalid_response"
    assert out["total_count"] is None


@pytest.mark.parametrize("status", [200, 503])
def test_complete_search_json_is_bounded_even_with_large_escaped_data_and_queries(github, status):
    requests = []
    query = '\\"\x00中文' * 2000
    items = [{"full_name": f"owner/project-{i}", "description": '\\"\x00描述' * 2000,
              "topics": ["topic" * 500] * 100, "language": "L" * 2000,
              "pushed_at": "date" * 2000} for i in range(20)]

    def respond(req):
        requests.append(req)
        return httpx.Response(status, json={"total_count": 12345,
            "incomplete_results": True, "items": items})

    registry = tools.registry()
    ctx = Ctx(gh=github(respond))
    out = registry.run(ctx, "search_repos", {"query": query, "top_n": 20})
    encoded = json.dumps(out, ensure_ascii=False)
    assert len(encoded) <= 6000
    assert json.loads(encoded) == out
    assert out["truncated"] is True
    assert out["query_truncated"] is True
    assert out["count"] == out["returned_count"] == len(out["results"])
    if status == 200:
        assert out["total_count"] == 12345
        assert out["incomplete_results"] is True
        assert 1 <= out["returned_count"] < 20
        assert out["results"][0]["repo"] == "owner/project-0"
        assert out["results"][0]["url"] == "https://github.com/owner/project-0"
        assert [r["rank"] for r in out["results"]] == list(range(1, out["count"] + 1))
    else:
        assert out["status"] == "failed"
        assert out["error_code"] == "upstream_unavailable"
    assert registry.run(ctx, "search_repos", {"query": query, "top_n": 20}) == out
    assert len(requests) == 1


@pytest.mark.parametrize("changed", [
    {"query": "desktop"}, {"sort": "stars"}, {"top_n": 6}, {"min_star": 10},
])
def test_changing_any_effective_search_parameter_makes_a_new_request(github, changed):
    requests = []

    def respond(req):
        requests.append(req)
        return httpx.Response(200, json={"total_count": 1, "incomplete_results": False,
            "items": [{"full_name": f"a/attempt-{len(requests)}"}]})

    ctx = Ctx(gh=github(respond))
    registry = tools.registry()
    first = registry.run(ctx, "search_repos", {"query": "remote"})
    second = registry.run(ctx, "search_repos", {"query": "remote", **changed})
    assert first["results"][0]["repo"] == "a/attempt-1"
    assert second["results"][0]["repo"] == "a/attempt-2"
    assert len(requests) == 2


def test_registry_rejects_invalid_arguments_before_search_or_budget_consumption(github):
    requests = []

    def respond(req):
        requests.append(req)
        return httpx.Response(200, json={"total_count": 0, "incomplete_results": False, "items": []})

    ctx = Ctx(gh=github(respond))
    registry = tools.registry()
    for args in ({"sort": "updated"}, {"top_n": 21}, {"top_n": True}, {"min_star": -1},
                 {"order": "desc"}, {"query": ["remote"]}):
        out = registry.run(ctx, "search_repos", {"query": "remote", **args})
        assert out["invalid_arguments"]
    assert requests == []
    for query in ("one", "two", "three"):
        assert registry.run(ctx, "search_repos", {"query": query})["status"] == "empty"
    assert len(requests) == 3


def test_transport_errors_and_non_json_search_bodies_are_structured_failures(github):
    def disconnected(req):
        raise httpx.ConnectError("private diagnostic", request=req)

    out = tools.registry().run(Ctx(gh=github(disconnected)), "search_repos", {"query": "remote"})
    assert out["status"] == "failed"
    assert out["error_code"] == "upstream_unavailable"
    assert out["http_status"] is None
    assert "private diagnostic" not in json.dumps(out)

    out = tools.registry().run(Ctx(gh=github(lambda req: httpx.Response(200, text="<html>"))),
                               "search_repos", {"query": "remote"})
    assert out["status"] == "failed"
    assert out["error_code"] == "invalid_response"
    assert out["http_status"] == 200


def test_batch_422_termination_and_resolution_success_list_contract_are_preserved(github):
    requests = []

    def respond(req):
        requests.append(req)
        if req.url.params["q"] == "project":
            return httpx.Response(200, json={"items": [{"full_name": "a/project"}]})
        return httpx.Response(422)

    gh = github(respond)
    assert gh.search("project") == [{"full_name": "a/project"}]
    assert gh.keyword_sweep(["project"], min_star=100) == {}
    assert len(requests) == 2
    assert all(req.url.params["sort"] == "stars" for req in requests)
    assert all(req.url.params["order"] == "desc" for req in requests)


@pytest.mark.parametrize("status", [422, 503])
def test_fuzzy_repo_resolution_never_treats_rejected_search_as_a_missing_repo(github, status):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(status)

    out = tools.registry().run(Ctx(gh=github(respond)), "repo_profile", {"repo": "project"})

    assert out["status"] == "failed"
    assert "error" in out
    assert "没找到" not in out["error"]
    assert paths == ["/search/repositories"]


def test_agent_search_budget_and_cached_results_reset_on_the_next_chat(github):
    requests = []

    def respond(req):
        requests.append(req)
        return httpx.Response(200, json={"total_count": 1, "incomplete_results": False,
            "items": [{"full_name": f"a/attempt-{len(requests)}"}]})

    first_turn = {"tool_calls": [
        _call("search_repos", json.dumps({"query": query}), cid)["tool_calls"][0]
        for query, cid in (("one", "s1"), ("two", "s2"), ("three", "s3"), ("four", "s4"))]}
    second_turn = {"tool_calls": [
        _call("search_repos", json.dumps({"query": query}), cid)["tool_calls"][0]
        for query, cid in (("four", "s5"), ("one", "s6"))]}
    llm = _LLM(first_turn, {"content": "第一轮结束"}, second_turn, {"content": "第二轮结束"})
    agent = Agent(client=llm, tools=tools.registry(), gh=github(respond))

    agent.chat("找远程工具")
    observations = [json.loads(m["content"]) for m in llm.calls[1] if m["role"] == "tool"]
    assert [o["status"] for o in observations] == ["ok", "ok", "ok", "budget_exhausted"]
    assert len(requests) == 3
    agent.chat("再找一轮")
    observations = [json.loads(m["content"]) for m in llm.calls[-1]
                    if m["role"] == "tool" and m["tool_call_id"] in ("s5", "s6")]
    assert [o["results"][0]["repo"] for o in observations] == ["a/attempt-4", "a/attempt-5"]
    assert len(requests) == 5

    other = Agent(client=_LLM(_call("search_repos", '{"query":"one"}'), {"content": "结束"}),
                  tools=tools.registry(), gh=agent.ctx.gh)
    other.chat("新会话")
    assert len(requests) == 6


def test_agent_passes_search_identity_constraints_and_readme_sources_to_the_final_model_input(github):
    # 脚本只证明真实工具链传递证据,不声称证明推荐质量或模型自主选词能力。
    paths = []
    source = "https://github.com/canonical/project/blob/main/README.md"
    text = "# Project\n" + "intro " * 1000 + "\n## Deployment\nSelf-hosted with a private relay."

    def respond(req):
        paths.append(req.url.path)
        if req.url.path == "/search/repositories":
            return httpx.Response(200, json={"total_count": 42, "incomplete_results": False,
                "items": [{"full_name": "canonical/project", "description": "Remote mobile agent",
                           "html_url": "https://github.com/canonical/project"}]})
        if req.url.path == "/repos/canonical/project":
            return httpx.Response(200, json={"full_name": "canonical/project",
                "html_url": "https://github.com/canonical/project", "topics": ["remote", "mobile"]})
        if req.url.path == "/repos/canonical/project/readme":
            return httpx.Response(200, json={"encoding": "base64", "sha": "a" * 40,
                "path": "README.md", "html_url": source,
                "content": base64.b64encode(text.encode()).decode()})
        if req.url.path in ("/repos/canonical/project/releases", "/repos/canonical/project/commits"):
            return httpx.Response(200, json=[])
        pytest.fail(f"unexpected endpoint: {req.url.path}")

    llm = _LLM(
        _call("search_repos", '{"query":"mobile remote agent"}', "search"),
        _call("repo_profile", '{"repo":"canonical/project"}', "profile"),
        _call("read_repo_readme", json.dumps({"repo": "canonical/project", "sha": "a" * 40,
                                              "keyword": "Deployment"}), "readme"),
        {"content": "证据已到达,以下是脚本回答。"})
    agent = Agent(client=llm, tools=tools.registry(), gh=github(respond))
    user_request = "只看能在手机上远程控制桌面 agent、可自部署且不要求公网开放端口的项目;不要榜单。"

    agent.chat(user_request)

    final_input = llm.calls[-1]
    assert {"role": "user", "content": user_request} in final_input
    observations = {m["tool_call_id"]: json.loads(m["content"])
                    for m in final_input if m["role"] == "tool"}
    candidate = observations["search"]["results"][0]
    profile, readme = observations["profile"], observations["readme"]
    assert candidate["repo"] == profile["repo"] == readme["repo"] == "canonical/project"
    assert candidate["url"] == profile["html_url"] == "https://github.com/canonical/project"
    assert profile["readme_source"] == readme["source"] == source
    assert profile["readme_sha"] == readme["sha"] == "a" * 40
    assert "Self-hosted with a private relay." not in profile["readme_excerpt"]
    assert "Self-hosted with a private relay." in readme["text"]
    assert paths.count("/search/repositories") == 1
    assert paths.count("/repos/canonical/project/readme") == 2
