"""仓库取证的公开边界:真实 Registry/GitHub,仅把 HTTP 换成 MockTransport。"""

import base64
import json

import httpx
import pytest

from hot_project import tools
from hot_project.agent.history import Session
from hot_project.provider.github import request
from hot_project.provider.github.client import GitHub
from hot_project.provider.github.tokens import TokenPool
from hot_project.tools.spec import Ctx


@pytest.fixture
def github(monkeypatch):
    def make(handler):
        monkeypatch.setattr(request, "build_client", lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True))
        # 足够的独立租约,不等待真实配速时间。
        return GitHub(TokenPool([f"test-{i}" for i in range(100)]))
    return make


def test_a_temporary_metadata_failure_never_searches_for_another_repo(github):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if req.url.path == "/repos/a/project":
            return httpx.Response(503, json={"message": "unavailable"})
        return httpx.Response(200, json={"items": [{"full_name": "other/project"}]})

    out = tools.registry().run(Ctx(gh=github(respond), state=Session()),
                               "repo_profile", {"repo": "a/project"})

    assert out["status"] == "failed"
    assert "error" in out
    assert set(paths) == {"/repos/a/project"}


def test_profiles_keep_partial_success_empty_missing_and_failed_distinct(github):
    def respond(req):
        path = req.url.path
        if path == "/repos/a/good":
            return httpx.Response(200, json={"full_name": "a/good", "stargazers_count": 0})
        if path == "/repos/a/good/releases":
            return httpx.Response(200, json=[])
        if path == "/repos/a/good/readme":
            return httpx.Response(404)
        return httpx.Response(503, json={"message": "unavailable"})

    packs = github(respond).profiles(["a/good", "a/broken"])

    assert packs["a/good"]["info"]["stargazers_count"] == 0
    assert packs["a/good"]["releases"] == []
    assert packs["a/good"]["status"] == {
        "info": {"status": "ok"}, "releases": {"status": "empty"},
        "readme": {"status": "not_found"},
        "commits": {"status": "failed"},
    }
    assert packs["a/broken"]["status"]["info"]["status"] == "failed"
    assert packs["a/broken"]["status"]["releases"]["status"] == "failed"
    assert "releases" not in packs["a/broken"]


@pytest.mark.parametrize("metadata, star, archived", [
    ({"stargazers_count": 0, "archived": False}, 0, False),
    ({}, None, None),
    ({"stargazers_count": "unknown", "archived": "unknown"}, None, None),
])
def test_profile_reuses_resolved_metadata_and_never_fills_unknown_values(
        github, metadata, star, archived):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project", **metadata})
        return httpx.Response(404)

    out = tools.registry().run(Ctx(gh=github(respond), state=Session()),
                               "repo_profile", {"repo": "a/project"})

    assert out["star"] is star
    assert out["archived"] is archived
    assert out["status"]["info"]["status"] == "ok"
    assert paths.count("/repos/a/project") == 1


def _readme(text, sha="a" * 40):
    return {"content": base64.b64encode(text.encode()).decode(), "encoding": "base64",
            "sha": sha, "path": "README.md",
            "html_url": "https://github.com/a/project/blob/main/README.md"}


def test_readme_after_the_old_cutoff_can_be_located_and_continued_without_gaps(github):
    text = "# Project\n" + "intro " * 800 + "\n## Installation\n" + "安装步骤 abc " * 450
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme(text))
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    overview = registry.run(ctx, "repo_profile", {"repo": "a/project"})
    assert overview["readme_sha"] == "a" * 40
    assert overview["readme_source"].endswith("/blob/main/README.md")
    assert overview["readme_excerpt"] == text[:overview["readme_next_offset"]]
    assert {"heading": "Installation", "offset": text.index("## Installation")} in overview["readme_outline"]

    first = registry.run(ctx, "read_repo_readme", {
        "repo": "a/project", "sha": overview["readme_sha"], "keyword": "Installation"})
    second = registry.run(ctx, "read_repo_readme", {
        "repo": "a/project", "sha": overview["readme_sha"], "offset": first["next_offset"]})

    assert first["status"] == second["status"] == "ok"
    assert first["offset"] > 4000
    assert first["text"] + second["text"] == text[first["offset"]:second["next_offset"] or len(text)]
    assert len(json.dumps(ctx.state.tool_state, ensure_ascii=False)) < 1000
    assert all(p in {"/repos/a/project", "/repos/a/project/readme",
                     "/repos/a/project/commits", "/repos/a/project/releases"} for p in paths)


def test_readme_version_changes_require_new_evidence_without_returning_text(github):
    version = ["a" * 40]

    def respond(req):
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme("same page", version[0]))
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    registry.run(ctx, "repo_profile", {"repo": "a/project"})
    version[0] = "b" * 40
    changed = registry.run(ctx, "read_repo_readme", {"repo": "a/project", "sha": "a" * 40})
    assert changed["status"] == "version_changed"
    assert "text" not in changed
    assert "重新" in changed["error"]
    again = registry.run(ctx, "read_repo_readme", {"repo": "a/project", "sha": "b" * 40})
    assert again["status"] == "not_confirmed"
    fresh = registry.run(ctx, "repo_profile", {"repo": "a/project"})
    assert fresh["readme_sha"] == "b" * 40


def test_readme_budget_is_shared_across_repos_and_reset_only_by_a_new_turn(github):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if req.url.path in {"/repos/a/first", "/repos/a/second"}:
            return httpx.Response(200, json={"full_name": req.url.path.removeprefix("/repos/")})
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme("说明"))
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    for name in ["a/first", "a/second"]:
        registry.run(ctx, "repo_profile", {"repo": name})
    for i in range(4):
        out = registry.run(ctx, "read_repo_readme", {
            "repo": "a/first" if i % 2 else "a/second", "sha": "a" * 40})
        assert out["remaining_reads"] == 3 - i
    before = len(paths)
    exhausted = registry.run(ctx, "read_repo_readme", {"repo": "a/first", "sha": "a" * 40})
    assert exhausted["status"] == "budget_exhausted"
    assert len(paths) == before
    ctx.turn_state.clear()
    assert registry.run(ctx, "read_repo_readme", {
        "repo": "a/first", "sha": "a" * 40})["status"] == "ok"


def test_complete_json_stays_bounded_even_after_quotes_and_control_characters_escape(github):
    text = '# Overview\n' + '"\\\x00中文' * 2500 + '\n## Usage\n'

    def respond(req):
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project",
                "description": '"\\\x00' * 3000, "topics": ["huge" * 500] * 30})
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme(text))
        if req.url.path.endswith("/releases"):
            return httpx.Response(200, json=[{"name": "release" * 2000}] * 5)
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    overview = registry.run(ctx, "repo_profile", {"repo": "a/project"})
    page = registry.run(ctx, "read_repo_readme", {
        "repo": "a/project", "sha": overview["readme_sha"],
        "offset": overview["readme_next_offset"]})

    for result in (overview, page):
        assert len(json.dumps(result, ensure_ascii=False)) <= 6000
    assert overview["status"]["releases"]["status"] == "ok"
    assert overview["sources"]["readme"] == _readme("")["html_url"]
    assert page["source"] == overview["readme_source"]
    assert overview["readme_excerpt"] + page["text"] == text[:page["next_offset"]]


@pytest.mark.parametrize("response, expected", [(503, "failed"), (404, "not_found")])
def test_failed_readme_rereads_are_not_reported_as_a_version_change(github, response, expected):
    status = [200]

    def respond(req):
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        if req.url.path.endswith("/readme"):
            return httpx.Response(status[0], json=_readme("hello"))
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    registry.run(ctx, "repo_profile", {"repo": "a/project"})
    status[0] = response
    out = registry.run(ctx, "read_repo_readme", {"repo": "a/project", "sha": "a" * 40})
    assert out["status"] == expected
    assert out["remaining_reads"] == 3
    assert "text" not in out
    if response == 503:
        status[0] = 200
        assert registry.run(ctx, "read_repo_readme", {
            "repo": "a/project", "sha": "a" * 40})["status"] == "ok"


def test_readme_confirmations_are_bounded_and_failed_refresh_removes_old_identity(github):
    fail = [False]

    def respond(req):
        if req.url.path.endswith("/readme"):
            return httpx.Response(503 if fail[0] else 200, json=_readme("small"))
        if req.url.path.endswith(("/commits", "/releases")):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"full_name": req.url.path.removeprefix("/repos/")})

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    for i in range(13):
        registry.run(ctx, "repo_profile", {"repo": f"a/project-{i}"})
    old = registry.run(ctx, "read_repo_readme", {"repo": "a/project-0", "sha": "a" * 40})
    assert old["status"] == "not_confirmed"
    latest = registry.run(ctx, "read_repo_readme", {"repo": "a/project-12", "sha": "a" * 40})
    assert latest["status"] == "ok"
    fail[0] = True
    overview = registry.run(ctx, "repo_profile", {"repo": "a/project-12"})
    assert not overview["readme_available"]
    assert registry.run(ctx, "read_repo_readme", {
        "repo": "a/project-12", "sha": "a" * 40})["status"] == "not_confirmed"


def test_readme_unicode_offsets_and_out_of_range_are_explicit(github):
    text = "İ前言\n## INSTALL\n步骤"

    def respond(req):
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme(text))
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    registry.run(ctx, "repo_profile", {"repo": "a/project"})
    page = registry.run(ctx, "read_repo_readme", {
        "repo": "a/project", "sha": "a" * 40, "keyword": "install"})
    assert page["offset"] == 7
    assert page["text"] == "INSTALL\n步骤"
    assert page["next_offset"] is None
    invalid = registry.run(ctx, "read_repo_readme", {
        "repo": "a/project", "sha": "a" * 40, "offset": 100})
    assert invalid["status"] == "invalid_offset"
    assert "text" not in invalid


@pytest.mark.parametrize("response", [httpx.Response(200, content="null"),
                                     httpx.Response(200, content="not json"),
                                     httpx.Response(422, json={"message": "invalid"})])
def test_invalid_metadata_is_failed_not_missing_and_does_not_trigger_search(github, response):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        return response

    out = tools.registry().run(Ctx(gh=github(respond), state=Session()),
                               "repo_profile", {"repo": "a/project"})
    assert out["status"] == "failed"
    assert set(paths) == {"/repos/a/project"}


def test_disambiguation_and_invalid_repo_names_do_not_break_the_result_budget(github):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(200, json={"items": [
            {"full_name": f"a/project-{i}", "description": '"\\' * 5000} for i in range(5)]})

    ctx = Ctx(gh=github(respond), state=Session())
    registry = tools.registry()
    out = registry.run(ctx, "repo_profile", {"repo": "project"})
    assert out["disambiguation"]
    assert len(json.dumps(out, ensure_ascii=False)) <= 6000
    before = len(paths)
    invalid = registry.run(ctx, "repo_profile", {"repo": "a/project/../../other"})
    assert "error" in invalid
    assert len(paths) == before


@pytest.mark.parametrize("source", ["https://evil.example/README", "https://github.com/a/\x00bad"])
def test_readme_sources_never_use_external_or_malformed_links(github, source):
    def respond(req):
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json={**_readme("hello"), "html_url": source})
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        return httpx.Response(200, json=[])

    ctx = Ctx(gh=github(respond), state=Session())
    out = tools.registry().run(ctx, "repo_profile", {"repo": "a/project"})
    assert out["readme_source"] == "https://github.com/a/project"


def test_stateless_overview_and_unconfirmed_readme_paths_are_predictable(github):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme("hello"))
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        return httpx.Response(200, json=[])

    gh = github(respond)
    registry = tools.registry()
    stateless = Ctx(gh=gh)
    overview = registry.run(stateless, "repo_profile", {"repo": "a/project"})
    assert not overview["readme_available"]
    assert registry.run(stateless, "read_repo_readme", {
        "repo": "a/project", "sha": "a" * 40})["status"] == "state_required"
    ctx = Ctx(gh=gh, state=Session())
    registry.run(ctx, "repo_profile", {"repo": "a/project"})
    before = len(paths)
    for name in ("https://github.com/a/project", "a/project/docs", "a/other"):
        out = registry.run(ctx, "read_repo_readme", {"repo": name, "sha": "a" * 40})
        assert out["status"] == "not_confirmed"
    assert len(paths) == before
    assert not registry.get("read_repo_readme").expensive


@pytest.mark.parametrize("mode", ["redirect", "wrong_owner", "ambiguous"])
def test_http_resolution_preserves_renames_corrections_and_ambiguity(github, mode):
    paths = []

    def respond(req):
        path = req.url.path
        paths.append(path)
        if path == "/repos/old/project":
            if mode == "redirect":
                return httpx.Response(301, headers={"Location": "https://api.github.com/repositories/42"})
            return httpx.Response(404)
        if path == "/repositories/42":
            return httpx.Response(200, json={"full_name": "new/project"})
        if path == "/search/repositories":
            names = ["new/project"] if mode == "wrong_owner" else ["a/project", "b/project"]
            return httpx.Response(200, json={"items": [{"full_name": name} for name in names]})
        return httpx.Response(404)

    out = tools.registry().run(Ctx(gh=github(respond), state=Session()),
                               "repo_profile", {"repo": "old/project"})
    if mode == "ambiguous":
        assert out["disambiguation"]
        assert len(out["candidates"]) == 2
    else:
        assert out["repo"] == "new/project"
        assert "/repos/new/project/readme" in paths
        assert "/repos/new/project" not in paths


@pytest.mark.parametrize("payload, expected", [(_readme(""), "empty"),
    ({**_readme(""), "content": "???"}, "failed"),
    ({**_readme(""), "encoding": "none", "size": 2000000}, "failed")])
def test_empty_readmes_are_not_confused_with_decode_failures(github, payload, expected):
    gh = github(lambda req: httpx.Response(200, json=payload))
    pack = gh.profile("a/project", want=("readme",))
    assert pack["status"]["readme"]["status"] == expected


def test_real_agent_passes_readme_sources_to_the_model_and_resets_budget_next_chat(github):
    from hot_project.agent.loop import Agent

    text = "preface " * 650 + "\n## Deployment\nRun the local server."

    def respond(req):
        if req.url.path.endswith("/readme"):
            return httpx.Response(200, json=_readme(text))
        if req.url.path == "/repos/a/project":
            return httpx.Response(200, json={"full_name": "a/project"})
        return httpx.Response(200, json=[])

    def call(name, args, cid):
        return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}

    class ScriptedLLM:
        def __init__(self):
            self.inputs = []
            self.sha = None

        def chat(self, messages, **kwargs):
            self.inputs.append(messages)
            step = len(self.inputs)
            if step == 1:
                reply = {"tool_calls": [call("repo_profile", {"repo": "a/project"}, "profile")]}
            elif step == 2:
                overview = json.loads(messages[-1]["content"])
                self.sha = overview["readme_sha"]
                assert "Run the local server" not in overview["readme_excerpt"]
                reply = {"tool_calls": [call("read_repo_readme", {
                    "repo": "a/project", "sha": self.sha, "keyword": "Deployment"}, f"read-{i}")
                    for i in range(5)]}
            elif step == 4:
                reply = {"tool_calls": [call("read_repo_readme", {
                    "repo": "a/project", "sha": self.sha, "keyword": "Deployment"}, "next-turn")]}
            else:
                reply = {"content": "来源 https://github.com/a/project/blob/main/README.md"}
            return {"choices": [{"message": reply}]}

    llm = ScriptedLLM()
    agent = Agent(client=llm, gh=github(respond))
    answer = agent.chat("了解此项目的部署方式")
    assert "https://github.com/a/project/blob/main/README.md" in answer
    final_input = llm.inputs[2]
    evidence = [json.loads(m["content"]) for m in final_input if m["role"] == "tool"]
    assert [e["status"] for e in evidence[1:]] == ["ok"] * 4 + ["budget_exhausted"]
    assert evidence[1]["source"] == evidence[0]["readme_source"]
    assert evidence[1]["text"] == "Deployment\nRun the local server."
    assert evidence[1]["offset"] > 4000
    assert all(len(m["content"]) <= 6000 for m in final_input if m["role"] == "tool")

    agent.chat("再确认部署要求")
    new_evidence = json.loads(llm.inputs[-1][-1]["content"])
    assert new_evidence["status"] == "ok"
    assert new_evidence["remaining_reads"] == 3
