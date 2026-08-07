import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from datetime import date
from src.youtube import week_start, playlist_name, playlist_date, parse_duration


def test_week_helpers():
    # 2026-07-05 is a Sunday; week runs Sunday..Saturday
    assert week_start(date(2026, 7, 5)) == date(2026, 7, 5)
    assert week_start(date(2026, 7, 8)) == date(2026, 7, 5)   # Wednesday
    assert week_start(date(2026, 7, 11)) == date(2026, 7, 5)  # Saturday
    assert playlist_name(date(2026, 7, 8)) == "05-07-2026"
    assert playlist_date("05-07-2026") == date(2026, 7, 5)
    assert playlist_date("My Mixtape") is None


def test_parse_duration():
    assert parse_duration("PT1H2M30S") == 62.5
    assert parse_duration("PT15M") == 15
    assert parse_duration("PT45S") == 0.75
    assert parse_duration("") == 0
    assert parse_duration(None) == 0


def test_model_for():
    from src.llm import model_for
    assert "/" in model_for("research")
    assert "/" in model_for("curation")
    assert "/" in model_for("deep_research")


def test_complete_surfaces_error_body():
    import os
    from src import llm

    class FakeResp:  # a 400 as OpenRouter actually returns one
        ok, status_code = False, 400
        text = '{"error":{"message":"bad-slug is not a valid model ID","code":400}}'

        def raise_for_status(self):  # what requests would do: status only, no body
            raise llm.requests.HTTPError("400 Client Error: Bad Request for url: x")

    old_post = llm.requests.post
    old_key = os.environ.get("OPENROUTER_API_KEY")
    llm.requests.post = lambda *a, **k: FakeResp()
    os.environ["OPENROUTER_API_KEY"] = "test-key-not-used"
    try:
        llm.complete("curation", "hi")
    except llm.requests.HTTPError as e:
        # the message must name the rejected model AND echo the body — without both,
        # a typo in models.yaml is indistinguishable from any other 400
        assert "not a valid model ID" in str(e)
        assert llm.model_for("curation") in str(e)
    else:
        raise AssertionError("complete() swallowed a 400")
    finally:
        llm.requests.post = old_post
        if old_key is None:
            del os.environ["OPENROUTER_API_KEY"]
        else:
            os.environ["OPENROUTER_API_KEY"] = old_key


def test_citation_urls():
    from src.llm import _citation_urls
    top = {"citations": ["http://a", "http://b"],
           "choices": [{"message": {"content": "x"}}]}
    assert _citation_urls(top) == ["http://a", "http://b"]
    annotated = {"choices": [{"message": {"content": "x", "annotations": [
        {"url_citation": {"url": "http://c"}}, {"type": "other"}]}}]}
    assert _citation_urls(annotated) == ["http://c"]
    none = {"choices": [{"message": {"content": "x"}}]}
    assert _citation_urls(none) == []


def test_recent_entries():
    import time
    from types import SimpleNamespace
    from src.sources import recent_entries
    now = time.gmtime()
    old = time.gmtime(time.time() - 90 * 3600)
    parsed = SimpleNamespace(entries=[
        {"title": "fresh", "link": "http://a", "summary": "x", "published_parsed": now},
        {"title": "stale", "link": "http://b", "summary": "y", "published_parsed": old},
        {"title": "undated", "link": "http://c", "summary": "z"},
    ])
    got = recent_entries(parsed, hours=24)
    assert [e["title"] for e in got] == ["fresh"]


def test_parse_llm_json():
    from src.curate import parse_llm_json
    fenced = 'Here you go:\n```json\n{"a": [1, 2]}\n```'
    assert parse_llm_json(fenced) == {"a": [1, 2]}
    assert parse_llm_json('{"b": 1}') == {"b": 1}
    # LLMs emit literal newlines inside markdown string values (json is strict by default)
    assert parse_llm_json('{"md": "line1\nline2"}') == {"md": "line1\nline2"}


def test_seen_video_ids(tmp_dir="tests/_tmp_newsletters"):
    import shutil
    from src import curate
    p = pathlib.Path(tmp_dir)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    (p / "2026-07-04.md").write_text(
        "watch [this](https://www.youtube.com/watch?v=abcdefghijk) "
        "and [that](https://youtu.be/AAAAAAAAAAA)", encoding="utf-8")
    old_ledger = curate.SEEN_LEDGER  # isolate from the real docs/ ledger
    curate.SEEN_LEDGER = p / "seen_videos.txt"
    try:
        assert curate.seen_video_ids(p) == {"abcdefghijk", "AAAAAAAAAAA"}
    finally:
        curate.SEEN_LEDGER = old_ledger
    shutil.rmtree(p)


def test_strip_leading_title():
    from src.site import _strip_leading_title
    # H1 + subtitle + hr before the first section is dropped; internal --- kept
    body = ("# The Daily Brief\n**a subtitle**\n\n---\n\n"
            "## Breaking News\n\nx\n\n---\n\n## Headlines\n\ny\n")
    out = _strip_leading_title(body)
    assert out.startswith("## Breaking News")
    assert out.count("---") == 1  # the section separator survives
    # a body already starting at '## ' is untouched
    clean = "## Breaking News\n\nx\n"
    assert _strip_leading_title(clean) == clean


def test_parse_issue(tmp_dir="tests/_tmp_issue"):
    import shutil
    from src.site import parse_issue
    p = pathlib.Path(tmp_dir)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    f = p / "2026-07-06-weekly.md"
    f.write_text("---\ntitle: Weekly — 2026-07-06\nsummary: A test week.\n"
                 "tags: [dbt, local-llm]\n---\n\n## Breaking News\n\nBig.\n\n"
                 "## Headlines\n\nStuff.\n", encoding="utf-8")
    it = parse_issue(f)
    assert it["kind"] == "weekly" and it["date"] == "2026-07-06"
    assert it["tags"] == ["dbt", "local-llm"] and it["breaking"]
    assert it["headings"] == ["Breaking News", "Headlines"]
    assert it["summary"] == "A test week."
    # missing front matter fields degrade gracefully
    g = p / "2026-07-05.md"
    g.write_text("---\ntitle: Daily\n---\n\nFirst paragraph here.\n", encoding="utf-8")
    it2 = parse_issue(g)
    assert it2["kind"] == "daily" and it2["tags"] == []
    assert it2["summary"] == "First paragraph here."
    shutil.rmtree(p)


def test_site_build(tmp_dir="tests/_tmp_site"):
    import shutil
    from src.site import build
    root = pathlib.Path(tmp_dir)
    shutil.rmtree(root, ignore_errors=True)
    (root / "newsletters").mkdir(parents=True)
    for day in ("2026-07-04", "2026-07-05"):
        (root / "newsletters" / f"{day}.md").write_text(
            f"---\ntitle: Daily — {day}\nsummary: s\ntags: [dbt]\n---\n\n"
            "## Headlines\n\n[link](https://example.com)\n", encoding="utf-8")
    n = build(root)
    assert n == 2
    idx = (root / "index.html").read_text(encoding="utf-8")
    assert idx.index("2026-07-05") < idx.index("2026-07-04")  # newest first
    page = (root / "newsletters" / "2026-07-05.html").read_text(encoding="utf-8")
    assert '<a href="https://example.com">link</a>' in page  # markdown rendered
    assert 'id="headlines"' in page                          # toc anchor exists
    assert "2026-07-04.html" in page                         # prev/next nav
    assert (root / "search.json").exists() and (root / "feed.xml").exists()
    assert (root / ".nojekyll").exists()
    shutil.rmtree(root)


def test_collapsible_watchlist():
    from src.site import _collapsible_watchlists
    html_in = ('<h2 id="todays-watchlist">Today’s Watchlist</h2>'
               "<ul><li>vid</li></ul>"
               '<h2 id="repo-watchlist">Repo Watchlist</h2><p>repos</p>')
    out = _collapsible_watchlists(html_in)
    assert '<details class="watchlist" id="todays-watchlist" open>' in out
    assert "<summary>Today’s Watchlist</summary>" in out
    assert "<ul><li>vid</li></ul></details>" in out
    assert '<h2 id="repo-watchlist">' in out  # repo section untouched


def test_style_citations():
    from src.site import _style_citations
    # consecutive [1][12] cites must render as separate bracketed superscripts
    out = _style_citations('layers.<a href="http://a/x">1</a><a href="http://b/y">12</a>')
    assert out == ('layers.<sup class="cite"><a href="http://a/x">1</a></sup>'
                   '<sup class="cite"><a href="http://b/y">12</a></sup>')
    # links with non-numeric text (e.g. video titles) are left alone
    assert _style_citations('<a href="http://v">Watch this</a>') == '<a href="http://v">Watch this</a>'


def test_fix_tight_lists():
    import markdown
    from src.site import _fix_tight_lists
    tight = "**How to make the most of it:**\n1. First\n2. Second"
    fixed = _fix_tight_lists(tight)
    assert "<ol>" in markdown.markdown(fixed)          # now renders as a real list
    # already-spaced lists are unchanged (no spurious blank lines)
    ok = "Intro\n\n- a\n- b"
    assert _fix_tight_lists(ok) == ok
    # list-like lines inside a code fence are left alone
    fenced = "```\n1. not a list\n```"
    assert _fix_tight_lists(fenced) == fenced


def test_seen_repos(tmp_dir="tests/_tmp_repos"):
    import shutil
    from src.curate import seen_repos
    p = pathlib.Path(tmp_dir)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    (p / "2026-07-04.md").write_text(
        "[a](https://github.com/Foo/Bar) and [b](https://github.com/baz/qux.git)",
        encoding="utf-8")
    assert seen_repos(p) == {"foo/bar", "baz/qux"}
    shutil.rmtree(p)


def test_video_ledger(tmp_dir="tests/_tmp_ledger"):
    import shutil
    from src import curate
    p = pathlib.Path(tmp_dir)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    old_ledger = curate.SEEN_LEDGER
    curate.SEEN_LEDGER = p / "seen_videos.txt"
    try:
        curate.remember_videos(["AAAAAAAAAAA", "BBBBBBBBBBB"])
        curate.remember_videos(["BBBBBBBBBBB", "CCCCCCCCCCC"])  # dedupes
        assert curate.SEEN_LEDGER.read_text(encoding="utf-8").split() == [
            "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"]
        assert curate.seen_video_ids(p) == {"AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"}
    finally:
        curate.SEEN_LEDGER = old_ledger
    shutil.rmtree(p)


def test_week_watchlist():
    from src.weekly import week_watchlist
    d1 = ("## Headlines\n\nx\n\n## Today's Watchlist\n\n"
          "- [A](https://www.youtube.com/watch?v=AAAAAAAAAAA) – why a\n"
          "- [B](https://www.youtube.com/watch?v=BBBBBBBBBBB) – why b\n\n"
          "## Repo Watchlist\n\n- stuff\n")
    d2 = ("## Today’s Watchlist\n\n"  # curly apostrophe, as LLMs often emit
          "- [B again](https://www.youtube.com/watch?v=BBBBBBBBBBB) – dupe\n"
          "- [C](https://youtu.be/CCCCCCCCCCC) – why c\n")
    items = week_watchlist([d1, d2])
    assert len(items) == 3  # B deduped
    assert "AAAAAAAAAAA" in items[0] and "CCCCCCCCCCC" in items[2]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
