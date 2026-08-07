import os
import pathlib

import requests
import yaml

CONFIG = pathlib.Path(__file__).resolve().parent.parent / "config" / "models.yaml"
URL = "https://openrouter.ai/api/v1/chat/completions"

# ponytail: per-process running total — each `python -m src.daily/weekly` is its
# own process, so a module global is enough; not thread-safe, don't need it to be.
_run_cost = 0.0


def run_cost():
    return _run_cost


def model_for(role):
    return yaml.safe_load(CONFIG.read_text())[role]


def _citation_urls(data):
    """Perplexity/Sonar return source URLs out-of-band, not inline in content.

    OpenRouter exposes them either as a top-level `citations` list of URL strings
    or as OpenAI-style `annotations` (url_citation) on the message. Return [] if none.
    """
    msg = data["choices"][0]["message"]
    urls = data.get("citations")
    if not urls:
        urls = [a.get("url_citation", {}).get("url")
                for a in (msg.get("annotations") or [])]
    return [u for u in (urls or []) if isinstance(u, str)]


def complete(role, prompt, system=None, timeout=1800, with_citations=False):
    """One OpenRouter chat completion. role is a key in models.yaml.

    with_citations=True appends a numbered Sources list so [n] markers from
    research models resolve to real URLs. Leave False for JSON-returning calls.
    """
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        URL,
        timeout=timeout,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": model_for(role), "messages": messages,
              "usage": {"include": True}},  # OpenRouter returns actual cost
    )
    if not r.ok:
        # raise_for_status() alone says only "400 Bad Request"; the reason lives in
        # the body (unknown model id, out of credit, context overflow). A dropped
        # character in a models.yaml slug cost 5 days of silent failures once.
        raise requests.HTTPError(
            f"{r.status_code} from OpenRouter (model={model_for(role)!r}): "
            f"{r.text[:1000]}", response=r)
    data = r.json()
    u = data.get("usage") or {}
    cost = u.get("cost")
    if isinstance(cost, (int, float)):
        global _run_cost
        _run_cost += cost
    print(f"[llm] {model_for(role)} prompt={u.get('prompt_tokens', '?')} "
          f"completion={u.get('completion_tokens', '?')} tokens"
          + (f" cost=${cost:.4f}" if isinstance(cost, (int, float)) else ""))
    content = data["choices"][0]["message"]["content"]
    if with_citations:
        urls = _citation_urls(data)
        if urls:
            content += "\n\nSources:\n" + "\n".join(
                f"[{i + 1}] {u}" for i, u in enumerate(urls))
    return content
