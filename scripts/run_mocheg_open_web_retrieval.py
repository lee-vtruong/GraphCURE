"""Build an immutable, resumable Phase-C web-evidence snapshot."""
from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from tqdm import tqdm

from graphcure.open_web import (
    QUERY_POLICY_VERSION,
    assert_public_url,
    fuse_results,
    html_to_text,
    load_jsonl,
    query_plan,
    sha256_file,
    sha256_text,
)


USER_AGENT = "GraphCURE-Research/0.1 (evidence snapshot; contact repository owner)"


class SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def http_json(request: Request, timeout: float) -> dict:
    context = ssl.create_default_context()
    opener = build_opener(HTTPSHandler(context=context), SafeRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("search response exceeded 10 MiB")
        return json.loads(payload.decode("utf-8"))


def normalize_search(provider: str, payload: dict) -> list[dict]:
    if provider == "brave":
        source = payload.get("web", {}).get("results", [])
        return [{
            "url": row.get("url", ""), "title": row.get("title", ""),
            "snippet": row.get("description", ""),
            "published_at": row.get("page_age") or row.get("age"),
        } for row in source]
    if provider == "serper":
        return [{
            "url": row.get("link", ""), "title": row.get("title", ""),
            "snippet": row.get("snippet", ""), "published_at": row.get("date"),
        } for row in payload.get("organic", [])]
    raise ValueError(f"unsupported provider: {provider}")


def live_search(provider: str, query: str, count: int, api_key: str,
                timeout: float) -> tuple[dict, list[dict]]:
    if provider == "brave":
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({
            "q": query, "count": min(count, 20), "safesearch": "moderate",
        })
        request = Request(url, headers={
            "Accept": "application/json", "X-Subscription-Token": api_key,
            "User-Agent": USER_AGENT,
        })
    elif provider == "serper":
        body = json.dumps({"q": query, "num": min(count, 100)}).encode("utf-8")
        request = Request("https://google.serper.dev/search", data=body, headers={
            "Content-Type": "application/json", "X-API-KEY": api_key,
            "User-Agent": USER_AGENT,
        }, method="POST")
    else:
        raise ValueError(f"unsupported provider: {provider}")
    payload = http_json(request, timeout)
    return payload, normalize_search(provider, payload)[:count]


def fixture_search(fixture: dict, query: str, count: int) -> tuple[dict, list[dict]]:
    rows = fixture.get(query, [])
    if not isinstance(rows, list):
        raise ValueError(f"fixture value for {query!r} must be a list")
    payload = {"fixture_query": query, "results": rows[:count]}
    return payload, rows[:count]


def cached_search(cache_root: Path, provider: str, query: str, count: int,
                  api_key: str, timeout: float, fixture: dict | None) -> list[dict]:
    key = sha256_text(json.dumps({
        "provider": provider, "query": query, "count": count,
    }, sort_keys=True))
    path = cache_root / "queries" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["normalized_results"]
    started = time.perf_counter()
    if provider == "fixture":
        payload, rows = fixture_search(fixture or {}, query, count)
    else:
        payload, rows = live_search(provider, query, count, api_key, timeout)
    record = {
        "provider": provider, "query": query, "count": count,
        "retrieved_at": utc_now(), "elapsed_seconds": time.perf_counter() - started,
        "provider_payload": payload, "normalized_results": rows,
    }
    atomic_json(path, record)
    return rows


def fetch_page(url: str, timeout: float, max_bytes: int,
               max_text_chars: int) -> dict:
    canonical = assert_public_url(url)
    request = Request(canonical, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    })
    started = time.perf_counter()
    context = ssl.create_default_context()
    opener = build_opener(HTTPSHandler(context=context), SafeRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        final_url = assert_public_url(response.geturl())
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise ValueError(f"unsupported content type: {content_type}")
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"page exceeded {max_bytes} bytes")
        charset = response.headers.get_content_charset() or "utf-8"
        decoded = payload.decode(charset, errors="replace")
        text = decoded if content_type == "text/plain" else html_to_text(decoded)
        return {
            "fetch_status": "ok", "final_url": final_url,
            "fetched_at": utc_now(), "elapsed_seconds": time.perf_counter() - started,
            "content_type": content_type, "bytes": len(payload),
            "raw_sha256": sha256_text(decoded), "text": text[:max_text_chars],
            "text_sha256": sha256_text(text),
        }


def cached_page(cache_root: Path, row: dict, timeout: float, max_bytes: int,
                max_text_chars: int) -> dict:
    key = sha256_text(row["canonical_url"])
    path = cache_root / "pages" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        record = fetch_page(row["canonical_url"], timeout, max_bytes, max_text_chars)
    except (ValueError, HTTPError, URLError, TimeoutError, OSError) as error:
        record = {
            "fetch_status": "error", "fetched_at": utc_now(),
            "error_type": type(error).__name__, "error": str(error)[:500],
            "text": row.get("snippet", ""),
        }
    atomic_json(path, record)
    return record


def validate_test_unlock(path: Path | None) -> None:
    if path is None:
        raise ValueError("test requires --phase-c-freeze-manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("status") != "frozen_after_validation" or
            payload.get("protocol") != "P2_open_web"):
        raise ValueError("test requires a frozen P2_open_web manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("data/processed/mocheg_manifest_strict"))
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New immutable snapshot directory")
    parser.add_argument("--provider", choices=("brave", "serper", "fixture"),
                        default="brave")
    parser.add_argument("--api-key-env", default="BRAVE_SEARCH_API_KEY")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--results-per-query", type=int, default=10)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument("--query-budget", type=int, default=0,
                        help="Maximum constraint queries per claim; 0 uses all")
    parser.add_argument("--fetch-pages", action="store_true")
    parser.add_argument("--fetch-top-k", type=int, default=0,
                        help="Fetch only the first k results; 0 fetches output-k")
    parser.add_argument("--fetch-workers", type=int, default=4)
    parser.add_argument("--max-page-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-text-chars", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--phase-c-freeze-manifest", type=Path)
    args = parser.parse_args()
    if args.results_per_query <= 0 or args.output_k <= 0:
        parser.error("retrieval cutoffs must be positive")
    if args.query_budget < 0 or args.fetch_top_k < 0 or args.fetch_workers <= 0:
        parser.error("query/fetch limits must be non-negative and workers positive")
    if "test" in args.splits:
        try:
            validate_test_unlock(args.phase_c_freeze_manifest)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            parser.error(str(error))
    fixture = None
    if args.provider == "fixture":
        if args.fixture is None:
            parser.error("fixture provider requires --fixture")
        fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        api_key = ""
    else:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(f"missing API key environment variable {args.api_key_env}")
    manifest_hashes = {}
    for split in args.splits:
        manifest_path = args.manifest_root / f"{split}.jsonl"
        if not manifest_path.is_file():
            parser.error(f"missing manifest: {manifest_path}")
        manifest_hashes[split] = sha256_file(manifest_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    signature_payload = {
        "protocol": "P2_open_web", "query_policy": QUERY_POLICY_VERSION,
        "provider": args.provider, "results_per_query": args.results_per_query,
        "output_k": args.output_k, "fetch_pages": args.fetch_pages,
        "query_budget": args.query_budget,
        "fetch_top_k": args.fetch_top_k,
        "fetch_workers": args.fetch_workers,
        "max_page_bytes": args.max_page_bytes,
        "max_text_chars": args.max_text_chars,
        "manifest_hashes": manifest_hashes,
    }
    signature = sha256_text(json.dumps(signature_payload, sort_keys=True))
    snapshot_manifest = args.output_root / "snapshot_manifest.json"
    if snapshot_manifest.exists():
        existing = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            parser.error("snapshot settings changed; choose a new --output-root")
    else:
        atomic_json(snapshot_manifest, {
            **signature_payload, "signature": signature, "created_at": utc_now(),
            "api_key_stored": False, "test_unlocked": "test" in args.splits,
        })
    summary = {}
    for split in args.splits:
        manifest_path = args.manifest_root / f"{split}.jsonl"
        claims = load_jsonl(manifest_path)
        selected = claims[args.offset:]
        if args.limit:
            selected = selected[:args.limit]
        target = args.output_root / f"{split}.jsonl"
        completed_rows = load_jsonl(target)
        completed = {row["id"] for row in completed_rows}
        if len(completed) != len(completed_rows):
            parser.error(f"duplicate completed IDs in {target}")
        pending = [row for row in selected if row["id"] not in completed]
        with target.open("a", encoding="utf-8", buffering=1) as handle:
            for claim in tqdm(pending, desc=f"{split} open-web snapshot"):
                started = time.perf_counter()
                plan = query_plan(claim.get("claim", ""))
                if args.query_budget:
                    plan = plan[:args.query_budget]
                query_results = []
                for item in plan:
                    rows = cached_search(
                        args.output_root, args.provider, item["query"],
                        args.results_per_query, api_key, args.timeout, fixture,
                    )
                    query_results.append((item["constraint"], rows))
                    if args.delay and args.provider != "fixture":
                        time.sleep(args.delay)
                evidence = fuse_results(query_results)[:args.output_k]
                fetch_count = (
                    min(args.fetch_top_k or len(evidence), len(evidence))
                    if args.fetch_pages else 0
                )
                if fetch_count:
                    def retrieve_page(row):
                        return cached_page(
                            args.output_root, row, args.timeout,
                            args.max_page_bytes, args.max_text_chars,
                        )

                    with ThreadPoolExecutor(
                        max_workers=min(args.fetch_workers, fetch_count)
                    ) as executor:
                        page_records = list(executor.map(
                            retrieve_page, evidence[:fetch_count]
                        ))
                    for row, page_record in zip(
                        evidence[:fetch_count], page_records, strict=True
                    ):
                        row.update(page_record)
                for row in evidence[fetch_count:]:
                    row.update({
                        "fetch_status": "snippet_only",
                        "text": row.get("snippet", ""),
                    })
                result = {
                    "id": claim["id"], "claim_id": claim.get("claim_id"),
                    "label": claim.get("label"), "claim": claim.get("claim", ""),
                    "queries": plan, "evidence": evidence,
                    "search_calls": len(plan),
                    "elapsed_seconds": time.perf_counter() - started,
                    "snapshot_signature": signature,
                    "gold_evidence_used": False,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        all_rows = load_jsonl(target)
        selected_ids = {row["id"] for row in selected}
        relevant = [row for row in all_rows if row["id"] in selected_ids]
        fetched = [item for row in relevant for item in row["evidence"]]
        attempted = [
            row for row in fetched if row.get("fetch_status") in {"ok", "error"}
        ]
        status_counts = Counter(
            row.get("fetch_status", "missing") for row in fetched
        )
        error_type_counts = Counter(
            row.get("error_type", "unknown")
            for row in attempted if row.get("fetch_status") == "error"
        )
        domains_per_claim = [
            len({item.get("domain") for item in row["evidence"] if item.get("domain")})
            for row in relevant
        ]
        elapsed_seconds = [float(row.get("elapsed_seconds", 0.0)) for row in relevant]
        usable = [row for row in fetched if len(row.get("text", "").strip()) >= 80]
        split_summary = {
            "selected_claims": len(selected), "completed_claims": len(relevant),
            "complete": len(relevant) == len(selected),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hashes[split],
            "search_calls": sum(row["search_calls"] for row in relevant),
            "evidence_rows": len(fetched),
            "unique_domains": len({row.get("domain") for row in fetched if row.get("domain")}),
            "median_domains_per_claim": (
                statistics.median(domains_per_claim) if domains_per_claim else 0.0
            ),
            "fetch_status_counts": dict(status_counts),
            "fetch_error_type_counts": dict(error_type_counts),
            "attempted_page_fetches": len(attempted),
            "fetch_success_rate": (
                status_counts["ok"] / len(attempted) if attempted else None
            ),
            "usable_evidence_rate": len(usable) / len(fetched) if fetched else 0.0,
            "mean_seconds_per_claim": (
                statistics.mean(elapsed_seconds) if elapsed_seconds else 0.0
            ),
            "median_seconds_per_claim": (
                statistics.median(elapsed_seconds) if elapsed_seconds else 0.0
            ),
            "gold_evidence_used": False, "test_split_used": split == "test",
        }
        summary[split] = split_summary
        print(json.dumps({split: split_summary}, indent=2))
    atomic_json(args.output_root / "summary.json", {
        "protocol": "P2_open_web", "snapshot_signature": signature,
        "provider": args.provider, "splits": summary,
    })


if __name__ == "__main__":
    main()
