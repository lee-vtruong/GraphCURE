import json
import subprocess
import sys

import pytest

from graphcure.open_web import (
    assert_public_url,
    canonicalize_url,
    fuse_results,
    html_to_text,
    query_plan,
)
from scripts.run_mocheg_open_web_retrieval import normalize_search


def test_query_plan_is_deterministic_and_constraint_typed():
    claim = "Barack Obama said unemployment fell by 10% in 2012."
    first = query_plan(claim)
    assert first == query_plan(claim)
    assert first[0] == {"constraint": "semantic", "query": claim}
    assert {row["constraint"] for row in first} == {
        "semantic", "entity", "temporal", "contextual",
    }
    assert len({row["query"].casefold() for row in first}) == len(first)


def test_url_canonicalization_and_rrf_deduplicate_tracking_urls():
    assert canonicalize_url("HTTPS://Example.COM/a/?utm_source=x&b=2#part") == (
        "https://example.com/a?b=2"
    )
    fused = fuse_results([
        ("semantic", [{"url": "https://example.com/a?utm_source=x", "title": "A"}]),
        ("entity", [{"url": "https://example.com/a/", "title": "A again"}]),
    ])
    assert len(fused) == 1
    assert fused[0]["query_ranks"] == {"semantic": 1, "entity": 1}


def test_private_and_non_http_urls_are_rejected_without_dns():
    with pytest.raises(ValueError):
        assert_public_url("http://127.0.0.1/private", resolve_dns=False)
    with pytest.raises(ValueError):
        assert_public_url("file:///etc/passwd", resolve_dns=False)


def test_visible_html_and_provider_normalization():
    assert html_to_text("<p>Hello <b>world</b></p><script>secret</script>") == (
        "Hello world"
    )
    brave = normalize_search("brave", {"web": {"results": [{
        "url": "https://example.com", "title": "T", "description": "S",
    }]}})
    serper = normalize_search("serper", {"organic": [{
        "link": "https://example.com", "title": "T", "snippet": "S",
    }]})
    assert brave[0]["snippet"] == serper[0]["snippet"] == "S"


def test_fixture_cli_creates_complete_resumable_snapshot(tmp_path):
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    claim = "Barack Obama said unemployment fell by 10% in 2012."
    (manifest_root / "val.jsonl").write_text(
        json.dumps({
            "id": "claim-1", "claim_id": "1", "label": 0,
            "claim": claim,
        }) + "\n",
        encoding="utf-8",
    )
    fixture = {
        item["query"]: [{
            "url": "https://example.com/evidence?utm_source=test",
            "title": "Evidence", "snippet": "Relevant public evidence.",
        }]
        for item in query_plan(claim)
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    output_root = tmp_path / "snapshot"
    command = [
        sys.executable, "-m", "scripts.run_mocheg_open_web_retrieval",
        "--manifest-root", str(manifest_root),
        "--output-root", str(output_root),
        "--provider", "fixture", "--fixture", str(fixture_path),
        "--splits", "val", "--results-per-query", "5", "--output-k", "3",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)

    rows = [json.loads(line) for line in (
        output_root / "val.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    snapshot = json.loads((
        output_root / "snapshot_manifest.json"
    ).read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["gold_evidence_used"] is False
    assert len(rows[0]["evidence"]) == 1
    assert summary["splits"]["val"]["complete"] is True
    assert "val" in snapshot["manifest_hashes"]
