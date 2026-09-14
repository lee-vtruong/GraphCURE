"""Deterministic, provider-neutral utilities for Phase-C web retrieval."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


QUERY_POLICY_VERSION = "graphcure-c1-v1"
_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'\u2019-]*")
_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'\u2019-]*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'\u2019-]*|of|the)){0,4})\b"
)
_TEMPORAL = re.compile(
    r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2}|January|February|March|April|May|"
    r"June|July|August|September|October|November|December|Monday|Tuesday|"
    r"Wednesday|Thursday|Friday|Saturday|Sunday|\d+(?:[.,]\d+)?%?)\b",
    re.IGNORECASE,
)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "in", "into", "is", "it", "its", "of", "on", "or",
    "said", "says", "she", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "will", "with", "would",
}
_TRACKING_KEYS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "source",
}


def normalize_space(value: str) -> str:
    return _SPACE.sub(" ", value or "").strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_plan(claim: str) -> list[dict[str, str]]:
    """Build observable constraint views without labels, qrels, or gold evidence."""
    claim = normalize_space(claim)
    if not claim:
        return []

    candidates: list[tuple[str, str]] = [("semantic", claim)]
    entities: list[str] = []
    for value in _ENTITY.findall(claim):
        value = normalize_space(value)
        if value.casefold() not in {item.casefold() for item in entities}:
            entities.append(value)

    keywords: list[str] = []
    for token in _TOKEN.findall(claim):
        lower = token.lower()
        if lower not in _STOP and lower not in keywords:
            keywords.append(lower)

    if entities:
        entity_query = " ".join(f'"{value}"' for value in entities[:3])
        entity_query = normalize_space(entity_query + " " + " ".join(keywords[:6]))
        candidates.append(("entity", entity_query))

    temporal = list(dict.fromkeys(_TEMPORAL.findall(claim)))
    if temporal:
        candidates.append((
            "temporal",
            normalize_space(" ".join(keywords[:8] + temporal[:5])),
        ))

    candidates.append(("contextual", f'"{claim}" fact check evidence'))
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for constraint, query in candidates:
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            result.append({"constraint": constraint, "query": query})
    return result


def canonicalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"unsupported public URL: {url!r}")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    port = parts.port
    netloc = host
    if port and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        netloc += f":{port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_KEYS
    ]
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(query), ""))


def assert_public_url(url: str, resolve_dns: bool = True) -> str:
    """Reject local/private targets before fetching search-result URLs."""
    canonical = canonicalize_url(url)
    host = urlsplit(canonical).hostname
    if host is None or host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"local URL is forbidden: {url!r}")

    addresses = []
    try:
        addresses.append(ipaddress.ip_address(host))
    except ValueError:
        if resolve_dns:
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
                addresses.append(ipaddress.ip_address(item[4][0]))
    for address in addresses:
        if not address.is_global:
            raise ValueError(f"non-public URL is forbidden: {url!r}")
    return canonical


def fuse_results(
    query_results: list[tuple[str, list[dict]]],
    rank_constant: float = 60.0,
) -> list[dict]:
    """Deduplicate URLs and fuse query rankings without mixing raw scores."""
    fused: dict[str, dict] = {}
    for constraint, rows in query_results:
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            try:
                url = canonicalize_url(str(row.get("url", "")))
            except (TypeError, ValueError):
                continue
            if url in seen:
                continue
            seen.add(url)
            current = fused.setdefault(url, {
                "url": str(row.get("url", "")),
                "canonical_url": url,
                "domain": urlsplit(url).hostname,
                "title": normalize_space(str(row.get("title", ""))),
                "snippet": normalize_space(str(row.get("snippet", ""))),
                "published_at": row.get("published_at"),
                "query_ranks": {},
                "rrf_score": 0.0,
            })
            current["query_ranks"][constraint] = rank
            current["rrf_score"] += 1.0 / (rank_constant + rank)
            if len(str(row.get("snippet", ""))) > len(current["snippet"]):
                current["snippet"] = normalize_space(str(row["snippet"]))
    return sorted(
        fused.values(),
        key=lambda row: (
            -row["rrf_score"],
            min(row["query_ranks"].values()),
            row["canonical_url"],
        ),
    )


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            value = normalize_space(data)
            if value:
                self.parts.append(value)


def html_to_text(value: str) -> str:
    parser = VisibleTextParser()
    parser.feed(value)
    return normalize_space(" ".join(parser.parts))


def load_jsonl(path: Path) -> list[dict]:
    """Read complete JSONL rows and tolerate only a truncated final line."""
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows
