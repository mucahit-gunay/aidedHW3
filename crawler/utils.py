"""
crawler/utils.py — URL normalization, HTML parsing, tokenization.

All language-native: urllib, re, html. No third-party deps.
"""

from __future__ import annotations

import html
import re
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_URL_LEN = 2048
_BINARY_EXTENSIONS = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp|tiff|"
    r"pdf|doc|docx|xls|xlsx|ppt|pptx|"
    r"mp3|mp4|wav|ogg|flac|avi|mov|mkv|wmv|"
    r"zip|tar|gz|bz2|rar|7z|exe|dmg|pkg|deb|rpm|"
    r"woff|woff2|ttf|eot|otf|"
    r"css|js|json|xml|rss|atom"
    r")$",
    re.IGNORECASE,
)

_SCRIPT_RE  = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE   = re.compile(r"<style[^>]*>.*?</style>",  re.DOTALL | re.IGNORECASE)
_TAG_RE     = re.compile(r"<[^>]+>")
_WS_RE      = re.compile(r"\s+")
_TITLE_RE   = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_HREF_RE    = re.compile(r"""href\s*=\s*(?P<q>['"])(?P<url>[^'"]+)(?P=q)""", re.IGNORECASE)
_TOKEN_RE   = re.compile(r"[a-zğüşıöçÀ-ÿ]{2,}")

_STOP_WORDS = frozenset(
    # English
    "a an the and or but in on at to for of with is are was were "
    "be been being have has had do does did will would could should "
    "may might must shall can this that these those it its "
    "we you he she they them their there here when where "
    "what which who whom how from by up about out so than then "
    "if as into through during before after above below between "
    "each other both few more most only own same such no nor not "
    "just because while although though despite since until unless "
    # Turkish
    "ve de bir bu da ile için değil ama daha çok olan bu şu her "
    "gibi kadar sonra önce nasıl neden hangi hem ya ne mi mı mu mü "
    "olan olarak olan var yok ise en hem bile sadece artık ayrıca "
    "ki göre beri dolayı rağmen üzere karşı tarafından içinde "
    "üst alt yan ön arka dış iç üstü altı yanı önü arkası dışı içi".split()
)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str, base: Optional[str] = None) -> Optional[str]:
    """
    Canonicalize a URL for deduplication.
    Returns None for invalid / non-HTTP(S) URLs.
    """
    if not url:
        return None
    try:
        if base:
            url = urljoin(base, url.strip())
        p = urlparse(url.strip())
        if p.scheme not in ("http", "https"):
            return None
        host = p.hostname or ""
        if not host:
            return None
        port = p.port
        netloc = host.lower()
        if port and not ((p.scheme == "http" and port == 80) or
                         (p.scheme == "https" and port == 443)):
            netloc = f"{netloc}:{port}"
        path = p.path or "/"
        if not path:
            path = "/"
        # Strip trailing slash unless path is exactly "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        result = urlunparse((p.scheme, netloc, path, "", p.query, ""))
        if len(result) > _MAX_URL_LEN:
            return None
        return result
    except Exception:
        return None


def extract_domain(url: str) -> str:
    """Return the lowercase hostname (netloc without port) of `url`."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def extract_links(html_text: str, base_url: str) -> list[str]:
    """
    Return deduplicated list of absolute, canonicalized URLs from <a href=...>.
    Uses regex — no DOM parser.
    """
    seen: set[str] = set()
    results: list[str] = []
    for m in _HREF_RE.finditer(html_text):
        raw = m.group("url").strip()
        norm = normalize_url(raw, base=base_url)
        if norm and norm not in seen:
            seen.add(norm)
            results.append(norm)
    return results


def extract_text(html_text: str, max_chars: int = 5_000_000) -> str:
    """
    Return plain-text content of an HTML document.
    Removes script/style blocks, strips tags, decodes entities.
    """
    if not html_text:
        return ""
    text = html_text[:max_chars]
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def extract_title(html_text: str) -> Optional[str]:
    """Return text of the first <title> tag, entities decoded, max 512 chars."""
    if not html_text:
        return None
    m = _TITLE_RE.search(html_text)
    if not m:
        return None
    title = html.unescape(_TAG_RE.sub("", m.group(1))).strip()
    return title[:512] if title else None


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, min length 2, drop stop words."""
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOP_WORDS]


def word_frequencies(text: str) -> dict[str, int]:
    """Return {word: count} for plain text, filtered by tokenize()."""
    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    return freq


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------

def is_valid_crawl_url(
    url: str,
    origin_host: str,
    same_host_only: bool,
) -> bool:
    """
    Policy filter before enqueueing a discovered link.
    Rejects binary extensions and (optionally) off-host links.
    """
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path or ""
    if _BINARY_EXTENSIONS.search(path):
        return False
    if same_host_only:
        host = (parsed.hostname or "").lower()
        if host != origin_host.lower():
            return False
    return True
