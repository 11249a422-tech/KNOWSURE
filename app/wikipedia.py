"""Live evidence from Wikipedia: search for articles, then fetch their plain text."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

_HEADING = re.compile(r"\n=+\s*[^=\n]+?\s*=+\n")
CONTACT_HINT = ("Wikipedia requires a contact in the User-Agent: set KNOWSURE_WIKI_CONTACT in .env "
                "to your email address or project URL.")


def user_agent(contact: str) -> str:
    """Wikimedia's API policy: identify the client and give a way to contact its operator."""
    return f"KnowSure/0.2 ({contact}; Team ErrorX hackathon prototype) python-httpx"


@dataclass(frozen=True)
class Page:
    title: str
    url: str
    text: str
    section: str = "lead"  # "lead" (summary) or "full" (whole article, adaptive retrieval)

    @property
    def key(self) -> str:
        return f"{self.url}#{self.section}"


class EvidenceUnavailable(Exception):
    """The evidence source could not be reached."""


class EvidenceSource(Protocol):
    def search(self, query: str, full_articles: int = 0) -> list[Page]:
        """Return the most relevant pages for the query (may be empty); full_articles=N returns full texts."""


class WikipediaClient:
    def __init__(self, lang: str = "en", max_results: int = 3, max_chars: int = 15000, timeout: float = 15.0,
                 contact: str = "", client: httpx.Client | None = None, cache_size: int = 256,
                 full_max_chars: int = 40000):
        self.full_max_chars = full_max_chars
        self._full: dict[str, Page] = {}
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.max_results = max_results
        self.max_chars = max_chars
        self.contact = contact
        self.cache_size = cache_size
        self._client = client or httpx.Client(timeout=timeout, headers={"User-Agent": user_agent(contact)})
        self._pages: dict[str, Page] = {}  # in-memory only; nothing is written to disk
        self._cache_lock = threading.Lock()  # searches run in parallel threads

    def search(self, query: str, full_articles: int = 0) -> list[Page]:
        """Lead sections of the top results; with full_articles=N, the full text of the top N results instead
        (used by adaptive retrieval when the leads didn't settle a claim)."""
        try:
            data = self._get({"action": "query", "list": "search", "srsearch": query,
                              "srlimit": self.max_results, "format": "json", "formatversion": 2})
            titles = [hit["title"] for hit in data["query"]["search"]]
            if full_articles:
                return [p for p in (self._full_page(t) for t in titles[:full_articles]) if p.text]
            self._fetch_missing(titles)
            with self._cache_lock:
                pages = [self._pages[t] for t in titles if t in self._pages]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                raise EvidenceUnavailable(f"Wikipedia rejected the request (403). {CONTACT_HINT}") from exc
            raise EvidenceUnavailable(f"Wikipedia request failed (HTTP {exc.response.status_code}).") from exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EvidenceUnavailable(f"Wikipedia request failed ({exc.__class__.__name__}).") from exc
        return [p for p in pages if p.text]

    def _fetch_missing(self, titles: list[str]) -> None:
        """Fetch the lead (summary) section of every uncached title in a single request."""
        with self._cache_lock:
            missing = [t for t in titles if t not in self._pages]
        if not missing:
            return
        data = self._get({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                          "exlimit": len(missing), "titles": "|".join(missing), "format": "json",
                          "formatversion": 2})
        for page in data["query"].get("pages", []):
            title = page.get("title")
            if not title:
                continue
            text = _HEADING.sub("\n", page.get("extract") or "")[: self.max_chars].strip()
            url = f"https://{self.lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            with self._cache_lock:
                if len(self._pages) >= self.cache_size:
                    self._pages.pop(next(iter(self._pages)))
                self._pages[title] = Page(title, url, text)

    def _full_page(self, title: str) -> Page:
        with self._cache_lock:
            if title in self._full:
                return self._full[title]
        # Full-article extracts can only be fetched one title per request.
        data = self._get({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                          "format": "json", "formatversion": 2})
        page = data["query"]["pages"][0]
        real_title = page.get("title", title)
        text = _HEADING.sub("\n", page.get("extract") or "")[: self.full_max_chars].strip()
        url = f"https://{self.lang}.wikipedia.org/wiki/{quote(real_title.replace(' ', '_'))}"
        result = Page(real_title, url, text, section="full")
        with self._cache_lock:
            if len(self._full) >= 32:
                self._full.pop(next(iter(self._full)))
            self._full[title] = result
        return result

    def _get(self, params: dict) -> dict:
        response = self._client.get(self.api, params=params)
        response.raise_for_status()
        return response.json()
