# -*- coding: utf-8 -*-
"""DuckDuckGo(무키) 또는 Google CSE(유료키) 웹 검색 래퍼"""
from typing import List, Dict, Optional
import os

# DuckDuckGo (no key)
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
    _HAS_DDG = True
except Exception:
    _HAS_DDG = False

# Google CSE (optional)
try:
    from googleapiclient.discovery import build  # pip install google-api-python-client
    _HAS_GCSE = True
except Exception:
    _HAS_GCSE = False


def ddg_search(query: str, max_results: int = 5) -> List[Dict]:
    if not _HAS_DDG:
        return []
    try:
        with DDGS() as ddg:
            hits = list(ddg.text(query, max_results=max_results))
        # 표준화
        return [
            {
                "title": h.get("title"),
                "href": h.get("href") or h.get("link"),
                "body": h.get("body") or h.get("snippet")
            }
            for h in hits
        ]
    except Exception:
        return []


def google_cse_search(query: str, max_results: int = 5) -> List[Dict]:
    cx = os.getenv("GOOGLE_CSE_ID")
    key = os.getenv("GOOGLE_API_KEY")
    if not (_HAS_GCSE and cx and key):
        return []
    try:
        service = build("customsearch", "v1", developerKey=key)
        res = service.cse().list(q=query, cx=cx, num=max_results).execute()
        items = res.get("items", [])
        return [
            {
                "title": it.get("title"),
                "href": it.get("link"),
                "body": it.get("snippet")
            }
            for it in items
        ]
    except Exception:
        return []


def web_search_notes(queries: List[str], max_per_query: int = 3) -> str:
    """여러 쿼리를 순회하며 요약형 노트 문자열 구성"""
    if not queries:
        return ""
    notes = []
    use_web = os.getenv("USE_WEB_SEARCH", "0").strip() == "1"
    if not use_web:
        return ""
    for q in queries:
        hits = ddg_search(q, max_per_query) or google_cse_search(q, max_per_query)
        if not hits:
            continue
        notes.append(f"[검색: {q}]")
        for h in hits:
            notes.append(f"- {h['title']} — {h['href']} :: {h['body']}")
    return "\n".join(notes[:1000])  # 노트 길이 제한