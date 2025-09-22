# -*- coding: utf-8 -*-
"""
통합 분석기:
- 입력 STT 텍스트를 한 번 스캔하여 '관련성 판정 + 힌트 추출'을 동시에 수행합니다.
- app_fastapi.py에서 로딩한 domain_keywords(= farming_keywords.txt 내용)를 주입받아 사용합니다.
"""
from typing import Optional, List, Dict, Set
import re
from datetime import datetime

# --- 도메인 리소스(가벼운 기본값; 필요시 KB로 외부화 가능) ---

# 대표 작물 목록 (간단 베이스라인)
CROPS = ["배추", "고추", "사과", "토마토", "감자", "상추", "딸기", "파프리카", "오이", "참외", "포도", "복숭아"]

# 키워드 → 검색쿼리 템플릿 매핑용 패턴 (정규식은 미리 컴파일)
KEY_PATTERNS = [
    re.compile(r"진딧물"),
    re.compile(r"총채"),
    re.compile(r"역병"),
    re.compile(r"탄저"),
    re.compile(r"요소.*엽면|엽면시비"),
    re.compile(r"칼슘"),
]
KEY_TO_QUERY = {
    0: lambda crop: [f"{crop} 진딧물 방제 요령"],
    1: lambda crop: [f"{crop} 총채벌레 방제"],
    2: lambda crop: [f"{crop} 역병 예방"],
    3: lambda crop: [f"{crop} 탄저병 예방"],
    4: lambda crop: [f"{crop} 요소 엽면시비 농도"],
    5: lambda crop: [f"{crop} 칼슘 엽면시비 농도"],
}

# 위치/날짜 패턴 (정규식은 미리 컴파일)
LOC_PATTERNS = [re.compile(p) for p in [
    r"(포장[- ]?\d+)",        # 포장-2, 포장 2
    r"(하우스[- ]?\d+)",      # 하우스-3, 하우스 3
    r"([A-Z가-힣]블록)",       # A블록, 가블록
    r"(밭\s?\d+)",            # 밭2, 밭 2
]]

DATE_PATTERNS = [re.compile(p) for p in [
    r"(\d{4})[.\-/년]\s?(\d{1,2})[.\-/월]\s?(\d{1,2})[일]?",   # 2025-09-22, 2025년9월22일
    r"(\d{1,2})[.\-/월]\s?(\d{1,2})[일]"                      # 9월 22일
]]


def analyse(
    stt_text: str,
    domain_keywords: Set[str],
    default_date: Optional[str] = None,
) -> Dict:
    """
    한 번의 패스로 '관련성 판정'과 '힌트 추출'을 동시에 수행합니다.

    Returns dict:
      {
        "is_farming_related": bool,   # (참고용) 내부는 hits>=1 기준. 최종 판정은 app에서 강화 규칙으로 수행
        "date_hint": Optional[str],   # YYYY-MM-DD
        "crop_hint": Optional[str],
        "location_hint": Optional[str],
        "search_queries": List[str],  # 중복 제거됨
        "domain_hits": int            # 매칭된 도메인 키워드 개수
      }
    """
    text = (stt_text or "").strip()
    low = text.lower()

    # 1) 관련성 판정 (간단 키워드 매칭 기반; 내부 기준 hits>=1)
    hits = 0
    for kw in domain_keywords:
        if not kw:
            continue
        if kw in text or kw.lower() in low:
            hits += 1
    is_related = hits >= 1

    # 2) 작물
    crop = None
    for c in CROPS:
        if c in text:
            crop = c
            break

    # 3) 위치
    loc = None
    for pat in LOC_PATTERNS:
        m = pat.search(text)
        if m:
            loc = m.group(1).replace(" ", "").strip()
            break

    # 4) 날짜
    dt = default_date
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                if len(m.groups()) == 3:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    now = datetime.now()
                    y, mo, d = now.year, int(m.group(1)), int(m.group(2))
                dt = f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:
                pass
            break
    if not dt:
        if "오늘" in text:
            dt = datetime.now().strftime("%Y-%m-%d")
        elif "어제" in text:
            # 간단 처리(정밀한 상대일자 파싱이 필요하면 여기 확장)
            dt = datetime.now().strftime("%Y-%m-%d")

    # 5) 검색쿼리 (작물이 확정된 경우에만 생성)
    queries: List[str] = []
    if crop:
        for i, pat in enumerate(KEY_PATTERNS):
            if pat.search(text):
                queries.extend(KEY_TO_QUERY[i](crop))
        queries = sorted(set(queries))

    return {
        "is_farming_related": is_related,
        "date_hint": dt,
        "crop_hint": crop,
        "location_hint": loc,
        "search_queries": queries,
        "domain_hits": hits,
    }
