# -*- coding: utf-8 -*-
from typing import Optional, List, Dict
import re
from datetime import datetime

# 간단한 도메인 사전 (필요시 확장)
CROPS = ["배추", "고추", "사과", "토마토", "감자", "상추", "딸기", "파프리카", "오이", "참외", "포도", "복숭아"]
# 키워드 → 검색쿼리 템플릿 (간단 매핑; 실제로는 더 정교하게)
KEYWORDS = {
    r"진딧물": lambda crop: [f"{crop} 진딧물 방제 요령"],
    r"총채": lambda crop: [f"{crop} 총채벌레 방제"],
    r"역병": lambda crop: [f"{crop} 역병 예방"],
    r"탄저": lambda crop: [f"{crop} 탄저병 예방"],
    r"요소.*엽면|엽면시비": lambda crop: [f"{crop} 요소 엽면시비 농도"],
    r"칼슘": lambda crop: [f"{crop} 칼슘 엽면시비 농도"],
}

# 위치 패턴들
LOC_PATTERNS = [
    r"(포장[- ]?\d+)",        # 포장-2, 포장 2
    r"(하우스[- ]?\d+)",      # 하우스-3, 하우스 3
    r"([A-Z가-힣]블록)",       # A블록, 가블록
    r"(밭\s?\d+)",            # 밭2, 밭 2
]

DATE_PATTERNS = [
    r"(\d{4})[.\-/년]\s?(\d{1,2})[.\-/월]\s?(\d{1,2})[일]?",   # 2025-09-22, 2025년9월22일
    r"(\d{1,2})[.\-/월]\s?(\d{1,2})[일]"                      # 9월 22일
]

def _extract_crop(text: str) -> Optional[str]:
    for c in CROPS:
        if c in text:
            return c
    return None

def _extract_location(text: str) -> Optional[str]:
    for pat in LOC_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(" ", "").strip()
    return None

def _extract_date(text: str, default_date: Optional[str]) -> Optional[str]:
    # 우선 STT에서 절대 날짜가 보이면 파싱
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                if len(m.groups()) == 3:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    now = datetime.now()
                    y, mo, d = now.year, int(m.group(1)), int(m.group(2))
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:
                pass
    # 상대 표현(“오늘/어제”) 등은 간단 처리 (정확도가 필요하면 확장)
    if "오늘" in text:
        return datetime.now().strftime("%Y-%m-%d")
    if "어제" in text:
        return (datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0) \
               .strftime("%Y-%m-%d")  # 간단화
    # 기본값 있으면 사용
    return default_date

def _generate_search_queries(text: str, crop: Optional[str]) -> List[str]:
    if not crop:
        return []
    queries = []
    for pat, fn in KEYWORDS.items():
        if re.search(pat, text):
            queries.extend(fn(crop))
    # 중복 제거
    return sorted(set(queries))

def extract_hints(stt_text: str,
                  default_date: Optional[str] = None) -> Dict[str, Optional[str]]:
    stt_text = (stt_text or "").strip()
    crop = _extract_crop(stt_text)
    loc = _extract_location(stt_text)
    dt  = _extract_date(stt_text, default_date=default_date)
    queries = _generate_search_queries(stt_text, crop)
    return {
        "date_hint": dt,
        "crop_hint": crop,
        "location_hint": loc,
        "search_queries": queries
    }
