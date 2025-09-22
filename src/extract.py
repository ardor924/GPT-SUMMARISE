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

# 대표 작물/품종 (과수 강화)
CROPS = [
    "배추", "고추", "사과", "토마토", "감자", "상추", "딸기",
    "파프리카", "오이", "참외", "포도", "복숭아", "샤인머스켓"
]

# 농작업(오퍼레이션) 키워드
OPERATION_KEYWORDS = {
    "파종","정식","이식","정지","전정","적심","유인","결속","적과","수확","예취","선별",
    "관수","灌水","점적","급수","양액","시비","추비","밑거름","엽면시비","방제","약제","살포",
    "제초","예초","멀칭","피복","경운","로터리","두둑","소독","예찰","검정","엽분석","토양검정",
    "환기","차광","보온","난방","가온","저온","하우스관리","포장관리"
}

# ✅ 과수·현장 표현 화이트리스트 (히트하면 강한 영농 신호)
AGRI_WHITELIST_PATTERNS = [
    r"알솎기", r"봉지\s?씌우기", r"착색", r"당도\s?측정", r"낙과",
    r"일소", r"열과", r"보르도액", r"칼슘\s?엽면시비", r"병반",
    r"분무", r"환풍기", r"송이\s?모양", r"원통형", r"그늘진\s?과실",
    r"과원", r"[A-Z가-힣]블록", r"하우스\s?\d+동", r"포장[- ]?\d+",
    r"진딧물\s?예방", r"탄저병\s?예방", r"예방\s?방제",
]

# ❌ 레시피/비농업 차단 패턴 (보수적으로 수정)
# - '간' 같은 단일 음절 제거 (과도 매칭 방지)
# - 단위는 반드시 '숫자+단위'로만 매칭
NON_FARMING_PATTERNS = [
    r"(레시피|recipe)\b",
    r"\b재료\b",
    r"(스푼|큰술|작은술|티스푼)\b",
    r"\b\d+(\.\d+)?\s?(g|kg|ml|l|리터|밀리리터|그램)\b",
    r"(볶|끓이|졸이|굽|튀기|양념하|간\s?맞|간을\s?보|조리하|썰|다지|절이|데치|삶)\w*",
    r"(간장|설탕|소금|후추|다진마늘|대파|참기름|식용유|고춧가루|된장|고추장|김치|묵은지)",
    r"(시리야|유튜브\s?뮤직|플레이리스트|볼륨|마이크|녹음\s?종료|음악\s?틀어줘)"
]

# 키워드 → 검색쿼리 템플릿
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

# 위치/날짜 패턴
LOC_PATTERNS = [re.compile(p) for p in [
    r"(포장[- ]?\d+)",
    r"(하우스[- ]?\d+)",
    r"([A-Z가-힣]블록)",
    r"(밭\s?\d+)",
]]

DATE_PATTERNS = [re.compile(p) for p in [
    r"(\d{4})[.\-/년]\s?(\d{1,2})[.\-/월]\s?(\d{1,2})[일]?",
    r"(\d{1,2})[.\-/월]\s?(\d{1,2})[일]"
]]


def analyse(
    stt_text: str,
    domain_keywords: Set[str],
    default_date: Optional[str] = None,
) -> Dict:
    """
    한 번의 패스로 '관련성 판정'과 '힌트 추출'을 동시에 수행합니다.
    """
    text = (stt_text or "").strip()
    low = text.lower()

    # 1) 도메인 키워드 매칭 수
    domain_hits = 0
    for kw in domain_keywords:
        if not kw:
            continue
        if kw in text or kw.lower() in low:
            domain_hits += 1
    is_related = domain_hits >= 1

    # 2) 농작업 키워드 매칭 수
    op_hits = 0
    for op in OPERATION_KEYWORDS:
        if op in text:
            op_hits += 1

    # 3) 비농업 패턴 매칭 수 (보수 패턴)
    non_farm_hits = 0
    for pat in NON_FARMING_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            non_farm_hits += 1

    # 4) 화이트리스트 매칭 수
    agri_hits = 0
    for pat in AGRI_WHITELIST_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            agri_hits += 1

    # 5) 작물 (샤인머스켓 → 포도로 정규화)
    crop = None
    if "샤인머스켓" in text:
        crop = "포도"
    else:
        for c in CROPS:
            if c in text:
                crop = c
                break

    # 6) 위치
    loc = None
    for pat in LOC_PATTERNS:
        m = pat.search(text)
        if m:
            loc = m.group(1).replace(" ", "").strip()
            break

    # 7) 날짜
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
            dt = datetime.now().strftime("%Y-%m-%d")

    # 8) 검색쿼리 (작물 확정 시)
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
        "domain_hits": domain_hits,
        "op_hits": op_hits,
        "non_farm_hits": non_farm_hits,
        "agri_hits": agri_hits,
    }
