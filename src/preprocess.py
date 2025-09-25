# src/preprocess.py
# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────────────
RE_INTERJECTION_FRONT = re.compile(
    r'^\s*(아니요|아뇨|아니|아냐|응|음|어|어후|예|네|노|NO|no|뇨)\s*[, ]*\s*',
    flags=re.IGNORECASE
)

NONEISH_RE = re.compile(
    r'^\s*(없(음|어요|다)?|미사용|사용\s*안(함|했|합니다)?|안\s*(쳤|치|줬|주|했|함|씀|썼(어|다|어요)?)|no|무)\s*[,\.]*\s*$',
    flags=re.IGNORECASE
)

def collapse_commas_spaces(s: str) -> str:
    s = re.sub(r"\s*,\s*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def drop_front_interjection(s: str) -> str:
    return RE_INTERJECTION_FRONT.sub("", s or "")

def strip_tail_speech(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = s.rstrip(",.，。 ")
    s = re.sub(r"(했어|했어요|했습니다|하였음|하였어요|하였고|야|이야|에요|예요|입니다|이요|요)$", "", s).strip()
    return s

# ──────────────────────────────────────────────────────────────────────────────
# 표준화 사전
# ──────────────────────────────────────────────────────────────────────────────
OP_CANON = {
    "파종·정식": {"파종","정식","이식"},
    "재배관리": {
        "재배관리","생육관리","하우스관리","관수","灌水","점적","양액","시비","추비","환기","차광",
        "봉지","알솎기","착색","전정","가지치기","제초","예초","잡초","풀","제초제"
    },
    "병해충관리": {"병해","해충","방제","약제","살포","진딧물","탄저","역병","보르도"},
    "수확": {"수확","예취","따기","거둬","거둠"},
    "출하·유통": {"출하","유통","선별","포장","상차"}
}

KNOWN_CROPS = [
    "배추","고추","사과","토마토","감자","상추","딸기","파프리카","오이","참외","포도","복숭아",
    "샤인머스켓","사과나무","포도나무","감귤","귤"
]

KOR_LOC_RE = re.compile(
    r"(서울|부산|대구|인천|광주|대전|울산|세종|제주|제주도|강원도|강원|경기|경기도|충청남도|충남|충청북도|충북|"
    r"전라남도|전남|전라북도|전북|경상남도|경남|경상북도|경북)\s*[가-힣A-Za-z0-9]*"
)

# 메모 비농업 패턴(여행/요리/엔터/일상/건강)
NONFARM_MEMO_RE_LIST = [
    re.compile(r"(레시피|recipe)\b", re.I),
    re.compile(r"\b재료\b"),
    re.compile(r"\b\d+(\.\d+)?\s?(g|kg|ml|l|리터|밀리리터|그램)\b"),
    re.compile(r"(여행|항공권|비행기|렌터카|체크인|호텔|숙소|예약|제주)", re.I),
    re.compile(r"(영화|드라마|넷플릭스|콘서트|유튜브|플레이리스트|볼륨|음악\s?틀어줘)"),
    re.compile(r"(병원|약국|파스|어깨|허리|두통|감기|치과|건강검진)"),
    re.compile(r"(핸드폰|스마트폰|카톡|카카오톡|택배|영수증|세탁소|회의|출근|야근|카페|커피)"),
    re.compile(r"(게임|스팀|롤|배그)"),
]

def is_nonfarm_memo(memo: Optional[str]) -> bool:
    if not memo:
        return False
    t = memo.strip()
    for r in NONFARM_MEMO_RE_LIST:
        if r.search(t):
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# 표준화 함수들
# ──────────────────────────────────────────────────────────────────────────────
def normalize_site(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = collapse_commas_spaces(drop_front_interjection(s))
    s = strip_tail_speech(s)
    m = re.search(r"(포장[- ]?\d+|하우스[- ]?\d+|[A-Z가-힣]블록|밭\s?\d+|과원)", s)
    if m:
        return m.group(1).replace(" ", "")
    m = KOR_LOC_RE.search(s)
    if m:
        return m.group(0).replace("  "," ").strip()
    return s or None

def normalize_crop(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = collapse_commas_spaces(drop_front_interjection(s))
    s = strip_tail_speech(s)
    s = re.sub(r"(나무)(야|지|인데|에요|예요)?$", "", s).strip()
    if "샤인머스켓" in s: return "포도"
    if "사과나무"   in s: return "사과"
    if "포도나무"   in s: return "포도"
    if "귤" in s and "감귤" not in s: return "감귤"
    for c in KNOWN_CROPS:
        if c in s:
            return "포도" if c=="샤인머스켓" else ("사과" if c=="사과나무" else ("포도" if c=="포도나무" else c))
    toks = s.split()
    return toks[0] if toks else s

def _canon_op_free(answer: str) -> Optional[str]:
    a = (answer or "").replace(" ", "")
    for key in OP_CANON.keys():
        if key.replace(" ", "") == a:
            return key
    for key, kws in OP_CANON.items():
        for w in kws:
            if w in (answer or ""):
                return key
    return (answer or "").strip() or None

def normalize_operation(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = collapse_commas_spaces(drop_front_interjection(s))
    s = strip_tail_speech(s)
    s = s.replace("작업", "").strip()
    if any(k in s for k in ["수확","따기","거둬","거둠","수확함","수확했"]):
        return "수확"
    if any(k in s for k in ["가지치기","전정"]):
        return "재배관리"
    if any(k in s for k in ["잡초","제초","예초","풀 뽑","풀뽑"]):
        return "재배관리"
    if any(k in s for k in ["방제","약제","살포","진딧물","탄저","역병","보르도"]):
        return "병해충관리"
    if any(k in s for k in ["파종","정식","이식"]):
        return "파종·정식"
    if any(k in s for k in ["봉지","알솎기","착색","관수","灌水","점적","양액","시비","추비","환기","차광"]):
        return "재배관리"
    return _canon_op_free(s)

def normalize_agri_input(s: Optional[str]) -> Optional[str]:
    """
    농약/비료 입력 문자열을 정규화.
    - '아뇨/아니요/응,' 같은 문두 추임새 제거
    - '이번엔/오늘은/지금은' 등 맥락어 제거
    - '안 쳤어/안 줬어/미사용/없음' → None
    - '전에 줬고/지난번에 줌' → 오늘은 안함 → None
    """
    if not s: 
        return None
    s = collapse_commas_spaces(drop_front_interjection(s))
    s = strip_tail_speech(s)
    # 맥락어 제거
    s = re.sub(r"\b(이번엔|이번에는|오늘은|지금은|금번엔|이번|금번)\b", "", s).strip()
    # 부정/미사용 패턴
    if NONEISH_RE.search(s):
        return None
    if re.search(r'안\s*(줬|주|쳤|치|했|함|씀|썼)(었|었어|었어요|네|다|요)?', s):
        return None
    # "전에/지난번에/예전에 줬(쳤)" → 오늘은 안함
    if re.search(r'(전\s*(에|엔)\s*(줬|주었|쳤|치었)|지난번(에)?\s*(줬|쳤)|예전에\s*(줬|쳤))', s):
        return None
    return s or None

# ──────────────────────────────────────────────────────────────────────────────
# 메모 요약
# ──────────────────────────────────────────────────────────────────────────────
def summarize_memo(memo: Optional[str], crop: Optional[str] = None, op: Optional[str] = None) -> Optional[str]:
    if not memo:
        return None
    # 비농업 메모면 버림
    if is_nonfarm_memo(memo):
        return None

    t = collapse_commas_spaces(drop_front_interjection(memo))
    t = strip_tail_speech(t)

    tokens: List[str] = []
    if crop:
        tokens.append(crop)
    # 수확/병해충/제초/관수 등 키워드 요약
    if re.search(r"수확", t): tokens.append("수확")
    if re.search(r"(잡초|제초|예초|풀)", t): 
        if "재배관리" not in tokens and op != "재배관리":
            tokens.append("재배관리")
    if re.search(r"(방제|약제|살포|진딧물|탄저|역병|보르도)", t): tokens.append("병해충")
    # 당도
    bx = re.search(r"(\d+(\.\d+)?)\s*(brix|브릭스|bx)", t, flags=re.I)
    if bx:
        tokens.append(f"당도 {bx.group(1)} Brix")
    elif "당도" in t:
        tokens.append("당도 높음")
    # 날씨/상태
    if re.search(r"(맑|비\s?옴|비\s?와|더움|추움|바람|날씨)", t):
        tokens.append("날씨 좋음" if "맑" in t else "날씨")

    # 기본: op 있으면 포함
    if op and op not in tokens:
        tokens.append(op)

    # 너무 비어있으면 원문 축약 반환
    if not tokens:
        # 문장을 1~2개 토막만
        sent = re.split(r"[\.!?]\s*", t)
        return " ".join(sent[:2]).strip()

    return " · ".join(tokens)
