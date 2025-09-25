# src/gates.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, Optional, Set, Tuple

from .intent_gate import semantic_gate
from .extract import analyse
from .preprocess import is_nonfarm_memo

# 비농업(요리/여행/엔터/건강/일상) 패턴: memo-only일 때 차단
NONFARM_MEMO_HARD_RE = [
    re.compile(r"(레시피|recipe)\b", re.I),
    re.compile(r"\b재료\b"),
    re.compile(r"\b\d+(\.\d+)?\s?(g|kg|ml|l|리터|밀리리터|그램)\b"),
    re.compile(r"(여행|항공권|비행기|렌터카|체크인|호텔|숙소|예약|제주)", re.I),
    re.compile(r"(영화|드라마|넷플릭스|콘서트|유튜브|플레이리스트|볼륨|음악\s?틀어줘)"),
    re.compile(r"(병원|약국|파스|어깨|허리|두통|감기|치과|건강검진)"),
    re.compile(r"(핸드폰|스마트폰|카톡|카카오톡|택배|영수증|세탁소|회의|출근|야근|카페|커피)"),
    re.compile(r"(게임|스팀|롤|배그)"),
]

def _qa_to_text(qa: Dict[str, Optional[str]]) -> str:
    pairs = []
    label_map = {"site":"재배지","crop":"작물","operation":"작업","pesticide":"농약","fertiliser":"비료","memo":"메모"}
    for k, v in qa.items():
        if v:
            pairs.append(f"{label_map.get(k,k)}: {v}")
    return " / ".join(pairs)

def _is_nonfarm_memo_hard(memo: str) -> bool:
    if not memo:
        return False
    for r in NONFARM_MEMO_HARD_RE:
        if r.search(memo):
            return True
    return False

def gate_csv_qa(
    qa: Dict[str, Optional[str]],
    domain_kws: Set[str],
    kb_dir: str,
    csv_gate_lenient: bool = True,
) -> bool:
    """
    CSV 게이트:
    - memo만 있고 비농업이면 차단(안내문 반환)
    - 완화 모드: crop 또는 operation 있으면 통과
    - 그 외에는 semantic_gate + 규칙 analyse 결합
    """
    has_core = any(qa.get(k) for k in ("site","crop","operation","pesticide","fertiliser"))
    memo = qa.get("memo")

    # 1) 메모만 있고 비농업 → 차단
    if memo and not has_core:
        if _is_nonfarm_memo_hard(memo) or is_nonfarm_memo(memo):
            return False

    # 2) 완화: 핵심 키 하나라도 있으면 통과
    if csv_gate_lenient and (qa.get("crop") or qa.get("operation")):
        return True

    # 3) 의미 게이트 + 규칙 게이트
    text = _qa_to_text(qa)

    try:
        is_semantic_ok, _, _ = semantic_gate(text, kb_dir=kb_dir)
    except Exception:
        is_semantic_ok = False

    res = analyse(text, domain_kws, default_date=None)
    domain_hits  = int(res.get("domain_hits") or 0)
    op_hits      = int(res.get("op_hits") or 0)
    nonfarm_hits = int(res.get("non_farm_hits") or 0)
    agri_hits    = int(res.get("agri_hits") or 0)

    if is_semantic_ok or agri_hits >= 1:
        return True
    if (domain_hits >= 1) or (op_hits >= 1):
        return True
    if nonfarm_hits >= 2 and agri_hits == 0:
        return False
    return False
