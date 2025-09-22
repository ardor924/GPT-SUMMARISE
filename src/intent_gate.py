# -*- coding: utf-8 -*-
"""
임베딩 기반 의미적 게이트:
- 입력 텍스트가 '영농일지/농업' 의미 공간에 충분히 가까운지,
  '비농업(레시피/일상/엔터테인먼트 등)' 의미 공간과의 차이가 충분한지로 판정합니다.
- 앵커 텍스트는 kb/intent/positive.txt, kb/intent/negative.txt 에서 라인 단위로 가져오며,
  파일이 없으면 내장 기본 앵커를 사용합니다.
- 환경변수(기본값):
  * OPENAI_EMBED_MODEL=text-embedding-3-small
  * INTENT_POS_SIM=0.30           # 양성(영농) 코사인 유사도 최소값
  * INTENT_MARGIN=0.08            # (pos_sim - neg_sim) 최소 마진
  * INTENT_MIN_POS_ANCHORS=3      # 양성 앵커 최소 개수 (없으면 기본 앵커 사용)
  * INTENT_MIN_NEG_ANCHORS=3      # 음성 앵커 최소 개수
"""
from __future__ import annotations
import os
from typing import List, Tuple
import math

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- 기본 앵커 (파일이 없을 때 폴백) ----------
_DEFAULT_POSITIVE = [
    "과원 A블록에서 착색 상태 점검, 잔가지 정리, 엽면시비 여부 확인, 병해 예방 방제 계획 기록",
    "하우스 2동 관수 30분, 양액 전도도 체크, 진딧물 예찰 결과 기록",
    "포장-3 배추 정식 후 활착 상태 확인, 재식거리, 멀칭, 제초 관리 기록",
    "탄저병 예방 약제 살포 계획, REI 준수, PHI 확인, 작업자 보호구 착용 기록",
    "수확 예정 시기 판단, 낙과 처리, 작업 시간 및 자재 투입량 기록",
]
_DEFAULT_NEGATIVE = [
    "김치찌개 레시피: 재료, 조리 과정, 양념, 불 조절, 맛 평가",
    "오늘 저녁 뭐 먹지 고민, 햄버거와 피자 추천 요청",
    "어제 영화 보고 감상 후기, 엔터테인먼트 이야기",
    "스마트폰 보이스 어시스턴트 호출, 음악 재생, 볼륨 조절, 마이크 감도",
    "쇼핑 목록과 마트에서 장 본 내역, 가정 요리 계획",
]

# ---------- 파일 로더 ----------
def _load_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

# ---------- 코사인 유사도 ----------
def _cos_sim(u: List[float], v: List[float]) -> float:
    a = sum(x * y for x, y in zip(u, v))
    b = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    if b == 0.0:
        return 0.0
    return a / b

# ---------- OpenAI 임베딩 호출 ----------
def _embed(texts: List[str], model: str) -> List[List[float]]:
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    # 최신 SDK는 resp.data[i].embedding
    return [d.embedding for d in resp.data]

# ---------- 메인: 의미적 판정 ----------
def semantic_gate(
    text: str,
    kb_dir: str,
) -> Tuple[bool, float, float]:
    """
    Returns: (is_farm_semantic, pos_sim, neg_sim)
    - is_farm_semantic: 의미적으로 영농일지에 해당하는가
    - pos_sim: 양성 앵커 평균 유사도
    - neg_sim: 음성 앵커 평균 유사도
    """
    text = (text or "").strip()
    if not text:
        return False, 0.0, 0.0

    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    pos_path = os.path.join(kb_dir, "intent", "positive.txt")
    neg_path = os.path.join(kb_dir, "intent", "negative.txt")

    pos_anchors = _load_lines(pos_path) or list(_DEFAULT_POSITIVE)
    neg_anchors = _load_lines(neg_path) or list(_DEFAULT_NEGATIVE)

    # 앵커 최소 개수 보장
    try:
        min_pos = int(os.getenv("INTENT_MIN_POS_ANCHORS", "3"))
        min_neg = int(os.getenv("INTENT_MIN_NEG_ANCHORS", "3"))
    except Exception:
        min_pos, min_neg = 3, 3
    if len(pos_anchors) < min_pos:
        pos_anchors = list(_DEFAULT_POSITIVE)
    if len(neg_anchors) < min_neg:
        neg_anchors = list(_DEFAULT_NEGATIVE)

    # 임베딩 계산
    all_texts = [text] + pos_anchors + neg_anchors
    vecs = _embed(all_texts, model=model)
    q = vecs[0]
    pos_vecs = vecs[1:1 + len(pos_anchors)]
    neg_vecs = vecs[1 + len(pos_anchors):]

    # 평균 유사도
    pos_sim = sum(_cos_sim(q, v) for v in pos_vecs) / max(len(pos_vecs), 1)
    neg_sim = sum(_cos_sim(q, v) for v in neg_vecs) / max(len(neg_vecs), 1)

    # 판정 기준
    try:
        th_sim = float(os.getenv("INTENT_POS_SIM", "0.30"))
        th_margin = float(os.getenv("INTENT_MARGIN", "0.08"))
    except Exception:
        th_sim, th_margin = 0.30, 0.08

    is_farm = (pos_sim >= th_sim) and ((pos_sim - neg_sim) >= th_margin)
    return is_farm, pos_sim, neg_sim
