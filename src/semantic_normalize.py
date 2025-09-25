# src/semantic_normalize.py
# -*- coding: utf-8 -*-
"""
의미 기반 CSV 정규화기:
- 규칙(정규식)으로 다 못 잡는 자연어/구어체를 LLM이 문맥 파악해 6필드로 '정규화'합니다.
- 출력 스키마는 site/crop/operation/pesticide/fertiliser/memo (모두 Optional[str]) 고정.
- 부정(안 쳤어/안 줬어/미사용/없음/전에 줬다 등) -> None, 제품/용량/배수는 요점만 남깁니다.
"""
from typing import Optional, Dict
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────────────────────────────────────────────────────────────
# 출력 스키마
# ──────────────────────────────────────────────────────────────────────────────
class CsvNormalized(BaseModel):
    site: Optional[str] = Field(default=None, description="재배지/포장/하우스/블록 등. 군더더기 제거, 간결하게.")
    crop: Optional[str] = Field(default=None, description="작물명. 흔한 이름으로 간결하게 (샤인머스켓, 사과 등).")
    operation: Optional[str] = Field(default=None, description="파종·정식 / 재배관리 / 병해충관리 / 수확 / 출하·유통 중 하나.")
    pesticide: Optional[str] = Field(default=None, description="농약 정보. 제품명과 배수/농도/용량만 요점 표시. 부정/미사용이면 null.")
    fertiliser: Optional[str] = Field(default=None, description="비료 정보. 제품명과 용량만 요점 표시. 부정/미사용이면 null.")
    memo: Optional[str] = Field(default=None, description="1줄 핵심 메모. 작물/핵심토픽/수치만 간결하게 요약(비농업이면 null).")


# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM = """
너는 농작업 STT CSV를 '정확하고 간결한 6필드'로 정규화하는 도우미다.
반드시 다음 규칙을 지킨다:

[출력 스키마]
- site, crop, operation, pesticide, fertiliser, memo (모두 문자열 또는 null)
- 출력은 반드시 구조화 스키마에 '딱 맞게'만 반환한다. 불필요한 말/기호/설명 금지.

[정규화 규칙]
1) site: '포장-2', '하우스-3', 'A블록', '과원', '경북 상주시' 등 핵심만. 추임새/조사/문장부호 제거.
2) crop: 흔한 명칭으로 간결하게(배추/사과/감귤/샤인머스켓 등). 불필요 말투 제거.
3) operation: 다음 중 하나로 매핑
   - '파종·정식' / '재배관리' / '병해충관리' / '수확' / '출하·유통'
   예) 가지치기/제초/관수/봉지/알솎기/착색/잡초 제거 → '재배관리'
      방제/약제/진딧물/탄저/역병 → '병해충관리'
      수확/따기/거둠 → '수확'
4) pesticide(농약):
   - '안 쳤어/미사용/없음/전에 쳤다/이번엔 안 쳤다' 등 '금일 미사용'이면 null
   - 사용한 경우: 제품명 + (배/%, L, mL 등) 핵심만 (예: "HAL-900 1000배", "사파이어 수화제 200L")
   - 말끝(…했어요/뿌렸어요/보자..)·조사·군더더기는 제거
5) fertiliser(비료):
   - '안 줬어/미사용/없음/전에 줬다/이번엔 안 줬다' 등 '금일 미사용'이면 null
   - 사용한 경우: 제품명 + 용량만 (예: "백두산 A형 5kg")
   - 말끝/군더더기 제거
6) memo:
   - 농업 관련 핵심만 1줄로 압축 (작물/토픽/수치, 예: "샤인머스켓 · 수확 · 당도 18 Brix")
   - 비농업(여행/레시피/엔터/건강/일상) 내용이면 null
   - 감성문구/군더더기/중복 제거
"""

_USER_TMPL = """
[Q&A 원문]
- 재배지(site): {site}
- 작물(crop): {crop}
- 작업(operation): {operation}
- 농약(pesticide): {pesticide}
- 비료(fertiliser): {fertiliser}
- 메모(memo): {memo}

위 자료를 규칙에 따라 정규화해줘.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 실행 함수
# ──────────────────────────────────────────────────────────────────────────────
def normalize_csv_semantic(qa: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    qa: {"site":..,"crop":..,"operation":..,"pesticide":..,"fertiliser":..,"memo":..}
    return: 동일 키의 정규화된 값(dict). 실패 시 예외.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("user", _USER_TMPL),
    ])

    structured = llm.with_structured_output(CsvNormalized)
    msg = prompt.invoke({
        "site": qa.get("site") or "",
        "crop": qa.get("crop") or "",
        "operation": qa.get("operation") or "",
        "pesticide": qa.get("pesticide") or "",
        "fertiliser": qa.get("fertiliser") or "",
        "memo": qa.get("memo") or "",
    })
    out: CsvNormalized = structured.invoke(msg)
    return out.dict()
