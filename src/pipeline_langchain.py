# -*- coding: utf-8 -*-
"""
FarmLog 파이프라인 (LangChain 기반)
- STT 텍스트를 정제
- RAG 컨텍스트(로컬 KB)와 (선택) 웹 검색 노트를 주입
- GPT로 농업 페르소나/템플릿을 적용하여 구조화 JSON(FarmLog) 생성

※ 주의
- rag.build_or_load_vectorstore 가 (vectorstore, backend_name) 튜플을 반환하도록 되어 있으므로
  여기서 안전하게 언팩 처리합니다.
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from .prompts import AGRI_PERSONA, SYSTEM_INSTRUCTIONS, USER_TEMPLATE
from .rag import build_or_load_vectorstore, get_retriever


# ---------- 출력 스키마 ----------
class Operation(BaseModel):
    kind: str = Field(description="작업 종류 (파종/정식/관수/시비/방제/제초/수확/기타)")
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None

class Issue(BaseModel):
    title: str
    details: Optional[str] = None
    severity: Optional[str] = Field(default=None, description="low/medium/high")

class NextAction(BaseModel):
    action: str
    due_date: Optional[str] = None  # YYYY-MM-DD

class FarmLog(BaseModel):
    date: Optional[str] = None
    farmer: Optional[str] = None
    location: Optional[str] = None
    crop: Optional[str] = None
    weather: Optional[str] = None
    operations: List[Operation] = []
    issues: List[Issue] = []
    notes: Optional[str] = None
    next_actions: List[NextAction] = []
    references: List[str] = []  # RAG/웹 근거 링크 또는 문서명


# ---------- 파이프라인 ----------
class FarmLogPipeline:
    def __init__(self):
        load_dotenv()
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # build_or_load_vectorstore는 (vs, backend_name) 튜플을 반환
        vs_ret = build_or_load_vectorstore(
            kb_dir=os.getenv("KB_DIR", "./kb"),
            persist_dir=os.getenv("CHROMA_DIR", "./chroma"),
            force_vectorstore=os.getenv("FORCE_VECTORSTORE"),  # "chroma" or "docarray"
        )
        # 안전 언팩(구버전 호환)
        self.vs = vs_ret[0] if isinstance(vs_ret, tuple) and len(vs_ret) >= 1 else vs_ret

        # retriever 초기화(실패해도 앱은 동작)
        try:
            self.retriever = get_retriever(self.vs, k=int(os.getenv("RETRIEVE_TOP_K", "4")))
        except Exception:
            self.retriever = None

        # 구조화 출력 강제
        self.structured_llm = self.llm.with_structured_output(FarmLog)

        # 프롬프트
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGRI_PERSONA + "\n" + SYSTEM_INSTRUCTIONS),
            ("user", USER_TEMPLATE),
        ])

    # ------- 내부 유틸 -------
    def _clean_stt(self, text: str) -> str:
        text = (text or "").replace("\n", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def _rag_context(self, query_hint: Optional[str], stt_text: str):
        """retriever가 없거나 검색 실패해도 빈 컨텍스트로 안전 반환"""
        q = query_hint or (stt_text[:80] if stt_text else "영농일지")
        docs = []
        try:
            if self.retriever:
                docs = self.retriever.invoke(q)
        except Exception:
            docs = []
        ctx_chunks: List[str] = []
        refs: List[str] = []
        for d in docs or []:
            page = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}
            if page:
                ctx_chunks.append(page[:600])  # 과도한 길이 제한
            if meta and meta.get("source"):
                refs.append(str(meta["source"]))
        return "\n\n".join(ctx_chunks), refs

    # ------- 실행 진입점 -------
    def run(
        self,
        stt_text: str,
        date_hint: Optional[str] = None,
        crop_hint: Optional[str] = None,
        location_hint: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
    ) -> FarmLog:
        stt = self._clean_stt(stt_text)

        # RAG 컨텍스트
        rag_ctx, refs = self._rag_context(crop_hint or "영농", stt)

        # (선택) 웹 검색 노트
        try:
            from .search import web_search_notes
            web_notes = web_search_notes(search_queries or [])
        except Exception:
            web_notes = ""

        # 프롬프트 채우기
        filled = self.prompt.invoke({
            "stt_text": stt,
            "date_hint": date_hint or "",
            "crop_hint": crop_hint or "",
            "location_hint": location_hint or "",
            "rag_context": rag_ctx,
            "web_notes": web_notes or "",
        })

        # 구조화 결과 생성
        result: FarmLog = self.structured_llm.invoke(filled)

        # 참고 링크 병합(중복 제거)
        if web_notes:
            # 형식: "- 제목 — URL :: 스니펫"
            links: List[str] = []
            for ln in web_notes.splitlines():
                if not ln.startswith("- "):
                    continue
                # " — "와 " :: "가 모두 있어야 URL을 안전 추출
                if " — " in ln and " :: " in ln:
                    try:
                        url_part = ln.split(" — ", 1)[1]
                        url = url_part.split(" :: ", 1)[0].strip()
                        if url:
                            links.append(url)
                    except Exception:
                        pass
            if links:
                result.references.extend(links)

        if refs:
            result.references.extend(refs)

        # 고유화
        result.references = sorted(set([r for r in result.references if r]))
        return result
