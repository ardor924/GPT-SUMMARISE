# -*- coding: utf-8 -*-
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from .prompts import AGRI_PERSONA, SYSTEM_INSTRUCTIONS, USER_TEMPLATE
from .rag import build_or_load_vectorstore, get_retriever
from .search import web_search_notes

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
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.vs = build_or_load_vectorstore(
            kb_dir=os.getenv("KB_DIR", "./kb"),
            persist_dir=os.getenv("CHROMA_DIR", "./chroma"),
        )
        self.retriever = get_retriever(self.vs, k=int(os.getenv("RETRIEVE_TOP_K", "4")))
        self.structured_llm = self.llm.with_structured_output(FarmLog)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGRI_PERSONA + "\n" + SYSTEM_INSTRUCTIONS),
            ("user", USER_TEMPLATE),
        ])

    def _clean_stt(self, text: str) -> str:
        text = text.replace("\n", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def _rag_context(self, query_hint: Optional[str], stt_text: str) -> str:
        q = query_hint or (stt_text[:80] if stt_text else "영농일지")
        docs = self.retriever.invoke(q)
        ctx = []
        refs = []
        for d in docs:
            page = getattr(d, "page_content", "")
            meta = getattr(d, "metadata", {})
            if page:
                ctx.append(page[:600])
            if meta and meta.get("source"):
                refs.append(str(meta["source"]))
        return "\n\n".join(ctx), refs

    def run(self,
            stt_text: str,
            date_hint: Optional[str] = None,
            crop_hint: Optional[str] = None,
            location_hint: Optional[str] = None,
            search_queries: Optional[List[str]] = None) -> FarmLog:
        stt = self._clean_stt(stt_text)
        rag_ctx, refs = self._rag_context(crop_hint or "영농", stt)
        web_notes = web_search_notes(search_queries or [])

        filled = self.prompt.invoke({
            "stt_text": stt,
            "date_hint": date_hint or "",
            "crop_hint": crop_hint or "",
            "location_hint": location_hint or "",
            "rag_context": rag_ctx,
            "web_notes": web_notes or "",
        })
        result: FarmLog = self.structured_llm.invoke(filled)

        # 참고 링크 병합
        if web_notes:
            links = [ln.split(" — ")[1].split(" :: ")[0] for ln in web_notes.splitlines() if ln.startswith("-") and " — " in ln]
            result.references.extend(links)
        if refs:
            result.references.extend(refs)
        # 중복 제거
        result.references = sorted(set([r for r in result.references if r]))
        return result