# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os, glob
from datetime import datetime

# =========================
# bootstrap: requirements 설치 확인/자동 설치 (상대/절대 임포트 모두 시도)
# =========================
try:
    from .bootstrap import ensure_requirements_installed  # 패키지 상대 경로
except Exception:
    try:
        from bootstrap import ensure_requirements_installed  # 동일 디렉토리 절대 경로
    except Exception:
        def ensure_requirements_installed(requirements_path: str = "requirements.txt",
                                          lock_path: str = ".requirements.sha256"):
            return False, "bootstrap module missing"

# =========================
# 내부 모듈
# =========================
try:
    from .pipeline_langchain import FarmLogPipeline, FarmLog
except Exception as e:
    raise RuntimeError(f"cannot import pipeline_langchain: {e}")

try:
    from .rag import build_or_load_vectorstore
except Exception as e:
    raise RuntimeError(f"cannot import rag: {e}")

# 자동 추출기
try:
    from .extract import extract_hints
except Exception as e:
    raise RuntimeError(f"cannot import extract: {e}")

# =========================
# 경로/상수 (프로젝트 루트 기준 절대 경로)
# =========================
# src  ─┐
#       └─ app_fastapi.py  (현재 파일)
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQ_PATH   = os.getenv("REQUIREMENTS_PATH", os.path.join(PROJ_ROOT, "requirements.txt"))
REQ_LOCK   = os.getenv("REQUIREMENTS_LOCK", os.path.join(PROJ_ROOT, ".requirements.sha256"))
TEXT_DIR   = os.getenv("TEXT_DIR", os.path.join(PROJ_ROOT, "text"))
KB_DIR     = os.getenv("KB_DIR", os.path.join(PROJ_ROOT, "kb"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJ_ROOT, "chroma"))

# =========================
# FastAPI app
# =========================
app = FastAPI(title="FarmLog STT→RAG Baseline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models
# =========================
class SummariseRequest(BaseModel):
    stt_text: str
    date_hint: Optional[str] = None
    crop_hint: Optional[str] = None
    location_hint: Optional[str] = None
    search_queries: Optional[List[str]] = None

class IngestRequest(BaseModel):
    kb_dir: Optional[str] = None

class SummariseFileRequest(BaseModel):
    filename: Optional[str] = None  # ./text 안의 파일명(상대경로 금지)
    date_hint: Optional[str] = None
    crop_hint: Optional[str] = None
    location_hint: Optional[str] = None
    search_queries: Optional[List[str]] = None

class TextInfo(BaseModel):
    name: str
    size: int
    mtime: str  # ISO8601

class SummarisePathJSON(BaseModel):
    path: str
    date_hint: Optional[str] = None
    crop_hint: Optional[str] = None
    location_hint: Optional[str] = None
    search_queries: Optional[List[str]] = None

class SummariseAutoRequest(BaseModel):
    path: Optional[str] = None         # text/xxx.txt
    stt_text: Optional[str] = None     # 파일 대신 직접 텍스트
    date_hint: Optional[str] = None    # 있으면 자동 추출보다 우선

# =========================
# 전역 상태
# =========================
_pipeline: Optional[FarmLogPipeline] = None

# =========================
# 유틸: text/ 파일 접근
# =========================
def _list_text_files() -> List[TextInfo]:
    os.makedirs(TEXT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    out: List[TextInfo] = []
    for fp in files:
        st = os.stat(fp)
        out.append(TextInfo(
            name=os.path.basename(fp),
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        ))
    return out

def _read_text_file_safe(filename: str) -> str:
    if not filename or any(ch in filename for ch in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="filename must be a base name under TEXT_DIR")
    path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _normalise_to_basename(path: str) -> str:
    base = os.path.basename((path or "").strip())
    if not base or any(ch in base for ch in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="invalid path")
    return base

# =========================
# 스타트업
# =========================
@app.on_event("startup")
def _startup():
    load_dotenv()

    # 1) 요구 패키지 설치 확인 & 필요 시 설치
    installed, msg = ensure_requirements_installed(requirements_path=REQ_PATH, lock_path=REQ_LOCK)
    app.state.requirements_status = {"installed_or_ok": installed, "message": msg}

    # 2) KB 인덱싱(없으면 생성) — rag.py 내부에서 chroma→docarray 폴백 처리
    try:
        vs, backend = build_or_load_vectorstore(
            kb_dir=KB_DIR,
            persist_dir=CHROMA_DIR,
            force_vectorstore=os.getenv("FORCE_VECTORSTORE"),  # "chroma" or "docarray"
        )
        app.state.vector_backend = backend
    except Exception as e:
        # 인덱싱 실패해도 앱은 뜨도록
        app.state.vector_backend = f"indexing-error: {e}"

# =========================
# 헬스/리스트
# =========================
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "requirements": getattr(app.state, "requirements_status", None),
        "vector_backend": getattr(app.state, "vector_backend", None),
    }

@app.get("/texts", response_model=List[TextInfo])
def list_texts():
    return _list_text_files()

# =========================
# Summarise: 자유 텍스트 직접
# =========================
@app.post("/summarise", response_model=FarmLog)
def summarise(req: SummariseRequest):
    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()
    return _pipeline.run(
        stt_text=req.stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: text/ 파일 선택 (명시/자동)
# =========================
@app.post("/summarise_file", response_model=FarmLog)
def summarise_file(req: SummariseFileRequest):
    """./text 폴더 내 STT(.txt) 중 하나를 선택해 요약"""
    filename = req.filename
    if not filename:
        items = _list_text_files()
        if not items:
            raise HTTPException(status_code=404, detail="no .txt files under TEXT_DIR")
        filename = sorted(items, key=lambda x: x.mtime, reverse=True)[0].name

    stt_text = _read_text_file_safe(filename)

    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    return _pipeline.run(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: 경로만 보냄 (text/plain / JSON)
# =========================
@app.post("/summarise_path", response_model=FarmLog)
def summarise_path(path: str = Body(..., media_type="text/plain")):
    """본문이 순수 문자열(예: text/2025-09-22_farmlog_baechu.txt)인 경우"""
    filename = _normalise_to_basename(path)
    stt_text = _read_text_file_safe(filename)

    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    return _pipeline.run(
        stt_text=stt_text,
        date_hint=None,
        crop_hint=None,
        location_hint=None,
        search_queries=None,
    )

@app.post("/summarise_path_json", response_model=FarmLog)
def summarise_path_json(req: SummarisePathJSON):
    """JSON 형태로 경로 + 선택 힌트를 함께 보내는 경우"""
    filename = _normalise_to_basename(req.path)
    stt_text = _read_text_file_safe(filename)

    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    return _pipeline.run(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: 경로만 받아도 '자동 추출' → 요약
# =========================
@app.post("/summarise_auto_path", response_model=FarmLog)
def summarise_auto_path(path: str = Body(..., media_type="text/plain")):
    """
    text/plain 본문에 'text/파일명.txt' 경로만 보내면,
    STT 텍스트에서 crop/location/date/검색쿼리를 자동 추출한 뒤 요약합니다.
    """
    filename = _normalise_to_basename(path)
    stt_text = _read_text_file_safe(filename)

    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    hints = extract_hints(stt_text)
    return _pipeline.run(
        stt_text=stt_text,
        date_hint=hints["date_hint"],
        crop_hint=hints["crop_hint"],
        location_hint=hints["location_hint"],
        search_queries=hints["search_queries"],
    )

class _SummariseAutoRequest(BaseModel):
    path: Optional[str] = None         # text/xxx.txt
    stt_text: Optional[str] = None     # 파일 대신 직접 텍스트
    date_hint: Optional[str] = None    # 있으면 자동 추출보다 우선

@app.post("/summarise_auto", response_model=FarmLog)
def summarise_auto(req: _SummariseAutoRequest):
    """
    JSON으로 경로만 보내거나(path) 전사 텍스트(stt_text)를 직접 실어 보낼 수 있습니다.
    필요하면 date_hint만 덮어쓸 수 있고, crop/location/search_queries는 자동 추출합니다.
    """
    if not req.path and not req.stt_text:
        raise HTTPException(status_code=400, detail="either path or stt_text is required")

    if req.path:
        filename = _normalise_to_basename(req.path)
        stt_text = _read_text_file_safe(filename)
    else:
        stt_text = (req.stt_text or "").strip()

    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    hints = extract_hints(stt_text, default_date=req.date_hint)
    date_hint = req.date_hint or hints["date_hint"]

    return _pipeline.run(
        stt_text=stt_text,
        date_hint=date_hint,
        crop_hint=hints["crop_hint"],
        location_hint=hints["location_hint"],
        search_queries=hints["search_queries"],
    )

# =========================
# KB 재인덱싱
# =========================
@app.post("/ingest")
def ingest(req: IngestRequest):
    kb_dir = req.kb_dir or KB_DIR
    try:
        vs, backend = build_or_load_vectorstore(kb_dir=kb_dir, persist_dir=CHROMA_DIR,
                                                force_vectorstore=os.getenv("FORCE_VECTORSTORE"))
        # 파이프라인 재생성 유도
        global _pipeline
        _pipeline = None
        app.state.vector_backend = backend
        return {"status": "ok", "kb_dir": kb_dir, "vector_backend": backend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
