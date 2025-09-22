# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os, glob
from datetime import datetime

# ✅ 부트스트랩(요구 패키지 확인/설치) — 상대/절대 경로 모두 시도, 최종 폴백 제공
try:
    from .bootstrap import ensure_requirements_installed  # 패키지 상대 경로
except Exception:
    try:
        from bootstrap import ensure_requirements_installed  # 동일 디렉토리 절대 경로
    except Exception:
        def ensure_requirements_installed(requirements_path: str = "requirements.txt",
                                          lock_path: str = ".requirements.sha256"):
            return False, "bootstrap module missing"

from .pipeline_langchain import FarmLogPipeline, FarmLog
from .rag import build_or_load_vectorstore

# ---------- 경로 상수 (프로젝트 루트 기준) ----------
# src  ─┐
#       └─ app_fastapi.py (현재 파일)
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# requirements.txt / lock 파일은 루트에 있다고 가정
REQ_PATH = os.getenv("REQUIREMENTS_PATH", os.path.join(PROJ_ROOT, "requirements.txt"))
REQ_LOCK = os.getenv("REQUIREMENTS_LOCK", os.path.join(PROJ_ROOT, ".requirements.sha256"))

# STT 텍스트 폴더 (루트/text)
TEXT_DIR = os.getenv("TEXT_DIR", os.path.join(PROJ_ROOT, "text"))

# KB/Chroma 경로 (루트/kb, 루트/chroma)
KB_DIR = os.getenv("KB_DIR", os.path.join(PROJ_ROOT, "kb"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJ_ROOT, "chroma"))

app = FastAPI(title="FarmLog STT→RAG Baseline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 모델 ----------
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

# --------- 전역 ----------
_pipeline: Optional[FarmLogPipeline] = None

# --------- 유틸 ----------
def _list_text_files() -> List[TextInfo]:
    os.makedirs(TEXT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    out: List[TextInfo] = []
    for fp in files:
        st = os.stat(fp)
        out.append(TextInfo(
            name=os.path.basename(fp),
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
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

@app.on_event("startup")
def _startup():
    load_dotenv()

    # 1) 요구 패키지 설치 확인 & 필요 시 설치
    installed, msg = ensure_requirements_installed(requirements_path=REQ_PATH, lock_path=REQ_LOCK)
    app.state.requirements_status = {"installed_or_ok": installed, "message": msg}

    # 2) KB 인덱싱(없으면 생성) — 폴백 포함 (rag.py에서 자동 처리)
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

@app.post("/ingest")
def ingest(req: IngestRequest):
    # KB 폴더가 바뀐 경우 강제로 재인덱싱하고 파이프라인 리셋
    kb_dir = req.kb_dir or os.getenv("KB_DIR", "./kb")
    vs = build_or_load_vectorstore(kb_dir=kb_dir, persist_dir=os.getenv("CHROMA_DIR", "./chroma"))
    global _pipeline
    _pipeline = None
    return {"status": "ok", "kb_dir": kb_dir}

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "requirements": getattr(app.state, "requirements_status", None),
        "vector_backend": getattr(app.state, "vector_backend", None)
    }
