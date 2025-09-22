# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Set
from dotenv import load_dotenv
import os, glob
from datetime import datetime

# =========================
# bootstrap
# =========================
try:
    from .bootstrap import ensure_requirements_installed
except Exception:
    try:
        from bootstrap import ensure_requirements_installed
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

# 통합 분석기
try:
    from .extract import analyse
except Exception as e:
    raise RuntimeError(f"cannot import extract.analyse: {e}")

# =========================
# 경로/상수
# =========================
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQ_PATH   = os.getenv("REQUIREMENTS_PATH", os.path.join(PROJ_ROOT, "requirements.txt"))
REQ_LOCK   = os.getenv("REQUIREMENTS_LOCK", os.path.join(PROJ_ROOT, ".requirements.sha256"))
TEXT_DIR   = os.getenv("TEXT_DIR", os.path.join(PROJ_ROOT, "text"))
KB_DIR     = os.getenv("KB_DIR", os.path.join(PROJ_ROOT, "kb"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJ_ROOT, "chroma"))
KEYWORDS_PATH = os.getenv("KEYWORDS_PATH", os.path.join(KB_DIR, "farming_keywords.txt"))

REJECT_MSG = (
    "해당 내용은 분석결과 영농일지와 관련없는 내용으로 판단됩니다.\n"
    "영농일지/농업 관련 내용을 말해주세요."
)

# 게이트가 비어 있거나 너무 느슨할 때를 대비한 기본 키워드(폴백)
_DEFAULT_FARM_KEYWORDS: Set[str] = {
    "영농","농업","농사","작목","작물","재배","포장","하우스","과원","논","밭",
    "관수","灌水","시비","비료","엽면시비","방제","제초","파종","정식","수확","적과","전정","멀칭",
    "농약","REI","PHI","병해","해충","진딧물","총채","탄저","역병","흰가루","노균","응애","가루이",
    "배추","고추","사과","토마토","감자","상추","딸기","파프리카","오이","참외","포도","복숭아",
}

# =========================
# FastAPI
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
    filename: Optional[str] = None
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
    date_hint: Optional[str] = None    # 있으면 analyse 결과보다 우선 적용

# =========================
# 전역 상태
# =========================
_pipeline: Optional[FarmLogPipeline] = None

# =========================
# 유틸: text/ 파일
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
# 키워드 로딩 (farming_keywords.txt)
# =========================
def _load_farm_keywords(path: str) -> Set[str]:
    kws: Set[str] = set()
    try:
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                parts = []
                if "," in line:
                    for p in line.split(","):
                        parts.extend(p.strip().split())
                else:
                    parts = line.split()
                for p in parts:
                    p = p.strip()
                    if p:
                        kws.add(p)
    except Exception:
        return set()
    return kws

# =========================
# 스타트업
# =========================
@app.on_event("startup")
def _startup():
    load_dotenv()

    installed, msg = ensure_requirements_installed(requirements_path=REQ_PATH, lock_path=REQ_LOCK)
    app.state.requirements_status = {"installed_or_ok": installed, "message": msg}

    try:
        vs, backend = build_or_load_vectorstore(
            kb_dir=KB_DIR,
            persist_dir=CHROMA_DIR,
            force_vectorstore=os.getenv("FORCE_VECTORSTORE"),
        )
        app.state.vector_backend = backend
    except Exception as e:
        app.state.vector_backend = f"indexing-error: {e}"

    # 키워드 캐시 (빈 결과/실패 시 기본 키워드 폴백)
    try:
        kws = _load_farm_keywords(KEYWORDS_PATH)
        if not kws:
            kws = set(_DEFAULT_FARM_KEYWORDS)
        app.state.farm_keywords = kws
    except Exception:
        app.state.farm_keywords = set(_DEFAULT_FARM_KEYWORDS)

# =========================
# 헬스/리스트
# =========================
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "requirements": getattr(app.state, "requirements_status", None),
        "vector_backend": getattr(app.state, "vector_backend", None),
        "keywords_count": len(getattr(app.state, "farm_keywords", set())),
        "keywords_path": KEYWORDS_PATH,
        "gate_min_hits": int(os.getenv("FARM_GATE_MIN_HITS", "2")),
    }

@app.get("/texts", response_model=List[TextInfo])
def list_texts():
    return _list_text_files()

# =========================
# 공통 실행 헬퍼 (게이트 강화)
# =========================
def _run_with_analysis(
    stt_text: str,
    date_hint: Optional[str] = None,
    crop_hint: Optional[str] = None,
    location_hint: Optional[str] = None,
    search_queries: Optional[List[str]] = None,
) -> FarmLog:
    # 1) 통합 분석 (게이트 + 힌트 추출)
    domain_kws: Set[str] = getattr(app.state, "farm_keywords", set())
    if not domain_kws:
        # 혹시 모를 예외 상황(키워드 미로드) 폴백
        domain_kws = set(_DEFAULT_FARM_KEYWORDS)
        app.state.farm_keywords = domain_kws

    res = analyse(stt_text, domain_kws, default_date=date_hint)

    # 🔒 게이트 강화: MIN_HITS(기본 2) 규칙 + crop/location 대안
    try:
        min_hits = int(os.getenv("FARM_GATE_MIN_HITS", "2"))
    except Exception:
        min_hits = 2

    domain_hits = int(res.get("domain_hits") or 0)
    crop_auto = res.get("crop_hint")
    loc_auto  = res.get("location_hint")

    # 사용자가 명시한 힌트가 있으면 그것도 판단에 활용
    crop_eff = crop_hint or crop_auto
    loc_eff  = location_hint or loc_auto

    # 최종 게이트: 충분히 관련(키워드 매칭)하거나, 도메인스러운 구조 신호(작물/위치) 존재
    is_related_final = (domain_hits >= min_hits) or bool(crop_eff or loc_eff)

    if not is_related_final:
        return PlainTextResponse(REJECT_MSG)

    # 3) 힌트 병합(사용자 입력 우선 → 없으면 analyse 결과)
    merged_date = date_hint or res.get("date_hint")
    merged_crop = crop_hint or crop_auto
    merged_loc  = location_hint or loc_auto
    merged_qs   = search_queries or res.get("search_queries") or []

    # 4) 파이프라인 실행
    global _pipeline
    if _pipeline is None:
        _pipeline = FarmLogPipeline()

    return _pipeline.run(
        stt_text=stt_text,
        date_hint=merged_date,
        crop_hint=merged_crop,
        location_hint=merged_loc,
        search_queries=merged_qs,
    )

# =========================
# Summarise: 자유 텍스트
# =========================
@app.post("/summarise", response_model=FarmLog)
def summarise(req: SummariseRequest):
    return _run_with_analysis(
        stt_text=req.stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: 파일 선택
# =========================
@app.post("/summarise_file", response_model=FarmLog)
def summarise_file(req: SummariseFileRequest):
    filename = req.filename
    if not filename:
        items = _list_text_files()
        if not items:
            raise HTTPException(status_code=404, detail="no .txt files under TEXT_DIR")
        filename = sorted(items, key=lambda x: x.mtime, reverse=True)[0].name
    stt_text = _read_text_file_safe(filename)
    return _run_with_analysis(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: 경로만 (text/plain)
# =========================
@app.post("/summarise_path", response_model=FarmLog)
def summarise_path(path: str = Body(..., media_type="text/plain")):
    filename = _normalise_to_basename(path)
    stt_text = _read_text_file_safe(filename)
    return _run_with_analysis(stt_text=stt_text)

# =========================
# Summarise: 경로 JSON
# =========================
@app.post("/summarise_path_json", response_model=FarmLog)
def summarise_path_json(req: SummarisePathJSON):
    filename = _normalise_to_basename(req.path)
    stt_text = _read_text_file_safe(filename)
    return _run_with_analysis(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

# =========================
# Summarise: 자동(경로 or 텍스트)
# =========================
@app.post("/summarise_auto", response_model=FarmLog)
def summarise_auto(req: SummariseAutoRequest):
    if not req.path and not req.stt_text:
        raise HTTPException(status_code=400, detail="either path or stt_text is required")
    if req.path:
        filename = _normalise_to_basename(req.path)
        stt_text = _read_text_file_safe(filename)
    else:
        stt_text = (req.stt_text or "").strip()
    # date_hint만 사용자 override 허용(나머지는 analyse 결과 사용)
    return _run_with_analysis(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=None,
        location_hint=None,
        search_queries=None,
    )

# =========================
# KB 재인덱싱(+키워드 재로딩)
# =========================
@app.post("/ingest")
def ingest(req: IngestRequest):
    kb_dir = req.kb_dir or KB_DIR
    try:
        vs, backend = build_or_load_vectorstore(
            kb_dir=kb_dir, persist_dir=CHROMA_DIR,
            force_vectorstore=os.getenv("FORCE_VECTORSTORE")
        )
        global _pipeline
        _pipeline = None
        app.state.vector_backend = backend
        # 키워드 재로딩 (빈 결과면 기본 키워드 폴백)
        try:
            kws = _load_farm_keywords(KEYWORDS_PATH)
            if not kws:
                kws = set(_DEFAULT_FARM_KEYWORDS)
            app.state.farm_keywords = kws
        except Exception:
            app.state.farm_keywords = set(_DEFAULT_FARM_KEYWORDS)

        return {
            "status": "ok",
            "kb_dir": kb_dir,
            "vector_backend": backend,
            "keywords_count": len(app.state.farm_keywords)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
