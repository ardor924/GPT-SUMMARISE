# -*- coding: utf-8 -*-
from .intent_gate import semantic_gate
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict
from dotenv import load_dotenv
import os, glob, csv
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

# STT CSV 루트 (ID별 폴더/qa.csv)
_raw_csv_dir = os.getenv("STT_CSV_DIR", os.path.join(PROJ_ROOT, "stt_csv"))
STT_CSV_DIR = os.path.abspath(_raw_csv_dir if os.path.isabs(_raw_csv_dir) else os.path.join(PROJ_ROOT, _raw_csv_dir))
STT_CSV_FILENAME = os.getenv("STT_CSV_FILENAME", "qa.csv")  # 각 ID 폴더 내부 파일명

REJECT_MSG = (
    "해당 내용은 분석결과 영농일지와 관련없는 내용으로 판단됩니다.\n"
    "영농일지/농업 관련 내용을 말해주세요."
)

_DEFAULT_FARM_KEYWORDS: Set[str] = {
    "영농","농업","농사","작목","작물","재배","포장","하우스","과원","논","밭",
    "관수","灌水","시비","비료","엽면시비","방제","제초","파종","정식","수확","적과","전정","멀칭",
    "농약","REI","PHI","병해","해충","진딧물","총채","탄저","역병","흰가루","노균","응애","가루이",
    "배추","고추","사과","토마토","감자","상추","딸기","파프리카","오이","참외","포도","복숭아","샤인머스켓",
    "알솎기","봉지씌우기","착색","보르도액","낙과","일소","열과","하우스관리","예찰","약제","살포"
}

# CSV 필드 표준 키(내부) ↔ 질문 라벨(외부)
_FIELD_ALIASES: Dict[str, Set[str]] = {
    "site": {"재배지","포장","위치","장소"},
    "crop": {"작물","품목","품종"},
    "operation": {"작업","카테고리","작업구분"},
    "pesticide": {"농약","약제"},
    "fertiliser": {"비료","시비","자재"},
    "memo": {"메모","비고","기타"}
}

# 작업 카테고리 표준화(5종)
_OP_CANON = {
    "파종·정식": {"파종","정식","이식"},
    "재배관리": {"재배관리","생육관리","하우스관리","관수","灌水","점적","양액","시비","추비","환기","차광","봉지","알솎기","착색"},
    "병해충관리": {"병해","해충","방제","약제","살포","진딧물","탄저","역병","보르도"},
    "수확": {"수확","예취","따기","거둬"},
    "출하·유통": {"출하","유통","선별","포장","상차"}
}

def _canon_op(answer: str) -> Optional[str]:
    a = (answer or "").replace(" ", "")
    for key in _OP_CANON.keys():
        if key.replace(" ","") == a:
            return key
    for key, kws in _OP_CANON.items():
        for w in kws:
            if w in (answer or ""):
                return key
    return answer.strip() if answer else None

# 게이트 완화 옵션
CSV_GATE_LENIENT = os.getenv("CSV_GATE_LENIENT", "1").lower() in ("1","true","yes")

# =========================
# FastAPI
# =========================
app = FastAPI(title="FarmLog STT→RAG Baseline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =========================
# 기존 요약용 Pydantic models
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
    mtime: str

class SummarisePathJSON(BaseModel):
    path: str
    date_hint: Optional[str] = None
    crop_hint: Optional[str] = None
    location_hint: Optional[str] = None
    search_queries: Optional[List[str]] = None

class SummariseAutoRequest(BaseModel):
    path: Optional[str] = None
    stt_text: Optional[str] = None
    date_hint: Optional[str] = None

# =========================
# 새: CSV 요약 결과 모델 (한글 키로 응답)
# =========================
class CsvSummary(BaseModel):
    site: Optional[str]       = Field(default=None, alias="재배지")
    crop: Optional[str]       = Field(default=None, alias="작물")
    operation: Optional[str]  = Field(default=None, alias="작업")
    pesticide: Optional[str]  = Field(default=None, alias="농약")
    fertiliser: Optional[str] = Field(default=None, alias="비료")
    memo: Optional[str]       = Field(default=None, alias="메모")

    model_config = {
        "populate_by_name": True  # Pydantic v2: 내부필드명으로 세팅 허용
    }

# 요청 모델(Union 대신 단일)
class CsvJsonReq(BaseModel):
    id: Optional[str] = None
    path: Optional[str] = None

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
# CSV 도우미
# =========================
def _safe_id(id_text: str) -> str:
    base = os.path.basename((id_text or "").strip())
    if not base or any(ch in base for ch in ("..","/","\\")):
        raise HTTPException(status_code=400, detail="invalid id")
    return base

def _id_to_csv_path(id_text: str) -> str:
    safe = _safe_id(id_text)
    return os.path.join(STT_CSV_DIR, safe, STT_CSV_FILENAME)

def _canon_field(label: str) -> Optional[str]:
    lab = (label or "").strip()
    for k, al in _FIELD_ALIASES.items():
        if lab in al:
            return k
    low = lab.lower()
    if low in ("site","location","field","plot"): return "site"
    if low in ("crop","item","variety"): return "crop"
    if low in ("operation","category","work"): return "operation"
    if low in ("pesticide","agrochemical","chem"): return "pesticide"
    if low in ("fertiliser","fertilizer","nutrient"): return "fertiliser"
    if low in ("memo","note","remarks"): return "memo"
    return None

def _read_qa_csv(csv_path: str) -> Dict[str, Optional[str]]:
    """
    CSV를 읽어 {표준필드: 값}으로 반환.
    - UTF-8/UTF-8-SIG/CP949 자동 처리
    - question,answer / field,value / 한글 라벨 헤더 / 헤더없음(라벨,값) 모두 처리
    - 값 안의 쉼표도 허용(첫 쉼표만 라벨/값 경계로 보고 나머지는 값으로 재결합)
    """
    # 절대경로 정규화 & 상위 폴더 제한
    csv_abs = os.path.abspath(csv_path if os.path.isabs(csv_path) else os.path.join(PROJ_ROOT, csv_path))
    base_abs = os.path.abspath(STT_CSV_DIR)
    try:
        if os.path.commonpath([csv_abs, base_abs]) != base_abs:
            raise HTTPException(status_code=400, detail="csv must be under STT_CSV_DIR")
    except Exception:
        raise HTTPException(status_code=400, detail="csv must be under STT_CSV_DIR")

    if not os.path.exists(csv_abs):
        raise HTTPException(status_code=404, detail=f"csv not found: {csv_abs}")

    # --- 인코딩 로드(UTF-8-SIG → UTF-8 → CP949 순서) ---
    text: str
    try:
        with open(csv_abs, "rb") as fb:
            raw = fb.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp949")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"csv read error: {e}")

    # 줄 정리
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip() != ""]
    if not lines:
        return {}

    # 헤더 후보
    header = [c.strip() for c in lines[0].split(",")]
    header_l = [h.lower() for h in header]

    out: Dict[str, Optional[str]] = {}

    def _parse_rows_qna(rows: List[str], q_idx: int = 0, a_start_idx: int = 1) -> None:
        for ln in rows:
            cols = [c.strip() for c in ln.split(",")]
            if len(cols) <= q_idx:
                continue
            label = cols[q_idx]
            value = ",".join(cols[a_start_idx:]).strip() if len(cols) > a_start_idx else ""
            fld = _canon_field(label)
            if fld:
                out[fld] = value if value != "" else None

    # 1) question/answer, field/value 헤더
    if set(header_l) >= {"question","answer"} or set(header_l) >= {"field","value"} \
       or (header_l and header_l[0] in {"question","field","질문","항목","label","라벨"} and
           len(header_l) > 1 and header_l[1] in {"answer","value","답변","값"}):
        _parse_rows_qna(lines[1:], 0, 1)

    # 2) 한글 라벨 헤더 + 2행 값
    elif any(h in (_FIELD_ALIASES["site"] | _FIELD_ALIASES["crop"] | _FIELD_ALIASES["operation"] |
                   _FIELD_ALIASES["pesticide"] | _FIELD_ALIASES["fertiliser"] | _FIELD_ALIASES["memo"]) for h in header):
        if len(lines) >= 2:
            vals = [c.strip() for c in lines[1].split(",")]
            lab2idx = {h: i for i, h in enumerate(header)}
            for k, aliases in _FIELD_ALIASES.items():
                for al in aliases:
                    if al in lab2idx:
                        idx = lab2idx[al]
                        if idx < len(vals):
                            v = vals[idx]
                            out[k] = v if v != "" else None

    # 3) 헤더 없음: 각 행이 "라벨,값…" 구조
    else:
        for ln in lines:
            if "," not in ln:
                continue
            label, value = ln.split(",", 1)
            fld = _canon_field(label)
            if fld:
                v = value.strip()
                out[fld] = v if v != "" else None

    # 후처리: 작업 표준화, 빈문자열→None
    if "operation" in out and out.get("operation"):
        out["operation"] = _canon_op(out.get("operation") or "")
    for k, v in list(out.items()):
        if isinstance(v, str) and v.strip() == "":
            out[k] = None

    return out

def _qa_to_text(qa: Dict[str, Optional[str]]) -> str:
    pairs = []
    label_map = {
        "site": "재배지",
        "crop": "작물",
        "operation": "작업",
        "pesticide": "농약",
        "fertiliser": "비료",
        "memo": "메모",
    }
    for k, v in qa.items():
        if v:
            pairs.append(f"{label_map.get(k,k)}: {v}")
    return " / ".join(pairs)

# =========================
# 스타트업
# =========================
@app.on_event("startup")
def _startup():
    load_dotenv()
    os.makedirs(STT_CSV_DIR, exist_ok=True)

    installed, msg = ensure_requirements_installed(requirements_path=REQ_PATH, lock_path=REQ_LOCK)
    app.state.requirements_status = {"installed_or_ok": installed, "message": msg}

    try:
        vs, backend = build_or_load_vectorstore(
            kb_dir=KB_DIR, persist_dir=CHROMA_DIR,
            force_vectorstore=os.getenv("FORCE_VECTORSTORE"),
        )
        app.state.vector_backend = backend
    except Exception as e:
        app.state.vector_backend = f"indexing-error: {e}"

    try:
        kws = _load_farm_keywords(KEYWORDS_PATH) or set(_DEFAULT_FARM_KEYWORDS)
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
        "stt_csv_dir": STT_CSV_DIR,
        "stt_csv_filename": STT_CSV_FILENAME,
        "csv_gate_lenient": CSV_GATE_LENIENT,
    }

@app.get("/texts")
def list_texts():
    return _list_text_files()

# =========================
# 기존: 자유 텍스트/파일 요약
# =========================
def _run_with_analysis(
    stt_text: str,
    date_hint: Optional[str] = None,
    crop_hint: Optional[str] = None,
    location_hint: Optional[str] = None,
    search_queries: Optional[List[str]] = None,
):
    try:
        is_semantic_ok, _, _ = semantic_gate(stt_text, kb_dir=KB_DIR)
    except Exception:
        is_semantic_ok = False

    domain_kws: Set[str] = getattr(app.state, "farm_keywords", set()) or set(_DEFAULT_FARM_KEYWORDS)
    app.state.farm_keywords = domain_kws
    res = analyse(stt_text, domain_kws, default_date=date_hint)

    try:
        min_hits = int(os.getenv("FARM_GATE_MIN_HITS", "1"))
    except Exception:
        min_hits = 1
    try:
        min_op_hits = int(os.getenv("FARM_GATE_MIN_OP_HITS", "1"))
    except Exception:
        min_op_hits = 1
    block_nonfarm = os.getenv("FARM_GATE_BLOCK_NONFARM", "true").lower() in ("1","true","yes")
    try:
        nonfarm_block_min = int(os.getenv("NONFARM_BLOCK_MIN_HITS", "2"))
    except Exception:
        nonfarm_block_min = 2

    domain_hits  = int(res.get("domain_hits") or 0)
    op_hits      = int(res.get("op_hits") or 0)
    nonfarm_hits = int(res.get("non_farm_hits") or 0)
    agri_hits    = int(res.get("agri_hits") or 0)

    crop_auto = res.get("crop_hint")
    loc_auto  = res.get("location_hint")
    crop_eff = crop_hint or crop_auto
    loc_eff  = location_hint or loc_auto

    rule_ok = (
        (agri_hits >= 1) or
        (domain_hits >= min_hits) or
        (bool(crop_eff or loc_eff) and ((op_hits >= min_op_hits) or (domain_hits >= 1)))
    )
    is_related_final = False
    if is_semantic_ok and not (block_nonfarm and (nonfarm_hits >= nonfarm_block_min) and agri_hits == 0):
        is_related_final = True
    else:
        if block_nonfarm and (nonfarm_hits >= nonfarm_block_min) and agri_hits == 0:
            is_related_final = False
        else:
            is_related_final = rule_ok

    if not is_related_final:
        return PlainTextResponse(REJECT_MSG)

    merged_date = date_hint or res.get("date_hint")
    merged_crop = crop_hint or crop_auto
    merged_loc  = location_hint or loc_auto
    merged_qs   = search_queries or res.get("search_queries") or []

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

@app.post("/summarise")
def summarise(req: SummariseRequest):
    return _run_with_analysis(
        stt_text=req.stt_text,
        date_hint=req.date_hint,
        crop_hint=req.crop_hint,
        location_hint=req.location_hint,
        search_queries=req.search_queries,
    )

@app.post("/summarise_file")
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

@app.post("/summarise_path")
def summarise_path(path: str = Body(..., media_type="text/plain")):
    filename = _normalise_to_basename(path)
    stt_text = _read_text_file_safe(filename)
    return _run_with_analysis(stt_text=stt_text)

@app.post("/summarise_path_json")
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

@app.post("/summarise_auto")
def summarise_auto(req: SummariseAutoRequest):
    if not req.path and not req.stt_text:
        raise HTTPException(status_code=400, detail="either path or stt_text is required")
    if req.path:
        filename = _normalise_to_basename(req.path)
        stt_text = _read_text_file_safe(filename)
    else:
        stt_text = (req.stt_text or "").strip()
    return _run_with_analysis(
        stt_text=stt_text,
        date_hint=req.date_hint,
        crop_hint=None,
        location_hint=None,
        search_queries=None,
    )

# =========================
# 새: CSV 기반 요약 (정확 6필드)
# =========================
def _gate_csv_qa(qa: Dict[str, Optional[str]]) -> bool:
    # 완화: 작물 또는 작업만 있으면 통과
    if CSV_GATE_LENIENT and (qa.get("crop") or qa.get("operation")):
        return True

    text = _qa_to_text(qa)
    try:
        is_semantic_ok, _, _ = semantic_gate(text, kb_dir=KB_DIR)
    except Exception:
        is_semantic_ok = False

    domain_kws: Set[str] = getattr(app.state, "farm_keywords", set()) or set(_DEFAULT_FARM_KEYWORDS)
    res = analyse(text, domain_kws, default_date=None)
    domain_hits  = int(res.get("domain_hits") or 0)
    op_hits      = int(res.get("op_hits") or 0)
    nonfarm_hits = int(res.get("non_farm_hits") or 0)
    agri_hits    = int(res.get("agri_hits") or 0)

    if is_semantic_ok or agri_hits >= 1 or domain_hits >= 1 or op_hits >= 1:
        return True
    if nonfarm_hits >= 2 and agri_hits == 0:
        return False
    return False

def _qa_to_summary(qa: Dict[str, Optional[str]]) -> CsvSummary:
    return CsvSummary(
        site=qa.get("site"),
        crop=qa.get("crop"),
        operation=_canon_op(qa.get("operation") or ""),
        pesticide=qa.get("pesticide"),
        fertiliser=qa.get("fertiliser"),
        memo=qa.get("memo"),
    )

@app.post("/summarise_csv_id", response_model=CsvSummary, response_model_by_alias=True)
def summarise_csv_id(id_text: str = Body(..., media_type="text/plain")):
    csv_path = _id_to_csv_path(id_text)
    qa = _read_qa_csv(csv_path)
    if not _gate_csv_qa(qa):
        return PlainTextResponse(REJECT_MSG)
    return _qa_to_summary(qa)

@app.post("/summarise_csv_json", response_model=CsvSummary, response_model_by_alias=True)
def summarise_csv_json(req: CsvJsonReq):
    """
    JSON 예:
    { "id": "ID001" }
    또는
    { "path": "stt_csv/ID001/qa.csv" }
    """
    if req.id:
        csv_path = _id_to_csv_path(req.id)
    elif req.path:
        csv_path = req.path if os.path.isabs(req.path) else os.path.join(PROJ_ROOT, req.path)
    else:
        raise HTTPException(status_code=400, detail="must provide id or path")

    qa = _read_qa_csv(csv_path)
    if not _gate_csv_qa(qa):
        return PlainTextResponse(REJECT_MSG)
    return _qa_to_summary(qa)

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
        try:
            kws = _load_farm_keywords(KEYWORDS_PATH) or set(_DEFAULT_FARM_KEYWORDS)
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
