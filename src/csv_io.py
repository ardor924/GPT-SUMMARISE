# -*- coding: utf-8 -*-
import os, re
from typing import Optional, Dict, List, Set
from fastapi import HTTPException

FIELD_ALIASES: Dict[str, Set[str]] = {
    "site": {"재배지","포장","위치","장소"},
    "crop": {"작물","품목","품종"},
    "operation": {"작업","카테고리","작업구분"},
    "pesticide": {"농약","약제"},
    "fertiliser": {"비료","시비","자재"},
    "memo": {"메모","비고","기타"}
}

def _canon_field(label: str) -> Optional[str]:
    lab = (label or "").strip()
    for k, al in FIELD_ALIASES.items():
        if lab in al:
            return k
    low = lab.lower()
    if low in ("site","location","field","plot"): return "site"
    if low in ("crop","item","variety"):         return "crop"
    if low in ("operation","category","work"):   return "operation"
    if low in ("pesticide","agrochemical","chem"): return "pesticide"
    if low in ("fertiliser","fertilizer","nutrient"): return "fertiliser"
    if low in ("memo","note","remarks"):         return "memo"
    return None

def read_qa_csv(csv_path: str, base_dir: str) -> Dict[str, Optional[str]]:
    csv_abs = os.path.abspath(csv_path if os.path.isabs(csv_path) else os.path.join(base_dir, "..", csv_path))
    base_abs = os.path.abspath(base_dir)
    try:
        if os.path.commonpath([csv_abs, base_abs]) != base_abs:
            raise HTTPException(status_code=400, detail="csv must be under STT_CSV_DIR")
    except Exception:
        raise HTTPException(status_code=400, detail="csv must be under STT_CSV_DIR")
    if not os.path.exists(csv_abs):
        raise HTTPException(status_code=404, detail=f"csv not found: {csv_abs}")

    out: Dict[str, Optional[str]] = {}
    try:
        with open(csv_abs, "r", encoding="utf-8-sig", newline="") as f:
            lines = [ln.rstrip("\r\n") for ln in f if ln.strip() != ""]
        if not lines:
            return out

        header = [c.strip() for c in lines[0].split(",")]
        header_l = [h.lower() for h in header]

        def _parse_rows_qna(rows: List[str], q_idx: int, a_start_idx: int) -> None:
            for ln in rows:
                cols = [c.strip() for c in ln.split(",")]
                if len(cols) <= q_idx:
                    continue
                label = cols[q_idx]
                value = ",".join(cols[a_start_idx:]).strip() if len(cols) > a_start_idx else ""
                fld = _canon_field(label)
                if fld:
                    out[fld] = value or None

        if set(header_l) >= {"question","answer"} or set(header_l) >= {"field","value"}:
            _parse_rows_qna(lines[1:], 0, 1)
        elif any(h in (FIELD_ALIASES["site"] | FIELD_ALIASES["crop"] | FIELD_ALIASES["operation"] |
                       FIELD_ALIASES["pesticide"] | FIELD_ALIASES["fertiliser"] | FIELD_ALIASES["memo"]) for h in header):
            vals = [c.strip() for c in (lines[1].split(",") if len(lines) >= 2 else [])]
            lab2idx = {h: i for i, h in enumerate(header)}
            for k, aliases in FIELD_ALIASES.items():
                for al in aliases:
                    if al in lab2idx:
                        idx = lab2idx[al]
                        if idx < len(vals):
                            out[k] = vals[idx] or None
        else:
            for ln in lines:
                if "," not in ln:
                    continue
                label, value = ln.split(",", 1)
                fld = _canon_field(label)
                if fld:
                    out[fld] = (value or "").strip() or None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"csv parse error: {e}")

    for k, v in list(out.items()):
        if isinstance(v, str) and v.strip() == "":
            out[k] = None

    return out
