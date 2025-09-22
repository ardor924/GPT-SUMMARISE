# -*- coding: utf-8 -*-
"""
로컬 KB 인덱싱 및 Retriever (Chroma → 실패 시 DocArrayInMemorySearch 폴백)
- ENV FORCE_VECTORSTORE: "chroma" 또는 "docarray" 강제 가능
"""
from typing import List, Optional, Tuple
import os, glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Vectorstore 후보들 임포트
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch

def _load_kb_texts(kb_dir: str) -> List[str]:
    exts = ("*.txt", "*.md")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(kb_dir, ext)))
    texts = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception:
            pass
    return texts

def _split_docs(raw_texts: List[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.create_documents(raw_texts)

def _build_chroma(docs, persist_dir: str, embed_model: str):
    embeddings = OpenAIEmbeddings(model=embed_model)
    os.makedirs(persist_dir, exist_ok=True)
    # 기존 인덱스가 있으면 로드
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings), "chroma-load"
    # 신규 빌드
    vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vs.persist()
    return vs, "chroma-build"

def _build_docarray(docs, embed_model: str):
    embeddings = OpenAIEmbeddings(model=embed_model)
    return DocArrayInMemorySearch.from_documents(docs, embeddings), "docarray-build"

def build_or_load_vectorstore(
    kb_dir: str,
    persist_dir: str,
    force_vectorstore: Optional[str] = None,
) -> Tuple[object, str]:
    """
    returns: (vectorstore, backend_name)
    backend_name ∈ {"chroma-load","chroma-build","docarray-build"}
    """
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    raw_texts = _load_kb_texts(kb_dir)
    docs = _split_docs(raw_texts) if raw_texts else _split_docs(["영농일지 템플릿 참고용 빈 KB"])  # 빈 폴더 대비

    backend = (force_vectorstore or os.getenv("FORCE_VECTORSTORE", "")).strip().lower()

    # 1) 강제 설정이 chroma인 경우
    if backend == "chroma":
        return _build_chroma(docs, persist_dir, embed_model)

    # 2) 강제 설정이 docarray인 경우
    if backend == "docarray":
        return _build_docarray(docs, embed_model)

    # 3) 자동: chroma 시도 → 실패 시 docarray
    try:
        return _build_chroma(docs, persist_dir, embed_model)
    except Exception:
        return _build_docarray(docs, embed_model)

def get_retriever(vs, k: int = 4):
    return vs.as_retriever(search_kwargs={"k": k})
