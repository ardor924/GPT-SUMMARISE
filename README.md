# 영농일지 STT → RAG 요약 베이스라인

음성(STT)으로 전사된 텍스트를 받아, 농업/영농 도메인 페르소나와 템플릿을 적용해 **구조화된 영농일지 요약(FarmLog JSON)** 을 생성하는 FastAPI 서비스입니다.  
로컬 KB(RAG)와 규칙 기반 **통합 분석기(analyse)** 로 작물/위치/날짜/검색쿼리를 보조합니다.

## ✨ 주요 특징
- **경로만 보내면 동작**: `text/파일명.txt` 경로만 POST → 자동 판정·추출 → 요약
- **도메인 가드**: `kb/farming_keywords.txt` 기반으로 **농업/영농 관련성 판별** (무관 시 안내 문구 반환)
- **RAG**: `kb/` 문서를 인덱싱하여 참고 컨텍스트 제공
- **Windows 호환**: Chroma 이슈 시 **DocArray 폴백** 가능

---

## 1) 빠른 시작

### 요구 사항
- Python 3.10+ 권장
- (선택) Windows 10/11, macOS, Linux

### 0. 클론
```bash
git clone <your-repo-url> farm-log
cd farm-log
```

### 1. 가상환경 & 라이브러리
```bash
# Windows (PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> 서버는 기동 시 **requirements 자동 점검**을 수행하지만, 개발 환경에서는 사전 설치를 권장드립니다.

### 2. 환경 변수 (.env)
루트에 `.env`를 만들고 값을 채워주세요. (예시는 `.env.example` 참고)

```dotenv
OPENAI_API_KEY=YOUR_KEY
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# RAG/경로
KB_DIR=./kb
CHROMA_DIR=./chroma
TEXT_DIR=./text
KEYWORDS_PATH=./kb/farming_keywords.txt

# Windows에서 Chroma 이슈 시 폴백
# FORCE_VECTORSTORE=docarray
```

**보안 안내:** `.env`는 커밋 금지, `.env.example`만 커밋하십시오.

### 3. 폴더 준비
```bash
mkdir -p kb text chroma
```
- `kb/` : 참고 문서 및 `farming_keywords.txt` 배치
- `text/` : STT 전사 텍스트(.txt) 배치

#### 키워드 파일(필수)
`kb/farming_keywords.txt` – 영농/농업 관련 판단용 키워드(줄 단위, 공백/쉼표 구분, `#` 주석 허용).  
예시 파일이 포함돼 있지 않다면 리드미 하단의 예시를 참고해 생성하세요.

### 4. 서버 실행
```bash
python -m uvicorn src.app_fastapi:app --host 0.0.0.0 --port 8001
```

헬스 체크:
```bash
curl http://localhost:8001/healthz
```
예시 응답:
```json
{
  "status": "ok",
  "vector_backend": "docarray-build or chroma-...",
  "keywords_count": 123,
  "keywords_path": "./kb/farming_keywords.txt"
}
```

---

## 2) 엔드포인트

### 2.1 텍스트 파일 목록
```
GET /texts
```
`text/` 폴더의 `.txt` 파일 목록 반환.

### 2.2 경로만 보내서 요약(가장 간단)
```
POST /summarise_path
Content-Type: text/plain

text/2025-09-22_farmlog_baechu.txt
```
- 서버는 경로에서 **파일명만 추출**하여 `TEXT_DIR`에서 로드
- 관련성 판정 → 자동 추출 → RAG → FarmLog(JSON)  
- 영농/농업 무관 시 `text/plain` 안내 문구:
  ```
  해당 내용은 분석결과 영농일지와 관련없는 내용으로 판단됩니다.
  영농일지/농업 관련 내용을 말해주세요.
  ```

### 2.3 경로(JSON) + 선택 힌트
```
POST /summarise_path_json
Content-Type: application/json

{
  "path": "text/2025-09-22_farmlog_baechu.txt",
  "date_hint": "2025-09-22"
}
```

### 2.4 자동(경로 또는 텍스트)
```
POST /summarise_auto
Content-Type: application/json

{ "path": "text/2025-09-22_farmlog_baechu.txt" }
```
또는:
```json
{
  "stt_text": "…전사문…",
  "date_hint": "2025-09-22"
}
```

### 2.5 자유 텍스트
```
POST /summarise
Content-Type: application/json

{
  "stt_text": "…전사문…"
}
```

### 2.6 KB 재인덱싱(+키워드 재로딩)
```
POST /ingest
Content-Type: application/json

{ "kb_dir": "./kb" }  // 생략 가능
```
KB 변경 후 재인덱싱 및 `farming_keywords.txt` 재로딩.

---

## 3) 응답 스키마 (FarmLog)
성공 시 아래 스키마의 JSON 반환:

```json
{
  "date": "2025-09-22",
  "farmer": null,
  "location": "포장-2",
  "crop": "배추",
  "weather": null,
  "operations": [
    { "kind": "관수", "description": "점적 30분", "quantity": 0.5, "unit": "시간" }
  ],
  "issues": [
    { "title": "진딧물 발생", "details": "하엽에서 소집단 관찰", "severity": "medium" }
  ],
  "notes": "…",
  "next_actions": [
    { "action": "진딧물 예찰 강화 및 필요시 약제 살포", "due_date": "2025-09-24" }
  ],
  "references": [
    "https://…", "영농일지_모범_템플릿.md"
  ]
}
```

---

## 4) 동작 원리

1. **도메인 판정(게이트)**  
   - `kb/farming_keywords.txt` 키워드와 부분 문자열 매칭  
   - 무관 판단 시 즉시 안내 문구 반환

2. **통합 분석(`src/extract.py: analyse`)**  
   - 한 번의 패스로 **관련성 + 힌트(작물/위치/날짜/검색쿼리)** 동시 추출  
   - `date_hint` 등 요청 힌트가 있으면 **우선 적용**

3. **RAG + LLM**  
   - `kb/` 인덱스에서 관련 컨텍스트를 검색해 요약에 주입  
   - 결과를 FarmLog 스키마로 구조화

---

## 5) Postman 예시

### A) 경로만
- **POST** `http://localhost:8001/summarise_path`  
- **Headers**: `Content-Type: text/plain`  
- **Body (raw/Text)**:
  ```
  text/2025-09-22_farmlog_baechu.txt
  ```

### B) 자동(JSON)
- **POST** `http://localhost:8001/summarise_auto`  
- **Body**:
  ```json
  { "path": "text/2025-09-22_farmlog_baechu.txt" }
  ```

---

## 6) 프로젝트 구조

```
.
├─ src/
│  ├─ app_fastapi.py         # FastAPI 엔드포인트 (게이트+통합 분석 호출)
│  ├─ extract.py             # analyse(): 도메인 판정 + 힌트 추출
│  ├─ pipeline_langchain.py  # FarmLogPipeline (LLM/RAG 요약)
│  ├─ rag.py                 # KB 인덱싱/리트리버
│  ├─ prompts.py             # 페르소나·시스템·유저 템플릿
│  └─ bootstrap.py           # requirements 자동 확인/설치
├─ kb/
│  ├─ farming_keywords.txt   # 도메인 키워드 (게이트)
│  ├─ 영농일지_모범_템플릿.md
│  ├─ 농약_안전사용_가이드.txt
│  └─ 작물별_관리요령_요약.txt
├─ text/                     # STT 텍스트(.txt)
├─ chroma/                   # RAG 인덱스 (Chroma 사용 시)
├─ requirements.txt
├─ .env.example              # 환경변수 템플릿(실값 없음)
└─ README.md
```

---

## 7) 트러블슈팅

### Windows에서 `chromadb_rust_bindings` DLL 오류
- 증상: `DLL load failed while importing chromadb_rust_bindings`
- 조치: `.env`에 폴백 지정
  ```
  FORCE_VECTORSTORE=docarray
  ```
  후 재실행.

### Pydantic 경고
- 메시지: `pydantic.error_wrappers:ValidationError has been moved ...`
- 영향 없음(경고). 동작에는 문제 없습니다.

### 키워드가 너무 엄격/느슨할 때
- `kb/farming_keywords.txt` 보완 → `POST /ingest`로 재로딩.

---

## 8) 보안(토큰/비밀 관리)

- **`.env` 커밋 금지**: `.gitignore`에 `.env`, `.env.*` 포함
- `.env.example`만 커밋하여 키 목록을 공유
- GitHub Actions/서버 배포는 **GitHub Secrets** 사용 권장

`.gitignore` 예시:
```gitignore
__pycache__/
*.pyc
.venv/
.env
.env.*
!.env.example

.vscode/
.idea/
.DS_Store

chroma/
text/
*.log
```

---

## 9) 기여
PR/이슈 환영합니다.  
- 버그 리포트: 재현 단계/로그 포함  
- 기능 제안: 사용 사례와 기대 동작을 간단히 설명해 주세요.
