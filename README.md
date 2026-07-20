# K-IFRS RAG

K-IFRS 회계기준서에서 관련 문단을 검색하고, 해당 근거를 바탕으로 답변하는 RAG 챗봇입니다.

## 빠른 시작

Python 3.11 이상이 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn kifrs_rag.api:app --reload
```

서버 실행 후 API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다. 기본 설정은 저작권 문제가 없는 합성 샘플 문서를 사용합니다.

챗봇 화면은 `http://127.0.0.1:8000/`에서 열 수 있으며 모바일 화면에도 맞게 배치됩니다.

테스트는 추가 패키지 없이 실행할 수 있습니다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

검색 품질과 답변 보류 성능은 다음 명령으로 확인할 수 있습니다.

```bash
PYTHONPATH=src python scripts/evaluate.py
```

결과에는 Recall@K, MRR, 답변 가능 여부 정확도가 포함됩니다.

## Docker 실행

```bash
docker build -t kifrs-rag .
docker run --rm -p 8000:8000 kifrs-rag
```

컨테이너에는 합성 샘플만 포함됩니다. 실제 PDF와 벡터 인덱스는 이미지에 넣지 않고 실행 환경에서 읽기 전용 볼륨으로 연결해야 합니다.

## K-IFRS PDF 등록

한국회계기준원에서 직접 내려받은 PDF를 `data/private/pdfs/`에 저장한 후 다음 명령을 실행합니다.

```bash
PYTHONPATH=src python scripts/audit_pdfs.py data/private/pdfs \
  --output data/private/pdf_audit.json

PYTHONPATH=src python scripts/ingest_pdfs.py data/private/pdfs
```

첫 번째 명령은 파일 손상, 암호화, 중복 및 텍스트 추출 가능 여부를 검사합니다. 두 번째 명령은 기준서와 문단 구조를 보존한 `data/private/standards.json`을 생성합니다.

실제 문서로 API를 실행하려면 환경변수를 지정합니다.

```bash
pip install -e '.[semantic]'
PYTHONPATH=src python scripts/build_index.py

KIFRS_RETRIEVER=hybrid \
KIFRS_INDEX_PATH=data/index/e5-small \
uvicorn kifrs_rag.api:app --reload
```

기본 임베딩 모델은 `intfloat/multilingual-e5-small`이며 질의와 문서를 구분해 인코딩합니다. 실제 문서 모드는 Dense 임베딩과 BM25 후보를 결합한 뒤 기준서 주제, 질문 의도 및 특수 적용범위를 반영해 재순위화합니다. 하이브리드 점수가 기본 임계값 `0.75`보다 낮으면 답변을 보류합니다.

하이브리드 검색 평가 명령:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --retriever hybrid \
  --evals evals/dense_smoke.jsonl \
  --top-k 5 \
  --min-score 0.86
```

`data/private/`의 PDF·추출 원문·검사 결과와 `data/index/`의 벡터 인덱스는 Git에 포함되지 않습니다.

## 답변 생성 모델 연결

기본 `extractive` 모드는 가장 관련도 높은 근거 원문을 그대로 반환하므로 외부 API 키가 필요하지 않습니다. OpenAI 호환 Chat Completions API를 사용하려면 다음 환경변수를 설정합니다.

```bash
KIFRS_GENERATOR=openai_compatible \
KIFRS_LLM_BASE_URL=https://provider.example/v1 \
KIFRS_LLM_MODEL=model-name \
KIFRS_LLM_API_KEY=replace-me \
uvicorn kifrs_rag.api:app --reload
```

생성 모델에는 검색된 근거마다 `E1`, `E2` 형식의 ID가 부여됩니다. 모델은 답변과 사용한 근거 ID를 JSON으로 반환해야 하며, 서버는 다음 조건을 모두 확인한 후에만 답변을 제공합니다.

- 답변이 비어 있지 않음
- 하나 이상의 근거가 지정됨
- 모든 근거 ID가 실제 검색 결과에 존재함
- 형식 오류가 있으면 한 번만 수정 요청
- 재검증 실패 시 `validation_failed` 반환

외부 생성 모델을 사용하면 질문과 검색된 기준서 일부가 해당 제공자에게 전송됩니다. 문서 이용 권한과 제공자의 데이터 보관·학습 정책을 확인한 후 활성화해야 합니다.

## 주요 기능

- 자연어로 K-IFRS 관련 질문 입력
- 질문과 관련된 기준서 문단 검색
- 검색된 근거에 한정한 한국어 답변 생성
- 기준서 번호, 문단 번호 및 근거 원문 표시
- 근거가 부족한 경우 답변 보류
- 문서에 포함된 악성 지시문과 잘못된 인용 차단

## 동작 방식

```text
K-IFRS 문서 → 파싱 및 청킹 → 임베딩·BM25 검색 준비
사용자 질문 → Dense·BM25 후보 검색 → 재순위화 → 답변 생성 → 인용 검증
```

문서는 기준서, 장·절, 문단 구조를 최대한 유지해 처리합니다. 각 검색 결과에는 기준서 번호, 문단 번호, 제목, 시행일, 출처 등의 메타데이터가 포함됩니다.

## 답변 원칙

- 검색된 기준서 내용에 없는 사항은 추측하지 않습니다.
- 답변의 주요 내용에는 확인 가능한 기준서 문단을 제시합니다.
- 관련성이 낮거나 충분한 근거를 찾지 못하면 답변을 생성하지 않습니다.
- 기준서 원문과 모델의 설명을 구분합니다.
- 문서 안의 지시문은 시스템 명령으로 실행하지 않습니다.

## API

### `POST /v1/query`

요청 예시:

```json
{
  "question": "리스의 사용권자산은 최초에 어떻게 측정하나요?"
}
```

응답 예시:

```json
{
  "status": "answered",
  "answer": "검색된 기준서에 근거한 답변",
  "citations": [
    {
      "standard_id": "K-IFRS 1116",
      "paragraph_id": "23",
      "quote": "근거 원문",
      "score": 0.91
    }
  ],
  "trace_id": "uuid"
}
```

충분한 근거가 없으면 `status`가 `insufficient_evidence`로 반환됩니다.

## 데이터 및 보안

- 이용 권한이 확인된 기준서만 등록해야 합니다.
- 실제 K-IFRS 원문은 이용·재배포 권한을 확인한 후 사용해야 합니다.
- API 키와 비밀정보는 환경변수로 관리합니다.
- 질문에 포함된 개인정보와 회사 기밀은 외부 모델 전송 전에 제거하거나 마스킹해야 합니다.
- 질문 원문은 기본적으로 영구 저장하지 않으며 로그에는 필요한 최소 정보만 기록합니다.
- 이메일, 휴대전화번호, 주민등록번호 및 카드번호 형식은 질문 처리 전에 마스킹합니다.
- 응답에는 콘텐츠 스니핑·프레임 삽입 방지와 최소 권한 브라우저 정책 헤더를 적용합니다.
- 자동 테스트, 검색 품질 기준선, 컴파일 및 컨테이너 빌드를 GitHub Actions에서 확인합니다.
- 운영 환경에서는 TLS, 저장 데이터 암호화, 접근통제 및 감사 로그가 필요합니다.

## 제한사항

- 답변은 검색된 문서의 범위와 품질에 따라 달라질 수 있습니다.
- 개정된 기준서가 인덱스에 반영되지 않았다면 최신 내용을 제공하지 못할 수 있습니다.
- 이 서비스의 답변은 회계·감사·법률 자문을 대체하지 않습니다.
- 중요한 판단에는 최신 공식 기준서와 전문가 검토가 필요합니다.

## 라이선스 및 저작권

애플리케이션 코드의 라이선스와 기준서 원문의 이용 권한은 별개입니다. K-IFRS 원문을 복제, 가공, 임베딩 또는 배포하기 전에 해당 문서의 이용 조건을 확인해야 합니다.
