# FAISS 기반 Stack Overflow Q&A 검색기

 이 프로젝트는 Stack Overflow에서 얻은 질문-답변 데이터를 기반으로 사용자의 질문과 유사한 기존 질문을 검색하고 관련 답변을 제공하는 **FAISS 기반 검색 시스템**입니다. 간단한 질문 및 답변은 `.json` 파일로 시작하여 실제 Stack Overflow API를 통해 데이터를 확장할 수 있도록 설계하였습니다.



## 파일 구조

```
.
├── create_faiss_index.py        # 질문 및 답변 데이터를 임베딩하여 FAISS 인덱스 생성
├── query_faiss_index.py         # 사용자 질문을 기반으로 유사 질문 검색
├── stackoverflow_sample.json    # 샘플 질문 및 답변 데이터
├── test_pandas.py               # pandas 로드 테스트
├── test.py                      # 임베딩 및 데이터 로딩 실험 코드
├── test_faiss.py                # faiss 버전 확인용
├── faiss_index/                 # 생성된 인덱스 및 데이터 저장 위치
│   ├── faiss_index.index        # FAISS 인덱스 파일
│   └── df.pkl                   # 질문/답변 DataFrame 저장
```



## 설치 방법

```bash
pip install pandas faiss-cpu sentence-transformers
```



## 실행 방법

### 1. 인덱스 생성

```bash
python create_faiss_index.py
```

- `stackoverflow_sample.json` 파일에서 질문을 읽어 임베딩합니다.
- FAISS 인덱스와 함께 DataFrame(`df.pkl`)으로 저장됩니다.

### 2. 질문 검색

```bash
python query_faiss_index.py
```

- 사용자로부터 질문을 입력받고, 유사한 질문 Top 3을 출력합니다.
- 각 질문에 대한 Stack Overflow 스타일 답변도 함께 제공됩니다.



## 향후 확장 계획

### 1. Stack Overflow 실시간 데이터 수집

- StackExchange API 또는 웹 크롤러를 사용하여 실시간 질문/답변 수집

### 2. 데이터셋 확장 및 재임베딩

- 수집한 데이터를 기존 인덱스에 추가하거나 새롭게 구성

### 3. GPT 등의 LLM 연동

- 검색된 유사 질문들을 바탕으로 자연어 답변 생성

### 4. 웹 기반 UI 개발 (Streamlit, Flask 등)

- 입력창 + 검색결과 UI 구성으로 사용성 향상



## 기여하기

이 프로젝트는 학습 및 프로토타이핑 용도로 개발되었습니다. 개선 아이디어나 확장 기능이 있다면 자유롭게 제안해주세요!



## 라이선스

MIT License