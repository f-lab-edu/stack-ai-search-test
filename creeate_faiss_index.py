# create_faiss_index.py

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# 1. 파일 경로 설정
DATA_PATH = "stackoverflow_sample.json"
INDEX_SAVE_PATH = "faiss_index/faiss_index.index"
DF_SAVE_PATH = "faiss_index/df.pkl"

# 2. 디렉토리 없으면 생성
os.makedirs("faiss_index", exist_ok=True)

# 3. 데이터 로드
df = pd.read_json(DATA_PATH)

# 4. 모델 불러오기
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. 질문 문장 리스트
corpus = df["question"].tolist()

# 6. 임베딩 벡터화
embeddings = model.encode(corpus, convert_to_numpy=True)

# 7. FAISS 인덱스 생성
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 8. 인덱스 저장
faiss.write_index(index, INDEX_SAVE_PATH)

# 9. DataFrame 저장
df.to_pickle(DF_SAVE_PATH)

print("FAISS 인덱스와 DataFrame 저장 완료!")


# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pandas as pd
#
#
# # 1. 데이터 불러오기
# df = pd.read_json("stackoverflow_sample.json")
#
# # 2. 텍스트 임베딩 생성
# model = SentenceTransformer("all-MiniLM-L6-v2")
# corpus = df["question"].tolist()
# embeddings = model.encode(corpus, convert_to_numpy=True)  # numpy array로!
#
# # 3. FAISS 인덱스 생성
# embedding_dim = embeddings.shape[1]  # 예: 384
# index = faiss.IndexFlatL2(embedding_dim)  # L2 거리 기반
#
# # 4. 벡터 추가 (Index에 넣기!)
# index.add(embeddings)
#
# print("FAISS index size:", index.ntotal) #인덱스에 저장된 벡터 수
# print("총 문장 수:", len(corpus))
