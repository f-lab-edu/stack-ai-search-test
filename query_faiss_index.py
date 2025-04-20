# query_faiss_index.py

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# 1. 경로 정의
INDEX_LOAD_PATH = "faiss_index/faiss_index.index"
DF_LOAD_PATH = "faiss_index/df.pkl"

# 2. 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. 인덱스 & 데이터 로드
index = faiss.read_index(INDEX_LOAD_PATH)
df = pd.read_pickle(DF_LOAD_PATH)

# 4. 사용자 질문 입력 받기
user_question = input("StackOverflow에 물어볼 질문을 입력하세요: ")

# 5. 임베딩 벡터화
query_embedding = model.encode([user_question], convert_to_numpy=True)

# 6. FAISS 검색
k = 3  # 검색할 유사 질문 개수
D, I = index.search(query_embedding, k)

# 7. 결과 출력
print("\n 유사한 질문 결과 Top 3:")
for rank, idx in enumerate(I[0], 1):
    print(f"\n {rank}. 질문: {df['question'].iloc[idx]}")
    print(f"    ▶ 답변: {df['answer'].iloc[idx]}")


# # 사용자 입력 질문
# from creeate_faiss_index import index, df, model
#
# usre_question = "How to handle async code in Python?"
#
# # 1. 벡터화
# query_embedding = model.encode([usre_question], convert_to_numpy=True)
#
# # 2. FAISS 검색(TOP 3 유사 질문 가져오기)
# k = 3 # 가져올 개수
# D, I = index.search(query_embedding, k)
#
# # 3. 결과
# print("유사한 질문들 : ")
# for idx in I[0]:
#     print("-", df["question"].iloc[idx])
#     print("  →", df["answer"].iloc[idx])
