import pandas as pd
import faiss
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer

#LangChain Agent에서 사용할 수 있는 StackOverflow 검색툴 만들기
#초기화
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index/faiss_index.index")
df = pd.read_pickle("faiss_index/df.pkl")

#검색함수
def search_stackoverflow(query: str) -> str: #LangChain Agent 사용 함수 정의
    embedding = model.encode([query], convert_to_numpy=True) #사용자 질문을 임베딩(벡터화) faiss는 숫자로 된 벡터만 비교할 수 있음
    D, I = index.search(embedding, k=3) #k=3는 유사질문 3개 찾는다 / I 유사한 질문의 인덱스 리스트 / D 거리값(몇 점 차이)
    results = []
    for idx in I[0]: #유사 질문 3개 반복
        q = df["question"].iloc[idx] #질문 가져오기
        a = df["answer"].iloc[idx] #답변 가져오기
        results.append(f"Q: {q}\nA: {a}") #포맷팅
    return "\n\n".join(results) #결과를 문자열로 묶어서 return -> Agent가 이걸 받아서 사용자에게 보여줌

#LangChain 용 툴로 감싸기
faiss_tool = Tool( #LangChain용 Tool
    name="StackOverflowLocalSearch", #name 툴 이름
    func=search_stackoverflow, #실행할 함수
    description="StackOverflow local search tool", #LLM이 어떤 상황에서 이 툴을 사용해야 하는지 알려줌
)