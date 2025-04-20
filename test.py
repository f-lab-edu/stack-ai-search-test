# from sentence_transformers import SentenceTransformer
import pandas as pd

data = pd.read_json("stackoverflow_sample.json")
print(data.head())


# model = SentenceTransformer('all-MiniLM-L6-v2')
# sentences = ['나는 파이참을 설치했어!', '이건 정말 똑똑한 선택이야.']
# embeddings = model.encode(sentences)
#
# print('임베딩 결과:')
# print(embeddings)


