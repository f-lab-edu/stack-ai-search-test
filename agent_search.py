import warnings
warnings.filterwarnings("ignore") # 경고 숨기기

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, LLMSingleActionAgent
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline

# 1. 검색 툴
search = DuckDuckGoSearchRun()

# 2. LLM 모델
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small", # "tiiuae/falcon-7b-instruct"이 너무 커서 터짐
    device=-1, # CPU 강제 지정
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=generator)

# 3. Agent
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False, #verbose=Ture 일 경우 Agent 행동 로그 출력
    handle_parsing_errors=True,
    max_iterations=5, #반복제한
    max_execution_time=60 #실행시간제한(60초)
)

# 4. 테스트
if __name__ == "__main__":
    question = input("궁금한 질문을 입력하세요: ")
    response = agent.run(question)
    print("\n AI 응답:\n", response)