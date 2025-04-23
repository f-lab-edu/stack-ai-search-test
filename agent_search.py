from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, LLMSingleActionAgent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 1. 검색 툴
search = DuckDuckGoSearchRun()

# 2. LLM 모델
generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=generator)

# 3. Agent
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. 테스트
if __name__ == "__main__":
    question = input("궁금한 질문을 입력하세요: ")
    response = agent.run(question)
    print("\n AI 응답:\n", response)