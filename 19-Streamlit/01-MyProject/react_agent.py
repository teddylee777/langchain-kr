from typing import Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from custom_tools import WebSearchTool


def create_agent_executor(model_name="gpt-4o-mini", tools=[]):
    # 메모리 설정
    memory = MemorySaver()

    # 모델 설정
    model = ChatOpenAI(model_name=model_name)

    agent_executor = create_react_agent(model, tools=tools, checkpointer=memory)

    return agent_executor
