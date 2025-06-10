from langchain.tools import tool
from typing import Annotated, Optional
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class DataAnalysisAgent:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4o",
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
    ):
        self.df = dataframe
        self.model_name = model_name
        self.prefix_prompt = prefix_prompt
        self.postfix_prompt = postfix_prompt
        if column_guideline is not None and column_guideline.strip() != "":
            COLUMN_GUIDE_PREFIX = "###\n\n# Column Guideline\n\nHere's the column guideline you'll be working with:\n"
            self.column_guideline = COLUMN_GUIDE_PREFIX + column_guideline
        else:
            self.column_guideline = ""
        self.tools = [self.create_python_repl_tool()]
        self.store = {}
        self.setup_agent()

    def create_python_repl_tool(self):
        @tool
        def python_repl_tool(
            code: Annotated[str, "Any python code(pandas, matplotlib, seaborn) to run"],
        ):
            """Use this tool to run python, pandas query, matplotlib, and seaborn code."""
            try:
                python_tool = PythonAstREPLTool(
                    locals={"df": self.df, "sns": sns, "plt": plt}
                )
                return python_tool.invoke(code)
            except Exception as e:
                return f"Execution failed. Error: {repr(e)}"

        return python_repl_tool

    def build_system_prompt(self):

        system_prompt = load_prompt("prompts/data-analysis-V02.yaml", encoding="utf-8")

        system_prompt = system_prompt.format(
            dataframe_head=self.df.head().to_string(),
            column_guideline=self.column_guideline,
        )

        if self.prefix_prompt is not None:
            system_prompt = f"{self.prefix_prompt}\n\n{system_prompt}"

        if self.postfix_prompt is not None:
            system_prompt = f"{system_prompt}\n\n{self.postfix_prompt}"
        return system_prompt

    def setup_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.build_system_prompt(),
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        llm = ChatOpenAI(model=self.model_name, temperature=0)
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=60,
            handle_parsing_errors=True,
        )

    def get_session_history(self, session_id):
        return self.store.setdefault(session_id, ChatMessageHistory())

    def get_agent_with_chat_history(self):
        return RunnableWithMessageHistory(
            self.agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def stream(self, input_query, session_id="abc123"):
        agent_with_chat_history = self.get_agent_with_chat_history()
        response = agent_with_chat_history.stream(
            {"input": input_query},
            config={"configurable": {"session_id": session_id}},
        )
        return response
