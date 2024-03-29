from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
import streamlit as st
from streamlit_feedback import streamlit_feedback
import time
import os


st.set_page_config(page_title="피드백 데모", page_icon="🦜")
st.title("🦜 피드백 데모")

# Set LangSmith environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Customize if needed
client = Client()
ls_tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"], client=client)
run_collector = RunCollectorCallbackHandler()
cfg = RunnableConfig()
cfg["callbacks"] = [ls_tracer, run_collector]
cfg["configurable"] = {"session_id": "any"}


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

reset_history = st.sidebar.button("채팅 초기화")
if len(msgs.messages) == 0 or reset_history:
    msgs.clear()
    msgs.add_ai_message("무엇을 도와드릴까요?")
    st.session_state["last_run"] = None


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "한글로 간결하게 답변하세요."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# If user inputs a new prompt, generate and draw a new response
if user_input := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        chain = prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
        )
        response = chain_with_history.invoke({"question": user_input}, cfg)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )
    st.session_state.last_run = run_collector.traced_runs[0].id


@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url


if st.session_state.get("last_run"):
    run_url = get_run_url(st.session_state.last_run)
    st.sidebar.markdown(f"[LangSmith 추적🛠️]({run_url})")
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label=None,
        key=f"feedback_{st.session_state.last_run}",
    )
    if feedback:
        scores = {"👍": 1, "👎": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=feedback.get("text", None),
        )
        st.toast("피드백을 저장하였습니다.!", icon="📝")
