from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st

from dotenv import load_dotenv

# 페이지 설명
st.set_page_config(page_title="CustomGPT", page_icon="💬")

# API 키 설정
load_dotenv()

# 제목
st.title("💬 CustomGPT")

# 메모리 설정
# msgs =

# 채팅 초기화 버튼 삽입
with st.sidebar:
    reset_history = st.button("채팅 초기화")

# 채팅 기록
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# 채팅 메시지 추가를 위한 함수
def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 이전의 채팅 기록 출력을 위한 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# 채팅 초기화
if reset_history:
    st.session_state["messages"] = []

# 이전의 채팅 기록 출력
print_messages()


# 사용자의 질문을 입력으로 받습니다.
user_input = st.chat_input()

if user_input:
    # 사용자의 질문을 입력으로 받습니다.
    st.chat_message("user").write(user_input)

    # 사용자의 질문을 채팅 기록에 추가합니다.
    add_message("user", user_input)

    with st.chat_message("assistant"):
        chat_container = st.empty()

        # 프롬프트 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "한글로 간결하게 답변하세요."),
                ("human", "{question}"),
            ]
        )

        # LLM 정의
        llm = ChatOpenAI()

        # 체인 생성
        chain = prompt | llm | StrOutputParser()

        # 채팅 실행
        response = chain.stream({"question": user_input})

        # 스트리밍 출력
        answer = ""
        for token in response:
            answer += token
            chat_container.markdown(answer)

        add_message("assistant", answer)
