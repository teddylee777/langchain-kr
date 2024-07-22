from pdb import run
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response
from dotenv import load_dotenv


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Multi-Modal Chat")

st.title("이미지 인식 챗봇 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# Chain 저장용
if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 대화 내용을 기억하기 위한 저장소 생성
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    tab1, tab2 = st.tabs(["파일", "프롬프트"])
    # 파일 업로드
    uploaded_file = tab1.file_uploader("파일 업로드", type=["jpg", "jpeg", "png"])
    # 시스템 프롬프트
    system_prompt = tab2.text_area(
        "시스템 프롬프트",
        "당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.",
    )


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지 🏙️ 를 처리 중입니다...")
def uploade_image_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 체인 생성
def run_chain(image_filepath, system_prompt, user_prompt):
    # 프롬프트 정의
    # system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다.
    # 당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

    # user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 멀티모달 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 파일로 부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)

    return answer


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

main1, main2 = st.tabs(["이미지", "Chat"])
if uploaded_file:
    main1.image(uploade_image_file(uploaded_file))


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        main2.chat_message(chat_message.role).write(chat_message.content)


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    if uploaded_file:
        # 사용자의 입력
        main2.chat_message("user").write(user_input)

        # chain 을 생성
        response = run_chain(
            uploade_image_file(uploaded_file), system_prompt, user_input
        )

        with main2.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.warning("이미지 파일을 업로드 해주세요.")
