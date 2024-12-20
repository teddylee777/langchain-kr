import streamlit as st
from langchain_core.messages.chat import ChatMessage
from rag.pdf import PDFRetrievalChain
from langchain_teddynote import logging
from rag.evaluation import RagEvaluator
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG With Evaluation")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("RAG 평가 ✅")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

if "evaluator" not in st.session_state:
    # RAGAS 평가를 위한 객체 생성
    st.session_state["evaluator"] = RagEvaluator()

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    eval_toggle = st.toggle("평가 결과 출력", value=True)

    st.subheader("전체 평가")
    eval_all_btn = st.button(
        "결과 출력", key="eval_all", type="primary", use_container_width=True
    )

    if eval_all_btn:

        evaluator = st.session_state["evaluator"]
        if len(evaluator.get_samples()["question"]) > 0:
            with st.spinner("평가 중입니다. 잠시만 기다려 주세요"):
                eval_df = evaluator.evaluate_all()
                result_df = eval_df[["faithfulness", "answer_relevancy"]].mean()
                result_df.name = "평균 점수"
                st.dataframe(
                    result_df,
                    use_container_width=True,
                )
        else:
            st.error("평가할 데이터가 없습니다.")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


# 체인 생성
def create_rag_chain(file_path):
    # PDF 문서를 로드
    pdf = PDFRetrievalChain([file_path]).create_chain()

    # retriever 와 chain을 생성
    pdf_retriever = pdf.retriever
    pdf_chain = pdf.chain
    return pdf_retriever, pdf_chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 임베딩
    file_path = embed_file(uploaded_file)
    # RAG 체인 생성
    retriever, chain = create_rag_chain(file_path)
    st.session_state["retriever"] = retriever
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["evaluator"] = RagEvaluator()

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]
    retriever = st.session_state["retriever"]
    evaluator = st.session_state["evaluator"]
    if chain is not None and retriever is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        context = retriever.invoke(user_input)
        response = chain.stream(
            {
                "question": user_input,
                "context": context,
            }
        )
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # RAGAS 평가를 위한 결과 저장
            evaluator.add_sample(user_input, ai_answer, context)
            if eval_toggle:
                with st.spinner("평가 중입니다. 잠시만 기다려 주세요"):
                    evaluate_last = evaluator.evaluate_last()
                ai_answer += f'\n\n✅ 평가 결과\n- 관련성 점수: {evaluate_last.iloc[0]["answer_relevancy"]:.3f}\n- 신뢰도 점수: {evaluate_last.iloc[0]["faithfulness"]:.3f}'
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
