import os
from dotenv import load_dotenv
from langchain_text_splitters import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Python 파일을 로드하고 문서를 분할합니다.
repo_root = "/home/hellocosmos/telegram-bot/langchain/libs"
repo_core = repo_root + "/core/langchain_core"
repo_community = repo_root + "/community/langchain_community"
repo_experimental = repo_root + "/experimental/langchain_experimental"
repo_partners = repo_root + "/partners"
repo_text_splitter = repo_root + "/text_splitters/langchain_text_splitters"
repo_cookbook = repo_root + "/cookbook"

py_documents = []
for path in [repo_core, repo_community, repo_experimental, repo_partners, repo_cookbook]:
    loader = GenericLoader.from_filesystem(
        path, glob="**/*", suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=30),
    )
    py_documents.extend(loader.load())
print(f".py 파일의 개수: {len(py_documents)}")

py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
py_docs = py_splitter.split_documents(py_documents)
print(f"분할된 .py 파일의 개수: {len(py_docs)}")

# MDX 파일을 로드하고 문서를 분할합니다.
root_dir = "/home/hellocosmos/telegram-bot/langchain/"

mdx_documents = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if (file.endswith(".mdx")) and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                mdx_documents.extend(loader.load())
            except Exception:
                pass
print(f".mdx 파일의 개수: {len(mdx_documents)}")

mdx_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
mdx_docs = mdx_splitter.split_documents(mdx_documents)
print(f"분할된 .mdx 파일의 개수: {len(mdx_docs)}")

# Teddy님의 랭체인노트를 로드하고 문서를 분할합니다.
import pandas as pd
from langchain.schema import Document

df = pd.read_csv('data_list_with_content.csv')
df_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
teddy_docs = []
for index, row in df.iterrows():
    if pd.isna(row['content']):
        continue
    chunks = df_splitter.split_text(row['content'])
    for chunk in chunks:
        teddy_docs.append(Document(page_content=chunk, metadata={"title": row['title'], "source": row['source']}))
print(f"분할된 .df 파일 개수: {len(teddy_docs)}")

# PDF 파일로드 및 텍스트 분할합니다. (PDF 파일은 유료구매하셔야 합니다)
pdf_docs = []
document = PyPDFLoader("data/Generative_Al_with_LangChain.pdf").load_and_split()
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
pdf_docs = pdf_splitter.split_documents(document)
print(f"분할된 .pdf 파일의 개수: {len(pdf_docs)}")

# 파이썬 문서, MDX 문서, PDF 문서, 테디노트(Langhchin-KR) 문서를 결합합니다.
combined_documents = teddy_docs + py_docs + mdx_docs + pdf_docs
print(f"총 도큐먼트 개수: {len(combined_documents)}")

# 필요한 임베딩과 캐싱설정을 수행합니다. 
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./cache/")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace=embeddings.model)

# Kiwi Tokenizer를 설정합니다.
from kiwipiepy import Kiwi

kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

# FAISS 클래스를 가져와 검색 모델 인스턴스를 생성합니다.
from langchain_community.vectorstores import FAISS, Chroma

FAISS_DB_INDEX = "langchain_faiss"
db = FAISS.from_documents(combined_documents, cached_embeddings)
db.save_local(folder_path=FAISS_DB_INDEX)
db = FAISS.load_local(FAISS_DB_INDEX, cached_embeddings, allow_dangerous_deserialization=True)
faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# BM25Retriever 클래스를 가져와 검색 모델 인스턴스를 생성합니다.
from langchain_community.retrievers import BM25Retriever

kiwi_bm25_retriever = BM25Retriever.from_documents(combined_documents, preprocess_func=kiwi_tokenize, k=10)

# EnsembleRetriever 클래스를 사용하여 검색 모델을 결합하여 사용합니다.
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[kiwi_bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3], search_type="mmr",
)

# PromptTemplate을 생성하여 프롬프트를 설정합니다.
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
"""
당신은 20년차 AI 개발자이자 파이썬 전문가입니다. 당신의 임무는 주어진 질문에 대하여 최대한 문서의 정보를 활용하여 답변하는 것입니다. 아래의 숫자가 적힌 순서대로 절차를 지켜서 단계적으로 생각하고 진행하세요.

1.주어진 문서에 기반하여 답변하는 경우, "문서를 기반으로 답변드리겠습니다" 라고 시작한다. Python 코드에 대한 상세한 code snippet을 포함해야 하며, 코드 설명에 대한 주석도 작성해주세요. 답변은 자세하게 설명하고, 한글로 작성해 주세요. 주어진 문서에서 정보를 찾아 답변하는 경우 출처(source)를 반드시 표기해야 합니다. 출처는 절대경로로 출력되는 경우 "/home/hellocosmos/telegram-bot"은 생략하고 출력해주세요. 출처가 PDF 파일인 경우 "출처 소스, 페이지"를 형식으로 표기해야 합니다. 메타데이터의 title이 빈공백이 아닌 경우 반드시 "title, source" 형식으로 표기해야 합니다.
2.주어진 문서에 기반해 답변을 찾을 수 없는 경우에는 AI, Langchain 및 파이썬 전문가로써 당신이 알고 있는 관련 지식만을 활용해야 합니다. "문서에 관련 정보가 없지만, 알고 있는 지식을 활용해 답변드리겠습니다."라고 시작한다. 최대한 자세하게 답변하고, 출처는 당신이 알고 있는 출처를 표기해주세요.
3.주어진 문서에 기반해 답변을 찾을 수 없고, Langchain 및 파이썬 전문가로써 당신이 알고 있는 관련 지식만을 활용해도 답변을 찾을 수 없습니다. 이 경우에는 "AI 및 Langchain에 대해 문의해주세요 😂"라고 답변해야 하며, 출처는 생략 해주세요.

#참고문서:
{context}

#질문:
{question}

#답변: 

#출처:
- source1
- source2
- ...
"""
)

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# LLM 모델을 설정합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True, callbacks=[StreamCallback()])

# Retriever 문서를 포맷팅합니다.
def format_docs(documents):
    formatted_list = []
    for doc in documents:
        title = doc.metadata.get('title', '')  # title이 있으면 가져오고, 없으면 빈 문자열로 설정
        formatted_list.append(
            f"<doc><content>{doc.page_content}</content><title>{title}</title><source>{doc.metadata['source']}</source></doc>"
        )
    return formatted_list

# 체인을 생성합니다.
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 날짜와 시간 함수
from datetime import datetime

def get_current_datetime():
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

# 텔레그램 봇 설정 및 핸들러 정의
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.constants import ChatAction, ParseMode

# 텔레그램 봇 토큰을 환경 변수에서 가져옵니다.
bot = telegram.Bot(os.getenv("BOT_TOKEN"))

# RAG 체인을 사용하여 답변을 생성하는 함수
def generate_response(message):
    return rag_chain.invoke(message)

# 텍스트를 Telegram Markdown V2 형식으로 이스케이프하는 함수
def escape_markdown_v2(text):
    escape_chars = r'\`*_{}[]()#+-.!|>='
    return ''.join(['\\' + char if char in escape_chars else char for char in text])

# 응답을 나누어 마크다운 V2 형식으로 포맷팅하는 함수
def split_response(response):
    parts = response.split("```")
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result.append(escape_markdown_v2(part))
        else:
            result.append(f"```{part}```")
    return result

# 봇의 /start 명령에 대한 핸들러 함수
async def start(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="안녕하세요, Langchain 챗봇입니다! 🧑‍💻")

# 텔레그램 메시지에 대한 응답을 생성하는 핸들러 함수
async def answer_openai(update, context):
    message = update.message.text

    # 유저 이름 또는 사용자명 추출
    user = update.message.from_user  # 유저 정보 추출
    user_id = update.message.from_user.id  # 유저 ID 추출
    user_identifier = user.username if user.username else f"{user.first_name} {user.last_name if user.last_name else ''}"
    date_time = get_current_datetime()
    print(f"\n[User_Info] uid: {user_id},  name: {user_identifier}, date: {date_time}")
    print(f"\n[Question] {message}\n[Answer]\n")    

    chat_id = update.effective_chat.id

    loading_message = await context.bot.send_message(chat_id=chat_id, text="처리 중입니다... 🧑‍💻")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        response = generate_response(message)
        print("\n\n")
    except Exception as e:
        await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
        await context.bot.send_message(chat_id=chat_id, text=f"오류가 발생했습니다: {str(e)}")
        return
    
    await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
    
    # 코드 블록으로 감싸고 마크다운 V2 이스케이프 처리
    formatted_response_parts = split_response(response)
    
    # 디버ps깅 출력을 추가하여 이스케이프된 텍스트 확인
    # for part in formatted_response_parts: print(part)
    
    # 메시지가 너무 길면 나누어서 보내기
    for part in formatted_response_parts:
        if part.strip():  # part가 비어있지 않은 경우에만 메시지 전송
            await context.bot.send_message(chat_id=update.effective_chat.id, text=part, parse_mode=ParseMode.MARKDOWN_V2)
# 텔레그램 봇 애플리케이션 생성 및 핸들러 추가

application = Application.builder().token(os.getenv("BOT_TOKEN")).build()
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_openai))

# 봇 실행
application.run_polling()