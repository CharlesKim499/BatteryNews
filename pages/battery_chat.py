import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain_teddynote import logging
from langchain.schema import Document
import faiss
import hmac

import pandas as pd
from dotenv import load_dotenv

import os

load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] 배터리 동향 전문가 RAG")



# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.makedirs(".cache")

if not os.path.exists(".cache/files"):
    os.makedirs(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.makedirs(".cache/embeddings")

st.title("배터리 동향 전문가 QA💬 ")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "xlsx", "xls"])
    # 파일 타입 선택
    # uploaded_file = st.selectbox("파일 타입 선택", ["pdf", "excel"], index=0)

    # # 파일 업로드 되었을 때
    # if uploaded_file:
    #     if selected_file_type == "pdf":
    #         # 작업시간이 오래 걸릴 예정
    #         retriever = embed_file(uploaded_file)
    #         chain = create_chain(retriever, model_name=selected_model)
    #         st.session_state["chain"] = chain
    #     elif selected_file_type == "excel":
    #         # 엑셀 파일 처리 코드 추가
    #         # 작업시간이 오래 걸릴 예정
    #         retriever = embed_excel_file(uploaded_file)
    #         chain = create_chain(retriever, model_name=selected_model)
    #         st.session_state["chain"] = chain
    # 모델 선택 박스
    selected_model = st.selectbox(
        "LLM 모델 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 엑셀 파일을 행 단위로 불러오기
def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df


# 파일을 캐시 저장(시간이 오래걸리는 작업)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 디버깅을 위해 파일 타입과 이름 출력
    print(f"File type: {file.type}")
    print(f"File name: {file.name}")

    if file.type == "application/pdf":
        # 단계 1: 문서 로드(Load Documents)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        split_documents = text_splitter.split_documents(docs)

        # 단계 3: 임베딩(Embedding) 생성
        embeddings = OpenAIEmbeddings()

        # 단계 4: DB 생성(Create DB) 및 저장
        # 벡터스토어를 생성합니다.
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )

        # 단계 5: 검색기(Retriever) 생성
        # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
        retriever = vectorstore.as_retriever()
    # 엑셀 파일인 경우
    elif (
        file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        df = load_excel(file_path)

        # 각 행을 개별 문서로 처리
        docs = []
        for index, row in df.iterrows():
            document = row.to_string(index=False)
            docs.append(Document(page_content=document))

        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        split_documents = text_splitter.split_documents(docs)

        # 단계 3: 임베딩(Embedding) 생성
        embeddings = OpenAIEmbeddings()

        # 단계 4: DB 생성(Create DB) 및 저장
        embedded_documents = embeddings.embed_documents(
            [doc.page_content for doc in split_documents]
        )

        # 벡터 DB에 저장
        docstore = InMemoryDocstore()
        dimension = len(embedded_documents[0])  # 임베딩 차원 수
        index = faiss.IndexFlatL2(dimension)  # FAISS 인덱스 생성
        vector_db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
        )
        vector_db.add_documents(split_documents)

        # 단계 5: 검색기(Retriever) 생성
        # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
        retriever = vector_db.as_retriever()
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    if docs is None:
        raise ValueError("문서가 로드되지 않았습니다.")

    return retriever


# Retriever를 합차는 과정
def foramt_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def create_chain(retriever, model_name="gpt-4o"):
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("./prompts/battery_chat.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever | foramt_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 출력 파서
    outputp_parser = StrOutputParser()

    return chain


# 파일 업로드 되었을 때
if uploaded_file:
    # 작업시간이 오래 걸릴 예정
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면 :
if clear_btn:
    # 대화 초기화
    st.session_state["messages"] = []

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약 사용자 입력이 들어오면...
if user_input:

    # chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)

        # AI의 답변
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            containner = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                containner.markdown(ai_answer)

        # 대화 기록에 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일 업로드 경고
        warning_msg.error("파일을 업로드 해주세요.")
