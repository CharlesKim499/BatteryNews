from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
import pandas as pd
import faiss


def create_retriever_pdf(file_path):
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def create_retriever_excel_csv(file_path, file_type):
    if file_type == "text/csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # 각 행을 개별 문서로 처리
    docs = []
    for index, row in df.iterrows():
        document = row.to_string(index=False)
        docs.append(Document(page_content=document))

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
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
    return retriever
