from fastapi import APIRouter, status, Depends, HTTPException
from app.v1.utils_sample import extract_text_from_pdf, chunk_text, get_embeddings

from app.config import settings

from langchain_chroma import Chroma 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document

import pymupdf4llm


router_v1 = APIRouter(
    prefix="/embed",
    tags=['embed']
)

@router_v1.get("/")
def test_v1():
    return "embed"


@router_v1.get("/pdf")
def gpt_answer():
    pdf_path = "/Users/ramisalem/Desktop/github/one-dialog-callus/data/2103.15348v2.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    response = {f"chunk_{i}" : embedding for i, embedding in enumerate(embeddings)}
    
    return response



@router_v1.get("/langchain")
def embed_with_langchain(query):
    print(query)

    loader = TextLoader("./data/sample.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(docs, len(docs))

    # embedding_function = OpenAIEmbeddings(
    #     openai_api_key=settings.openai_api_key,
    #     model="text-embedding-3-small"
    # )

    # db = Chroma.from_documents(docs, embedding_function)

    # docs = db.similarity_search(query)

    # print(len(docs))

    # retriever = db.as_retriever()

    # response = retriever.invoke(query)[0]


    return {"response" : response }



@router_v1.get("/pdf/langchain")
def embed_with_langchain_pdf():
    # loader = PyMuPDFLoader("/Users/ramisalem/Desktop/github/one-dialog-callus/data/2103.15348v2.pdf")
    # pages = loader.load()
    
    # texts = ""
    # for i, page in enumerate(pages):
    #     if i == 0:
    #         continue

    #     if i==len(pages)-1:
    #         with open('./data/pdf.txt', 'w') as file:
    #             file.write(texts)

    #     texts += page.page_content
    
    import fitz

    # pdf_document = fitz.open("/Users/ramisalem/Desktop/github/one-dialog-callus/data/Itzehoer Connect Unterlagen/Marketing GeschÑftsberichte aller Itzehoer Gesellschaften/geschaeftsbericht-konzern-2022.pdf")
    md_text = pymupdf4llm.to_markdown("/Users/ramisalem/Desktop/github/one-dialog-callus/data/2103.15348v2.pdf")

    # cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # cleaned_text = cleaned_text.replace("\n", ". ")
    # print(cleaned_text)

    with open('./data/pdf.md', 'w') as file:
        file.write(md_text)

    full_text_doc = Document(page_content=md_text)

    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = text_splitter.split_documents([full_text_doc])
    print(docs, len(docs))

    # embedding_function = OpenAIEmbeddings(
    #     openai_api_key=settings.openai_api_key,
    #     model="text-embedding-3-small"
    # )

    # # print(full_text)

    # full_text_doc =  Document(page_content=texts)

    # text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    # docs = text_splitter.split_documents([full_text_doc])

    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small", 
    #     api_key=settings.openai_api_key
    #     )
    
    # query_result = embeddings.embed_query(docs)

    return "o"