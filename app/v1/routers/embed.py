from fastapi import APIRouter, status, Depends, HTTPException
from app.v1.utils import extract_text_from_pdf, chunk_text, get_embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

from app.config import settings

router_v1 = APIRouter(
    prefix="/embed",
    tags=['embed']
)

@router_v1.get("/")
def test_v1():
    return "embed"

@router_v1.get("/pdf")
def gpt_answer():
    pdf_path = "/Users/parkseohyun/Internship/one-dialog-callus/2103.15348v2.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    response = {f"chunk_{i}" : embedding for i, embedding in enumerate(embeddings)}
    
    return response

@router_v1.get("/langchain")
def embed_with_langchain():
    loader = PyPDFLoader("/Users/parkseohyun/Internship/one-dialog-callus/data/2103.15348v2.pdf")
    pages = loader.load()
    
    texts = []
    for page in pages:
        texts.append(page.page_content)

    full_text = " ".join(texts)
    doc =  Document(page_content=full_text)
    # print(full_text)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(doc)
    # docs = text_splitter.split_documents(pages)
    print(docs)

    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small", 
    #     api_key=settings.openai_api_key
    #     )
    
    # query_result = embeddings.embed_query(docs)

    return "o"