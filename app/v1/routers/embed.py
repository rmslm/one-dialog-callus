from fastapi import APIRouter, status, Depends, HTTPException
from app.v1.utils import extract_text_from_pdf, chunk_text, get_embeddings

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