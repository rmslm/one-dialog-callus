from openai import OpenAI
import fitz
import tiktoken
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    total_text = [
        doc.load_page(page_num).get_text() for page_num in range(doc.page_count)]

    return " ".join(total_text)

def chunk_text(text, max_tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk))
    return chunks

def get_embeddings(chunks, model="text-embedding-3-small"):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model=model)
        embeddings.append(response.data[0].embedding)
    return embeddings