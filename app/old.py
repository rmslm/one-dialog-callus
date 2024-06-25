from fastapi import FastAPI, status, Request

from openai import OpenAI
from langchain_openai import OpenAI as LangchainOpenAI
import fitz
import tiktoken

app = FastAPI()
client = OpenAI()

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


@app.get("/", status_code=status.HTTP_201_CREATED)
def root():
    return {"message": "Hello callus"}


@app.get("/openai", status_code=status.HTTP_201_CREATED)
def gpt_answer():
    pdf_path = "/Users/parkseohyun/Internship/one-dialog-callus/2103.15348v2.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    response = {f"chunk_{i}" : embedding for i, embedding in enumerate(embeddings)}

    return response

@app.get("/langchain", status_code=status.HTTP_201_CREATED)
def langchain_response():
    
    llm = LangchainOpenAI(
        model_name="gpt-3.5-turbo-instruct")
    reponse = llm.predict("Tell me a joke")

    return reponse