from openai import OpenAI
from dotenv import load_dotenv
import fitz
import tiktoken

load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    # text = ""
    total_text = [doc.load_page(page_num).get_text() for page_num in range(doc.page_count)]
    # for page_num in range(doc.page_count):
    #     page = doc.load_page(page_num)
    #     text += page.get_text()
    return " ".join(total_text)

def chunk_text(text, max_tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk))
    return chunks

if __name__ == "__main__":
    pdf_path = "/Users/parkseohyun/Internship/one-dialog-callus/2103.15348v2.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")

# client = OpenAI()    

# doc = client.embeddings.create(
#     input=s"plit_files", 
#     model="text-embedding-3-small"
#     )


# print(len(doc[0]))