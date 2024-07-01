# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# load the document and split it into chunks
loader = TextLoader("./sample.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the openai embedding function
embedding_function = OpenAIEmbeddings(
    openai_api_key="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",
    model="text-embedding-3-small"
)

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)