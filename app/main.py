from fastapi import FastAPI, status, Request

from .v1.routers import chat as chat_v1
from .v1.routers import embed as embed_v1

app = FastAPI()

app.include_router(chat_v1.router_v1, prefix="/v1")
app.include_router(embed_v1.router_v1, prefix="/v1")

@app.get("/", status_code=status.HTTP_201_CREATED)
def root():
    return {"message": "Hello ai"}