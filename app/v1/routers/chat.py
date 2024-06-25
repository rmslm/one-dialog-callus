from fastapi import APIRouter, status, Depends, HTTPException
from app.config import settings
from langchain_openai import OpenAI as LangchainOpenAI

router_v1 = APIRouter(
    prefix="/chat",
    tags=['chat']
)

@router_v1.get("/")
def test_v1():
    return "chat"


@router_v1.get("/jokes")
def tell_jokes():

    llm = LangchainOpenAI(
        openai_api_key=settings.openai_api_key,
        model_name="gpt-3.5-turbo-instruct")
    reponse = llm.invoke("Tell me a joke")

    return reponse