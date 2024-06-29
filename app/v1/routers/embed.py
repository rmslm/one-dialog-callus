from fastapi import APIRouter, status, Depends, HTTPException
from app.schema import ReadPDFIn
from app.v1.utils import PDF2Images


router_v1 = APIRouter(
    prefix="/embed",
    tags=['embed']
)

@router_v1.get("/")
def test_v1():
    return "embed"

@router_v1.post("/pdf")
def upload_to_vdb(payload:ReadPDFIn):
    """Creates embeddings for the pdf""" 

    if payload.threads > 1:
        print("start multprocessing")

    if payload.vision :
        print("run vision tools")

    return payload