from fastapi import APIRouter, status, Depends, HTTPException
from app.schema import ReadPDFIn
from app.v1.utils import PDF2Images
from app.v1.vision.openai import VisionOpenAI

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
        raise NotImplementedError("multprocessing not implemented.")

    if payload.vision :
        print("run vision tools")

        # PDF2Images(pdf_path=payload.pdf_path, batch_size=2).run()

        resp = VisionOpenAI(image_path="/Users/ramisalem/Desktop/github/one-dialog-callus/app/v1/tmp/batch_images/batch_0_page_1.png").run()

        print(resp)


    return payload