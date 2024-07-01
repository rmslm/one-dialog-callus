from fastapi import APIRouter, status, Depends, HTTPException
from app.schema import ReadPDFIn
from app.v1.utils import PDF2Images

from app.v1.vision.openai import VisionContextOpenAI
from app.v1.enhancer.openai import EnhanceContextOpenAI

import pymupdf4llm
import fitz
import os
import logging


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

    # TODO : check if pdf 

    pdf_path = payload.pdf_path.replace(" ", "_")

    file_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path).split(".pdf")[0]

    print(file_dir, file_name)

    if payload.threads > 1:
        raise NotImplementedError("multprocessing not implemented.")

    # get the text in markdown
    pdf_document = fitz.open(payload.pdf_path)
    page_numbers = pdf_document.page_count

    # for page in range(page_numbers):
    md_text = pymupdf4llm.to_markdown(pdf_document, pages=[1])

    if payload.vision :
        print("run vision tools")

        # should return all the path to the images
        # image_paths = PDF2Images(
        #     pdf_path=payload.pdf_path, 
        #     batch_size=2,
        #     filename=file_name
        # ).run()

        # for path in image_paths: 

        resp = VisionContextOpenAI(
            image_path="/Users/ramisalem/Desktop/github/one-dialog-callus/app/v1/tmp/batch_images/transparenzinformationen-kapitalanlagen-und-verguetungspolitik_batch_0_page_2.png"
        ).run()

        print(resp)

        print("start enhance") 
        chunks = EnhanceContextOpenAI(text=[md_text, resp]).run()

    # start the embedding

    return payload