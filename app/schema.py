from pydantic import BaseModel, EmailStr, Field, validator, json
from typing import Optional


class ReadPDFBase(BaseModel):
    pdf_path:str

class ReadPDFIn(ReadPDFBase):
    threads : Optional[int] = 1
    vision : Optional[bool] = False

class ReadPDFOut(ReadPDFBase):
    pass