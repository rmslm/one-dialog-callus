from fastapi import APIRouter, status, Depends, HTTPException


router_v1 = APIRouter(
    prefix="/embed",
    tags=['embed']
)

@router_v1.get("/")
def test_v1():
    return "embed"