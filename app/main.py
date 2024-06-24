from fastapi import FastAPI, status, Request

app = FastAPI()

@app.get("/", status_code=status.HTTP_201_CREATED)
def root():
    return {"message": "Hello callus"}