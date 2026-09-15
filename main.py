from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "SkinGuru agent is running"}