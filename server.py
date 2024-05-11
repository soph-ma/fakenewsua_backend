from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel
from typing import Optional
import uvicorn

from src.classifier import NewsClassifier

classifier = NewsClassifier()
app = FastAPI()
handler = Mangum(app)

class Input(BaseModel): 
    text: str 
    title: Optional[str] = None 

class ClassificationResult(BaseModel): 
    label: str

@app.get("/")
async def root() -> JSONResponse:
    """Checking connection"""
    return {"connection": True}

@app.post("/classify")
async def classify(input: Input) -> ClassificationResult:
    """News classification"""
    label = classifier.classify_news(text=input.text, title=input.title)
    return ClassificationResult(label=label)

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8085)