
from typing import Dict
from torch import FloatTensor

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    transfer: str
    confidence: float
    attention_matrix: list


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    sentiment, confidence, probabilities, attention_matrix = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities, attention_matrix= attention_matrix
    )
    
@app.post("/train")
def train( model: Model = Depends(get_model)):
    model.train_conf()
    return {'status': 'ok'}
