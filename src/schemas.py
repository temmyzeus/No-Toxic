from pydantic import BaseModel


class Text(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    tokens: list
    predictions: list
    number_of_labels: int
