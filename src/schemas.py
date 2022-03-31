from pydantic import BaseModel
from typing import Union

class Text(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predictions: Union[list, str]
    number_of_labels: int = 0
