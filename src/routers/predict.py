import logging
import time
from typing import Union

from fastapi import APIRouter
from keras.preprocessing.sequence import pad_sequences

from ..pipeline.preprocess import clean_text, tokenizer
from ..schemas import PredictionResponse, Text
from ..utils.load_model import model

router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post("", response_model=PredictionResponse)
def predict(text: Text):
    start_time = time.time()
    cleaned_text = clean_text(text.text)
    text_tokens = tokenizer.texts_to_sequences([cleaned_text])
    print("Text Tokens: ", text_tokens)
    text_tokens_padded = pad_sequences(text_tokens, maxlen=200)
    preds = model.predict(text_tokens_padded)

    def round_predictions_up(preds: list[list[float]]) -> list[float]:
        """Round Model prediction to 2 d.p"""
        decimals = 2
        return list(map(lambda x: round(x, decimals), preds[0]))

    def get_targets(pred_probs: list[float]) -> Union[list[str], str]:
        """Collect targets and based on THRESHOLD probability, return correspoding targets."""
        TARGETS: list = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        THRESHOLD: float = 0.4
        targets = [
            TARGETS[n]
            for n, pred_prob in enumerate(pred_probs)
            if pred_prob > THRESHOLD
        ]
        if targets:
            return targets
        else:
            return "Not Toxic"

    new_preds = round_predictions_up(preds)
    end_time = time.time()
    total_inference_time = end_time - start_time

    print("Prediction: ", preds)
    print("Prediction Probs: ", new_preds)
    print("Preds:", get_targets(new_preds))

    logging.info("Inference time is {} seconds".format(total_inference_time))

    target = get_targets(new_preds)
    if target == "Not Toxic":
        return {
            "predictions": target
        }
    else:
        return {
            "predictions": target,
            "number_of_labels": len(target),
        }
