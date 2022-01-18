import os
import sys
import logging
import time
import warnings
from typing import Union

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from pydantic import BaseModel
dirname = os.path.dirname(__file__)
print(f'DIR: {dirname}')
print(f'BEFORE: {sys.path}')
sys.path.append(dirname)
print(f'AFTER: {sys.path}')
from pipeline.preprocess import clean_text, tokenizer
from utils.load_model import model


api = FastAPI()
logging.basicConfig(level=logging.INFO)

if not tf.config.list_physical_devices('GPU'):
    # ignore all tensorflow warnings if gpu isn't available
    warnings.filterwarnings('ignore')

class Text(BaseModel):
    text: str

class Prediction(BaseModel):
    targts: list

@api.get('/')
def root():
    return 'No-Toxic API'

@api.post('/predict')
def predict(text: Text):
    start_time = time.time()
    cleaned_text = clean_text(text.text)
    text_tokens = tokenizer.texts_to_sequences([cleaned_text])
    print('Text Tokens: ', text_tokens)
    text_tokens_padded = pad_sequences(text_tokens, maxlen=200)
    preds = model.predict(text_tokens_padded)

    def round_predictions_up(preds: list[list[float]]) -> list[float]:
        """Round Model prediction to 2 d.p"""
        decimals = 2
        return list(map(lambda x: round(x, decimals), preds[0]))
    
    def get_targets(pred_probs: list[float]) -> Union[list[str], str]:
        """Collect targets and based on THRESHOLD probability, return correspoding targets."""
        TARGETS:list = ['toxic',
        'severe_toxic', 
        'obscene', 
        'threat',
        'insult',
        'identity_hate']
        THRESHOLD:float = 0.4
        targets = [TARGETS[n] for n, pred_prob in enumerate(pred_probs) if pred_prob > THRESHOLD]
        if targets:
            return targets
        else:
            return 'Not Toxic'
        

    new_preds = round_predictions_up(preds)
    end_time = time.time()
    total_inference_time = end_time - start_time
    print('Prediction: ', preds)
    print('Prediction Probs: ', new_preds)
    print('Preds:', get_targets(new_preds))
    logging.info('Inference time is {} seconds'.format(total_inference_time))
    return {
        'tokens': text_tokens,
        'predictions': get_targets(new_preds)
    }
