import os
from pathlib import Path

ROOT_DIR:Path = Path(os.path.abspath(__file__)).parents[2] #So models and others can be loaded irrespective of init path
MODELS_DIR:Path = Path('models')
MODEL_NAME:Path = Path('NoToxicModel.h5')
DATA_DIR:Path = Path('data')
DATASET:Path = Path('jigsaw-toxic-comment-classification-challenge.zip')
TOKENIZER_NAME:Path = Path('Tokenizer.joblib')
SEED:int = 45