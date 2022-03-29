import logging
import warnings

import tensorflow as tf

try:
    from config import MODEL_NAME, MODELS_DIR, ROOT_DIR
except ImportError:
    from .config import MODEL_NAME, MODELS_DIR, ROOT_DIR

logging.basicConfig(level=logging.INFO)

if not tf.config.list_physical_devices("GPU"):
    # ignore all tensorflow warnings if gpu isn't available
    warnings.filterwarnings("ignore")

model_path = ROOT_DIR / MODELS_DIR / MODEL_NAME
logging.info(f"Loading model from {model_path}")
model = tf.keras.models.load_model(model_path)
