import logging
import warnings

from fastapi import FastAPI
import tensorflow as tf

from .routers import predict

api = FastAPI()
api.include_router(predict.router)

logging.basicConfig(level=logging.INFO)

if not tf.config.list_physical_devices("GPU"):
    # ignore all tensorflow warnings if gpu isn't available
    warnings.filterwarnings("ignore")


@api.get("/")
def root():
    return {"message": "No-Toxic API"}
