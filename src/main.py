import logging
import warnings

from fastapi import FastAPI

from .routers import predict

api = FastAPI()
api.include_router(predict.router)

logging.basicConfig(level=logging.INFO)

@api.get("/")
def root():
    return {"message": "No-Toxic API"}
