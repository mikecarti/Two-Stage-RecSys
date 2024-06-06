import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.dtype import PredictRequest, PredictResponse, TrainResponse, ItemsResponse
from app.inference import Inference

log_level = "DEBUG"
logger.add("logs/debug.log", level=log_level, colorize=False, backtrace=True, diagnose=True)

app = FastAPI()
inference = Inference()


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    success = inference.train_inference()
    return TrainResponse(success=1 if success else 0)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictions = inference.predict(np.array(request.user_ids))
    if predictions is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet.")
    return PredictResponse(predictions=predictions.tolist())


@app.get("/get_items", response_model=ItemsResponse)
def get_items() -> ItemsResponse:
    return ItemsResponse(items=inference.get_item_ids())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
