from typing import List
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI
from app.train import train


class PredictRequest(BaseModel):
    user_ids: list[int]


class PredictResponse(BaseModel):
    predictions: list[float]


class Inference:
    def __init__(self):
        self.model = train()

    def predict(self, user_ids: np.ndarray) -> np.ndarray:
        """
        :param user_ids: np.array of dimension 1
        :return: np.ndarray of dimension len(user_ids) x len(item_ids
        """
        return self.model.predict(user_ids)


inference = Inference()
app = FastAPI()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictions = inference.predict(np.array(request.user_ids))
    return PredictResponse(predictions=predictions.tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
