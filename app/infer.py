from typing import List
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI
from train import train


class PredictRequest(BaseModel):
    user_ids: np.array


class PredictResponse(BaseModel):
    predictions: np.ndarray


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
def predict(request: PredictRequest):
    predictions = inference.predict(np.array(request.user_ids))
    return PredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
