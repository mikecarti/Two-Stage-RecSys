from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, HTTPException
from app.train import train as train_2stage
from loguru import logger
from app.models.two_stage import TwoStageModel


class PredictRequest(BaseModel):
    user_ids: List[int]


class PredictResponse(BaseModel):
    predictions: List[float]


class TrainResponse(BaseModel):
    success: int

class ItemsResponse(BaseModel):
    items: List[int]



class Inference:
    def __init__(self):
        self.model = TwoStageModel.load()
        self.test_model()

    def test_model(self):
        test_pred = self.predict(np.array([1, 2, 3, 4]))
        logger.info(f"Test preditions were made: {test_pred}")
        test_items = self.get_item_ids()
        logger.info(f"Test item ids: {test_items}")

    def train_inference(self) -> bool:
        try:
            self.model = train_2stage()
            return True
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            self.model = None
            return False

    def predict(self, user_ids: np.ndarray) -> Optional[np.ndarray]:
        """
        :param user_ids: np.array of dimension 1
        :return: np.ndarray of dimension len(user_ids) x len(item_ids)
        """
        if self.model is None:
            return None
        logger.info(f"Predicting")
        return self.model.predict(user_ids)

    def get_item_ids(self):
        return self.model.ensemble.get_item_ids()


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

@app.post("/get_items", response_model=ItemsResponse)
def get_items() -> ItemsResponse:
    return ItemsResponse(items=inference.get_item_ids())

# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
