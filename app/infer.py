from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, HTTPException
from app.train import train as train_2stage
from loguru import logger

class PredictRequest(BaseModel):
    user_ids: List[int]

class PredictResponse(BaseModel):
    predictions: List[float]

class TrainResponse(BaseModel):
    success: int

class Inference:
    def __init__(self):
        self.model = None

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
        return self.model.predict(user_ids)

inference = Inference()
app = FastAPI()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
