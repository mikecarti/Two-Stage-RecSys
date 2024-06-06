from typing import List
from pydantic import BaseModel

class PredictRequest(BaseModel):
    user_ids: List[int]


class PredictResponse(BaseModel):
    predictions: List[List[float]]


class TrainResponse(BaseModel):
    success: int


class ItemsResponse(BaseModel):
    items: List[int]
