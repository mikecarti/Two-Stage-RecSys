from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, HTTPException
from app.train import train as train_2stage
from loguru import logger

from app.models.base.popularity import PopModel
from app.models.base.user_history import UserHistoryModel
from app.models.base.lfm import MatrixFactorization
from app.models.base.neighbor import User2User
from app.models.rerank import Reranker
from app.models.ensemble import ModelEnsemble
from app.models.two_stage import TwoStageModel


# TODO:
# 1) load user_id_to_index in MF +
# 2) load self.item_ids+, user_id_to_index+ and self.sparse_data_fitted+ in U2U
#

class PredictRequest(BaseModel):
    user_ids: List[int]


class PredictResponse(BaseModel):
    predictions: List[float]


class TrainResponse(BaseModel):
    success: int


class Inference:
    POP_PATH = "model_files/pop_model.csv"

    HIST_PATH = "model_files/user_history_model.npz"
    HIST_MAP_PATH = "model_files/user_history_map.pkl"

    MF_PATH = "model_files/mf_model.pickle"
    MF_USER_MAP_PATH = "model_files/mf_user_mapping.pkl"
    MF_ITEM_MAP_PATH = "model_files/mf_item_mapping.pkl"
    MF_ITEM_IDS_PATH = "model_files/mf_item_ids.pkl"

    U2U_PATH = "model_files/knn.index"
    U2U_ITEMS_PATH = "model_files/u2u_item_ids.pkl"
    U2U_MAP_PATH = "model_files/u2u_mapping.pkl"
    U2U_SPARSE_PATH = "model_files/u2u_sparse_data_fitted.npz"

    CTB_PATH = "model_files/ctb_model.dill"

    def __init__(self):
        self.model = self._init_models()
        test_pred = self.predict(np.array([1,2,3,4]))
        logger.info(f"Test preditions were made: {test_pred}")

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

    def _init_models(self):
        popularity = PopModel.load(self.POP_PATH)
        history = UserHistoryModel.load(self.HIST_PATH, self.HIST_MAP_PATH)
        lfm = MatrixFactorization.load(self.MF_PATH, self.MF_USER_MAP_PATH, self.MF_ITEM_MAP_PATH, self.MF_ITEM_IDS_PATH)
        neighbor = User2User.load(self.U2U_PATH, self.U2U_MAP_PATH, self.U2U_ITEMS_PATH, self.U2U_SPARSE_PATH)

        base_models = {
            "Popularity Based Model": popularity,
            "User History Model": history,
            "Matrix Factorization": lfm,
            "User2User": neighbor
        }

        ensemble = ModelEnsemble(base_models)
        reranker = Reranker.load(self.CTB_PATH)
        logger.info('all models loaded')
        return TwoStageModel(ensemble, reranker)


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
