from typing import Optional

import numpy as np
from loguru import logger

from app.models.two_stage import TwoStageModel
from app.train import train as train_2stage


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
