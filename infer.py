from typing import Iterable

import numpy as np

from train import train


class Inference:
    def __init__(self):
        self.model = train()

    def predict(self, user_ids: Iterable) -> np.ndarray:
        """
        :param user_ids: np.array of dimension 1
        :return: np.ndarray of dimension len(user_ids) x len(item_ids
        """
        return self.model.predict(user_ids)
