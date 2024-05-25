from __future__ import annotations
import warnings
import numpy as np
from typing import Iterable
from scipy.sparse import csr_matrix

from models.core import BaseModel

warnings.simplefilter(action='ignore', category=FutureWarning)


class PopModel(BaseModel):
    def __init__(self):
        self.ranks = None
        self.n_items = None

    def fit(self, sparse_data: csr_matrix, item_ids: np.array, user_ids: np.array) -> PopModel:
        _ = user_ids
        assert sparse_data.shape == (len(user_ids), len(item_ids))
        self.n_items = len(item_ids)

        total_purchases = sparse_data.sum(axis=0)
        popularity_sorted_args = total_purchases.argsort()[:, ::-1]
        popularity_sorted_index = np.squeeze(np.asarray(popularity_sorted_args))
        total_purchases = np.squeeze(np.asarray(total_purchases))

        self.ranks = n_purchases = total_purchases[popularity_sorted_index].astype(np.ushort)

        return self

    def predict(self, user_ids: np.array | Iterable) -> np.ndarray:
        """
        Here we get same list of items for any user_ids, because it is Baseline.
        Returns N_USERS X N_ITEMS
        """
        n_users = len(user_ids)
        n_items = self.n_items

        ranks = self.ranks
        ranks_broadcasted = ranks[np.newaxis, :]

        assert ranks_broadcasted.shape == (1, n_items)

        return np.broadcast_to(ranks_broadcasted, (n_users, n_items))