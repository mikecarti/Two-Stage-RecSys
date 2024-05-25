from __future__ import annotations

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np

from typing import Iterable

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix


class BaseModel(ABC):
    @abstractmethod
    def fit(self, sparse_data: csr_matrix, item_ids: np.array, user_ids: np.array) -> 'BaseModel':
        """
    Fit the model to the data.

    Parameters:
    - sparse_data: A sparse matrix representing user-item interactions (number of purchases).
    - item_ids: An array of item IDs.
    - user_ids: An array of user IDs.

    Returns:
    A reference to the fitted model instance.
    """
        pass

    @abstractmethod
    def predict(self, user_ids: np.array) -> np.array:
        """
    Predict the ranking scores for a given array of user IDs.

    Parameters:
    - user_ids: An array of user IDs for whom predictions are to be made.

    Returns:
    N_USERS x N_ITEMS Rank Array
    """
        pass

    def _create_user_ids_to_user_index_map(self, user_ids: Iterable):
        return self._ids_to_index_map(user_ids)

    def _create_item_ids_to_item_index_map(self, item_ids: Iterable):
        return self._ids_to_index_map(item_ids)

    def _ids_to_index_map(self, ids: Iterable):
        n_ids = len(ids)
        return dict(zip(ids, range(n_ids)))