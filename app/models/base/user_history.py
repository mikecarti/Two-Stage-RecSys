from __future__ import annotations

import pickle
import warnings
from app.models.base.core import BaseModel
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
import scipy

warnings.simplefilter(action='ignore', category=FutureWarning)

class UserHistoryModel(BaseModel):
    def __init__(self, chunk_size=10000):
        self.train_data: csr_matrix = None
        self.user_ids_to_index = {}
        self.chunk_size = chunk_size  # Adjust the chunk size based on available memory and performance
        self.debug = False

    def fit(self, sparse_data: csr_matrix, item_ids: np.array, user_ids: np.array) -> UserHistoryModel:
        """
        This model must collect most popular items for every user
        """
        assert sparse_data.shape == (len(user_ids), len(item_ids))
        # it is itself ranks matrix
        self.train_data = sparse_data
        self.user_ids_to_index = self._create_user_ids_to_user_index_map(user_ids)
        return self

    def predict(self, user_ids: np.array) -> np.ndarray:
        """
        Returns N_USERS X N_ITEMS
        """
        user_indices = np.array([self.user_ids_to_index.get(id_) for id_ in user_ids])
        cold_start_users = np.where(user_indices == None)[0].shape[0]
        logger.warning(f"{cold_start_users} users are not recognizable by UserHistoryModel")
        dtype = np.uint8

        # Deal with Cold Start, Just return 0 vectors
        user_data = np.zeros((len(user_indices), self.train_data.shape[1]), dtype)  # Initialize with zeros

        chunk_size = self.chunk_size

        for i in range(0, len(user_indices), chunk_size):
            start_idx = i
            end_idx = min(start_idx + chunk_size, len(user_indices))

            user_indices_chunk = user_indices[start_idx:end_idx]
            non_null_users_mask = user_indices_chunk != None
            user_indices_chunk = user_indices_chunk[non_null_users_mask]
            user_data_pred_chunk = self.train_data[user_indices_chunk, :].astype(dtype)

            user_data_chunk = user_data[start_idx:end_idx, :]

            if self.debug:
                logger.debug((user_data_chunk[non_null_users_mask, :].shape, user_data_pred_chunk.shape))
                logger.debug((user_data_chunk[non_null_users_mask, :].dtype, user_data_pred_chunk.dtype))

            user_data_chunk[non_null_users_mask, :] = user_data_pred_chunk.toarray()

        return user_data

    @classmethod
    def load(cls, train_matrix_path: str, user_map_path: str):
        logger.info("User history loading...")
        model = UserHistoryModel()
        model.train_data = scipy.sparse.load_npz(train_matrix_path)

        with open(user_map_path, 'rb') as f:
            model.user_ids_to_index = pickle.load(f)
        return model
