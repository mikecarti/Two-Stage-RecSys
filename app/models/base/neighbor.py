# as NN already uses Approximate Nearest Neighbour, we will utilize this
import warnings
import faiss
import numpy as np
import pickle
from typing import Iterable
from loguru import logger
from scipy.sparse import csr_matrix

from app.models.base.core import BaseModel
from app.utils import user_purchases_from_sparse_optimized

warnings.simplefilter(action='ignore', category=FutureWarning)



class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices


class User2User(BaseModel):
    def __init__(self, n_neighbors=3, metric="", leaf_size=-1):
        self.model = FaissKNeighbors(k=n_neighbors)
        self.sparse_data_fitted = None
        self.item_ids = None
        self.user_ids = None
        self.user_id_to_index = {}
        self.user_embeddings = None

    def fit(self, sparse_data: csr_matrix, item_ids: Iterable, user_ids: Iterable, user_embeddings: np.ndarray):
        assert sparse_data.shape == (len(user_ids), len(item_ids))

        self.sparse_data_fitted = sparse_data
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.user_id_to_index = self._create_user_ids_to_user_index_map(user_ids)

        # add zero row for unknown users (cold start)
        zero_row = np.zeros((1, user_embeddings.shape[1]))
        self.user_embeddings = np.vstack((user_embeddings, zero_row))

        logger.info(f"{user_embeddings.shape[0]} users to fit left...")
        self.model.fit(user_embeddings)
        logger.info(f"Done.")
        return self

    def predict(self, user_ids: Iterable):
        # -1 will index just a zero row for unknown users
        user_indices = np.array([self.user_id_to_index.get(uid, -1) for uid in user_ids])
        user_embeddings = self.user_embeddings[user_indices]

        logger.info(f"Running neighbors search")
        logger.debug(f"embedding shape: {user_embeddings.shape}, \nUsers for prediction: {user_ids.shape}")
        similar_users_index_matrix = self.model.kneighbors(user_embeddings)

        all_users_ranks = np.zeros((len(user_ids), len(self.item_ids)), dtype=np.float16)
        rank_positions = 1 / (np.arange(len(similar_users_index_matrix[0])) + 1)

        for i, similar_users in enumerate(similar_users_index_matrix):
            if i % 100000 == 0:
                logger.info(f"{i} Users processed out of {similar_users_index_matrix.shape[0]}")
            ranks_for_user = self._find_and_rank_neighbors(similar_users, rank_positions)
            all_users_ranks[i, :] = ranks_for_user
        logger.info("Users predicted")
        return all_users_ranks

    def save(self):
        path = "knnpickle_file"
        knn_pickle = open('knnpickle_file', 'wb')
        pickle.dump(self.model, knn_pickle)
        knn_pickle.close()
        logger.info(f"KNN Model was saved with name {path}")

    def _find_and_rank_neighbors(self, similar_users: np.array, rank_positions: np.array):
        ranks_for_users = np.zeros((len(similar_users), len(self.item_ids)))

        for i, similar_user_index in enumerate(similar_users):
            similar_user_item_indices, n_purchases = self._get_user_items_and_ranks(similar_user_index)
            ranks_for_users[i, similar_user_item_indices] = n_purchases * rank_positions[i]

        return np.sum(ranks_for_users, axis=0)

    def _get_user_items_and_ranks(self, user_index: int):
        similar_user_item_indices, n_purchases = user_purchases_from_sparse_optimized(self.sparse_data_fitted,
                                                                                      self.item_ids, user_index,
                                                                                      return_index=True)
        return similar_user_item_indices, n_purchases.flatten()

# # Wall time 3:40
# %time knn_model = User2User().fit(debug_set_csr, item_ids, user_ids, user_emb)
# %time ranks = knn_model.predict(user_ids[:10])
# del knn_model, ranks