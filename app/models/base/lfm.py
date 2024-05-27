from __future__ import annotations
import warnings
from lightfm import LightFM
from lightfm.evaluation import precision_at_k as lightfm_precision_at_k
from app.models.base.core import BaseModel
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from loguru import logger
from scipy.sparse import csr_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)


class MatrixFactorization(BaseModel):
    def __init__(self, model_params={}) -> MatrixFactorization:
        # Используем warp так как это лучший лосс для кейсов, когда только положительные взаимодействия в датасете
        self.model = LightFM(**model_params)
        self.precision = lightfm_precision_at_k
        self.item_ids_fitted = None
        # num_threads=4 потому что у Kaggle 4 ядер.
        self.num_threads = 4

    def fit(self, sparse_data: coo_matrix, item_ids: np.array, user_ids: np.array, epochs=1) -> MatrixFactorization:
        assert sparse_data.shape == (len(user_ids), len(item_ids))

        self.user_id_to_index = self._create_user_ids_to_user_index_map(user_ids)
        self.item_id_to_index = self._create_item_ids_to_item_index_map(item_ids)
        self.item_ids_fitted = item_ids

        self.model.fit(sparse_data, epochs=epochs, num_threads=self.num_threads)
        return self

    def predict(self, user_ids: np.array) -> np.ndarray:
        """
        Returns np.ndarray of shape N_USERS X N_ITEMS
        """
        # Lets set 0 as unknown id (Cold Start)
        user_indices = np.array([self.user_id_to_index.get(uid, 0) for uid in user_ids])
        item_indices = np.array([self.item_id_to_index.get(iid, 0) for iid in self.item_ids_fitted])

        n_users = len(user_indices)
        n_items = len(item_indices)

        user_indices = user_indices.astype(int)
        users_flat = user_indices.reshape(-1, 1)
        repeated_user_indices = np.broadcast_to(users_flat, (users_flat.shape[0], n_items))
        predictions = np.empty((n_users, n_items), dtype=np.float16)

        for u in range(n_users):
            # [u_1, u_1,......]
            # [...............]
            # [u_n, u_n, .....]

            if u % 100_000 == 0:
                logger.info(f"{u} users were predicted")

            user_indices_repeated = repeated_user_indices[u, :]
            ranks_predicted = self.model.predict(user_indices_repeated, item_indices)
            predictions[u, :] = ranks_predicted

        return predictions

    def save(self):
        with open('savefile.pickle', 'wb') as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def get_latent_users(self):
        return self.model.user_embeddings

    def evaluate(self, test_data: csr_matrix, k=3):
        prec = self.precision(self.model, test_data, k=k)
        print(f"Train precision at {k}: %.2f" % prec.mean())
        return prec
