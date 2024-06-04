from __future__ import annotations

import warnings

from app.models.base.lfm import MatrixFactorization
from app.models.base.neighbor import User2User
from app.models.base.popularity import PopModel
from app.models.base.user_history import UserHistoryModel
from app.utils import all_user_purchases_from_sparse, mapk
import numpy as np
import time
from scipy.sparse import coo_matrix
from typing import Iterable, Dict
from loguru import logger
from scipy.sparse import csr_matrix
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)


class ModelEnsemble:
    DEFAULT_MODELS = {
        "Popularity Based Model": PopModel(),
        "User History Model": UserHistoryModel(chunk_size=100_000),
        "Matrix Factorization": MatrixFactorization(),
        "User2User": User2User()
    }

    def __init__(self, models: Dict = None, k: int = 6, evaluate_mf=True):
        self.top_k_metric = k
        self.evaluate_mf = evaluate_mf

        self.models = models if models else self.DEFAULT_MODELS.copy()

        for model_name in self.DEFAULT_MODELS:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} is not passed to initializer")

        self.pop_model: PopModel = self.models.get("Popularity Based Model")
        self.history_model: UserHistoryModel = self.models.get("User History Model")
        self.mf_model: MatrixFactorization = self.models.get("Matrix Factorization")
        self.knn_model: User2User = self.models.get("User2User")

        if self.knn_model.user_embeddings is None:
            self.knn_model.user_embeddings = np.array(self.mf_model.get_latent_users())

    def fit(self, data: coo_matrix, user_ids: Iterable, item_ids: Iterable) -> ModelEnsemble:
        simple_fit_models = [(k, v) for k, v in self.models.items() if k != "User2User"]
        assert data.shape == (len(user_ids), len(item_ids))

        for name, model in simple_fit_models:
            logger.info(f"Training model {name}")
            model.fit(data, user_ids=user_ids, item_ids=item_ids)

        if self.mf_model and self.knn_model:
            user_embeddings = self.mf_model.get_latent_users()
            logger.info(f"Training model User2User")
            self.knn_model.fit(data, item_ids, user_ids, user_embeddings)
        else:
            logger.warning(f"Model KNN is not inferencing because it was not passed as a model")
        return self

    def predict(self, user_ids: Iterable, val: csr_matrix = None, train_item_ids: Iterable = None,
                val_item_ids: Iterable = None) -> Dict[str, np.ndarray]:
        calculate_score = all([var is not None for var in [train_item_ids, val_item_ids, val]])
        # predictions = {"Model": ranks1, "Model2": ranks2, ...}
        predictions = {model_name: None for model_name in self.models.keys()}

        for name, model in self.models.items():
            logger.info(f"Inferencing model {name}")
            start_time = time.time()
            ranks_pred = model.predict(user_ids)
            end_time = time.time()
            prediction_time = end_time - start_time
            logger.info(f"Time taken to predict with {name}: {prediction_time / 60 :.2f} minutes")

            # based on model class
            if isinstance(model, MatrixFactorization) and not self.evaluate_mf:
                continue

            if calculate_score:
                start_time = time.time()

                score = self.evaluate(ranks_pred, val, train_item_ids, val_item_ids,
                                      k=self.top_k_metric)
                end_time = time.time()
                evaluation_time = end_time - start_time
                logger.info(
                    f"Score of model {name} for MAP@{self.top_k_metric}: {score} (Calculation took {evaluation_time / 60:.5f} minutes)")

            predictions[name] = ranks_pred
            del ranks_pred
            gc.collect()

        return predictions

    def get_item_ids(self):
        item_ids_mf = self.mf_model.item_ids_fitted
        item_ids_knn = self.knn_model.item_ids
        assert np.array_equal(item_ids_mf, item_ids_knn)
        return item_ids_mf

    @staticmethod
    def evaluate(ranks_pred: np.ndarray, ranks_true: np.ndarray,
                 train_item_ids: Iterable, val_item_ids: Iterable,
                 k) -> float:
        logger.debug(f"Shape for evaluation: {ranks_pred.shape}, dtype: {ranks_pred.dtype}")
        logger.info(f"Evaluating")
        ranks_list = all_user_purchases_from_sparse(ranks_pred, train_item_ids, top_k=k)
        val_list = all_user_purchases_from_sparse(ranks_true, val_item_ids)
        score = mapk(actual=val_list, predicted=ranks_list, k=k)

        return score

    def __del__(self):
        for name, model in self.models.items():
            del model
