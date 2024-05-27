import warnings

from app.models.base.core import BaseModel
from app.models.ensemble import ModelEnsemble
from app.models.rerank import Reranker

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from typing import Iterable, Dict
from loguru import logger


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class TwoStageModel(BaseModel):
    def __init__(self, ensemble: ModelEnsemble, reranker: Reranker):
        self.ensemble = ensemble
        self.reranker = reranker

    def fit(self):
        raise NotImplementedError()

    def predict(self, user_ids: Iterable, batch_size: int = 25000) -> np.ndarray:
        """
        Predicts ranks for the given user IDs using the ensemble model and reranker.

        Args:
            user_ids (Iterable): An iterable of user IDs.

        Returns:
            np.ndarray: A 2D array with columns: user_index, item_index, rank.
        """

        ctb_ranks = None
        logger.debug(f"Predicting with TwoStageModel for {len(user_ids)}")

        for i, user_batch in enumerate(batch(user_ids, batch_size)):
            base_model_ranks = self.ensemble.predict(user_batch)
            batch_ctb_ranks = self._predict_final_ranks(base_model_ranks)

            # Check the shape of batch_ctb_ranks before reshaping
            logger.debug(f"Batch {i}: batch_ctb_ranks shape before reshaping: {batch_ctb_ranks.shape}")

            if ctb_ranks is None:
                n_item_ids = list(base_model_ranks.values())[0].shape[1]
                ctb_ranks = np.empty((0, n_item_ids), dtype=np.float16)
            del base_model_ranks

            # Ensure batch_ctb_ranks can be reshaped into (len(user_batch), n_item_ids)
            expected_size = len(user_batch) * n_item_ids
            actual_size = batch_ctb_ranks.size
            if actual_size != expected_size:
                raise ValueError(f"Mismatch in expected size: expected {expected_size}, got {actual_size}")

            batch_ctb_ranks = batch_ctb_ranks.reshape(len(user_batch), n_item_ids)
            ctb_ranks = np.concatenate((ctb_ranks, batch_ctb_ranks), axis=0)
            logger.info(f"Two stage model prediction finished for batch №{i}.")
            logger.info(f"{batch_size * (i + 1)} users were processed out of {len(user_ids)}")
        return ctb_ranks

    def _predict_final_ranks(self, base_model_ranks: Dict[str, 'BaseModel']) -> np.ndarray:
        """
        Preprocesses the ranks from the base model predictions by adding user and item indices.

        Args:
            base_model_ranks (Dict[str, 'BaseModel']): The base model predictions.

        Returns:
            np.ndarray: A 2D array with user indices, item indices, and ranks.
        """
        logger.info("Ranks Preprocessing")
        all_ranks = list(base_model_ranks.values())

        n_users, n_items = all_ranks[0].shape
        assert len(all_ranks) == 4
        assert all(len(sublist) == len(all_ranks[0]) for sublist in all_ranks), "Not all sublists have the same length"

        all_ranks = np.stack(all_ranks, axis=0)  # shape: (4, user_id, item_id)
        all_ranks = all_ranks.transpose(2, 0, 1).reshape(-1, 4)  # shape (item_id * user_id, 4)
        assert all_ranks.shape == (n_users * n_items, 4), f"Shape of all_ranks: {all_ranks.shape}"
        logger.info("Inferencing Catboost")
        return self.reranker.predict_proba(all_ranks)
