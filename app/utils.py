from __future__ import annotations

import warnings
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Union

warnings.simplefilter(action='ignore', category=FutureWarning)


def user_purchases_from_sparse_optimized(sparse_data: Union[csr_matrix, np.ndarray], item_ids: np.ndarray,
                                         user_index: int, return_index=False) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(sparse_data, csr_matrix):
        user_interactions = sparse_data.getrow(user_index)
        user_interactions_dense = user_interactions.toarray().flatten()
    elif isinstance(sparse_data, np.ndarray):
        user_interactions = sparse_data[user_index, :]
        user_interactions_dense = user_interactions.flatten()
    else:
        raise Exception(f"Unexpected type: {type(sparse_data)}")

    indices = np.where(user_interactions_dense != 0)[0]
    n_purchases = user_interactions_dense[indices]
    if return_index:
        return indices, n_purchases

    item_ids_of_user = item_ids[indices]
    return item_ids_of_user, n_purchases


def all_user_purchases_from_sparse(sparse_data: Union[csr_matrix, np.ndarray], item_ids: np.ndarray,
                                   top_k: int = None) -> List[np.ndarray]:
    all_user_purchases = []

    for user_index in range(sparse_data.shape[0]):
        item_ids_of_user, n_purchases = user_purchases_from_sparse_optimized(sparse_data, item_ids, user_index)

        if top_k is not None:
            if len(n_purchases) > 0:
                # Get the indices of the top k purchases
                top_k_indices = np.argsort(n_purchases)[-top_k:][::-1]
                # Get the top k item ids and their respective purchase counts
                item_ids_of_user = item_ids_of_user[top_k_indices]
            else:
                # If user has no purchases, return an empty array
                item_ids_of_user = np.array([])
        all_user_purchases.append(item_ids_of_user)

    return all_user_purchases


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def set_seed(seed: int = 42) -> None:
    import random
    import os

    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    return seed


def long_to_csr(df: pl.DataFrame) -> (csr_matrix, (np.array, np.array)):
    # Get unique user and item IDs
    users = df['user_id'].unique().to_list()
    items = df['item_id'].unique().to_list()

    # Create mapping from user/item ID to row/column index
    user_to_row = {user: i for i, user in enumerate(users)}
    item_to_col = {item: i for i, item in enumerate(items)}

    # Create data, row indices, and column indices for the CSR matrix
    data = [1] * len(df)
    row_indices = [user_to_row[user] for user in df['user_id']]
    col_indices = [item_to_col[item] for item in df['item_id']]

    # Create the CSR matrix
    csr = csr_matrix((data, (row_indices, col_indices)),
                     shape=(len(users), len(items)),
                     dtype=np.int8)

    return csr, (np.array(users).flatten(), np.array(items).flatten())

def take_frac_of_data(df: pl.DataFrame, frac = 0.5):
    df = df.sort("user_id")
    return df.head(int(df.shape[0] * frac))