import warnings
from typing import List

from app.models.base.core import BaseModel
from catboost import CatBoostClassifier


import numpy as np
import polars as pl
import dill
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from typing import Tuple, Iterable, Dict
from loguru import logger
from abc import abstractmethod
from scipy.sparse import csr_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_pos_memo(positive_df: pl.DataFrame, user_id: Iterable, item_id: Iterable):
    pos_user_ids = positive_df["item_id"]
    pos_item_ids = positive_df["user_id"]

    positive_examples_memo = {(user_id, item_id): True for user_id, item_id in zip(pos_user_ids, pos_item_ids)}
    return positive_examples_memo


def get_rank_for_pair_index(user_index: int, item_index: int, ranks: np.ndarray) -> float:
    """
    Get the rank for a specific user-item pair from a model's prediction. Works only with all models except Sequential

    Parameters:
        user_index (int): Index of the user for whom to get the rank.
        item_index (int): Index of the item for which to get the rank.
        model_prediction (np.ndarray): Array of shape (N_USERS, N_ITEMS) containing candidates items and ranks.
    Returns:
        float: The rank of the given item for the specified user from row user_index and column item_index.
    """
    rank = ranks[user_index, item_index]
    if isinstance(rank, np.matrix):
        rank = np.array(rank, dtype=np.float16)
    elif isinstance(rank, (csr_matrix, coo_matrix)):
        rank = rank.toarray()
    elif isinstance(rank, (float, int)):
        return rank
    elif isinstance(rank, np.integer):
        return int(rank)
    elif isinstance(rank, (np.integer, np.float32, np.float64, np.float16)):
        return float(rank)
    else:
        assert isinstance(rank, np.ndarray), f"Type: {type(rank)}"

    if rank.size == 0:
        return None
    return float(rank.flatten()[0])


def get_rank_for_pair_ids(user_id: int, item_id: int, user_index_map: Dict[int, int], item_index_map: Dict[int, int],
                          ranks: np.ndarray) -> float:
    """
    Get the rank for a specific user-item pair from precomputed indices and ranks.

    Parameters:
        user_id (int): ID of the user for whom to get the rank.
        item_id (int): ID of the item for which to get the rank.
        user_index_map (dict): Mapping from user_id to index.
        item_index_map (dict): Mapping from item_id to index.
        ranks (np.ndarray): Array of shape (N_USERS, N_ITEMS) containing ranks.
    Returns:
        float: The rank of the given item for the specified user.
    """
    user_index = user_index_map.get(user_id)
    item_index = item_index_map.get(item_id)

    if user_index is not None and item_index is not None:
        return get_rank_for_pair_index(user_index, item_index, ranks)
    else:
        return 0.0


def append_ranks(ranks: np.ndarray, df: pl.DataFrame, user_index_map: Dict[int, int], item_index_map: Dict[int, int],
                 col_name: str) -> pl.DataFrame:
    user_index = 1
    item_index = 2

    rank_series = df.map_rows(
        lambda row: get_rank_for_pair_ids(row[user_index], row[item_index], user_index_map, item_index_map, ranks)
    )
    return df.with_columns(pl.Series(col_name, rank_series))


def get_positive_samples(df: pl.DataFrame, all_ranks: List[np.ndarray], user_ids: Iterable[int],
                         item_ids: Iterable[int], col_names: List[str]) -> pl.DataFrame:
    """
    Parameters:
        df (pl.DataFrame): Input dataframe with columns user_id, item_id, timestamp.
        all_ranks (list): List of np.ndarrays with shape (N_USERS, N_ITEMS).
        user_ids (np.ndarray): Array of user IDs corresponding to rows in ranks.
        item_ids (np.ndarray): Array of item IDs corresponding to columns in ranks.
        col_names (list): List of column names for the ranks.
    Returns:
        pl.DataFrame: Dataframe with ranks appended as columns.
    """
    logger.debug(f"Creating maps")
    user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

    logger.debug(f"Filtering")
    df_known_users = df.filter(pl.col("user_id").is_in(user_ids))
    df_with_ranks = df_known_users

    i = 0
    for ranks, col_name in zip(all_ranks, col_names):
        logger.debug(f"Appending ranks #{i + 1}")
        df_with_ranks = append_ranks(ranks, df_with_ranks, user_index_map, item_index_map, col_name)
        i += 1

    return df_with_ranks.with_columns(pl.lit(1).alias("target"))  # .alias("target") )


# Example usage (assuming you have the data loaded in df, all_ranks, user_ids, item_ids, col_names)
# df_optimized = get_positive_samples(df, all_ranks, user_ids, item_ids, col_names)
import random


def generate_negative_examples(user_ids: Iterable[int], item_ids: Iterable[int],
                               positive_examples_memo: Dict[Tuple[int, int], bool],
                               all_ranks: List[np.ndarray], df_schema: List[Tuple[str, pl.DataType]],
                               n=10) -> pl.DataFrame:
    """
    Generates negative examples by randomly choosing item_id and user_id pairs.

    Parameters:
        user_ids (Iterable[int]): List of user IDs.
        item_ids (Iterable[int]): List of item IDs.
        positive_examples_memo (Dict[Tuple[int, int], bool]): Memoization of positive examples.
        all_ranks (List[np.ndarray]): List of np.ndarrays with shape (N_USERS, N_ITEMS).
        df_schema (List[Tuple[str, pl.DataType]]): Schema definition for the resulting DataFrame.
        n (int): Number of negative examples to generate.

    Returns:
        pl.DataFrame: DataFrame containing negative examples.
    """
    logger.debug("Starting to generate negative examples")

    pos_set = set(positive_examples_memo.keys())
    logger.debug(f"Positive set size: {len(pos_set)}")

    # Sampling negative pairs
    negative_samples = set()
    users = np.array(user_ids)
    items = np.array(item_ids)
    np.random.shuffle(users)
    np.random.shuffle(items)
    logger.debug("Shuffled user_ids and item_ids")

    logger.debug(f"Unique users: {len(users)}, Unique items: {len(items)}")

    max_attempts = n * 100  # Set a limit on the number of attempts to prevent infinite looping
    attempts = 0

    while len(negative_samples) < n and attempts < max_attempts:
        user = random.choice(users)
        item = random.choice(items)
        attempts += 1

        if (user, item) not in pos_set:
            negative_samples.add((user, item))
            if len(negative_samples) % 100000 == 0:
                logger.debug(f"Negative samples collected: {len(negative_samples)}")

    if attempts >= max_attempts:
        logger.error("Unable to generate the required number of negative examples within the attempt limit.")
        raise RuntimeError("Unable to generate the required number of negative examples within the attempt limit.")

    logger.debug("Finished sampling negative pairs")

    # Convert negative_samples to DataFrame
    neg_samples_list = list(negative_samples)
    negative_examples = np.zeros((len(neg_samples_list), 5 + len(all_ranks)))

    user_index_map = {user_id: idx for idx, user_id in enumerate(users)}
    item_index_map = {item_id: idx for idx, item_id in enumerate(items)}
    logger.debug("Created user_index_map and item_index_map")

    for idx, (uid, iid) in enumerate(neg_samples_list):
        ranks_for_pair = [get_rank_for_pair_ids(uid, iid, user_index_map, item_index_map, ranks) for ranks in all_ranks]
        row = [0, uid, iid, 0] + ranks_for_pair + [0]
        negative_examples[idx, :] = np.array(row)

    logger.debug(f"Total negative samples: {len(negative_examples)}")
    return pl.DataFrame(negative_examples, schema=df_schema)


# Example usage (assuming you have the data loaded in user_ids, item_ids, positive_examples_memo, all_ranks, and df_schema)
# negative_samples_df = generate_negative_examples(user_ids, item_ids, positive_examples_memo, all_ranks, df_schema, n=1000)


def catboost_preprocess(candidate_predictions: Dict[str, BaseModel], candidate_predict_set,
                        cp_user_ids, cp_item_ids, ctb_train_users, ctb_test_users):
    all_ranks = list(candidate_predictions.values())
    col_names = ["pop_rank", "hist_rank", "mf_rank", "knn_rank"]

    pos_neg_split = 0.66

    logger.info(f"positive samples are generating...")
    pos = get_positive_samples(candidate_predict_set, all_ranks, cp_user_ids, cp_item_ids, col_names)
    logger.info(f"positive samples generated")
    pos_dict = create_pos_memo(pos, cp_user_ids, cp_item_ids)
    logger.info(f"positive memo created")

    n_positives = pos.shape[0]
    n_negatives = int(n_positives * (1 - pos_neg_split) / pos_neg_split)

    logger.debug(f"pos.shape: {pos.shape}")
    logger.debug(f"{n_negatives} negative samples will be sampled, %{1 - pos_neg_split}")

    neg = generate_negative_examples(cp_user_ids, cp_item_ids, pos_dict, all_ranks, pos.schema, n=n_negatives)
    logger.debug(f"neg.shape: {neg.shape}")

    print(pos.sample(5))
    print(neg.sample(5))

    ctb_train_users = ctb_train_users.unique()
    ctb_test_users = ctb_test_users.unique()

    ctb_train_users, ctb_eval_users = train_test_split(ctb_train_users,
                                                       random_state=1,
                                                       test_size=0.1)

    select_col = ['user_id', 'item_id', 'target'] + col_names

    logger.info(f"Train test split is in process")
    ctb_train = shuffle(
        pl.concat([
            pos.filter(pl.col('user_id').is_in(ctb_train_users)),
            neg.filter(pl.col('user_id').is_in(ctb_train_users))
        ])[select_col]
    )

    # Catboost test
    ctb_test = shuffle(
        pl.concat([
            pos.filter(pl.col('user_id').is_in(ctb_test_users)),
            neg.filter(pl.col('user_id').is_in(ctb_test_users))
        ])[select_col]
    )

    # for early stopping
    ctb_eval = shuffle(
        pl.concat([
            pos.filter(pl.col('user_id').is_in(ctb_eval_users)),
            neg.filter(pl.col('user_id').is_in(ctb_eval_users))
        ])[select_col]
    )

    logger.info(f"Class balance: {ctb_test['target'].value_counts()}")

    # X,y
    drop_col = ['user_id', 'item_id']
    target_col = ['target']
    cat_col = []

    X_train, y_train = ctb_train.drop(columns=drop_col + target_col), ctb_train[target_col]
    X_val, y_val = ctb_eval.drop(columns=drop_col + target_col), ctb_eval[target_col]
    logger.debug(
        f"Shapes of X_train.shape, y_train.shape, X_val.shape, y_val.shape: {X_train.shape, y_train.shape, X_val.shape, y_val.shape}")

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    return X_train, X_val, y_train, y_val


def catboost_preprocess_global_test(global_test_preds: Dict[str, 'BaseModel'], test_global_frac: pl.DataFrame,
                                    glob_user_ids: np.array, glob_item_ids: np.array, pos_neg_split=0.5):
    # Filter the rows of the DataFrame based on the split user sets

    test_global_users = test_global_frac["user_id"]

    all_ranks = list(global_test_preds.values())
    col_names = ["pop_rank", "hist_rank", "mf_rank", "knn_rank"]

    logger.info(f"positive samples are generating...")
    pos = get_positive_samples(test_global_frac, all_ranks, glob_user_ids, glob_item_ids, col_names)
    logger.info(f"positive samples generated")
    pos_dict = create_pos_memo(pos, glob_user_ids, glob_item_ids)
    logger.info(f"positive memo created")

    n_positives = pos.shape[0]
    n_negatives = int(n_positives * (1 - pos_neg_split) / pos_neg_split)

    logger.debug(f"pos.shape: {pos.shape}")
    logger.debug(f"{n_negatives} negative samples will be sampled, %{1 - pos_neg_split}")

    neg = generate_negative_examples(glob_user_ids, glob_item_ids, pos_dict, all_ranks, pos.schema, n=n_negatives)
    logger.debug(f"neg.shape: {neg.shape}")

    print(pos.sample(5))
    print(neg.sample(5))

    ctb_test_users = test_global_users.unique()
    select_col = ['user_id', 'item_id', 'target'] + col_names

    ctb_global_test = shuffle(
        pl.concat([
            pos.filter(pl.col('user_id').is_in(ctb_test_users)),
            neg.filter(pl.col('user_id').is_in(ctb_test_users))
        ])[select_col]
    )

    drop_col = ['user_id', 'item_id']
    target_col = ['target']
    cat_col = []

    X_test, y_test = ctb_global_test.drop(columns=drop_col + target_col), ctb_global_test[target_col]
    return X_test.to_numpy(), y_test.to_numpy()

class Reranker:
    DEFAULT_PARAMS = {
        'subsample': 0.9,
        'max_depth': 5,
        'n_estimators': 2000,
        'learning_rate': 0.1,
        'thread_count': 20,
        'random_state': 42,
        'verbose': 200,
    }

    def __init__(self, **params):
        if params is {}:
            params = self.DEFAULT_PARAMS
        self.cat_col = []
        self.ctb_model = CatBoostClassifier(**params)

    def fit(self, X_train, X_val, y_train, y_val) -> 'Reranker':
        """
        Fit the model to the data.

        Parameters (np.ndarray):
        - X_train
        - X_val
        - y_train
        - y_val

        Returns:
        A reference to the fitted model instance.
        """
        self.ctb_model.fit(X_train,
                           y_train,
                           eval_set=(X_val, y_val),
                           early_stopping_rounds=300,
                           cat_features=self.cat_col,
                           plot=True)

        return self

    @abstractmethod
    def predict_proba(self, X_test: np.array) -> np.array:
        """
        Predict the ranking scores for a given array ranks.

        Parameters:
        - X_test: An array of ranks for prediction

        Returns:
        y_pred
        """
        return self.ctb_model.predict_proba(X_test)[:, 1]

    def evaluate(self, y_val, y_pred):
        from sklearn.metrics import roc_auc_score
        metric = roc_auc_score(y_val, y_pred)
        logger.info(f"ROC AUC score = {metric:.3f}")
        return metric

    def save(self):
        with open(f"ctb_model.dill", 'wb') as f:
            dill.dump(self.ctb_model, f)
