import numpy as np

from app.models.ensemble import ModelEnsemble
from app.models.base.lfm import MatrixFactorization
from app.models.base.neighbor import User2User
from app.models.base.popularity import PopModel
from app.models.rerank import catboost_preprocess, Reranker
from app.models.two_stage import TwoStageModel
from app.models.base.user_history import UserHistoryModel
from app.utils import set_seed, long_to_csr, take_frac_of_data
import os
import polars as pl
from sklearn.model_selection import train_test_split
import dill
from catboost import CatBoostClassifier

def train():
    set_seed(42)
    knn_est_params = {'n_neighbors': 1, 'metric': 'cosine', 'leaf_size': 50}
    mf_est_params = {'no_components': 7, 'learning_rate': 0.05756873225072556, 'max_sampled': 85}
    ctb_est_params = {"subsample": 0.6493933232357054, "max_depth": 5, "n_estimators": 1100,
                      "learning_rate": 0.09618289423795023}

    data_path = "wb_data/"

    test_global = pl.read_csv(data_path+'test_global.csv')

    # Load train_base, val_base, train_val_base, and candidate_predict_set dataframes
    train_base = pl.read_csv(data_path+'train_base.csv')
    val_base = pl.read_csv(data_path + 'val_base.csv')
    train_val_base = pl.read_csv(data_path + 'train_val_base.csv')
    candidate_predict_set = pl.read_csv(data_path + 'candidate_predict_set.csv')

    # Load train_rerank and test_rerank dataframes
    train_rerank = pl.read_csv(data_path + 'train_rerank.csv')
    test_rerank = pl.read_csv(data_path + 'test_rerank.csv')

    # Create csr matrices
    train_base_csr, (trb_user_ids, trb_item_ids) = long_to_csr(train_base)
    val_base_csr, (vb_user_ids, vb_item_ids) = long_to_csr(val_base)

    FRACTION = 0.5
    train_val_base_csr, (tvb_user_ids, tvb_item_ids) = long_to_csr(train_val_base)
    candidate_predict_set_frac = take_frac_of_data(candidate_predict_set, FRACTION)
    cand_pred_csr, (cp_user_ids, cp_item_ids) = long_to_csr(candidate_predict_set_frac)
    test_global_frac = take_frac_of_data(test_global, FRACTION)
    test_global_csr, (glob_user_ids, glob_item_ids) = long_to_csr(test_global_frac)

    print(candidate_predict_set.shape, candidate_predict_set_frac.shape)
    print(cand_pred_csr.shape)
    print(test_global.shape, test_global_frac.shape)
    print(test_global_csr.shape)

    # Ensemble for catboost training
    models = {"Popularity Based Model": PopModel(),
              "User History Model": UserHistoryModel(),
              "Matrix Factorization": MatrixFactorization(mf_est_params),
              "User2User": User2User(**knn_est_params)  # change on 7
              }

    ensemble_final = ModelEnsemble(models=models).fit(train_val_base_csr, tvb_user_ids, tvb_item_ids)

    candidate_preds_frac = ensemble_final.predict(cp_user_ids)

    # Catboost preprocess
    users = candidate_predict_set_frac['user_id'].unique()

    # Split users into train and test sets
    users_train, users_test = train_test_split(users, test_size=0.2, random_state=42)

    # Filter the rows of the DataFrame based on the split user sets
    df = candidate_predict_set_frac
    train_rerank_frac = df.filter(pl.col("user_id").is_in(users_train))
    test_rerank_frac = df.filter(pl.col("user_id").is_in(users_test))

    ctb_train_users = train_rerank_frac["user_id"]
    ctb_test_users = test_rerank_frac["user_id"]

    X_train, X_val, y_train, y_val = catboost_preprocess(candidate_preds_frac, candidate_predict_set_frac,

                                                         # global_test_preds = ensemble_final.predict(glob_user_ids)
                                                         # X_test, y_test = catboost_preprocess_global_test(global_test_preds, test_global_frac,glob_user_ids, glob_item_ids)
                                                         cp_user_ids, cp_item_ids, ctb_train_users, ctb_test_users)

    # Load the Catboost Model or Fit it
    # Path to the model file
    model_path = 'wb_data/ctb_model_1500_est_25_05.dill'

    # Check if the model file exists
    if os.path.exists(model_path):
        # Load the model using dill
        with open(model_path, 'rb') as f:
            ctb_model = dill.load(f)

        # Ensure the loaded object is a CatBoostClassifier
        if isinstance(ctb_model, CatBoostClassifier):
            print("Model loaded successfully.")
        else:
            print("Loaded object is not a CatBoostClassifier.")
    else:
        print("Model file does not exist.")
        ctb_model = Reranker(n_estimators=1500).fit(X_train, X_val, y_train, y_val)
        ctb_model.save()

    y_pred = ctb_model.predict_proba(X_val)

    # Prepare global test
    del candidate_preds_frac

    warm_glob_user_ids = glob_user_ids[np.isin(glob_user_ids, tvb_user_ids)]
    print(glob_user_ids.shape, warm_glob_user_ids.shape)

    mask = np.isin(glob_user_ids, warm_glob_user_ids)
    warm_test_global_csr = test_global_csr[mask]
    print(f"test_global_csr.shape, warm_test_global_csr.shape, {test_global_csr.shape, warm_test_global_csr.shape}")

    # TwoStageModel
    reranker_model = Reranker()
    reranker_model.ctb_model = ctb_model
    two_stage_model = TwoStageModel(ensemble_final, reranker_model)
    # preds = two_stage_model.predict(warm_glob_user_ids)
    return two_stage_model