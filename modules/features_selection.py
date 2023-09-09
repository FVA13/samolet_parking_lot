from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from samolet_parking_lot.modules.utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def remove_null_columns(df, threshold=0.8):
    """
    Remove columns from a pandas DataFrame where more than a certain
    threshold of the values are null.

    Parameters:
    df (pd.DataFrame): the input DataFrame
    threshold (float): the ratio of null values at which to remove columns

    Returns:
    pd.DataFrame: a new DataFrame with null columns removed
    """
    # Calculate ratio of nulls for each column
    null_ratio = df.isnull().sum() / len(df)

    # Identify columns to keep (where null_ratio <= threshold)
    to_keep = null_ratio[null_ratio <= threshold].index

    # Return new DataFrame with only the columns to keep
    return df[to_keep]


def get_not_null_columns_names(df, threshold):
    null_counts = df.isnull().sum()  # Count the number of null values in each column
    total_rows = df.shape[0]  # Total number of rows in the DataFrame
    null_share = (
            null_counts / total_rows
    )  # Calculate the share of null values for each column
    null_columns_indices = null_share[
        null_share < threshold
        ].index.tolist()  # Get the names of columns with share of nulls lower than the threshold
    return null_columns_indices


# def get_features_importance_rand_feaut(X_train, y_train, X_valid, y_valid):
#     # Add a random feature
#     X_train["random"] = np.random.random(size=len(X_train))
#     X_valid["random"] = np.random.random(size=len(X_valid))
#
#     # Get categorical features
#     categorical_columns = X_train.select_dtypes(exclude=['float64', 'int64']).columns
#     categorical_features_indices = get_column_indices(X_train, categorical_columns)
#
#     model = CatBoostClassifier(
#         loss_function='Logloss',
#         random_seed=42,
#         logging_level='Silent',
#         max_depth=8,
#         iterations=200,
#         auto_class_weights='Balanced',
#         early_stopping_rounds=20,
#     )
#     # Train the model
#     model.fit(
#         X_train,
#         y_train,
#         eval_set=(X_valid, y_valid),
#         cat_features=categorical_features_indices,
#     )
#
#     # Get feature importance
#     importance = model.feature_importances_
#
#     # Create a dictionary that maps feature names to their importance
#     features_importance = {
#         name: importance for name, importance in zip(X_train.columns, importance)
#     }
#
#     return features_importance


def get_features_importance_rand_feat(
    X_train, y_train, X_valid, y_valid, n_iterations=10
):
    # Initialize a dictionary to store accumulated feature importance
    accumulated_importance = {name: 0 for name in X_train.columns}
    accumulated_importance["random"] = 0

    for _ in tqdm(range(n_iterations)):
        # Add a random feature
        X_train["random"] = np.random.random(size=len(X_train))
        X_valid["random"] = np.random.random(size=len(X_valid))

        # Get categorical features
        categorical_columns = X_train.select_dtypes(
            exclude=["float64", "int64"]
        ).columns
        categorical_features_indices = get_column_indices(X_train, categorical_columns)

        model = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=42,
            logging_level="Silent",
            max_depth=8,
            iterations=200,
            auto_class_weights="Balanced",
            early_stopping_rounds=20,
        )
        # Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical_features_indices,
        )

        # Get feature importance
        importance = model.feature_importances_

        # Accumulate feature importance
        for name, imp in zip(X_train.columns, importance):
            accumulated_importance[name] += imp

    # Average the feature importance
    features_importance = {
        name: imp / n_iterations for name, imp in accumulated_importance.items()
    }

    return features_importance


def get_random_feat_important_features(X_train, y_train, X_valid, y_valid):
    feat_importance = get_features_importance_rand_feat(
        X_train, y_train, X_valid, y_valid
    )
    feat_importance = (
        pd.DataFrame.from_records(
            [feat_importance],
        )
        .transpose()
        .rename(columns={0: "AVG_Importance"})
    )
    useful_column_indices = get_column_indices(
        X_train,
        feat_importance.query("AVG_Importance > 0 ")["AVG_Importance"].index.to_list(),
    )
    return useful_column_indices


def get_sklearn_important_features(model, X_train, X_valid, y_valid, plot=False):
    perm_raw = permutation_importance(
        model, X_valid, y_valid, n_repeats=10, random_state=42, n_jobs=10
    )
    perm = (
        pd.DataFrame(
            columns=["AVG_Importance", "STD_Importance"],
            index=[i for i in X_train.columns],
        )
        .assign(AVG_Importance=perm_raw.importances_mean)
        .assign(STD_Importance=np.std(perm_raw.importances, axis=1))
        .sort_values(by="AVG_Importance", ascending=False)
    )
    useful_column_indices = get_column_indices(
        X_train, perm.query("AVG_Importance > 0")["AVG_Importance"].index.to_list()
    )
    feat_types = get_data_feature_types(X_train, useful_column_indices)
    logger.info(
        "From {orig_n} feature {tr_n} were selected ({left_perc:.2f}%)."
        "Share of 'Object' type features is: {obj_feat:.2f}%".format(
            orig_n=len(X_train.columns),
            tr_n=len(useful_column_indices),
            left_perc=len(useful_column_indices) / len(X_train.columns),
            obj_feat=len(feat_types.query("dtype.isin(['object'])")) / len(feat_types),
        )
    )
    if plot:
        sns.barplot(x=perm.index, y=perm.AVG_Importance)
        plt.xticks([])
        plt.show()
    return useful_column_indices
