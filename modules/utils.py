import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_column_indices(df: pd.DataFrame, column_names: list) -> list:
    return [df.columns.get_loc(c) for c in column_names if c in df.columns]


def plot_roc_curve(model, X_test, y_test):
    # Predict probabilities for the test data.
    y_probs = model.predict_proba(X_test)

    # Keep only the positive class
    y_probs = y_probs[:, 1]

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # Compute the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_catboost_feature_importance(model, categorical_columns):
    feat_importances = model.get_feature_importance(prettified=True)
    feat_importances["feat_type"] = feat_importances["Feature Id"].apply(
        lambda x: "categorial" if x in categorical_columns else "numerical"
    )

    ax = plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x="Importances",
        y="Feature Id",
        data=feat_importances.loc[:30, :],
        hue="feat_type",
        palette="bright",
    )
    ax = plt.title("CatBoost features importance:")
    plt.show()


def plot_model_info(model, X_test, y_test, categorical_columns):
    y_pred = model.predict(X_test)
    print("ROC-AUC score is: ", roc_auc_score(y_test, y_pred))
    plot_catboost_feature_importance(model, categorical_columns)
    plot_roc_curve(model, X_test, y_test)


def plot_pca_variance(pca_model, save_to=None):
    # Calculate the cumulative explained variance ratio
    explained_variance_ratio = np.cumsum(pca_model.explained_variance_ratio_)

    # Plot the variance explained by each principal component
    plt.plot(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
        marker="o",
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Variance Explained")
    plt.show()
    if save_to:
        plt.savefig(save_to)
        print("Plot was saved to {}".format(save_to))


def get_data_feature_types(data, indices):
    df_feat_types = pd.DataFrame(
        data.iloc[:, indices].dtypes, columns=["dtype"]
    ).reset_index(names=["feature_name"])
    return df_feat_types
