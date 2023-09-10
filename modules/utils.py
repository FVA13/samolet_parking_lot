import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import os
import random


def weighted_mean(values, weights):
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def lists_intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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


def plot_catboost_feature_importance(model, X):
    categorical_columns = X.select_dtypes(
        exclude=["float64", "int64"]
    ).columns.to_list()
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


def plot_confusion_matrix(model, X_test, y_test):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix(y_test, model.predict(X_test)), display_labels=[False, True]
    )
    cm_display.plot(cmap="Blues")
    plt.show()


# add opportunity to plot w/o passing the model (and passing y_pred, y_pred_proba) instead
def plot_model_info(model, X_test, y_test, catboost=False):
    y_pred = model.predict(X_test)
    print("Classification report of the model\n", classification_report(y_test, y_pred))
    print("ROC-AUC score is: ", roc_auc_score(y_test, y_pred))
    if catboost:
        plot_catboost_feature_importance(model, X_test)
    plot_roc_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()


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


def lists_analysis(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1 & set2
    difference1 = set1 - set2
    difference2 = set2 - set1

    intersection_count = len(intersection)
    difference1_count = len(difference1)
    difference2_count = len(difference2)

    total_elements = len(set1.union(set2))

    intersection_percent = (intersection_count / total_elements) * 100
    difference1_percent = (difference1_count / total_elements) * 100
    difference2_percent = (difference2_count / total_elements) * 100

    return {
        "intersection": {
            "absolute": intersection_count,
            "percent": round(intersection_percent, 2),
        },
        "difference_list1": {
            "absolute": difference1_count,
            "percent": round(difference1_percent, 2),
        },
        "difference_list2": {
            "absolute": difference2_count,
            "percent": round(difference2_percent, 2),
        },
    }


def get_train_valid_test_split(data):
    X = data.drop(columns=["target", "report_date", "client_id", "col1454"])
    Y = data["target"]

    categorical_columns = X.select_dtypes(exclude=["float64", "int64"]).columns
    numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns

    X[numerical_columns] = X[numerical_columns].fillna(0)
    X[categorical_columns] = X[categorical_columns].astype(str)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=0.33, random_state=42, shuffle=True, stratify=Y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=0.33,
        random_state=42,
        shuffle=True,
        stratify=y_valid,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_train_valid_test_split_group(data):
    splitter = GroupShuffleSplit(test_size=0.30, n_splits=2, random_state=7)
    split = splitter.split(data, groups=data["client_id"])
    train_inds, test_inds = next(split)

    split = splitter.split(
        data.iloc[test_inds], groups=data.iloc[test_inds]["client_id"]
    )
    valid_inds, test_inds = next(split)

    train = data.iloc[train_inds]
    valid = data.iloc[valid_inds]
    test = data.iloc[test_inds]

    X_train = train.drop(columns=["target", "client_id", "report_date"])
    X_valid = valid.drop(columns=["target", "client_id", "report_date"])
    X_test = test.drop(columns=["target", "client_id", "report_date"])

    y_train = train["target"]
    y_valid = valid["target"]
    y_test = test["target"]

    categorical_columns = (
        data.drop(columns=["target", "client_id", "report_date"])
        .select_dtypes(exclude=["float64", "int64"])
        .columns
    )
    numerical_columns = (
        data.drop(columns=["target", "client_id", "report_date"])
        .select_dtypes(include=["float64", "int64"])
        .columns
    )
    X_train[numerical_columns] = X_train[numerical_columns].fillna(0)
    X_train[categorical_columns] = X_train[categorical_columns].astype(str)

    X_valid[numerical_columns] = X_valid[numerical_columns].fillna(0)
    X_valid[categorical_columns] = X_valid[categorical_columns].astype(str)

    X_test[numerical_columns] = X_test[numerical_columns].fillna(0)
    X_test[categorical_columns] = X_test[categorical_columns].astype(str)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# check technique one more time
def get_train_valid_test_split_group_balanced(data, groupby="client_id"):
    X = data.drop(columns=["target", "report_date"])
    y = data["target"]

    # Assuming X is your feature set, y is your target variable and groups is the array of group labels
    group_shuffle_split = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    groups = X[groupby]

    # First split to create train and a temporary set (test + validation)
    for train_index, temp_index in group_shuffle_split.split(X, y, groups):
        X_train, X_temp = X.iloc[train_index], X.iloc[temp_index]
        y_train, y_temp = y.iloc[train_index], y.iloc[temp_index]

    # Balance the training data
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # Second split to separate the temporary set into test and validation sets
    group_shuffle_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for test_index, val_index in group_shuffle_split.split(
        X_temp, y_temp, groups[temp_index]
    ):
        X_test, X_val = X_temp.iloc[test_index], X_temp.iloc[val_index]
        y_test, y_val = y_temp.iloc[test_index], y_temp.iloc[val_index]

        # Balance the test data
        ros = RandomOverSampler(random_state=42)
        X_test, y_test = ros.fit_resample(X_test, y_test)

        # Balance the validation data
        ros = RandomOverSampler(random_state=42)
        X_val, y_val = ros.fit_resample(X_val, y_val)

    categorical_columns = (
        data.drop(columns=["target", "client_id", "report_date"])
        .select_dtypes(exclude=["float64", "int64"])
        .columns
    )
    numerical_columns = (
        data.drop(columns=["target", "client_id", "report_date"])
        .select_dtypes(include=["float64", "int64"])
        .columns
    )

    X_train[numerical_columns] = X_train[numerical_columns].fillna(0)
    X_train[categorical_columns] = X_train[categorical_columns].astype(str)

    X_val[numerical_columns] = X_val[numerical_columns].fillna(0)
    X_val[categorical_columns] = X_val[categorical_columns].astype(str)

    X_test[numerical_columns] = X_test[numerical_columns].fillna(0)
    X_test[categorical_columns] = X_test[categorical_columns].astype(str)

    X_train, X_val, X_test = (
        X_train.drop(columns=["client_id"]),
        X_val.drop(columns=["client_id"]),
        X_test.drop(columns=["client_id"]),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
