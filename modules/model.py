from catboost import CatBoostClassifier, Pool, metrics, EFeaturesSelectionAlgorithm
from samolet_parking_lot.modules.utils import *


def catboost_model_classifier(x_train, x_test, y_train, y_test):
    # categorical_features_indices = np.where(~X_train.dtypes.isin(['float64', 'int64']))[0]
    categorical_columns = x_train.select_dtypes(exclude=["float64", "int64"]).columns
    categorical_features_indices = get_column_indices(x_train, categorical_columns)

    cb_model = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        logging_level="Silent",
        # custom_metric=['MAE', 'MAPE'],
        max_depth=8,
        iterations=200,
        # scale_pos_weight=26,
        auto_class_weights="Balanced",
        early_stopping_rounds=20,
        # eval_metric=[metrics.Precision(), metrics.Recall(), metrics.F1(), metrics.TotalF1(), metrics.Accuracy()]
    )

    if categorical_features_indices:
        cb_model.fit(
            x_train,
            y_train,
            eval_set=(x_test, y_test),
            cat_features=categorical_features_indices,
            plot=True,
        )
    else:
        cb_model.fit(
            x_train,
            y_train,
            eval_set=(x_test, y_test),
            cat_features=categorical_features_indices,
            plot=True,
        )

    return cb_model
