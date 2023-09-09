from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from catboost import CatBoostClassifier
from samolet_parking_lot.modules.utils import *


def search_best_params(X_train, X_test, y_train, y_test):
    def objective(search_space):
        categorical_columns = X_train.select_dtypes(
            exclude=["float64", "int64"]
        ).columns
        categorical_features_indices = get_column_indices(X_train, categorical_columns)

        cb_model = CatBoostClassifier(
            **search_space,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            early_stopping_rounds=20,
            random_seed=42,
        )

        cb_model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            cat_features=categorical_features_indices,
            plot=True,
            verbose=False,
        )
        return {
            "loss": cb_model.get_best_score()["validation"]["Logloss"],
            "status": STATUS_OK,
        }

    search_space = {
        "learning_rate": hp.uniform("learning_rate", 0.1, 0.5),
        "iterations": hp.randint("iterations", 100, 1000),
        "l2_leaf_reg": hp.randint("l2_leaf_reg", 1, 10),
        "depth": hp.randint("depth", 4, 10),
        # 'border_count': hp.uniform ('border_count', 32, 255),
    }

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1000,
        # verbose=False,
    )

    hyperparams = space_eval(search_space, best_params)
    return hyperparams
