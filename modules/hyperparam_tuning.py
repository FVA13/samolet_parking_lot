from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, early_stop
from catboost import CatBoostClassifier, cv, Pool
from samolet_parking_lot.modules.utils import *


def search_best_params(
    X_train, X_val, y_train, y_val, max_evals=100, early_stop_steps=50
):
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
            eval_set=(X_val, y_val),
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
        max_evals=max_evals,
        early_stop_fn=early_stop.no_progress_loss(early_stop_steps)
        # verbose=False,
    )

    hyperparams = space_eval(search_space, best_params)
    return hyperparams


def search_best_params_cv(
        X,
        y,
        max_evals=100,
        early_stop_steps=50,
        # preprocessor,
        # pca
):
    def objective(space):
        # Modify the parameters to be used in CatBoost

        # Perform cross-validation and return the average AUC
        params = {
            "learning_rate": space["learning_rate"],
            "iterations": int(space["iterations"]),
            "l2_leaf_reg": space["l2_leaf_reg"],
            "depth": space["depth"],
            # "random_strength": space["random_strength"],
            # "border_count": space["border_count"],
            # "bagging_temperature": space["bagging_temperature"],
            "loss_function": "Logloss",
            "random_seed": 42,
            "logging_level": "Silent",
            "auto_class_weights": "Balanced",
            "early_stopping_rounds": 20,
        }
        cv_results = cv(pool, params, fold_count=5)
        avg_auc = np.mean(cv_results["test-Logloss-mean"])

        # # Train LogisticRegressionCV
        # X_linear = preprocessor.transform(X)
        # X_pca = pca.transform(X_linear)[:, :70]
        #
        # # Make predictions
        # cat_boost_preds = model_best.predict_proba(X_test_cleaned)[:, 1]
        # log_reg_preds = logreg_cv.predict_proba(X_pca)[:, 1]
        #
        # # Combine predictions
        # preds = np.array([cat_boost_preds, log_reg_preds])
        # final_preds = np.average(preds, weights=[params['cat_weight'], 1-params['cat_weight']], axis=0)
        #
        # # Calculate the loss of the final prediction
        # loss = log_loss(y_test, final_preds) # if we use it we should change loss in return

        return {"loss": avg_auc, "status": STATUS_OK}

    # Get categorical and numerical columns
    categorical_columns = X.select_dtypes(exclude=["float64", "int64"]).columns
    numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns

    # Fill NA values and convert data types
    X[numerical_columns] = X[numerical_columns].fillna(0).astype(int)
    X[categorical_columns] = X[categorical_columns].astype(str)

    # Create a Pool object
    global pool
    pool = Pool(X, y, cat_features=categorical_columns.to_list())

    # Define the hyperparameter space
    search_space = {
        "learning_rate": hp.uniform("learning_rate", 0.1, 0.5),
        "iterations": hp.randint("iterations", 100, 500),
        "l2_leaf_reg": hp.randint("l2_leaf_reg", 1, 10),
        "depth": hp.randint("depth", 4, 10),
        # "random_strength": hp.choice("random_strength", (0.0, 1.0)),
        # "border_count": hp.qloguniform("border_count", np.log(32), np.log(255), 1),
        # "bagging_temperature": hp.loguniform(
        #     "bagging_temperature", np.log(1), np.log(3)
        # 'cat_weight': hp.uniform('cat_weight', 0.3, 0.7),
        # ),
    }

    # Run the hyperparameter optimization
    trials = Trials()
    best = fmin(
        objective,
        search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        early_stop_fn=early_stop.no_progress_loss(early_stop_steps),
    )

    # Train the final model on the entire dataset with the best found parameters
    best_params = {
        "learning_rate": best["learning_rate"],
        "iterations": int(best["iterations"]),
        "l2_leaf_reg": best["l2_leaf_reg"],
        "depth": best["depth"],
        # "random_strength": best["random_strength"],
        # "border_count": best["border_count"],
        # "bagging_temperature": best["bagging_temperature"],
        "loss_function": "Logloss",
        "random_seed": 42,
        "logging_level": "Silent",
        "auto_class_weights": "Balanced",
        "early_stopping_rounds": 20,
    }
    model = CatBoostClassifier(**best_params)

    model.fit(
        X,
        y,
        cat_features=categorical_columns.to_list(),
        plot=True,
        verbose=False,
    )

    # # Make predictions on the test set
    # X_test[numerical_columns] = X_test[numerical_columns].fillna(0).astype(int)
    # X_test[categorical_columns] = X_test[categorical_columns].astype(str)
    # predictions = model.predict(X_test)

    return best_params  # predictions
