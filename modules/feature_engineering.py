import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_date_features(X):
    X["day"] = pd.to_datetime(X["report_date"]).dt.day
    X["week"] = pd.to_datetime(X["report_date"]).dt.week
    X["weekday"] = pd.to_datetime(X["report_date"]).dt.weekday
    X["month"] = pd.to_datetime(X["report_date"]).dt.month
    X["year"] = pd.to_datetime(X["report_date"]).dt.year
    return X


def transform_data_for_regression(X):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [
        cname
        for cname in X.columns
        if X[cname].nunique() < 10 and X[cname].dtype == "object"
    ]

    # Select numerical columns
    numerical_cols = [
        cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
    ]

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant")),
            ("scaler", StandardScaler()),
            (
                "poly",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            ),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
