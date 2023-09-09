import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def create_date_features(X):
    X["day"] = pd.to_datetime(X["report_date"]).dt.day
    X["week"] = pd.to_datetime(X["report_date"]).dt.week
    X["weekday"] = pd.to_datetime(X["report_date"]).dt.weekday
    X["month"] = pd.to_datetime(X["report_date"]).dt.month
    X["year"] = pd.to_datetime(X["report_date"]).dt.year
    return X


def create_numerical_features(X):
    poly = PolynomialFeatures(degree=3, interaction_only=False)
    return poly.fit_transform(X)
