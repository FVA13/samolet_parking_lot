from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def get_feature_importances(X, y, model):
    # Add a random feature
    X['random'] = np.random.random(size=len(X))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a dictionary that maps feature names to their importances
    feature_importances = {name: importance for name, importance in zip(X.columns, importances)}

    return feature_importances


def get_ (model, X_train, X_test, y_test, plot=False):
    perm_raw = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=10)
    perm = (
        pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in X_train.columns])
        .assign(AVG_Importance=perm_raw.importances_mean)
        .assign(STD_Importance=np.std(perm_raw.importances, axis=1))
        .sort_values(by='AVG_Importance', ascending=False)
    )
    sns.barplot(x=perm.index, y=perm.AVG_Importance)
    plt.xticks([])
    plt.show()
    useful_column_indices = get_column_indices(X_train, perm.query("AVG_Importance > 0")["AVG_Importance"].index.to_list())
