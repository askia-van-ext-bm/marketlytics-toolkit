import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def create_linear_regression_model(df: pd.DataFrame, validation_start: pd.Timestamp,
                                   promo_start: pd.Timestamp, promo_end: pd.Timestamp,
                                   target: str, top_corr: int = 10):
    """
    Trains multiple linear regression models using top correlated features and selects the best one.

    Parameters:
    - df (pd.DataFrame): Input time-indexed DataFrame with target and covariates
    - validation_start (pd.Timestamp): Start date of validation period
    - promo_start (pd.Timestamp): Start date of promotion period
    - promo_end (pd.Timestamp): End date of promotion period
    - target (str): Name of the target column
    - top_corr (int): Maximum number of top correlated features to test

    Returns:
    - model (LinearRegression): Best performing linear regression model
    - top_features (list): List of selected features
    - mae (float): Mean Absolute Error on validation set
    - mape (float): Mean Absolute Percentage Error on validation set (in %)
    """
    train, test = _split_train_test(df, validation_start, promo_start)
    performances = _train_and_evaluate_models(train, test, target, top_corr)
    best_model, top_features, mae, mape = sorted(performances, key=lambda x: x[2])[0]

    print("Selected features:", top_features)
    print("MAE =", round(mae, 2))
    print("MAPE =", round(mape * 100, 2), "%")

    return best_model, top_features, round(mae, 2), round(mape * 100, 2)

def _split_train_test(df: pd.DataFrame, validation_start: pd.Timestamp, promo_start: pd.Timestamp):
    """Splits the dataset into training and validation sets based on date."""
    train = df[df.index < validation_start].dropna()
    test = df[(df.index >= validation_start) & (df.index < promo_start)].dropna()
    return train, test

def _train_and_evaluate_models(train: pd.DataFrame, test: pd.DataFrame,
                               target: str, top_corr: int):
    """
    Trains multiple models using increasing number of top correlated features and evaluates them.

    Returns:
    - List of tuples: (model, features, MAE, MAPE)
    """
    performances = []
    for i in range(1, top_corr):
        correlations = train.corr()[target].drop(target)
        top_features = correlations.abs().sort_values(ascending=False).head(i).index.tolist()

        model = LinearRegression()
        model.fit(train[top_features], train[target])

        y_pred = model.predict(test[top_features])
        y_true = test[target]

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        performances.append((model, top_features, mae, mape))

    return performances
