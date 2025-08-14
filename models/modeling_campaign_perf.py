import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm

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

def plot_and_compute_uplift(df, model, top_features, confidence_level,
                            validation_start, promo_start, promo_end,
                            target, granularity="D"):
    """
    Plots actual vs predicted values with confidence intervals and computes uplift during a promo period.

    Parameters:
    - df (pd.DataFrame): Input time-indexed DataFrame
    - model: Trained regression model with a .predict() method
    - top_features (list): List of features used for prediction
    - confidence_level (float): Confidence level for prediction interval (e.g., 0.95)
    - validation_start, promo_start, promo_end (datetime): Key dates for segmentation
    - target (str): Name of the target column
    - granularity (str): Resampling frequency (default: "D" for daily)

    Returns:
    - df_pred (pd.DataFrame): DataFrame with predictions and confidence bounds
    - diff_sum (float): Observed uplift during promo period
    - lower_bound (float): Lower bound of confidence interval
    - upper_bound (float): Upper bound of confidence interval
    - p_value (float): p-value of uplift significance
    """
    z_score_h0 = norm.ppf(1 - (1 - confidence_level) / 2)
    df_pred, residuals, std = _predict_with_confidence(df, model, top_features, validation_start, z_score_h0, target)
    _plot_prediction(df_pred, promo_start, promo_end, target, confidence_level, granularity)
    _plot_difference_bar(df_pred, validation_start, promo_start, promo_end, target, granularity)
    diff_sum, ci_margin_sum, p_value = _compute_statistical_uplift(df_pred, residuals, promo_start, promo_end, target, z_score_h0)

    print(f"Observed difference (actual sum - predicted sum): {round(diff_sum, 2)}")
    print(f"Confidence interval ± {round(ci_margin_sum, 2)}")
    print(f"p-value for the sum: {round(p_value, 4)}")

    return df_pred, round(diff_sum, 2), round(diff_sum - ci_margin_sum, 2), round(diff_sum + ci_margin_sum, 2), round(p_value, 4)

def _predict_with_confidence(df, model, top_features, validation_start, z_score_h0, target):
    df_pred = df.dropna().copy()
    df_pred['y_pred'] = model.predict(df_pred[top_features])

    train = df[df.index < validation_start].dropna()
    residuals = train[target] - model.predict(train[top_features])
    std = residuals.std()

    df_pred['lower'] = df_pred['y_pred'] - z_score_h0 * std
    df_pred['upper'] = df_pred['y_pred'] + z_score_h0 * std

    return df_pred, residuals, std
  
def _plot_prediction(df_pred, promo_start, promo_end, target, confidence_level, granularity):
    df_plot = df_pred[[target, 'y_pred', 'lower', 'upper']].resample(granularity).mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot.index, df_plot[target], label='Actual', color='black')
    ax.plot(df_plot.index, df_plot['y_pred'], label='Prediction', color='blue')
    ax.fill_between(df_plot.index, df_plot['lower'], df_plot['upper'], color='blue', alpha=0.2,
                    label=f'{int(confidence_level * 100)}% interval')
    ax.axvspan(promo_start, promo_end, color='orange', alpha=0.3, label='Promo period')
    ax.set_title(f"Prediction vs Actual — {target}")
    ax.set_ylabel("GMV (local currency)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
  
def _plot_difference_bar(df_pred, validation_start, promo_start, promo_end, target, granularity):
    df_diff = df_pred[[target, 'y_pred']].copy()
    df_diff['diff'] = df_diff[target] - df_diff['y_pred']
    df_diff_resampled = df_diff.resample(granularity).mean()

    colors = []
    for date in df_diff_resampled.index:
        if date < validation_start:
            colors.append('blue')
        elif validation_start <= date < promo_start:
            colors.append('gold')
        elif promo_start <= date <= promo_end:
            colors.append('green')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(df_diff_resampled.index, df_diff_resampled['diff'], color=colors)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"Diff btwn actual & pred — {target}")
    ax.set_ylabel("Diff (Actual - Predicted)")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
  
def _compute_statistical_uplift(df_pred, residuals, promo_start, promo_end, target, z_score_h0):
    promo_data = df_pred[(df_pred.index >= promo_start) & (df_pred.index <= promo_end)]
    y_true_promo = promo_data[target].sum()
    y_pred_promo = promo_data['y_pred'].sum()
    diff_sum = y_true_promo - y_pred_promo
    n = promo_data.shape[0]
    std_residuals = residuals.std()
    ci_margin_sum = z_score_h0 * std_residuals * np.sqrt(n)
    z_score = diff_sum / (std_residuals * np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return diff_sum, ci_margin_sum, p_value

def build_uplift_result_table(market: list, mae: float, mape: float,
                              diff_sum: float, diff_lower: float,
                              diff_upper: float, p_value: float) -> pd.DataFrame:
    """
    Builds a single-row DataFrame summarizing uplift evaluation metrics.

    Parameters:
    - market (list): List containing the market name (e.g., ['FR'])
    - mae (float): Mean Absolute Error of the model
    - mape (float): Mean Absolute Percentage Error of the model
    - diff_sum (float): Observed uplift (actual - predicted) during promo period
    - diff_lower (float): Lower bound of the confidence interval
    - diff_upper (float): Upper bound of the confidence interval
    - p_value (float): p-value of the uplift significance test

    Returns:
    - pd.DataFrame: A single-row DataFrame with named columns
    """
    data = np.array([
        market[0], mae, mape, diff_sum,
        round(diff_lower, 2), round(diff_upper, 2), p_value
    ]).reshape(1, 7)

    columns = ['market', 'mae', 'mape', 'uplift', 'uplift_lower', 'uplift_upper', 'p_value']
    return pd.DataFrame(data, columns=columns)

