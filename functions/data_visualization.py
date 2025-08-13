import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def filter_data(df, start, end, market, exclusion_list, exclusion_period):
    """Filters data according to period, market, and any exclusions."""
    df_filtered = df[(df["DATE_KPI"] >= start) & (df["DATE_KPI"] < end) & (df["MARKET"].isin(market))]
    if exclusion_period:
        df_filtered = df_filtered[(df_filtered["DATE_KPI"] < exclusion_period[0]) | (df_filtered["DATE_KPI"] > exclusion_period[1])]
    df_filtered['DATE_KPI'] = pd.to_datetime(df_filtered['DATE_KPI'])
    df_filtered = df_filtered[~df_filtered["CATEGORY_2_NAME"].isin(exclusion_list)]
    return df_filtered

def compute_daily_metric(df, by_variable, metric):
    """Aggregates data by day and target variable."""
    return df.groupby([df['DATE_KPI'].dt.date, by_variable])[metric].sum().reset_index(name=metric)

def get_top_correlated_features(daily_counts, by_variable, target, metric, top_corr, validation_start):
    """Identifies the variables most correlated with the target over the training period."""
    pivot_table = daily_counts.pivot(index='DATE_KPI', columns=by_variable, values=metric).fillna(0)
    pivot_train = pivot_table[pivot_table.index < validation_start.date()]
    correlations = pivot_table.corr()[target].drop(target)
    top_features = correlations.abs().sort_values(ascending=False).head(top_corr).index.tolist()
    return correlations, top_features, pivot_table

def plot_time_series(daily_counts, by_variable, top_features, target, metric, promo_start, promo_end):
    """Displays the time curves of the correlated variables and the target."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=daily_counts[daily_counts[by_variable].isin(top_features + [target])],
        x='DATE_KPI',
        y=metric,
        hue=by_variable,
        palette='Set2'
    )
    ax.axvspan(pd.Timestamp(promo_start), pd.Timestamp(promo_end), color='skyblue', alpha=0.3)
    plt.title("Daily GMV by correlated categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(pivot_table, top_features, target, market):
    """Displays a heat map of correlations between the selected variables and the target."""
    plt.figure(figsize=(8, 6))
    correlation_matrix = pivot_table[top_features + [target]].corr(method='pearson')
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f"{market} : Correlation for daily GMV by Base model")
    plt.tight_layout()
    plt.show()

def create_graph_daily_gmv(df, start, end, market, by_variable, target, metric, top_corr,
                           validation_start, promo_start, promo_end, exclusion_list, exclusion_period=()):
    """
    Generates daily GMV graphs and correlation analysis for a given target category.

    Parameters:
    - df (pd.DataFrame): Input dataset with daily GMV data
    - start, end (str): Date range for analysis
    - market (list): List of markets to include
    - by_variable (str): Column to group by (e.g., 'CATEGORY_2_NAME')
    - target (str): Target category to analyze
    - metric (str): Metric to aggregate (e.g., 'GMV_LOCAL_CURRENCY')
    - top_corr (int): Number of top correlated features to display
    - validation_start, promo_start, promo_end (datetime): Key dates for segmentation
    - exclusion_list (list): Categories to exclude from analysis
    - exclusion_period (tuple): Optional date range to exclude

    Returns:
    - daily_counts (pd.DataFrame): Aggregated daily metrics
    - correlations (pd.Series): Correlation scores with target
    - top_features (list): Most correlated categories
    """
    df_filtered = filter_data(df, start, end, market, exclusion_list, exclusion_period)
    daily_counts = compute_daily_metric(df_filtered, by_variable, metric)
    correlations, top_features, pivot_table = get_top_correlated_features(
        daily_counts, by_variable, target, metric, top_corr, validation_start
    )
    plot_time_series(daily_counts, by_variable, top_features, target, metric, promo_start, promo_end)
    plot_correlation_heatmap(pivot_table, top_features, target, market[0])
    return daily_counts, correlations, top_features
