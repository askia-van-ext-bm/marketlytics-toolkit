import pandas as pd

def compute_promocode_token_metrics(df: pd.DataFrame, groupby: list[str] = ['MARKET', 'TOKEN']) -> pd.DataFrame:
    """
    Computes aggregated metrics for promotional campaigns based on payment IDs and GMV.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns including 'TOKEN', 'PAYMENT_ID', 'GMV_LOCAL_CURRENCY'
    - groupby (list of str): Columns to group by for final aggregation (default: ['MARKET', 'TOKEN'])

    Returns:
    - pd.DataFrame: Aggregated metrics with number of distinct payments and average GMV
    """
    # Select token based payment
    paymentid_with_token = set(df.loc[df['TOKEN'].notna(), 'PAYMENT_ID'])

    # Agg GMV by payment
    df_sum = df[df['PAYMENT_ID'].isin(paymentid_with_token)].groupby(
        groupby + ['PAYMENT_ID']
    )['GMV_LOCAL_CURRENCY'].sum().reset_index()

    # Final agg by group
    df_agg = df_sum.groupby(groupby).agg(
        Campaign_Orders=('PAYMENT_ID', 'nunique'),
        Campaign_AOV=('GMV_LOCAL_CURRENCY', 'mean')
    ).reset_index()

    return df_agg

def pivot_and_reindex_daily_counts(daily_counts: pd.DataFrame, by_variable: str, metric: str) -> pd.DataFrame:
    """
    Pivots and reindexes a daily aggregated DataFrame to prepare it for time series analysis.

    Parameters:
    - daily_counts (pd.DataFrame): DataFrame containing daily aggregated metrics with columns 'DATE_KPI', the grouping variable, and the metric
    - by_variable (str): Column name used for grouping (e.g., 'CATEGORY_2_NAME')
    - metric (str): Name of the metric column to pivot (e.g., 'GMV_LOCAL_CURRENCY')

    Returns:
    - pd.DataFrame: Pivoted DataFrame with 'DATE_KPI' as datetime index and missing values filled with 0
    """
    daily_gmv_by_base_model = (
        daily_counts
        .pivot(index='DATE_KPI', columns=by_variable, values=metric)
        .reset_index()
        .set_index('DATE_KPI')
    )
    daily_gmv_by_base_model.index = pd.to_datetime(daily_gmv_by_base_model.index)
    return daily_gmv_by_base_model.fillna(0)

