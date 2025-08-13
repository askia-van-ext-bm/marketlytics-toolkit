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
