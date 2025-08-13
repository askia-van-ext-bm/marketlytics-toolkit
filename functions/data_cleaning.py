import pandas as pd

def convert_dataframe_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Converts specified columns of a DataFrame to given types.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - type_map (dict): Dictionary mapping types to lists of column names,
      e.g. {'int': ['QUANTITY'], 'float': ['GMV_LOCAL_CURRENCY'], 'datetime': ['DATE_KPI']}

    Returns:
    - pd.DataFrame: A copy of the DataFrame with converted column types
    """
    df_converted = df.copy()

    for dtype, columns in type_map.items():
        for col in columns:
            if dtype == 'datetime':
                df_converted[col] = pd.to_datetime(df_converted[col], format='%Y-%m-%d', errors='coerce')
            else:
                df_converted[col] = df_converted[col].astype(dtype, errors='ignore')

    return df_converted
