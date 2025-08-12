import pandas as pd
from google.colab import auth
from google.cloud import bigquery

def authenticate():
    """Authenticate user in Google Colab."""
    auth.authenticate_user()
    print("Authenticated")

def load_orderline_kpi(markets, start_date, dataset_name, orderline_table, project_id='data-backmarket-user', sql_path='sql/orderline_kpi_by_market.sql'):
    """
    Load KPI data from BigQuery using a parameterized SQL query.

    Args:
        markets (list): List of market codes (e.g., ['US', 'FR'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        dataset_name (str): Name of the BigQuery dataset
        orderline_table (str): Name of the table inside the dataset
        project_id (str): GCP project ID
        sql_path (str): Path to the SQL query file

    Returns:
        pd.DataFrame: Resulting dataframe from BigQuery
    """
    with open(sql_path, 'r') as file:
        query = file.read()

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("markets", "STRING", markets),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name),
            bigquery.ScalarQueryParameter("orderline_table", "STRING", orderline_table)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    return df
