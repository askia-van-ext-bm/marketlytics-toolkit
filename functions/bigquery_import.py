import pandas as pd
from google.colab import auth

def fetch_orderline_kpi(markets, start_date, project_id, dataset_name, orderline_table):
    """
    Exécute une requête BigQuery pour récupérer les KPIs des orderlines par marché.

    Parameters:
    - markets (list of str): Liste des marchés à inclure (ex: ['FR', 'DE'])
    - start_date (str): Date de début au format 'YYYY-MM-DD'
    - project_id (str): ID du projet GCP
    - dataset_name (str): Nom du dataset BigQuery
    - orderline_table (str): Nom de la table des orderlines

    Returns:
    - pd.DataFrame: Résultat de la requête
    """
    auth.authenticate_user()
    print("Authenticated")

    query = f"""
    DECLARE markets ARRAY<STRING>;
    SET markets = {markets};

    SELECT
        o.MARKET,
        DATE(o.DATETIME_CREATION_ORDERLINE_LOCAL_TIME) AS DATE_KPI,
        o.CLIENT_COUNTRY, o.CLIENT_GMA_CODE, o.CLIENT_GMA_NAME,
        SUM(o.QUANTITY) AS QUANTITY,
        COUNT(DISTINCT o.ORDERLINE_ID) AS ORDERS,
        SUM(o.GMV_LOCAL_CURRENCY) AS GMV_LOCAL_CURRENCY,
        COUNT(DISTINCT o.CLIENT_ID) AS CLIENTS
    FROM `{dataset_name}.{orderline_table}` o
    WHERE 1=1
        AND o.ORDERLINE_STATE IN (1,2,3,4,5)
        AND DATE(DATE_CREATION_ORDERLINE_LOCAL_TIME) >= '{start_date}'
        AND o.MARKET IN UNNEST(markets)
    GROUP BY 1,2,3,4,5
    ORDER BY 1,2,3,4,5
    """

    df = pd.io.gbq.read_gbq(query, project_id=project_id, dialect='standard')
    return df

