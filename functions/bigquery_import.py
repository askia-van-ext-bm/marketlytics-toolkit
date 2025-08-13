import pandas as pd
from google.colab import auth

def fetch_orderline_kpi_by_client_gma(markets, start_date, project_id, dataset_name, orderline_table):
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

def fetch_orderline_kpi_with_promocodes(markets, tokens, start_year, project_id, join_type='LEFT'):
    """
    Récupère les KPIs des orderlines enrichis avec les codes promo.

    Parameters:
    - markets (list of str): Liste des marchés (ex: ['FR'])
    - tokens (list of str): Liste des tokens promo (ex: ['LAPTOPSUMMER'])
    - start_year (int): Année de début (ex: 2025)
    - project_id (str): ID du projet GCP
    - join_type (str): 'LEFT' ou 'INNER' pour le type de jointure

    Returns:
    - pd.DataFrame: Résultat de la requête
    """
    auth.authenticate_user()
    print("Authenticated")

    query = f"""
    DECLARE markets ARRAY<STRING>;
    DECLARE tokens ARRAY<STRING>;
    DECLARE start_year INT64;

    SET markets = {markets};
    SET tokens = {tokens};
    SET start_year = {start_year};

    WITH ORDERS AS (
        SELECT 
            o.MARKET,
            DATE(o.DATETIME_CREATION_ORDERLINE_LOCAL_TIME) AS DATE_KPI,
            o.CATEGORY_1_NAME,
            o.CATEGORY_2_NAME,
            o.CATEGORY_3_NAME,
            o.PRODUCT_MODEL,
            o.ORDERLINE_ID,
            o.PAYMENT_ID,
            o.CLIENT_ID,
            MAX(CASE WHEN o.SOURCE_PAYMENT_LABEL = "App" THEN 1 ELSE 0 END) AS IS_APP,
            SUM(o.QUANTITY) AS QUANTITY,
            SUM(o.FEE_LOCAL_CURRENCY) AS FEE_LOCAL_CURRENCY,
            SUM(o.GMV_LOCAL_CURRENCY) AS GMV_LOCAL_CURRENCY
        FROM `universe-prod-20220914.finance.universe_orderlines` o
        WHERE o.MARKET IN UNNEST(markets)
            AND o.ORDERLINE_STATE IN (1,2,3,4,5,6)
            AND o.DATE_CREATION_ORDERLINE_LOCAL_TIME IS NOT NULL
            AND EXTRACT(YEAR FROM o.DATE_CREATION_ORDERLINE_LOCAL_TIME) >= start_year
        GROUP BY 1,2,3,4,5,6,7,8,9
    ), PROMO_CODES AS (
        SELECT CONCAT(pn.PAYMENT_ID,'EU') AS PAYMENT_ID, pc.TOKEN, SUM(DISCOUNT_VALUE) AS DISCOUNT_VALUE
        FROM `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionnew` pn
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionconstraint` pc ON pn.CODE_PROMOTION_ID = pc.ID
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_categorypromotion` pcat ON pc.CATEGORY_ID = pcat.ID
        WHERE pn.IS_PAID = 1 AND pc.TOKEN IN UNNEST(tokens)
        GROUP BY 1,2
        UNION ALL
        SELECT CONCAT(pn.PAYMENT_ID,'US'), pc.TOKEN, SUM(DISCOUNT_VALUE)
        FROM `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionnew` pn
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionconstraint` pc ON pn.CODE_PROMOTION_ID = pc.ID
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_categorypromotion` pcat ON pc.CATEGORY_ID = pcat.ID
        WHERE pn.IS_PAID = 1 AND pc.TOKEN IN UNNEST(tokens)
        GROUP BY 1,2
        UNION ALL
        SELECT CONCAT(pn.PAYMENT_ID,'AP'), pc.TOKEN, SUM(DISCOUNT_VALUE)
        FROM `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionnew` pn
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_promotionconstraint` pc ON pn.CODE_PROMOTION_ID = pc.ID
        LEFT JOIN `customer-prod-220914.promotions_silver_eu.bo_merchant_categorypromotion` pcat ON pc.CATEGORY_ID = pcat.ID
        WHERE pn.IS_PAID = 1 AND pc.TOKEN IN UNNEST(tokens)
        GROUP BY 1,2
    )

    SELECT
        o.*, p.TOKEN, p.DISCOUNT_VALUE
    FROM ORDERS o
    {join_type.upper()} JOIN PROMO_CODES p ON o.PAYMENT_ID = p.PAYMENT_ID
    """

    df = pd.io.gbq.read_gbq(query, project_id=project_id, dialect='standard')
    return df
