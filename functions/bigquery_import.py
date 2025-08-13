import pandas as pd
from google.colab import auth

def fetch_orderline_kpi_by_client_gma(markets, start_date, project_id, dataset_name, orderline_table):
    """
    Executes a BigQuery query to retrieve orderline KPIs by client gma.
    
    Parameters:
    - markets (list of str): List of markets to include (e.g., ['FR', 'DE'])
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - project_id (str): GCP project ID
    - dataset_name (str): Name of the BigQuery dataset
    - orderline_table (str): Name of the orderline table
    
    Returns:
    - pd.DataFrame: Query result
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
    Retrieves orderline KPIs enriched with promotional codes.
    
    Parameters:
    - markets (list of str): List of markets (e.g., ['FR'])
    - tokens (list of str): List of promotional tokens (e.g., ['LAPTOPSUMMER'])
    - start_year (int): Starting year (e.g., 2025)
    - project_id (str): GCP project ID
    - join_type (str): 'LEFT' or 'INNER' to specify the type of join
    
    Returns:
    - pd.DataFrame: Result of the query
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
