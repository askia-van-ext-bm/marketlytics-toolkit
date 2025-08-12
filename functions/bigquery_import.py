from google.cloud import bigquery

def fetch_orderline_kpi(dataset_name: str, orderline_table: str, markets: list[str], start_date: str) -> list[dict]:
    client = bigquery.Client()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("markets", "STRING", markets),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        ]
    )

    query = f"""
    DECLARE dataset_name STRING DEFAULT '{dataset_name}';
    DECLARE orderline_table STRING DEFAULT '{orderline_table}';
    DECLARE markets ARRAY<STRING>;
    DECLARE start_date DATE;

    SET markets = @markets;
    SET start_date = @start_date;

    EXECUTE IMMEDIATE FORMAT(
      '''
      SELECT
          o.MARKET,
          DATE(o.DATETIME_CREATION_ORDERLINE_LOCAL_TIME) AS DATE_KPI,
          o.CLIENT_COUNTRY, o.CLIENT_GMA_CODE, o.CLIENT_GMA_NAME,
          SUM(o.QUANTITY) AS QUANTITY,
          COUNT(DISTINCT o.ORDERLINE_ID) AS ORDERS,
          SUM(o.GMV_LOCAL_CURRENCY) AS GMV_LOCAL_CURRENCY,
          COUNT(DISTINCT o.CLIENT_ID) AS CLIENTS
      FROM `%s.%s` o
      WHERE 1=1
          AND o.ORDERLINE_STATE IN (1,2,3,4,5)
          AND DATE(DATE_CREATION_ORDERLINE_LOCAL_TIME) >= start_date
          AND o.MARKET IN UNNEST(markets)
      GROUP BY 1,2,3,4,5
      ORDER BY 1,2,3,4,5
      ''',
      dataset_name, orderline_table
    );
    """

    job = client.query(query, job_config=job_config)
    return [dict(row) for row in job.result()]
