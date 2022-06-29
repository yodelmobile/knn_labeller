# Load sql_from_bq into the workspace
def sql_from_bq(project, dataset, table):
    """Set up SQL query and returns a dataFrame df2 from table."""
    
    from google.cloud import bigquery
    import pandas as pd
    #from decouple import config
    #from google.oauth2 import service_account
    #import os

    # Dev branch: Set google credentials below:
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config('auth_file_path')

    #credentials = service_account.Credentials.from_service_account_file(
    #    os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=["https://www.googleapis.com/auth/cloud-platform"],
    #    )

    # Dev branch: Set google client below:
    #client = bigquery.Client(credentials=credentials)
    client = bigquery.Client()
    
    table_id = project + '.' + dataset + '.' + table
    
    sql = "SELECT * FROM {} ORDER BY date DESC".format(table_id)
    
    # The client library uses the BigQuery Storage API to download results to a
    # pandas dataframe if the API is enabled on the project, the
    # `google-cloud-bigquery-storage` package is installed, and the `pyarrow`
    # package is installed.
    df = client.query(sql).to_dataframe()
    return(df)
