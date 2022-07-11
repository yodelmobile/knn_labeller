def df_to_bq(df, project, dataset, table):
    """Function to write df to specific project, dataset and table in Google BigQuery"""
    
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
    
    # Construct a BigQuery client object.
    # Dev branch: Change google client definition below:
    #client = bigquery.Client(credentials=credentials)
    client = bigquery.Client()


    # TODO(developer): Set table_id to the ID of the table to create.
    table_id = project + '.' + dataset + '.' + table

    job_config = bigquery.LoadJobConfig(
        # Specify a (partial) schema. All columns are always written to the
        # table. The schema is used to assist in data type definitions.
        schema=[
            # Specify the type of columns whose type cannot be auto-detected. For
            # example the "title" column uses pandas dtype "object", so its
            # data type is ambiguous.
            bigquery.SchemaField("date", bigquery.enums.SqlTypeNames.TIMESTAMP),
            bigquery.SchemaField("__instert_date", bigquery.enums.SqlTypeNames.DATETIME),
            # Indexes are written if included in the schema by name.
            bigquery.SchemaField("campaign_name", bigquery.enums.SqlTypeNames.STRING),
        ],
        write_disposition="WRITE_TRUNCATE",
        
        time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date"
        )
        # Optionally, set the write disposition. BigQuery appends loaded rows
        # to an existing table by default, but with WRITE_TRUNCATE write
        # disposition it replaces the table with the loaded data.
    )

    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )  # Make an API request.
    job.result()  # Wait for the job to complete.

    table = client.get_table(table_id)  # Make an API request.
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
            )
        )
    print(
        "Partitioning: {}".format(table.time_partitioning
            )
        )
