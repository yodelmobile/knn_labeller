
def if_tbl_exists(**bqarg):
  """Check if table indicated by project, dataset and table keyword arguments exists and return True/False"""
  
  from google.cloud import bigquery
  from decouple import config
  from google.oauth2 import service_account
  #import os

  # Dev branch: Set google credentials below:
  #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config('auth_file_path')
  #credentials = service_account.Credentials.from_service_account_file(
  #  os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=["https://www.googleapis.com/auth/cloud-platform"],
  #  )

  # Dev branch: Set google client below:
  #client = bigquery.Client(credentials=credentials)
  client = bigquery.Client()
  
  #client = bigquery.Client(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
  table_id = bqarg['project']+'.'+bqarg['dataset']+'.'+bqarg['table']
  
  # Using method from https://cloud.google.com/bigquery/docs/samples/bigquery-table-exists
  from google.cloud.exceptions import NotFound
  try:
    client.get_table(table_id)
    return True
  except NotFound:
    return False
