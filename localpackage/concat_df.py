def concat_df(df1, df2, join_date, id_val):
  """Function for concatenating two dataFrames with identical columns including a date column, dropping all rows that overlap in date and id_val from df2"""
  
  import pandas as pd
  
  if df1 is None:
    print("df1 from concat_df() {}".format(type(df1)))
  else:
    # Normalise 'date' column values
    df1['date'] = pd.to_datetime(df1['date'])#.dt.date
    df2['date'] = pd.to_datetime(df2['date'])#.dt.date
    
    # Concatenate the new values (df1) with the values from the the datawarehouse where any values in date range and from this  are dropped 
    df3 = pd.concat([df1, df2.drop(df2[(df2['date'].dt.date >= join_date) & (df2['account_id'] == id_val)].index).reindex()], ignore_index=True)
    
    # Ensure date and __insert_date columns are in correct format
    df3['date'] = pd.to_datetime(df3['date'])#.dt.date
    df3['__insert_date'] = pd.to_datetime(df3['__insert_date'], utc=True)
    
    # Sort the concatenated column by date ascending
    df3.sort_values(by=['date'], inplace=True, ascending=True, ignore_index=True)
    return(df3)
