### 0.0 Add libraries
from datetime import date, timedelta, datetime
import pandas as pd
# from google.cloud import bigquery
# from google.cloud import storage
import numpy as np
import string
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

### 1.0 Assign required variables

#### Variable Assignment ####

# Column name of the target column for prediction
target_col = 'channel'

# Set drop threshold for low_row_drop() function
drop_thresh = 100

# Set pattern and replacement variables for strip_non_values() function call
pattern = '(^$|^ $|^None$|^-$|^Unknown$)'
replacement = np.nan

# Set u_val_cap, maximum number of unique values for columns
u_val_cap=800

# Null thresh is the max percentage threshold for null values per column.
null_thresh = 0.85

# Number of columns that must be non-null for all rows
no_null_cols = 12

# Set training set random sample length variable
sample_length = 50000

#List of Hyperparameter lists that to use for GridSearch.
list_leaf_size = [1,2,28,29,30,31,32]
list_n_neighbors = list(range(3,18))
list_p = [1,2]
list_algorithm=['auto']
list_metric=['minkowski']
list_weights=['distance']

# Set test_pct variable to determine the train/test split share
test_pct = 0.40

#
#
#
#
#
### 2.0 Import data

#### Data import and selection ####



#
#
#
#
#
### 3.0 Select and clean relevant data

#### Function definitions ####
def low_row_drop(df, **kwargs):
    """Function to drop any rows from *df* that have a *column* value that appears less than *thresh*  
    times and return truncated DataFrame accepts kwargs """
    thresh = kwargs.get("thresh", 100)
    defalut_colname = df.columns[0]
    column = kwargs.get("column", defalut_colname)
    # Check  
    if isinstance(column,str):
        if column in df.columns:
            # Count number of rows per column value 
            counts = df[column].value_counts()
            # Mask df with counts, dropping all rows that occur less than thresh times
            df = df[~df[column].isin(counts[counts <= thresh].index)]
            return df
        else:
            print(column + ": str-like column name not present in df")
    elif isinstance(column,list):
        for c in column:
            counts = df[c].value_counts()
            df = df[~df[c].isin(counts[counts <= thresh].index)]
    else:
        print('Keyword argument "column" is neither string nor list type')
    return df

# Function to strip out missing values
def strip_non_values(df, **kwargs):
    """Funciton to take in dataframe df and strip out whitespace in object columns 
    and replace selected/missing values with a replacement eg. NULL.
    ###### Keyword arguments ######
    pos arg df: DataFrame to search for pattern and replace with repl
    kwarg pattern: regex or string to be replaced - defaults to '^None$'
    kwarg repl: value used to replace pattern - defaults to np.nan
    kwarg regex: Boolean to decide treat pattern as regex? - defaults to True
    kwarg columns: Column or column list to trim whitespace - defaults all object dtype columns
    kwarg selectall: Boolean, apply pattern replacement to all columns in df? - defaults to True 
        - If false only column or columns in columns list will be searched
    """
    # Set kwarg arguments and related defaults
    pattern = kwargs.get("pattern", '^None$')
    repl = kwargs.get("replacement", np.nan)
    regex = kwargs.get("regex", True)
    # Check for presents of columns kwarg and sets 'object' all dtype if not
    objects = df.select_dtypes(include=['object']).columns.tolist()
    columns = kwargs.get("columns", objects)
    # Set selectall value to default True - This selectall 
    selectall = kwargs.get("selectall", True)
    # Strip out any whitespace on either side of values in the selected columns
    df[columns] = df[columns].apply(lambda x: x.str.strip())
    # Replace all spaces, empty and 'None' values with np.nan/NULL
    df.replace(pattern, repl, regex=regex, inplace=True)
    return df

# Define cleaning function for 
def drop_majority_null(df, **kwargs):
    """Funciton to drop majority null columns from dataFrame df
    pos arg df: DataFrame to check for majority NULL columns
    kwarg thresh: Percentage NULL value per column threshold above which column is dropped from df
    """
    thresh = kwargs.get("thresh", 0.6)
    df = df.loc[:, df.isnull().mean() < thresh]
    return df

# Define funciton to drop all columns with eaqual or less than x distinct value from df
def drop_low(df, **kwargs):
    """Funciton to drop all columns from df that have only x or less distinct values
    pos arg df: DataFrame to check for number of unique values per column
    kwarg max_vals: Number of unique vals per col and under 
    """
    min_vals = kwargs.get("min_vals", 1)
    res = df
    for col in df.columns:
        if len(df[col].unique()) <= min_vals:
            res = res.drop(col,axis=1)
    return res

# Define funciton to drop all columns with eaqual or more than x distinct value from df
def drop_high(df, **kwargs):
    """Funciton to drop allcolumns from df that have only x distinct values
    pos arg df: DataFrame to check for number of unique values per column
    kwarg min_vals: Number of unique vals per col and under - defaults to 1
    kwarg max_vals: Number of unique vals per col and under 
    """
    max_vals = kwargs.get("max_vals")
    res = df
    for col in df.select_dtypes(include=['object']).columns:
        if len(df[col].unique()) >= max_vals:
            res = res.drop(col,axis=1)
    return res

# Define function to take in dataFrame and change all type-selected columns to another type
def col_type_change(df, **kwargs):
    """Funciton to select all columns of one type and change to another datatype allcolumns from 
    df that have only x distinct values
    pos arg df: DataFrame to check for datatype columns
    kwarg select_type: type of col to select - defaults to 'object' 
    kwarg change_type: column typ to change all seldct type columns to - defaults to 'category'
    """
    dtype_1 = kwargs.get("select_type", 'object')
    dtype_2 = kwargs.get("change_type", 'category')
    LABELS = df.select_dtypes(include=[dtype_1]).columns.tolist()
    # Define the lambda function named categorize_label to convert column x 
    # from any data type into categorical data types using 
    # the .apply() method: x.astype('category').
    categorize_label = lambda x: x.astype('category')
    # Pass df[LABELS] to the lambda function above and assign back to df[LABELS] 
    #to convert the subset of data df[LABELS] to type 'category'.
    df[LABELS] = categorize_label(df[LABELS])
    return(df)

# Function to take in one dataFrame and return two dataframes,
# one 'train' with all non null column values for use in train/test operations 
# and 'predict' dataFrame all null column value rows to use trained algorythom to predict
def split_by_mask(df, mask, **kwargs):
    """Funciton that takes two dataframes, one with NULL vlaues and one LabelEncoded and 
    splits encoded df based on null/notnull of selected column in unencoded dataFrame
    and returns two separate dataFrame copys
    pos arg df1: DataFrame to mask for NULL column values
    pos arg df2: DataFrame to split by notnull() mask
    kwarg selected_col: column name to use to mask df1 and split df2 
    """
    selected_col = kwargs.get('selected_col', df.columns[0])
    # default_mask = df[[selected_col]].notnull().all(1)
    # mask = df[[selected_col]].notnull().all(1)
    df_unknown = df[~mask]  # Filter by inverse of mask
    df_known = df[mask]  # Filter by mask
    return(df_known, df_unknown)

# LabelEncoder() specific functions for establishing column specific label encoders.
from sklearn.preprocessing import LabelEncoder

# Define text_to_numbers() which runs LabelEncoder() against each non-float/bool/int64 column
def text_to_numbers(df):
        le_dict = dict()
        for i in df.columns:
            if df[i].dtype not in ["float64", "bool", "int64"]:
                le_dict[i] = LabelEncoder()
                df[i] = le_dict[i].fit_transform(df[i])
    
        return df, le_dict

# Define text_to_numbers() which runs LabelEncoder() against each non-float/bool/int64 column
def text_to_numbers_2(df, le_dict):
        for i in le_dict.keys():
            df[i] = le_dict[i].transform(df[i])
                
        return df

# Check if key exists in dict.
def check_key_exist(test_dict, key):
    try:
        value = test_dict[key]
        return True
    except KeyError:
        return False

    # Define numbers_to_text which runs dictionary le_dict using inverse_transform() on each row in the dataframe df
def numbers_to_text(df, le_dict):
    for i in df.columns:
        if check_key_exist(le_dict, i):
            df[i] = le_dict[i].inverse_transform(df[i])
        else:
            pass
    return df

#
#
#
#
#

#### Data Cleaning, normalization and encoding of dataframe ####

# Normalise the channel column of df using regex
target_col = 'channel'

#### Regex to normailze existing target_col values ####
# Use regex replace on target col with pattern to find and normalize values
df[target_col] = df[target_col].replace(
    [r'^.*Organic.*$', 'Apple.*$','Google (Ads|Ad[Ww]ords).*$','^.*(Facebook|Instagram|Messenger|IG).*$',
     'TikTok.*$','^.*Snap.*$','^.*MOLOCO.*$'],
    [r'Organic','Apple Search','Google Ads','Facebook','TikTok Ads','Snapchat Ads','MOLOCO'], 
    regex=True
)


# Use above defined low_row_drop() function to drop all rows that occur less than 100 times per unique channel value  
df = low_row_drop(df, column=target_col, thresh=drop_thresh)

# Use above defined strip_non_values() function remove whitespace in object cols and replace NULL equivalent values
df = strip_non_values(df, pattern = pattern, repl = replacement, regex = True)

# Drop columns with only one unique value
df=drop_low(df, min_vals=1)

# Drop columns with too many (over 800) unique values
df=drop_high(df, max_vals=u_val_cap)

# Call above defined drop_majority_null() function with threshhold set at 85% NULL columns to drop
df = drop_majority_null(df, thresh=null_thresh)

# Drop all rows that don't have at least 12 non-null column values
df = df.dropna(thresh=no_null_cols)

# Use Null 'channel' column values to create 
mask = df[[target_col]].notnull().all(1)

# set up object and float colnames from cleaned df
objects = df.select_dtypes(include=['object']).columns.tolist()
floats = df.select_dtypes(include=['float64']).columns.tolist()

# Replace nan floats with 0
df[floats] = df[floats].replace(np.nan, 0)

# Replace nan objects with "" empty string
df[objects] = df[objects].fillna('')

# Call above defined df_col_type_change() function to change all 'object' dtype columns to 'category' dtype
df = col_type_change(df, select_type='object', change_type='category')

# Set copy of df
df_conv = df.copy()

# Run text_to_numbers() on the cleaned and prepared dataframe df to produce encoded df
df_conv, le_dict = text_to_numbers(df_conv)

# Run split_by_mask using null value 'channel' mask from above to split out train and prediction data
train, pred = split_by_mask(df_conv, mask, selected_col=target_col)

#
#
#
#
#

#### Sort values and reduce size of train set to reduce training time on machine ####
# Set length of train
if (train.shape[0]) <= sample_length:
    train_r = train.sort_values('date',ascending = False)
else:
    train_r = train.sample(n=sample_length)#.sort_values('date',ascending = False)
    
pred_r = pred.sort_values('date',ascending = False)

train_r = train_r.sort_values('date',ascending = False)

#
#
#
#
#

### 4.0 Train the model

#### Split encoded non-null dataFrame into train and test data sets ####
def tts_func(train, pred, **kwargs):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    
    target_col = kwargs.get('target_col', df.columns[0])
    test_size = kwargs.get('test_size', 0.20)
    
    # Separate the features X of df from the target variable y
    y = train[target_col].values
    X = train.drop(target_col, axis=1).values
    
    # Separate the features X of pred from the target variable y
    y_pred = pred[target_col].values
    X_pred = pred.drop(target_col, axis=1).values
    
    # Use train_test_split() to set values for the training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    # Return the train and test  data
    return(X_train, X_test, y_train, y_test, X_pred, y_pred)

import json

#### Hyper parameter tuning - use train-test data to select best parameters for Scalar|Classifier Pipeline ####
def model_tuning(X_train, y_train, X_test, y_test, y_pred, **kwargs):
    
    #List Hyperparameters that we want to tune.
    leaf_size = kwargs.get('leaf_size', list(range(1,4)))
    n_neighbors = kwargs.get('n_neighbors', list(range(1,8)))
    p = kwargs.get('p', list(range(1,3)))
    algorithm= kwargs.get('algorithm', ['auto'])
    metric= kwargs.get('metric', ['minkowski'])
    weights= kwargs.get('weights', ['distance'])

    #Convert to dictionary
    hyperparameters = {
        'knn__leaf_size': leaf_size, 
        'knn__n_neighbors': n_neighbors, 
        'knn__p': p,
        'knn__algorithm': algorithm,
        'knn__weights': weights,
        'knn__metric': metric
    }

    # Create new standard scalar.
    sc = StandardScaler()

    # Create new KNN classifier.
    knn = KNeighborsClassifier()

    steps = [('sc',sc),
             ('knn', knn)]

    pipeline = Pipeline(steps)

    # Use GridSearch to discern highest accuracy hyperparameters
    clf = GridSearchCV(pipeline, hyperparameters, cv=10, scoring='accuracy')

    # Fit the model
    result = clf.fit(X_train,y_train)

    # Set the value of best classifier
    best_clf = result.best_estimator_
    
    # Set score for clf using test data
    score = clf.score(X_test, y_test)
    
    # Predict unknown labels
    y_pred = clf.predict(X_pred)

    # Create array of confidence probabilities for predicted values
    y_prob = clf.predict_proba(X_pred)
    
    # Select the max prob for each prediction
    max_probs = np.amax(y_prob, axis=1)
    
    # Create 2d array of predictions and confidence values fore each prediction
    y_preds = np.array((y_pred,max_probs)).T
    
    return(y_preds, score)
#
#
#
#
#

#### KNN missing value imputation method from https://www.askpython.com/python/examples/impute-missing-data-values

# Use above defined tts_func() to split the 'train' dataFrame into train and test sets
X_train, X_test, y_train, y_test, X_pred, y_pred = tts_func(train_r, pred_r, target_col= target_col, test_size=test_pct)

y_preds, score  = model_tuning(X_train, y_train, X_test, y_test, X_pred, 
                               leaf_size= list_leaf_size, 
                               n_neighbors= list_n_neighbors, p= list_p,
                               algorithm= list_algorithm, metric= list_metric, 
                               weights= list_weights)

# Turn numpy array of predicted values into dataFrame using index of pred_r
df_pred = pd.DataFrame(y_preds, index=pred_r.index.copy())

# Create column header for target col probability column
prob_head = target_col+'_probs'

# Name column of df_pred with the target_col variable
df_pred.columns = [target_col, prob_head]

# Set target col of df_pred as int64 dtype
df_pred.loc[:,target_col] = df_pred.loc[:,target_col].astype('int64')

# Convert df_pred back to text using the LabelEncoder() dictionary le_dict
df_pred = numbers_to_text(df_pred, le_dict)

# convert pred dataFrame back to text
df_k, df_u = split_by_mask(df, mask, selected_col= target_col)

# Set new variable target_new wich is target_col(_new)
target_new = target_col+'_new'

# Change colname of target_col in df_pred
df_pred=df_pred.rename(columns = {target_col:target_new})

# Join the df_pred target_col_new onto pred DF
df_u = df_u.join(df_pred)

# Copy target col of known DF df_k and name target_new
df_k.loc[:,target_new] = df_k.loc[:,target_col]

# Add in probability = 1.0 column for every known column label value in df_k
df_k.loc[:, prob_head] = 1.0

# Concatenate the train df and the predictions df together
df_new = pd.concat([df_k, df_u])

df_new_j = df_new.loc[:,[target_new,prob_head]].copy()

# Join the 'channel_new' column of the concatenated dfs from above onto df_final
df_final = df_init.join(df_new_j)

#
#
#
#
#

### 4.0 Merge and export data

#### Merge processed data with previous by date and export to BigQuery ####