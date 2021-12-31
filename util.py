import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import calendar
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
sns.set_theme(style="whitegrid")

encoder_dic = {}

def get_gen_date(x):
    date = x.split("-")
    month = str(int(date[1]))
    day = str(int(date[2]))
    return  month + "/" + day + "/" + date[0]

def gb_util(df,cols,target,target_col,op = "sum"):    
    df = df.groupby(cols, as_index = False).agg({target : op})
    cols.append(target_col)
    df.columns = cols
    return df

def encode(df,target):
    encoder = LabelEncoder()
    encoder.fit(df[target])
    df[target] = encoder.transform(df[target])
    encoder_dic[target] = encoder
    
def map_time(time):
    time = time.split(":")[0]
    time = int(time)
    if time >= 6 and time <= 12:
        return "Morning"
    elif time > 12 and time <= 16:
        return "Noon"
    elif time > 16 and time <= 20:
        return "Evening"
    elif time > 20 and time <= 24:
        return "Night"
    else:
        return "Mid night"

# def get_week(date):
#     return int(int(date.split("/")[1])/8) + 1

def get_week(date_str):
    year  = int(date_str.split("/")[2])
    month = int(date_str.split("/")[0])
    day   = int(date_str.split("/")[1])
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return week_of_month
 
def calc_mean(df):
    ans = np.zeros(df.shape[0])
    for index, row in tqdm(df.iterrows()):
        ans[index] = df[df["chw_store_id"] == row["chw_store_id"]][df["min_date"] < row["min_date"]].visitors.mean()
    return ans

def calc_median(df):
    ans = np.zeros(df.shape[0])
    for index, row in df.iterrows():
        ans[index] = df[df["chw_store_id"] == row["chw_store_id"]][df["min_date"] < row["min_date"]].visitors.mean()
    return ans

def calc_mode(df):
    ans = np.zeros(df.shape[0])
    for index, row in df.iterrows():
        ans[index] = df[df["chw_store_id"] == row["chw_store_id"]][df["min_date"] < row["min_date"]].visitors.mean()
    return ans

def create_lags_customs(df,cols,lag,index_cols,lag_col,null_value=-1):
    copy_cols = index_cols + cols
    df_temp = df[copy_cols].copy()
    df_temp[lag_col] = df_temp[lag_col] + lag
    columns = []
    for col in df_temp.columns.values:
        if col in index_cols:
            columns.append(col)
        else:
            columns.append(col + "_lag_" + str(lag))
    df_temp.columns = columns
    df = df.merge(df_temp, how = "left", on=index_cols)
    for col in [col for col in columns if not col in index_cols]:
        df[col].fillna(null_value, inplace = True)
    return df
    
def create_lags(df,cols, lag, index_cols, null_value = 0):
    copy_cols = index_cols + cols
    df_temp = df[copy_cols].copy()
    df_temp["date"] = pd.to_datetime(df_temp['date']) + pd.to_timedelta(lag,unit='d')
    df_temp["date"] = pd.to_datetime(df_temp['date']).dt.strftime("%m/%d/%y")
    columns = []
    for col in df_temp.columns.values:
        if col in index_cols:
            columns.append(col)
        else:
            columns.append(col + "_lag_" + str(lag))
            
    df_temp.columns = columns
    df = df.merge(df_temp, how = "left", on=index_cols)
    for col in [col for col in columns if not col in index_cols]:
        df[col].fillna(null_value, inplace = True)
    return df

def create_lags_with_name(df,cols, lag, index_cols, name,null_value = 0):
    copy_cols = index_cols + cols
    df_temp = df[copy_cols].copy()
    df_temp["date"] = pd.to_datetime(df_temp['date']) + pd.to_timedelta(lag,unit='d')
    df_temp["date"] = pd.to_datetime(df_temp['date']).dt.strftime("%m/%d/%y")
    columns = []
    for col in df_temp.columns.values:
        if col in index_cols:
            columns.append(col)
        else:
            columns.append(name)
            
    df_temp.columns = columns
    df = df.merge(df_temp, how = "left", on=index_cols)
    for col in [col for col in columns if not col in index_cols]:
        df[col].fillna(null_value, inplace = True)
    return df

def get_lag_mean(df,cols,lags,index_cols,null_value = 0):
    for lag in lags:
        df = create_lags(df,cols,lag,index_cols,null_value)
    mean_cols = [cols[0] + "_lag_" + str(lag) for lag in lags]
    ans = df[mean_cols].mean(axis = 1).values
    df.drop(mean_cols, inplace = True, axis="columns")
    return ans
    
def get_lag_median(df,cols,lags,index_cols,null_value = 0):
    for lag in lags:
        df = create_lags(df,cols,lag, index_cols, null_value)
    mean_cols = [cols[0] + "_lag_" + str(lag) for lag in lags]
    ans = df[mean_cols].median(axis = 1).values
    df.drop(mean_cols, inplace = True, axis="columns")
    return ans

def prominent(df,target_col, cols, index_cols,map_col,name, null_value):
    copy_cols = index_cols + cols
    df_temp = df[copy_cols].copy()
    df_temp = df.loc[df.groupby(index_cols)[target_col].idxmax()]
    return df_temp
