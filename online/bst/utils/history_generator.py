import numpy as np

def rolling_mean(df, window):
    return df.groupby("ATM_ID").apply(lambda x: x['CLIENT_OUT'].rolling(window).mean().reset_index()).to_dict()['CLIENT_OUT']

def rolling_median(df, window):
    return df.groupby("ATM_ID").apply(lambda x: x['CLIENT_OUT'].rolling(window).median().reset_index()).to_dict()['CLIENT_OUT']

def rolling_std(df, window):
    return df.groupby("ATM_ID").apply(lambda x: x['CLIENT_OUT'].rolling(window).std().reset_index()).to_dict()['CLIENT_OUT']

def rolling_skew(df, window):
    return df.groupby("ATM_ID").apply(lambda x: x['CLIENT_OUT'].rolling(window).skew().reset_index()).to_dict()['CLIENT_OUT']

def rolling_zeros(df, window):
    return df.groupby("ATM_ID").apply(lambda x: x['CLIENT_OUT'].rolling(window).apply(lambda x: (x == 0.0).sum()).reset_index()).to_dict()['CLIENT_OUT']
    
def get_value_from_dict(gb_dict, days, atm_ids):
    return [gb_dict[(x,y)]  if (x,y) in gb_dict else np.nan for x, y in zip(atm_ids, days)]
