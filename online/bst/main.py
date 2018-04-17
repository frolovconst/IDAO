import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from utils.date_utils import *

PARAM_DAYS = 105
train = pd.read_csv('train.csv.zip', parse_dates=['DATE'])

# The dates to predict
pred_dates  = ['2017-08-16', '2017-08-17', '2017-08-18', '2017-08-19',
               '2017-08-20', '2017-08-21', '2017-08-22', '2017-08-23',
               '2017-08-24', '2017-08-25', '2017-08-26', '2017-08-27',
               '2017-08-28', '2017-08-29', '2017-08-30', '2017-08-31',
               '2017-09-01', '2017-09-02', '2017-09-03', '2017-09-04',
               '2017-09-05', '2017-09-06', '2017-09-07', '2017-09-08',
               '2017-09-09', '2017-09-10', '2017-09-11', '2017-09-12',
               '2017-09-13', '2017-09-14', '2017-09-15', '2017-09-16',
               '2017-09-17']

cols = ['DIP', 'DOW_0', 'DOW_1', 'DOW_2',
       'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6', 'DIM_1', 'DIM_2', 'DIM_3', 'DIM_4',
       'DIM_5', 'DIM_6', 'DIM_7', 'DIM_8', 'DIM_9', 'DIM_10', 'DIM_11',
       'DIM_12', 'DIM_13', 'DIM_14', 'DIM_15', 'DIM_16', 'DIM_17', 'DIM_18',
       'DIM_19', 'DIM_20', 'DIM_21', 'DIM_22', 'DIM_23', 'DIM_24', 'DIM_25',
       'DIM_26', 'DIM_27', 'DIM_28', 'DIM_29', 'DIM_30', 'DIM_31',
        'is_weeknd', 'days_till_weeknd', 'days_from_weeknd', 'weeknds_this_week',
#        'weeknds_prev_week', 'weeknds_next_week',
#         'atm_mean', 'atm_median', 'atm_std', 'atm_skew', 'atm_zeros'
       
       ]

weeknds = get_weeknds_set()
till_weeknd = get_days_till_weeknd(weeknds)
from_weeknd = get_days_from_weeknd(weeknds)
num_of_weeknds = get_number_of_weeknds(weeknds)

test_date = train.DATE.iloc[-992]
sep_date = train.DATE.iloc[-992-PARAM_DAYS+1]
train.drop(index=train[train.DATE<sep_date].index, inplace=True)
train["DOW"] = train.DATE.dt.dayofweek
train["DIM"] = train.DATE.dt.day
train["DIP"] = train.DATE.dt.dayofyear
prd_start_DOY = train.DIP.min()
train["DIP"] = train.DIP - prd_start_DOY
train = pd.get_dummies(train, columns=['DOW', 'DIM'])

train['is_weeknd'] = train.DATE.apply(lambda x: x in weeknds)
train['days_till_weeknd'] = train.DATE.apply(lambda x: till_weeknd[x])
train['days_from_weeknd'] = train.DATE.apply(lambda x: from_weeknd[x])
train['weeknds_this_week'] = train.DATE.apply(lambda x: num_of_weeknds[x])

train['prev_week_date'] = train.DATE.shift(7)
#train.fillna(pd.to_datetime("20141228", format='%Y%m%d'), inplace=True)
train['next_week_date'] = train.DATE.shift(-7)
#train.fillna(pd.to_datetime("20170924", format='%Y%m%d'), inplace=True)

train['weeknds_prev_week'] = train.prev_week_date.apply(lambda x: num_of_weeknds[x] if x in num_of_weeknds else 2)
train['weeknds_next_week'] = train.next_week_date.apply(lambda x: num_of_weeknds[x] if x in num_of_weeknds else 2)

train.drop(['prev_week_date', 'next_week_date'], axis=1, inplace=True)

train['atm_mean'] = train.groupby("ATM_ID").CLIENT_OUT.expanding().mean().shift().reset_index().CLIENT_OUT
train['atm_median'] = train.groupby("ATM_ID").CLIENT_OUT.expanding().median().shift().reset_index().CLIENT_OUT
train['atm_std'] = train.groupby("ATM_ID").CLIENT_OUT.expanding().std().shift().reset_index().CLIENT_OUT
train['atm_skew'] = train.groupby("ATM_ID").CLIENT_OUT.expanding().skew().shift().reset_index().CLIENT_OUT
train['atm_zeros'] = train.groupby("ATM_ID").CLIENT_OUT.expanding().apply(lambda x: (x == 0.0).sum()).shift().reset_index().CLIENT_OUT





hyper_param = {'n_estimators' : 100, 'learning_rate' : 0.1, 'max_depth' : 10, 'min_child_weight':20, 'subsample':1}
gen = lambda x : generate_month_rf(x, **hyper_param)
def generate_month_rf(series, **kwargs):
    dt_srs = pd.Series(pd.to_datetime(pred_dates))
    gs = pd.DataFrame({'DATE': dt_srs, 'DOW': dt_srs.dt.dayofweek, 'DIM': dt_srs.dt.day, 'DIP': dt_srs.dt.dayofyear - prd_start_DOY})
    gs = pd.get_dummies(gs, columns=['DOW', 'DIM'])
    gs['is_weeknd'] = gs.DATE.apply(lambda x: x in weeknds)
    gs['days_till_weeknd'] = gs.DATE.apply(lambda x: till_weeknd[x])
    gs['days_from_weeknd'] = gs.DATE.apply(lambda x: from_weeknd[x])
    gs['weeknds_this_week'] = gs.DATE.apply(lambda x: num_of_weeknds[x])

    gs['prev_week_date'] = gs.DATE.shift(7)
    #train.fillna(pd.to_datetime("20141228", format='%Y%m%d'), inplace=True)
    gs['next_week_date'] = gs.DATE.shift(-7)
    #train.fillna(pd.to_datetime("20170924", format='%Y%m%d'), inplace=True)

    gs['weeknds_prev_week'] = gs.prev_week_date.apply(lambda x: num_of_weeknds[x] if x in num_of_weeknds else 2)
    gs['weeknds_next_week'] = gs.next_week_date.apply(lambda x: num_of_weeknds[x] if x in num_of_weeknds else 2)

    gs.drop(['prev_week_date', 'next_week_date'], axis=1, inplace=True)
       
    gs['atm_mean'] = series['atm_mean']
    gs['atm_median'] = series['atm_median']
    gs['atm_std'] = series['atm_std']
    gs['atm_skew'] = series['atm_skew']
    gs['atm_zeros'] = series['atm_zeros']
    rgs = XGBRegressor(n_jobs=-1, **kwargs)
    rgs.fit(series[cols], series.CLIENT_OUT)
    
    gs['CLIENT_OUT'] = rgs.predict(gs[cols])
    return gs[['DATE', 'CLIENT_OUT']]

gb = train.groupby('ATM_ID').apply(gen).reset_index()[['DATE', 'ATM_ID', 'CLIENT_OUT']]
gb.CLIENT_OUT = np.where(gb['CLIENT_OUT'] <0, 0,gb['CLIENT_OUT']) 

gb.to_csv('submission.csv', index=False)