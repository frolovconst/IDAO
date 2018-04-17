import pandas as pd
import numpy as np
import xgboost as xgb

# The dates to predict
train = pd.read_csv('train.csv.zip')
date_list = train.drop_duplicates('DATE').DATE.tolist()

pred_dates = ['2017-08-16', '2017-08-17', '2017-08-18', '2017-08-19',
              '2017-08-20', '2017-08-21', '2017-08-22', '2017-08-23',
              '2017-08-24', '2017-08-25', '2017-08-26', '2017-08-27',
              '2017-08-28', '2017-08-29', '2017-08-30', '2017-08-31',
              '2017-09-01', '2017-09-02', '2017-09-03', '2017-09-04',
              '2017-09-05', '2017-09-06', '2017-09-07', '2017-09-08',
              '2017-09-09', '2017-09-10', '2017-09-11', '2017-09-12',
              '2017-09-13', '2017-09-14', '2017-09-15', '2017-09-16',
              '2017-09-17']
ATM_IDs = train.ATM_ID.unique()

rows = []
for ATM in ATM_IDs:
    for date in pred_dates:
        rows.append([date, ATM, -1])

test = pd.DataFrame(rows, columns=['DATE', 'ATM_ID', 'CLIENT_OUT'])

train = pd.concat([train, test])

train['DATE'] = pd.to_datetime(train['DATE'])
train['Month'] = train.DATE.dt.month
train['Year'] = train.DATE.dt.year
train['Day'] = train.DATE.dt.day
train['WeekDays'] = train.DATE.dt.weekday

dict_days = {d:i for i,d in enumerate(pd.date_range('2015-01-01', '2017-09-17'))}
train['Num_day'] = train['DATE'].map(dict_days)
train['Num_Week'] = train['Num_day'] // 7

train = train.sort_values(['ATM_ID', 'DATE'])
train = train.reset_index(drop = True)

atm_week_dict = train.groupby(['ATM_ID', 'WeekDays']).CLIENT_OUT.apply(list).to_dict()


def add_week(x, d=10):
    ind_st = x[2] - d - 5
    if ind_st < 0:
        ind_st = 0
    ind_f = x[2] - 5
    if ind_f < 1:
        return 0
    else:
        return np.mean(atm_week_dict[(x[0], x[1])][ind_st:ind_f])


train['mean_week_10'] = train[['ATM_ID', 'WeekDays', 'Num_Week']].apply(lambda x: add_week(x.values, d=10), axis=1)
train['mean_week_7'] = train[['ATM_ID', 'WeekDays', 'Num_Week']].apply(lambda x: add_week(x.values, d=7), axis=1)
train['mean_week_4'] = train[['ATM_ID', 'WeekDays', 'Num_Week']].apply(lambda x: add_week(x.values, d=4), axis=1)


def count_zero(window=33, dif=33):
    dict_zero = []
    dict_mean = []
    dict_median = []
    dict_std = []
    dict_dif = []

    for ATM in train.ATM_ID.unique():
        atm_mask = train.ATM_ID == ATM
        client_list = train[atm_mask].CLIENT_OUT.tolist()
        date_list = train[atm_mask].Num_day.tolist()
        for i, d in enumerate(date_list):
            ind_start = i - dif - window
            ind_finish = i - dif
            calc_list = np.array(client_list[ind_start: ind_finish])
            if len(calc_list) != 0:
                dict_zero += [calc_list[calc_list == 0].shape[0]]
                dict_mean += [calc_list.mean()]
                dict_median += [np.median(calc_list)]
                dict_std += [calc_list.std()]
                dict_dif += [calc_list.max() - calc_list.min()]
            else:
                dict_zero += [0]
                dict_mean += [0]
                dict_median += [0]
                dict_std += [0]
                dict_dif += [0]

    train['client_out_zero_' + str(window)] = dict_zero
    train['client_out_mean_' + str(window)] = dict_mean
    train['client_out_median_' + str(window)] = dict_median
    train['client_out_std_' + str(window)] = dict_std
    train['client_out_dif_' + str(window)] = dict_dif

window = 7
dif = 35
count_zero(window = window, dif = dif)

train_col = ['Month', 'Year', 'Day', 'WeekDays','ATM_ID', 'mean_week_10', 'mean_week_7', 'mean_week_4',
       'client_out_zero_7', 'client_out_mean_7', 'client_out_median_7',
       'client_out_std_7', 'client_out_dif_7']


class XGB():
    
    dict_pred = {}
    numb_trees = []
    
    def xgb_train(self, train_data, train_target, test_data, test_target, pred_data, n_e = 10000):
        ts = xgb.DMatrix(np.array(pred_data))
        tr = xgb.DMatrix(np.array(train_data), np.array(train_target))
        te = xgb.DMatrix(np.array(test_data), np.array(test_target))
        param = {'max_depth': 3,'eta': 0.1, 'objective': 'reg:linear', 'eval_metric': 'mae', 'subsample':0.9,
                    'colsample_bytree':1}

        evallist  = [(tr, 'train'), (te, 'test')]

        bst = xgb.train(param, tr, n_e, evallist, verbose_eval = False,
                       maximize = False, early_stopping_rounds = 40)
        
        self.numb_trees += [bst.best_ntree_limit]
        return bst.predict(ts, ntree_limit = bst.best_ntree_limit)
        
    def fit_predict(self, train_data, test_data, train_col):
        pred = np.zeros(len(test_data))
        dates = train_data.drop_duplicates('DATE')['DATE'].tolist()
        start_date = dates[-30]
        for ID in train_data.ATM_ID.unique():
            cur_train_df = train_data[(train_data.DATE <= start_date) & (train_data.ATM_ID == ID)]
            cur_valid_df = train_data[(train_data.DATE > start_date) & (train_data.ATM_ID == ID)]
            cur_test_df = test_data[test_data.ATM_ID == ID] 
            pred[cur_test_df.index] = self.xgb_train(cur_train_df[train_col], cur_train_df['CLIENT_OUT'], 
                                                 cur_valid_df[train_col], cur_valid_df['CLIENT_OUT'], cur_test_df[train_col])
        ans = pd.DataFrame()
        ans['DATE'] = test_data.DATE.tolist()
        ans['ATM_ID'] = test_data.ATM_ID.tolist()
        ans['CLIENT_OUT'] = pred
        self.numb_trees = []
        return ans


CM = XGB()

submission =  CM.fit_predict(train[train.Num_day <= 957].reset_index(drop = True), train[train.Num_day > 957].reset_index(drop = True), train_col)
submission.to_csv('submission.csv', index=False)