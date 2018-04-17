import pandas as pd

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

train['DOW'] = train.DATE.dt.dayofweek

def generate_month(series):
    generated_ = series.tail(28).groupby('DOW').CLIENT_OUT.mean()
    generated = pd.concat((generated_, generated_, generated_, generated_, generated_[:5]))
#     return generated
    return pd.DataFrame({'DATE': pred_dates, 'CLIENT_OUT': generated})

gb = train.groupby('ATM_ID').apply(generate_month)

submit = gb.reset_index().drop('DOW', axis=1)

submit.to_csv('submission.csv', index=False)