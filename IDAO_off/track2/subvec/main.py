import pandas as pd
import numpy as np

start_date = 62
train_set = pd.read_csv('train.csv.zip')
#train_set = pd.read_csv('train_sample.csv')
gb_f = train_set.groupby('user_id')
gb_f = gb_f.filter(lambda x: (x.id3.unique().size>2) & (x.shape[0]<400) & (x.date.diff(1).mean()+x.date.max()>=start_date) )
ptfidf_ipca = pd.read_csv('pcaed_products.csv', index_col=0)

rdm_5k_usrs = np.random.choice(gb_f.user_id.unique(),5000, replace=False)
sbm = np.empty((5000,6), dtype=int)

for i,usr in enumerate(rdm_5k_usrs):
    usr_watches = gb_f[gb_f.user_id==usr]['id3']
    crossed_watches = gb_f[(gb_f.user_id==usr) & (gb_f.date>=start_date-21)]['id3']
    mean_usr_vec = ptfidf_ipca.loc[usr_watches].mean(axis=0)
    best_matches = ptfidf_ipca.dot(mean_usr_vec).sort_values(ascending=False)
    sbm[i] = np.array([usr,]+list(best_matches[~best_matches.index.isin(crossed_watches.as_matrix())][:5].index))

submission = pd.DataFrame(sbm,  columns=['user_id', 'id3_1', 'id3_2', 'id3_3', 'id3_4', 'id3_5'])
submission.to_csv('submission.csv', index=False)