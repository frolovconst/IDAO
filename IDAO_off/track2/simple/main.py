import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from pathlib import Path

dct = dict()
#path = '../../data/train00_bcp'
#dtypes = {'id1': np.int16, 'id2': np.int16, 'id3': np.int16, 'user_id': np.int32, 'date': np.int16}
dt = pd.read_csv('train.csv.zip')#, dtype=dtypes)
#dt = pd.read_csv(path)#, dtype=dtypes)
for uid in dt['user_id']:
    if uid in dct:
        dct[uid] += 1
    else:
        dct[uid] = 1

user_visits = pd.DataFrame({'usr' : list(dct.keys()),
                             'visits': list(dct.values())})

bot_users = user_visits[(user_visits.visits>150) & (user_visits.visits<1000)].usr[:5000]

# good_indicies = np.array(dt[dt.user_id.isin(bot_users)].index)
id3s = dt['id3'].unique()



submission = pd.DataFrame(list(bot_users), columns = ['user_id'])
print(submission.shape)
submission['id3_1'] = np.random.choice(id3s, 5000)
submission['id3_2'] = np.random.choice(id3s, 5000)
submission['id3_3'] = np.random.choice(id3s, 5000)
submission['id3_4'] = np.random.choice(id3s, 5000)
submission['id3_5'] = np.random.choice(id3s, 5000)

submission.to_csv('submission.csv', index=False)
