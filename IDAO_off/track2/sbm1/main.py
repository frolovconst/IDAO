import pandas as pd
import numpy as np

#train = pd.read_csv('train.csv.zip')
#print(train.shape)

submission = pd.read_csv('submission.csv')
#submission = pd.DataFrame(train.user_id.unique()[:5000], columns = ['user_id'])

submission['id3_1'] = 0
submission['id3_2'] = 1
submission['id3_3'] = 2
submission['id3_4'] = 3
submission['id3_5'] = 4

#train = pd.read_csv('train.csv.zip', nrows=5001)
#users = pd.read_csv('frq_users.csv', names=['uid'])
#id3s = np.loadtxt('unique_id3s.csv', dtype=np.int16)

#submission = pd.DataFrame(users.uid.unique()[:5000], columns = ['user_id'])

#submission['id3_1'] = np.random.choice(id3s, 5000)
#submission['id3_2'] = np.random.choice(id3s, 5000)
#submission['id3_3'] = np.random.choice(id3s, 5000)
#submission['id3_4'] = np.random.choice(id3s, 5000)
#submission['id3_5'] = np.random.choice(id3s, 5000)


                       
#submission.to_csv('submission.csv', index=False)