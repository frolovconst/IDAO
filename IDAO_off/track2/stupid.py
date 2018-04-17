import pandas as pd

train = pd.read_csv('../data/train.csv.zip')

submission = pd.DataFrame(train.user_id.unique()[:5000], columns = ['user_id'])
submission.to_csv('submission.csv', index=False)