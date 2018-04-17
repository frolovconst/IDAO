import pandas as pd
import numpy as np

ptfidf = pd.read_csv('../data/products_vecs.csv', index_col=0)
tfidf = ptfidf.as_matrix()
mu = tfidf.mean(axis=0)
prods_sc = (tfidf-mu)/np.sqrt(((tfidf-mu)**2).mean(axis=0)*50/50)
sgma = prods_sc.T.dot(prods_sc)
np.savetxt('cov.csv')