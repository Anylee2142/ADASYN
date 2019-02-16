import numpy as np
import pandas as pd
import adasyn.generate as source

data = pd.read_csv('dataset/breast_cancer/breast-cancer-wisconsin.data', header=None)
data = data.rename(columns={each:each if each < 10 else 'Class' for each in range(11)})
# 0th column is ID number so that drop
data.drop(0, axis=1, inplace=True)

data = data.astype(np.object)
null = data[data=='?']
null_idx = null.dropna(how='all', axis=0).index

data.drop(null_idx, axis=0, inplace=True)
data = data.astype(np.float64)

data.to_csv('dataset/breast_cancer/breast-cancer-wisconsin.prep.csv', index=False)

source.analysis(data, 'Class')

generated = source.adasyn(data, 'Class')

oversampled_data = pd.concat([data, generated])
print(oversampled_data.shape)

oversampled_data.to_csv('dataset/breast_cancer/breast-cancer-wisconsin.adasyn.csv', index=False)

