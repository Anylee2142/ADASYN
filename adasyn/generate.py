import numpy as np
import pandas as pd
import random

def knn_and_ri(minor=None, major_idx=None, X=None, K=None):
    ri = dict()
    knn = dict()

    for idx, xi in minor.iterrows():
        X_ = X.drop(idx, axis=0)

        e_dist = np.sqrt(np.sum((X_ - xi)**2, axis=1)).sort_values()
        K_neighbors = e_dist[:K].index
        knn[idx] = K_neighbors

        delta = sum([1 for each in K_neighbors if each in major_idx])
        ri[idx] = delta / K

    return ri, knn

def generating_samples(minor=None, gi=None, X=None, knn=None, lamda=None):
    generated_samples = dict()
    for idx, xi in minor.iterrows():

        if gi[idx] == 0:
            continue

        si = []

        for each in range(gi[idx]):
            xzi = X.loc[random.choice(knn[idx])]
            si.append(xi + (xzi - xi) * lamda)

        generated_samples[idx] = si

    gen_list = []
    for each in generated_samples.values():
        gen_list.extend(each)
    np_gen = np.array([each.values for each in gen_list])

    gen_df = pd.DataFrame(columns=gen_list[0].index, data=np_gen)

    return gen_df

def adasyn(data, target, dth=0.6, b=1, K=5, lamda=0.5):

    class_table = data[target].value_counts()

    ms = class_table.min()
    ml = class_table.max()

    d = ms / ml

    assert d < dth, 'd = {}, dth ={}, it should be d < dth'.format(d, dth)

    minor_label = class_table[class_table == ms].index[0]
    major_label = class_table[class_table == ml].index[0]

    G = int(round((ml - ms) * b))

    X = data.drop(target, axis=1)
    minor = data[data[target] == minor_label].drop(target, axis=1)
    major_idx = data[data[target] == major_label].drop(target, axis=1).index

    ri, knn = knn_and_ri(minor, major_idx, X, K)
    gi = {k: round((v / sum(ri.values())) * G) for k, v in ri.items()}
    gen_df = generating_samples(minor, gi, X, knn, lamda)

    return gen_df

def analysis(data, target, b=1):

    class_table = data[target].value_counts()

    ms = class_table.min()
    ml = class_table.max()

    print('total number of data = ',data.shape[0])
    print('majority = ', ml)
    print('minority = ', ms)

    print('expected total number after adasyn = ', data.shape[0] + (ml - ms) * b)

