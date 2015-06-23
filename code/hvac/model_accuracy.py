import time
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

features = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
            'work_energy', 'sleep_energy', 'morning_energy',
            'evening_energy', 'morning_mins', 'work_mins', 'sleep_mins', 'evening_mins', 'overall_energy',
            'overall_mins']
features_small = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3']
features_mid = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3', 'work_energy',
                'evening_energy', 'work_mins', 'evening_mins']


def powerset(iterable, N_max):
    xs = list(iterable)
    return combinations(xs, N_max)


def accuracy_multiclass(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    length = len(confusion)
    out = 0.0
    for i in xrange(length):
        out += confusion[i, i] * 1.0 / np.sum(confusion[i])
    return out / length


out_fold1 = {}
out_fold2 = {}
y1_best_score = 0
y1_best_pred = None
y2_best_score = 0
y2_best_pred = None
best_random_state = -1
best_random_score = 0

df = pd.read_csv("../../data/hvac/minutes_a3.csv")
np.random.seed(42)

train_idx, test_idx = cross_validation.train_test_split(range(len(df)), train_size=0.5)
train = df.ix[train_idx]
test = df.ix[test_idx]

d = {}

for N_max in range(2, 3):
    print N_max, time.time()
    cls = {"RF": RandomForestClassifier(), "DT": DecisionTreeClassifier(), "SVM" :SVC()
           }
    out_fold1 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = test['hvac_class']
    for f in powerset(features, N_max):
        for cl_name, clf in cls.iteritems():
            np.random.seed(42)
            clf.fit(train[list(f)], train["hvac_class"])
            y_pred = clf.predict(test[list(f)])

            accur = accuracy_multiclass(y_pred, y_true)
            out_fold1[cl_name][f] = accur

    out_fold2 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = train['hvac_class']
    for f in powerset(features, N_max):
        np.random.seed(42)

        for cl_name, clf in cls.iteritems():
            clf.fit(test[list(f)], test["hvac_class"])
            y_pred = clf.predict(train[list(f)])

            accur = accuracy_multiclass(y_pred, y_true)
            out_fold2[cl_name][f] = accur

    d[N_max] = {}
    for technique in cls.iterkeys():
        temp = ((pd.Series(out_fold1[technique]) + pd.Series(out_fold2[technique])) / 2).dropna()
        temp.sort()
        d[N_max][technique] = {"feature": temp.index.values[-1],
                               "accuracy": temp.values[-1]}
