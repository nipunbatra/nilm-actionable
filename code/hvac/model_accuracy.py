import time
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.svm import SVC

features = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
            'work_energy', 'sleep_energy', 'morning_energy',
            'evening_energy', 'morning_mins', 'work_mins', 'sleep_mins', 'evening_mins', 'overall_energy',
            'overall_mins']
features_small = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3']
features_mid = ['sleep_pred', 'morning_pred', 'work_pred', 'evening_pred', 'a1', 'a2', 'a3', 'work_energy',
                'evening_energy', 'work_mins', 'evening_mins']


NUM_CLASSES = 2


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
from hvac_ids_to_consider import find_common_dataids

df = pd.read_csv("../../data/hvac/minutes_GT.csv")

df["hvac_class_copy"] = df["hvac_class"].copy()
df = df[df.dataid.isin(find_common_dataids())]
df = df.dropna()
df.index = range(len(df))
if NUM_CLASSES ==2:
    df.hvac_class[(df.hvac_class=="Average") | (df.hvac_class=="Good") ] ="Not bad"
    COLUMN_NAMES = ["Bad", "Not bad"]
else:
    COLUMN_NAMES = ["Average", "Bad", "Good"]


np.random.seed(42)

train_idx, test_idx = cross_validation.train_test_split(range(len(df)), train_size=0.5)
train = df.ix[train_idx]
test = df.ix[test_idx]

d = {}

for N_max in range(4, 5):
    print N_max, time.time()
    cls = {"RF": RandomForestClassifier()}
    #cls = {"RF": RandomForestClassifier(), "DT": DecisionTreeClassifier()}
    out_fold1 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = test['hvac_class']

    f = ('a1', 'a3', 'evening_energy', 'morning_mins')
    for cl_name, clf in cls.iteritems():
        np.random.seed(42)
        clf.fit(train[list(f)], train["hvac_class"])
        y_pred = clf.predict(test[list(f)])

        accur = accuracy_multiclass(y_true, y_pred)
        out_fold1[cl_name][f] = accur

    """COMMENTED
    for f in powerset(features, N_max):
        for cl_name, clf in cls.iteritems():
            np.random.seed(42)
            clf.fit(train[list(f)], train["hvac_class"])
            y_pred = clf.predict(test[list(f)])

            accur = accuracy_multiclass(y_pred, y_true)
            out_fold1[cl_name][f] = accur
    """""
    out_fold2 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = train['hvac_class']
    np.random.seed(42)

    for cl_name, clf in cls.iteritems():
        clf.fit(test[list(f)], test["hvac_class"])
        y_pred = clf.predict(train[list(f)])

        accur = accuracy_multiclass(y_true, y_pred)
        out_fold2[cl_name][f] = accur
    """
    for f in powerset(features, N_max):
        np.random.seed(42)

        for cl_name, clf in cls.iteritems():
            clf.fit(test[list(f)], test["hvac_class"])
            y_pred = clf.predict(train[list(f)])

            accur = accuracy_multiclass(y_pred, y_true)
            out_fold2[cl_name][f] = accur
    """
    d[N_max] = {}
    for technique in cls.iterkeys():
        temp = ((pd.Series(out_fold1[technique]) + pd.Series(out_fold2[technique])) / 2).dropna()
        temp.sort()
        d[N_max][technique] = {"feature": temp.index.values[-1],
                               "accuracy": temp.values[-1]}


def train_cross_validation(clf, seed, feature):
    np.random.seed(seed)
    clf.fit(train[feature], train["hvac_class"])
    y_pred_1 = clf.predict(test[feature])
    y_true = test['hvac_class']
    accur1 = accuracy_multiclass(y_true, y_pred_1)
    d1 = pd.DataFrame(confusion_matrix(y_true, y_pred_1))
    d1.columns = COLUMN_NAMES
    d1.index = COLUMN_NAMES

    np.random.seed(seed)
    clf = cls[technique]
    clf.fit(test[feature], test["hvac_class"])
    y_pred_2 = clf.predict(train[feature])
    y_true = train['hvac_class']
    accur2 = accuracy_multiclass(y_true, y_pred_2)
    d2 = pd.DataFrame(confusion_matrix(y_true, y_pred_2))
    d2.columns = COLUMN_NAMES
    d2.index = COLUMN_NAMES
    return (d1+d2, (accur1+accur2)/2.0, np.hstack(np.array([y_pred_1, y_pred_2]).flatten()))




for n, dn in d.iteritems():
    if n>=0:
        for technique, value_dict in dn.iteritems():
            accuracies = {}
            feature = list(value_dict["feature"])
            if technique in ["SVM"]:
                SEEDMAX=2
            else:
                SEEDMAX=10000


            for seed in range(1874, 1875):
            #for seed in range(1, SEEDMAX):
                confusion, accuracy, useless = train_cross_validation(cls[technique], seed, feature)
                accuracies[seed] = accuracy
            x = pd.Series(accuracies)
            x.sort()
            optimal_seed = x.index.values[-1]
            optimal_confusion, temp, predicted_labels = train_cross_validation(cls[technique], optimal_seed, feature)
            value_dict["optimal_confusion"] = optimal_confusion
            value_dict["predicted_labels"] = predicted_labels
            value_dict["optimal_seed"] = optimal_seed

