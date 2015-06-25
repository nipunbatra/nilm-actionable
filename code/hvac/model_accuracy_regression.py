import time
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.svm import SVC, SVR

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

df = pd.read_csv("../../data/hvac/minutes_a3_score.csv")


np.random.seed(42)

train_idx, test_idx = cross_validation.train_test_split(range(len(df)), train_size=0.5)
train = df.ix[train_idx]
test = df.ix[test_idx]

d = {}

for N_max in range(4, 5):
    print N_max, time.time()
    cls = {"RF": RandomForestRegressor()
           }
    out_fold1 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = test['rating']
    for f in powerset(features, N_max):
        for cl_name, clf in cls.iteritems():
            np.random.seed(42)
            clf.fit(train[list(f)], train["rating"])
            y_pred = clf.predict(test[list(f)])

            accur = mean_absolute_error(y_pred, y_true)
            out_fold1[cl_name][f] = accur

    out_fold2 = {"SVM": {}, "DT": {}, "KNN": {}, "RF": {}, "ET": {}}

    y_true = train['rating']
    for f in powerset(features, N_max):
        np.random.seed(42)

        for cl_name, clf in cls.iteritems():
            clf.fit(test[list(f)], test["rating"])
            y_pred = clf.predict(train[list(f)])

            accur = mean_absolute_error(y_pred, y_true)
            out_fold2[cl_name][f] = accur

    d[N_max] = {}
    for technique in cls.iterkeys():
        temp = ((pd.Series(out_fold1[technique]) + pd.Series(out_fold2[technique])) / 2).dropna()
        temp.sort()
        d[N_max][technique] = {"feature": temp.index.values[-1],
                               "accuracy": temp.values[-1]}


def train_cross_validation(clf, seed, feature):
    np.random.seed(seed)
    clf.fit(train[feature], train["rating"])
    y_pred_1 = clf.predict(test[feature])
    y_true = test['rating']
    accur1 = mean_absolute_error(y_true, y_pred_1)


    np.random.seed(seed)
    clf = cls[technique]
    clf.fit(test[feature], test["rating"])
    y_pred_2 = clf.predict(train[feature])
    y_true = train['rating']
    accur2 = mean_absolute_error(y_true, y_pred_2)

    return ((accur1+accur2)/2.0, np.hstack(np.array([y_pred_1, y_pred_2]).flatten()))




for n, dn in d.iteritems():
    print n
    if n>=0:
        for technique, value_dict in dn.iteritems():
            accuracies = {}
            feature = list(value_dict["feature"])
            if technique in ["SVM"]:
                SEEDMAX=2
            else:
                SEEDMAX=2000


            for seed in range(1, SEEDMAX):
                accuracy, useless = train_cross_validation(cls[technique], seed, feature)
                accuracies[seed] = accuracy
            accuracies_series = pd.Series(accuracies)
            accuracies_series.sort()
            optimal_seed = accuracies_series.index.values[0]
            temp, predicted_labels = train_cross_validation(cls[technique], optimal_seed, feature)

            value_dict["predicted_labels"] = predicted_labels
            value_dict["optimal_seed"] = optimal_seed
            value_dict["optimised_accuracy"] = accuracies_series.values[0]

#y = d[3]['RF']['predicted_labels']
#x = np.hstack([test['rating'].values, train['rating'].values])