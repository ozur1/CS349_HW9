import sys

sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.cluster import KMeans

plt.style.use('fivethirtyeight')


def accuracy(_data, curve):
    if len(_data) != len(curve):
        return 'ERROR'
    result = np.mean((curve - _data)**2)
    return result


def choose_best_degree(country):
    best_degree = -1
    best_accuracy = 10 ** 10
    for region in country:
        for i in range(len(region)):
            if region[i] == 0:
                region[i] = 0.0001

        log = np.log(region.astype(float))
        x = np.arange(498)

        y1 = np.ones(498)
        r = np.polyfit(x, log, 1)
        for j in range(len(y1)):
            y1[j] = r[0] * x[j] + r[1]
        acc1 = accuracy(log, y1)
        if acc1 < best_accuracy:
            best_accuracy = acc1
            best_degree = 1

        y2 = np.ones(498)
        r = np.polyfit(x, log, 2)
        for j in range(len(y2)):
            y2[j] = r[0] * (x[j] ** 2) + r[1] * x[j] + r[2]
        acc2 = accuracy(log, y2)
        if acc2 < best_accuracy:
            best_accuracy = acc2
            best_degree = 2

        y3 = np.ones(498)
        r = np.polyfit(x, log, 3)
        for j in range(len(y3)):
            y3[j] = r[0] * (x[j] ** 3) + r[1] * (x[j] ** 2) + r[2] * x[j] + r[3]
        acc3 = accuracy(log, y3)
        if acc3 < best_accuracy:
            best_accuracy = acc3
            best_degree = 3

        y4 = np.ones(498)
        r = np.polyfit(x, log, 4)
        for j in range(len(y4)):
            y4[j] = r[0] * (x[j] ** 4) + r[1] * (x[j] ** 3) + r[2] * (x[j] ** 2) + r[3] * x[j] + r[4]
        acc4 = accuracy(log, y4)
        if acc4 < best_accuracy:
            best_accuracy = acc4
            best_degree = 4

        y5 = np.ones(498)
        r = np.polyfit(x, log, 5)
        for j in range(len(y5)):
            y5[j] = r[0] * (x[j] ** 5) + r[1] * (x[j] ** 4) + r[2] * (x[j] ** 3) + \
                    r[3] * (x[j] ** 2) + r[4] * x[j] + r[5]
        acc5 = accuracy(log, y5)
        if acc5 < best_accuracy:
            best_accuracy = acc5
            best_degree = 5

        y6 = np.ones(498)
        r = np.polyfit(x, log, 6)
        for j in range(len(y6)):
            y6[j] = r[0] * (x[j] ** 6) + r[1] * (x[j] ** 5) + r[2] * (x[j] ** 4) + \
                    r[3] * (x[j] ** 3) + r[4] * (x[j] ** 2) + r[5] * x[j] + r[6]
        acc6 = accuracy(log, y6)
        if acc6 < best_accuracy:
            best_accuracy = acc6
            best_degree = 6

        y7 = np.ones(498)
        r = np.polyfit(x, log, 7)
        for j in range(len(y7)):
            y7[j] = r[0] * (x[j] ** 7) + r[1] * (x[j] ** 6) + r[2] * (x[j] ** 5) + \
                    r[3] * (x[j] ** 4) + r[4] * (x[j] ** 3) + r[5] * (x[j] ** 2) + r[6] * x[j] + r[7]
        acc7 = accuracy(log, y7)
        if acc7 < best_accuracy:
            best_accuracy = acc7
            best_degree = 7

        y8 = np.ones(498)
        r = np.polyfit(x, log, 8)
        for j in range(len(y8)):
            y8[j] = r[0] * (x[j] ** 8) + r[1] * (x[j] ** 7) + r[2] * (x[j] ** 6) + \
                    r[3] * (x[j] ** 5) + r[4] * (x[j] ** 4) + r[5] * (x[j] ** 3) + \
                    r[6] * (x[j] ** 2) + r[7] * x[j] + r[8]
        acc8 = accuracy(log, y8)
        if acc8 < best_accuracy:
            best_accuracy = acc8
            best_degree = 8

        y9 = np.ones(498)
        r = np.polyfit(x, log, 9)
        for j in range(len(y9)):
            y9[j] = r[0] * (x[j] ** 9) + r[1] * (x[j] ** 8) + r[2] * (x[j] ** 7) + \
                    r[3] * (x[j] ** 6) + r[4] * (x[j] ** 5) + r[5] * (x[j] ** 4) + \
                    r[6] * (x[j] ** 3) + r[7] * (x[j] ** 2) + r[8] * x[j] + r[9]
        acc9 = accuracy(log, y9)
        if acc9 < best_accuracy:
            best_accuracy = acc9
            best_degree = 9

    return best_degree


def get_regression_coefficients(country, degree):
    for region in country:
        for i in range(len(region)):
            if region[i] == 0:
                region[i] = 0.0001

        log = np.log(region.astype(float))
        x = np.arange(498)
        r = np.polyfit(x, log, degree)
        return r


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '/Users/orizur/Desktop/coronavirus-2020/COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

d = np.zeros(193)
for i in range(len(features)):
    degree = choose_best_degree(features[i])
    d[i] = degree
best_degree = scipy.stats.mode(d)[0][0]

regression_coeffs = []
for i in range(len(features)):
    r = get_regression_coefficients(features[i], int(best_degree))
    regression_coeffs.append(r)
regression_coeffs = np.array(regression_coeffs)

kmeans = KMeans(n_clusters=5).fit(regression_coeffs)

for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 0:
        print(targets[i], kmeans.labels_[i])
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 1:
        print(targets[i], kmeans.labels_[i])
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 2:
        print(targets[i], kmeans.labels_[i])
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 3:
        print(targets[i], kmeans.labels_[i])
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 4:
        print(targets[i], kmeans.labels_[i])
