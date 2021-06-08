"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

import sys

sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def accuracy(_data, curve):
    if len(_data) != len(curve):
        return 'ERROR'
    result = np.mean((curve - _data)**2)
    # result = np.mean(result)
    return result


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

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
NUM_COLORS = 0
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

for val in np.unique(confirmed["Country/Region"]):
    if val == 'US':
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)
        features.append(cases)
        targets.append(labels)

        if cases.sum() > MIN_CASES:
            NUM_COLORS += 1

colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []

for val in np.unique(confirmed["Country/Region"]):
    if val == 'US':
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)
        cases = cases.sum(axis=0)

        if cases.sum() > MIN_CASES:
            i = len(legend)
            lines = ax.plot(cases, label=labels[0, 1])
            handles.append(lines[0])
            lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])
            lines[0].set_color(colors[i])
            legend.append(labels[0, 1])
            for i in range(len(cases)):
                if cases[i] == 0:
                    cases[i] = 0.0001
            log_cases = np.log(cases.astype(float))
            x = np.arange(498)
            r = np.polyfit(x, log_cases, 3)
            y = np.ones(498)

            for j in range(len(y)):
                y[j] = r[0] * (x[j] ** 3) + r[1] * (x[j] ** 2) + r[2] * x[j] + r[3]

            plt.clf()

            plt.plot(x, log_cases)
            plt.plot(x, y)
            acc = accuracy(log_cases, y)
            print(acc)
            # plt.show()

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

ax.set_yscale('log')
ax.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results/cases_by_country.png')
