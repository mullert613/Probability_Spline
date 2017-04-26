	#!/usr/bin/python3
'''
Test cross-validating the splines with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.model_selection

import prob_spline
import test_common
import pandas as pd


bm_file = "Days_BloodMeal.csv"
bm_data = pd.read_csv(bm_file,index_col=0)
bm_time = numpy.array([int(x) for x in bm_data.columns])
bm_mat = bm_data.as_matrix()

birdnames = pd.read_csv(bm_file,index_col=0).index

p = len(bm_mat)
grid = numpy.ceil(numpy.sqrt(p))

# Get Poisson samples around mu(x).
X = prob_spline.time_transform(bm_time)
Y = bm_mat.T

spline = prob_spline.MultinomialSpline(period=prob_spline.period())
# Find the best sigma value by K-fold cross-validation.
kfold = sklearn.model_selection.KFold(n_splits = 5, shuffle = True)
param_grid = dict(sigma = numpy.logspace(-1, 1, 21))
gridsearch = sklearn.model_selection.GridSearchCV(spline,
                                                  param_grid,
                                                  cv = kfold,
                                                  error_score = 0)
gridsearch.fit(X, Y)
spline = gridsearch.best_estimator_

pyplot.figure(1)
pyplot.title("Bloodmeals")
pyplot.plot(param_grid['sigma'],
            gridsearch.cv_results_['mean_test_score'],
            marker = 'o')
i = numpy.argmax(gridsearch.cv_results_['mean_test_score'])
pyplot.scatter(param_grid['sigma'][i],
               gridsearch.cv_results_['mean_test_score'][i],
               color = 'red', marker = '*',
               zorder = 3)
pyplot.xscale('log')
pyplot.yscale('log')
pyplot.xlabel('$\sigma$')
pyplot.ylabel('Mean CV Likelihood')

pyplot.figure(2)
x = numpy.linspace(numpy.min(X), numpy.max(X), 1001)
handles = []
Y=Y.T
for j in range(len(Y)):
  pyplot.subplot(3,3,j+1)
  s = pyplot.scatter(prob_spline.inv_time_transform(X), Y[j]/numpy.sum(Y,axis=0))
  l = pyplot.plot(prob_spline.inv_time_transform(x), spline(x)[j])

  pyplot.tight_layout()
  pyplot.show()
