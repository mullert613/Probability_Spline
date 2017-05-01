	#!/usr/bin/python3
'''
Test cross-validating the splines with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.model_selection
import pickle

import prob_spline
import test_common
import pandas as pd


bc_file = "Days_BirdCounts.csv"
bc_data = pd.read_csv(bc_file,index_col=0)
bc_time = numpy.array([int(x) for x in bc_data.columns])
bc_mat = bc_data.as_matrix()

birdnames = pd.read_csv(bc_file,index_col=0).index

p = len(bc_mat)
grid = numpy.ceil(numpy.sqrt(p))
sigma_vals = []
for bird in range(p):
  pyplot.figure(bird+1)
  X = prob_spline.time_transform(bc_time)
  Y = numpy.squeeze(bc_mat[bird,:].T)

  spline = prob_spline.PoissonSpline(period=2)
  # Find the best sigma value by K-fold cross-validation.
  kfold = sklearn.model_selection.KFold(n_splits = 5, shuffle = True)
  param_grid = dict(sigma = numpy.logspace(-8, 4, 21))
  gridsearch = sklearn.model_selection.GridSearchCV(spline,
                                                    param_grid,
                                                    cv = kfold,
                                                    error_score = 0)
  gridsearch.fit(X, Y)
  spline = gridsearch.best_estimator_

  pyplot.subplot(2, 1, 1)
  pyplot.title(birdnames[bird])
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

  pyplot.subplot(2, 1, 2)
  x = numpy.linspace(numpy.min(X), numpy.max(X), 1001)
  handles = []
  s = pyplot.scatter(prob_spline.inv_time_transform(X), Y, color = 'black',
                     label = 'Poisson($\mu(x)$) samples')
  handles.append(s)
  l = pyplot.plot(prob_spline.inv_time_transform(x), spline(x),
                  label = 'Fitted {}($\sigma =$ {:g})'.format(
                      spline.__class__.__name__,
                      spline.sigma))
  handles.append(l[0])
  sigma_vals.append(spline.sigma)
  #pyplot.xlabel('$x$')
  #pyplot.legend(handles, [h.get_label() for h in handles])
  with open('%s_spline.pkl' %birdnames[bird], 'wb') as output:
    pickle.dump(spline,output) 
with open('sigma_vals.pkl', 'wb') as output:
  pickle.dump(sigma_vals,output)

  #pyplot.tight_layout()
#pyplot.show()


