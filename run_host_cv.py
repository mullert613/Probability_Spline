import joblib
from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.model_selection
import pickle

import prob_spline
import test_common
import pandas as pd

def run_host_cv(param_grid,index):
  X = prob_spline.time_transform(bc_time)
  Y = numpy.squeeze(bc_mat[index,:].T)   
  spline = prob_spline.PoissonSpline(period=2)
  kfold = sklearn.model_selection.KFold(n_splits = 5, shuffle = True)
  gridsearch = sklearn.model_selection.GridSearchCV(spline,
                                                    param_grid,
                                                    cv=kfold,
                                                    error_score = 0)
  gridsearch.fit(X,Y)
  spline = gridsearch.best_estimator
  return(spline)

bc_file = "Days_BirdCounts.csv"
bc_data = pd.read_csv(bc_file,index_col=0)
bc_time = numpy.array([int(x) for x in bc_data.columns])
bc_mat = bc_data.as_matrix()

birdnames = pd.read_csv(bc_file,index_col=0).index

p = len(bc_mat)
grid = numpy.ceil(numpy.sqrt(p))
param_grid = dict(sigma = numpy.logspace(-8, 2, 21))

with joblib.Parallel(n_jobs=-1) as parallel:
  splines = parallel(joblib.delayed(run_host_cv)(param_grid,j) for j in range(p))

with open('cross_validation_splines.pkl','wb') as output:
  pickle.dump(splines,output)

sigma_vals = [splines[j].sigma for j in range(p)]

with open('cross_validation_sigma_vals.pkl','wb') as output:
  pickle.dump(sigma_vals,output)






