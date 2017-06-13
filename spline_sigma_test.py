from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd
import pickle
import joblib

bc_file = "Days_BirdCounts.csv"
param_grid = numpy.logspace(-8, 4, 13)
N = len(param_grid)
with joblib.Parallel(n_jobs=-1) as parallel:
	splines = parallel(joblib.delayed(prob_spline.HostSpline)(bc_file,sigma=param_grid[j],sample=1) for j in range(N))

for j in range(N):
	with open('splines_with_sigma=%d' %param_grid[j],'wb') as output:
		pickle.dump(splines[j],output)
