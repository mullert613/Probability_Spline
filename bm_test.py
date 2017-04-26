#!/usr/bin/python3
'''
Test the module with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn

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
period = 2


x = numpy.linspace(prob_spline.time_transform(0), prob_spline.time_transform(365), 1001)

#time transform X

# Build a spline using the Poisson loglikelihood.
multinomial_spline = prob_spline.MultinomialSpline(sigma = 0,period = period)
multinomial_spline.fit(X, Y)
Y=Y.T
Y = Y/numpy.sum(Y,axis=0)
for j in range(len(Y)):
	pyplot.subplot(3,3,j+1)
	pyplot.plot(prob_spline.inv_time_transform(x), multinomial_spline(x)[j])
	pyplot.scatter(prob_spline.inv_time_transform(X),Y[j])

# Add decorations to plot.
#pyplot.xlabel('$x$')
#pyplot.legend(handles, [h.get_label() for h in handles])
#pyplot.show()
