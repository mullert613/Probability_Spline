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

bc_file = "Days_BirdCounts.csv"
bc_data = pd.read_csv(bc_file,index_col=0)
bc_time = numpy.array([int(x) for x in bc_data.columns])
bc_mat = bc_data.as_matrix()

birdnames = pd.read_csv(bc_file,index_col=0).index

p = len(bc_mat)
grid = numpy.ceil(numpy.sqrt(p))

# Get Poisson samples around mu(x).

p=1   #forcing only a single iteration to run
for i in range(p):
	pyplot.subplot(grid,grid,i+1)
	X = prob_spline.time_transform(bc_time)
	Y = numpy.squeeze(bc_mat[i,:])


	# Plot mu
	x = numpy.linspace(numpy.min(X), numpy.max(X), 1001)
	handles = []
	# Get Poisson samples around mu(x) and plot.
	s = pyplot.scatter(prob_spline.inv_time_transform(X), Y, color = 'black',
	                   label = birdnames[i])
	# Build a spline using the normal loglikelihood.
	handles.append(s)

	# Build a spline using the Poisson loglikelihood.
	poisson_spline = prob_spline.PoissonSpline(sigma = 0.2, period=prob_spline.period())
	poisson_spline.fit(X, Y)
	l = pyplot.plot(prob_spline.inv_time_transform(x), poisson_spline(x),
	                label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(
	                    poisson_spline.sigma))
	handles.append(l[0])

	# Add decorations to plot.
	pyplot.xlabel('$x$')
	pyplot.legend(handles, [h.get_label() for h in handles])
pyplot.show()
