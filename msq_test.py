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

msq_file = "Vector_Data(NoZeros).csv"
msq_data = pd.read_csv(msq_file,index_col=0)
msq_time = numpy.array([int(x) for x in msq_data.columns])
msq_mat = msq_data.as_matrix()



# Get Poisson samples around mu(x).
X = prob_spline.time_transform(msq_time)
Y = numpy.squeeze(msq_mat.T)


# Plot mu
x = numpy.linspace(numpy.min(X), numpy.max(X), 1001)
handles = []
# Get Poisson samples around mu(x) and plot.
s = pyplot.scatter(prob_spline.inv_time_transform(X), Y, color = 'black',
                   label = 'Poisson($\mu(x)$) samples')
# Build a spline using the normal loglikelihood.
handles.append(s)

# Build a spline using the Poisson loglikelihood.
poisson_spline = prob_spline.PoissonSpline(sigma = 10, period = prob_spline.period())
poisson_spline.fit(X, Y)
l = pyplot.plot(prob_spline.inv_time_transform(x), poisson_spline(x),
                label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(
                    poisson_spline.sigma))
handles.append(l[0])

# Add decorations to plot.
pyplot.xlabel('$x$')
pyplot.legend(handles, [h.get_label() for h in handles])
pyplot.show()
