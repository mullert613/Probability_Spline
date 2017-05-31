
from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import sklearn.model_selection
import pickle

import prob_spline
import test_common
import pandas as pd

sigma_vals = pickle.load(open('sigma_vals.pkl','rb'))

def bc_sigma_test(bc_file,sigma,species_index):
	bc_data = pd.read_csv(bc_file,index_col=0)
	bc_time = numpy.array([int(x) for x in bc_data.columns])
	bc_mat = bc_data.as_matrix()
	X = prob_spline.time_transform(bc_time)
	Y = numpy.squeeze(bc_mat[species_index,:].T)
	spline = prob_spline.PoissonSpline(sigma=sigma,period=2)
	spline.fit(X,Y)
	x=numpy.linspace(X[0],X[-1],1001)
	s = pyplot.scatter(prob_spline.inv_time_transform(X), Y, color = 'black',
                     label = 'Poisson($\mu(x)$) samples')
	l = pyplot.plot(prob_spline.inv_time_transform(x), spline(x),
		label = 'Fitted {}($\sigma =$ {:g})'.format(spline.__class__.__name__,spline.sigma))
	return()