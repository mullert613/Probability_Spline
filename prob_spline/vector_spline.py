import abc
import numbers

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation
import pandas as pd
from . import base
import prob_spline
import matplotlib.pyplot as pyplot
import scipy.stats


class VectorSpline():
	'''
	Utilizing the MultinomialSpline code and a datafile consisting of
	the sampled bird counts to generate the individual splines for each bird
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period()):



		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		self.time,self.Y = self.read_data(data_file)
		self.X=prob_spline.time_transform(self.time)

		assert (sigma >= 0), 'sigma must be nonnegative.'
		self.spline = self.get_vector_spline(sigma,period)

	def read_data(self,data_file):

		count_data = pd.read_csv(data_file,index_col=0)
		self.birdnames = count_data.index
		time = numpy.array([int(x) for x in count_data.columns])
		mat = count_data.as_matrix()
		return(time,mat.T)
	
	def get_vector_spline(self,sigma,period):
		multinomial_spline = prob_spline.MultinomialSpline(sigma = sigma,period = period)
		multinomial_spline.fit(self.X, self.Y)
		return(multinomial_spline)

	def evaluate(self,X):			# Evaluate the splines at given values X
		return(numpy.array([self.splines[i](X) for i in range(len(self.splines))]))

	__call__ = evaluate

	def derivative(self,X):
		return(numpy.array([self.splines[i].derivative(i) for i in range(len(self.splines))]))

	def plot(self):

		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		p = len(self.Y.T)
		grid = numpy.ceil(numpy.sqrt(p))
		Y = self.Y.T/numpy.sum(self.Y.T,axis=0)
		plot_counter = 1
		for j in range(len(Y)):
			pyplot.subplot(3,3,j+1)
			handles=[]
			s = pyplot.scatter(prob_spline.inv_time_transform(self.X),Y[j],label = self.birdnames[j])
			handles.append(s)
			l = pyplot.plot(prob_spline.inv_time_transform(x), self.spline(x)[j],
				label = 'Fitted MultinomialSpline($\sigma =$ {:g})'.format(self.spline.sigma))
			if j==0:
				handles.append(l[0])
			pyplot.legend(handles, [h.get_label() for h in handles])

		pyplot.show()
		return()
