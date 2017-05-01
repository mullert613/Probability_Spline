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


class HostSpline():
	'''
	Utilizing the PoissonSpline code and a datafile consisting of
	the sampled bird counts to generate the individual splines for each bird
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period(), n_samples = 0):



		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		self.data_file = data_file

		self.read_data()
		self.X=prob_spline.time_transform(self.time)

		if hasattr(sigma,"__len__"):
			for j in sigma:
				assert (j >=0 ), 'sigma must be nonnegative'
		else:
			assert (sigma >= 0), 'sigma must be nonnegative.'
			sigma = sigma*numpy.ones(len(self.Y))


		self.splines = self.get_host_splines(self.X,self.Y,sigma,period)

	def read_data(self):

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.birdnames = count_data.index
		self.time = numpy.array([int(x) for x in count_data.columns])
		self.Y = count_data.as_matrix()
		return()
	
	def get_host_splines(self,X,Y_mat,sigma,period):
		splines=[]
		for i in range(len(Y_mat)):
			Y = numpy.squeeze(Y_mat[i,:])
			poisson_spline = prob_spline.PoissonSpline(sigma = sigma[i], period=period)
			poisson_spline.fit(X, Y)
			splines.append(poisson_spline)
		return(splines)

	def evaluate(self,X):			# Evaluate the splines at given values X
		return(numpy.array([self.splines[i](X) for i in range(len(self.splines))]))

	__call__ = evaluate

	def derivative(self,X):
		return(numpy.array([self.splines[i].derivative(i) for i in range(len(self.splines))]))

	def pos_der(self,X):
		return(numpy.array([numpy.max((j,0)) for j in self.derivative(X)]))
	
	def neg_der(self,X):
		return(numpy.array([numpy.min((j,0)) for j in self.derivative(X)]))
	

	def plot(self,p=range(7)):
		'''
		A function to plot the data and spline fit of the specified species
		Defaults to all species given, but allows for input of specified species index
		'''
		msg = 'p must be a list or array, not at integer'
		assert hasattr(p, "__len__"), msg
		val = len(p)
		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		grid = numpy.ceil(numpy.sqrt(val))
		plot_counter = 1
		for j in p:
			handles = []
			pyplot.subplot(grid,grid,plot_counter)
			s = pyplot.scatter(prob_spline.inv_time_transform(self.X), self.Y[j,:], color = 'black',
	                   label = self.birdnames[j])
			handles.append(s)
			l = pyplot.plot(prob_spline.inv_time_transform(x), self.splines[j](x),
				label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(self.splines[j].sigma))
			handles.append(l[0])
			pyplot.xlabel('$x$')
			pyplot.legend(handles, [h.get_label() for h in handles],fontsize = 'xx-small',loc=0)
			plot_counter+=1
		pyplot.show()
		return()

	def generate_samples(self,distribution = 'p'):
		return (numpy.random.poisson(lam=data,size = (n_samples,len(data))))
