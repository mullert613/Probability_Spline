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
import joblib
from time import gmtime, strftime

# Current works on Jan's computer, but not on mine


#msq_file = "Vector_Data(NoZeros).csv"

class MosConstant():
	'''
	Return a constant mosquito function (the count value doesn't matter in the proportion case, but we set it up in case it does)
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period(), n_samples = 0):

		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		assert (sigma >= 0), 'sigma must be nonnegative.'

		assert (n_samples >=0), 'number of samples must be nonnegative.'

		assert isinstance(n_samples,int), 'number of samples must be an integer'

		self.data_file = data_file

		self.read_data()
		self.X=prob_spline.time_transform(self.time)
		self.sigma = sigma
		self.period = period
		self.n_samples = n_samples

		
		self.read_data()

		self.generate_samples()
		
		if (self.n_samples > 0):
			with joblib.Parallel(n_jobs = -1) as parallel:
				output = parallel(joblib.delayed(self.get_host_splines)(self.samples[j]) for j in range(len(self.samples)))
			self.constant = output
		else:
			self.constant = self.get_host_splines(self.Y)

	def read_data(self):

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.birdnames = count_data.index
		self.time = numpy.array([int(x) for x in count_data.columns])
		self.Y = count_data.as_matrix()
		self.X = prob_spline.time_transform(self.time)
		return()
	
	def get_host_splines(self,Y):   # Fit a degree 0 polynomial to the data
		Y_val = numpy.squeeze(Y.T)
		constant = numpy.polyfit(self.X,Y_val,0)
		return(constant)

	def evaluate(self,X):			# Returns the constant value
		return(self.constant[0])

	__call__ = evaluate

	def derivative(self,X):
		return(numpy.array(0))

	def pos_der(self,X):
		return(numpy.array(numpy.max((self.derivative(X),0))))
	
	def neg_der(self,X):
		return(numpy.array(numpy.min((self.derivative(X),0))))
	

	def plot(self):
		'''
		A function to plot the data and spline fit of the specified species
		Defaults to all species given, but allows for input of specified species index
		'''
		return()

	def generate_samples(self):
		self.samples = numpy.random.poisson(lam=self.Y,size = (self.n_samples,len(self.Y.T))) 
		return()
