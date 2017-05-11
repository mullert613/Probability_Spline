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
import joblib
from time import gmtime, strftime
'''
bm_file = "Days_BloodMeal.csv"
'''
class BloodmealSpline():
	'''
	Utilizing the MultinomialSpline code and a datafile consisting of
	the sampled bird counts to generate the individual splines for each bird
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period(),n_samples=0):

		self.data_file = data_file

		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		self.n_samples = n_samples
		self.read_data()
		self.generate_samples()

		self.X=prob_spline.time_transform(self.time)

		assert (sigma >= 0), 'sigma must be nonnegative.'
		if (self.n_samples>0):
			print('Start Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			with joblib.Parallel(n_jobs = -1) as parallel:
				output = parallel(joblib.delayed(self.get_vector_spline)(self.X,self.samples[j],sigma,period) for j in range(len(self.samples)))
			self.splines = output
			print('Finish Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		else:
			self.splines = self.get_vector_spline(self.X,self.Y,sigma,period)

	def read_data(self):	
		
		'''
		Read the given data_file to pull the required data
		'''

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.birdnames = count_data.index
		self.time = numpy.array([int(x) for x in count_data.columns])
		self.Y = count_data.as_matrix()
		return()
	
	def get_vector_spline(self,X,Y,sigma,period):
		multinomial_spline = prob_spline.MultinomialSpline(sigma = sigma,period = period)
		multinomial_spline.fit(X,Y.T)
		return(multinomial_spline)

	def evaluate(self,X,index=0):			# Evaluate the splines at given values X
		if (self.n_samples<=1):
			return(self.splines(X))
		else:
			return(self.splines[index](X))

	__call__ = evaluate

	def plot(self):
		# Note to self, fix axis values so that all the graphs display the same range
		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		p = len(self.Y)
		grid = numpy.ceil(numpy.sqrt(p))
		Y = self.Y/numpy.sum(self.Y,axis=0)
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

	def generate_samples(self):
		p_vals = self.Y/numpy.sum(self.Y,axis=0)
		total_bites = numpy.random.poisson(numpy.sum(self.Y,axis=0),size=(self.n_samples,len(numpy.sum(self.Y,axis=0)))) # the sampled total number of bites
		self.samples=[]
		for i in range(self.n_samples):		# number of samples
			bm_temp = numpy.zeros(self.Y.shape)
			for j in range(len(self.Y.T)):	# number of time points
					bm_temp[:,j] = numpy.random.multinomial(total_bites[i][j],p_vals[:,j])
			self.samples.append(bm_temp)

		return()

