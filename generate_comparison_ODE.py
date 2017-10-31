from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas
import pickle
import joblib
from time import gmtime, strftime
import sys
if 'pandas.indexes' not in sys.modules:
	sys.modules['pandas.indexes']=pandas.core.indexes
if 'pandas.core.indexes' not in sys.modules:
	sys.modules['pandas.core.indexes']=pandas.indexes


def generate_ODE(bc_splines,bm_splines,mos_curve,beta_vals,tstart,tend):

	N = len(bc_splines)
	print('Start Time')
	print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	with joblib.Parallel(n_jobs=-1) as parallel:
		output = parallel(joblib.delayed(prob_spline.Seasonal_Spline_ODE)(
				bc_splines[j],bm_splines[j],mos_curve[j],tstart,tend,beta_1=beta_vals[j],counter=j) 
			for j in range(N))
	print('Finish Time')
	print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	return(output)

def ODE_test(bc,bm,mos,beta,tstart,tend):
	j = numpy.random.choice(range(len(bc)))
	val = prob_spline.Seasonal_Spline_ODE(bc[j],bm[j],mos[j],tstart,tend,beta_1 = beta[j])
	return(val)

def generate_comparison_ODE(combine_index):

	bc_splines = pickle.load(open('host_splines_sample_combine_index=%s.pkl' %combine_index,'rb'))
	bm_splines = pickle.load(open('vectors_splines_sample_combine_index=%s.pkl' %combine_index,'rb'))
	mos_curve  = pickle.load(open('mos_curve_sample.pkl','rb'))
	tstart = prob_spline.time_transform(90)
	tend   = prob_spline.time_transform(270)
	x      = numpy.linspace(tstart,tend,1001) 

	previous_ODE = pickle.load(open('corrected_ODE_sampled_combined_index=[6].pkl','rb'))
	beta1_samples = numpy.array([previous_ODE[j].beta1 for j in range(len(previous_ODE))])

	# This following beta1 generation should be thought about more closely
	beta1_vals = numpy.random.choice(beta1_samples,len(bc_splines))

	ODE = generate_ODE(bc_splines,bm_splines,mos_curve,beta1_vals,tstart,tend)
	with open('sampled_comparison_ODE_combined_index%s.pkl' %combine_index,'wb') as output:
		pickle.dump(ODE,output)
	'''
	s_vals=[]
	i_vals=[]
	r_vals==[]
	sv_vals==[]
	iv_vals==[]
	c_vals==[]
	e_vals==[]
	for j in range(len(ODE)):
		s,i,r,sv,iv,c,e = ODE[j].get_SIR_vals(ODE[j].Y)
		s_vals.append(s)
		i_vals.append(i)
		r_vals.append(r)
		sv_vals.append(sv)
		iv_vals.append(iv)
		c_vals.append(c)
		e_vals.append(e)
	'''
	return()

