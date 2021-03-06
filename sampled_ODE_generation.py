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


def generate_ODE(bc_splines,bm_splines,mos_curve,tstart,tend):
	ODE = prob_spline.Seasonal_Spline_ODE(
		bc_splines[0],bm_splines[0],mos_curve,tstart,tend,find_beta=1,beta_1=68)

	beta_val = ODE.beta1
	N = len(bc_splines)
	print('Start Time')
	print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	with joblib.Parallel(n_jobs=-1) as parallel:
		output = parallel(joblib.delayed(prob_spline.Seasonal_Spline_ODE)(
				bc_splines[j+1],bm_splines[j+1],mos_curve,tstart,tend,find_beta=1,
				beta_1=beta_val,counter=j) 
			for j in range(N-1))
	print('Finish Time')
	print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	return(output)

bc_splines = pickle.load(open('host_splines_sample_combine_index=[6].pkl','rb'))
bm_splines = pickle.load(open('vectors_splines_sample_combine_index=[6].pkl','rb'))
mos_curve  = pickle.load(open('mos_curve_sample.pkl','rb'))
tstart = prob_spline.time_transform(90)
tend   = prob_spline.time_transform(270)
x      = numpy.linspace(tstart,tend,1001) 

ODE = generate_ODE(bc_splines,bm_splines,mos_curve,tstart,tend)
with open('sampled_ODE_combined_index[6].pkl','wb') as output:
	pickle.dump(ODE,output)
s_vals=[]
i_vals=[]
r_vals==[]
sv_vals==[]
iv_vals==[]
c_vals==[]
e_vals==[]
for j in length(ODE):
	s,i,r,sv,iv,c,e = ODE[j].get_SIR_vals(ODE[j].Y)
	s_vals.append(s)
	i_vals.append(i)
	r_vals.append(r)
	sv_vals.append(sv)
	iv_vals.append(iv)
	c_vals.append(c)
	e_vals.append(e)

