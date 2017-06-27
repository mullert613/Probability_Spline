from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd
import pickle

bc_file = "Test_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"
bm_file = "Test_BloodMeals.csv"

'''
For running a combined count
bc_file = "Combined_BirdCounts.csv"
bm_file = 'Combined_BloodMeal.csv'
bc_splines = prob_spline.HostSpline(bc_file,sample=1)
bm_splines = prob_spline.BloodmealSpline(bm_file,sample=1)
'''

'''
bc_splines = pickle.load(open('host_splines_sample.pkl', 'rb'))
bm_splines = pickle.load(open('vectors_splines_sample.pkl', 'rb'))
mos_curve = pickle.load(open('mos_curve_sample.pkl', 'rb'))
'''

#bc_sigma = pickle.load(open('sigma_vals.pkl','rb'))
bm_sigma = 0.199

#bc_splines = prob_spline.HostSpline(bc_file,sigma=bc_sigma,sample=1)
#bm_splines = prob_spline.BloodmealSpline(bm_file,sigma=bm_sigma,sample=1)

mos_curve = prob_spline.MosCurve(msq_file,prob_spline.MosConstant,sample=1)


bc_splines = prob_spline.HostSpline(bc_file)
bm_splines = prob_spline.BloodmealSpline(bm_file)

tstart = prob_spline.time_transform(98)
tend = prob_spline.time_transform(301)
x = numpy.linspace(tstart,tend,1001)


ODE = prob_spline.Seasonal_Spline_ODE(bc_splines,bm_splines,mos_curve,tstart,tend,find_beta=0,beta_1=100)

#ODE.eval_ode_results()

s,i,r,sv,iv,c,e = ODE.get_SIR_vals(ODE.Y)
counts = bc_splines(x)
S = s * counts.T
I = i * counts.T
R = r * counts.T
