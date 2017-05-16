from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd
import pickle

bc_file = "Days_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"
bm_file = "Days_BloodMeal.csv"

'''
bc_splines = pickle.load(open('host_splines_sample.pkl', 'rb'))
bm_splines = pickle.load(open('vectors_splines_sample.pkl', 'rb'))
mos_curve = pickle.load(open('mos_curve_sample.pkl', 'rb'))
'''

bc_splines = prob_spline.HostSpline(bc_file,sample=1)
bm_splines = prob_spline.BloodmealSpline(bm_file,sample=1)
mos_curve = prob_spline.MosCurve(msq_file,prob_spline.MosConstant,sample=1)

tstart = prob_spline.time_transform(90)
tend = prob_spline.time_transform(270)
x = numpy.linspace(tstart,tend,1001)


ODE = prob_spline.Seasonal_Spline_ODE(bc_splines,bm_splines,mos_curve,tstart,tend,find_beta=0)

ODE.eval_ode_results()