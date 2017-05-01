#!/usr/bin/python3
'''
Test the module with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd

bc_file = "Days_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"
bm_file = "Days_BloodMeal.csv"

bc_splines = prob_spline.HostSpline(bc_file)
bm_splines = prob_spline.BloodmealSpline(bm_file)
mos_curve = prob_spline.MosCurve(msq_file)
tstart = prob_spline.time_transform(90)
tend = prob_spline.time_transform(270)

ODE = prob_spline.Seasonal_Spline_ODE(bc_splines,bm_splines,mos_curve,tstart,tend)
with open('ODE_test.pkl', 'wb') as output:
	pickle.dump(ODE,output) 

#ODE = pickle.load(open("ODE_test.pkl",'rb'))