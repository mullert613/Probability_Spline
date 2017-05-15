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
bc_sigma = pickle.load(open('sigma_vals.pkl','rb'))
bm_sigma = 0.199

N = 1000 # number of samples to be generated

MosClass = prob_spline.MosConstant

bc_splines = prob_spline.HostSpline(bc_file,sigma=bc_sigma,n_samples = N)

bm_splines = prob_spline.BloodmealSpline(bm_file,sigma=bm_sigma,n_samples=N)

mos_curve = prob_spline.MosCurve(msq_file,MosClass,n_samples=N)

with open('host_splines_sample.pkl', 'wb') as output:
	pickle.dump(bc_splines,output) 

with open('vectors_splines_sample.pkl', 'wb') as output:
	pickle.dump(bm_splines,output) 

with open('mos_curve_sample.pkl', 'wb') as output:
	pickle.dump(mos_curve,output) 		