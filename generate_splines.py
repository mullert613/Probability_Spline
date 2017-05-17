from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd
import pickle
import joblib
from time import gmtime, strftime

def generate_splines(to_be_run,file_name,N,Mos_Class=0,sigma=0,sample=0):
	if Mos_Class==0:
		print('Start Time')
		print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		with joblib.Parallel(n_jobs=-1) as parallel:
			output = parallel(joblib.delayed(to_be_run)(file_name,sigma=sigma,sample=sample) for j in range(N))
		print('Finish Time')
		print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		
	else:		#change function to take *kargs?
		print('Start Time')
		print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		with joblib.Parallel(n_jobs=-1) as parallel:
			output = parallel(joblib.delayed(to_be_run)(file_name,Mos_Class,sigma=sigma,sample=sample) for j in range(N))

		print('Finish Time')
		print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	return(output)

bc_file = "Days_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"
bm_file = "Days_BloodMeal.csv"
#bc_sigma = 0		# for testing purposes
bc_sigma = pickle.load(open('sigma_vals.pkl','rb'))
bm_sigma = 0.199
#bm_sigma= 0

MosClass = prob_spline.MosConstant

N = 1000 # number of samples to be generated

bc_splines = generate_splines(prob_spline.HostSpline,bc_file,N,sigma=bc_sigma,sample=1)

bm_splines = generate_splines(prob_spline.BloodmealSpline,bm_file,N,sigma = bm_sigma,sample=1)

mos_curve = generate_splines(prob_spline.MosCurve,msq_file,N,Mos_Class = prob_spline.MosConstant,sample=1)

with open('host_splines_sample.pkl', 'wb') as output:
	pickle.dump(bc_splines,output) 

with open('vectors_splines_sample.pkl', 'wb') as output:
	pickle.dump(bm_splines,output) 

with open('mos_curve_sample.pkl', 'wb') as output:
	pickle.dump(mos_curve,output) 	
'''
print('Start Time')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# as written this doesn't output what I want
with joblib.Parallel(n_jobs = -1) as parallel:
	output = parallel(joblib.delayed(prob_spline.BloodmealSpline)(bm_file,sigma=bm_sigma,sample=1) for j in range(N))
bm_splines = output
print('Finish Time')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

print('Start Time')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# as written this doesn't output what I want
with joblib.Parallel(n_jobs = -1) as parallel:
	output = parallel(joblib.delayed(prob_spline.MosCurve)(msq_file,MosClass,sample=1) for j in range(N))
mos_curve = output
print('Finish Time')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

with open('host_splines_sample.pkl', 'wb') as output:
	pickle.dump(bc_splines,output) 

with open('vectors_splines_sample.pkl', 'wb') as output:
	pickle.dump(bm_splines,output) 

with open('mos_curve_sample.pkl', 'wb') as output:
	pickle.dump(mos_curve,output) 		
'''