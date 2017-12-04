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




def parallel_splines(to_be_run,file_name,N,Mos_Class=0,sigma=0,sample=0,combine_index=[],remove_index=[]):
	if Mos_Class==0:
		if N==1:
			output = to_be_run(file_name,sigma=sigma,sample=sample,combine_index=combine_index,remove_index=remove_index,seed=1)
		else:

			print('Start Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			with joblib.Parallel(n_jobs=-1) as parallel:
				output = parallel(joblib.delayed(to_be_run)(file_name,sigma=sigma,sample=sample,combine_index=combine_index,remove_index=remove_index,
					seed=j+1) for j in range(N))
			print('Finish Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
		
	else:		#change function to take *kargs?
		if N==1:
			output = to_be_run(file_name,Mos_Class,sigma=sigma,sample=sample)
		else:
			print('Start Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			with joblib.Parallel(n_jobs=-1) as parallel:
				output = parallel(joblib.delayed(to_be_run)(file_name,Mos_Class,sigma=sigma,sample=sample) for j in range(N))

			print('Finish Time')
			print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	return(output)

def get_splines(bc_file,bm_file,bc_sigma,bm_sigma,N,combine_index,remove_index):
	bc_splines = parallel_splines(prob_spline.HostSpline,bc_file,N,sigma=bc_sigma,sample=1,combine_index=combine_index,remove_index=remove_index)
	bm_splines = parallel_splines(prob_spline.BloodmealSpline,bm_file,N,sigma = bm_sigma,sample=1,combine_index=combine_index,remove_index=remove_index)
	if combine_index!=[]:
		with open('host_splines_sample_combine_index=%s.pkl' %str(combine_index), 'wb') as output:
			pickle.dump(bc_splines,output) 
		with open('vectors_splines_sample_combine_index=%s.pkl' %str(combine_index), 'wb') as output:
			pickle.dump(bm_splines,output)
	elif remove_index!=[]:
		with open('host_splines_sample_remove_index=%s.pkl' %str(remove_index), 'wb') as output:
			pickle.dump(bc_splines,output) 
		with open('vectors_splines_sample_remove_index=%s.pkl' %str(remove_index), 'wb') as output:
			pickle.dump(bm_splines,output)
	return()

def generate_splines_fun(combine_index=[],remove_index=[]):
	bc_file = "Days_BirdCounts.csv"
	msq_file = "Vector_Data(NoZeros).csv"
	bm_file = "Days_BloodMeal.csv"
	bc_sigma = 0		# for testing purposes
	#bc_sigma = pickle.load(open('Sigma_Vals_Cross_Validation_1mil_iter.pkl','rb'))
	#bm_sigma = 0.199
	bm_sigma= 0

	MosClass = prob_spline.MosConstant

	N = 2 # number of samples to be generated
	if not isinstance(bc_sigma,int):
		if combine_index != []:
			for j in combine_index:
				if j!=6:
					bc_sigma.pop(j)
		elif remove_index != []:
			bc_sigma.pop(remove_index)
	get_splines(bc_file,bm_file,bc_sigma,bm_sigma,N,combine_index,remove_index)
	#mos_curve = parallel_splines(prob_spline.MosCurve,msq_file,N,Mos_Class = prob_spline.MosConstant,sample=1)
	#with open('mos_curve_sample.pkl', 'wb') as output:
	#	pickle.dump(mos_curve,output) 	
	return()

if __name__ == '__main__':
	remove_index = [0,1,2,3,4,5,6]
	for x in remove_index:
		generate_splines_fun(remove_index=x)