import sys
import pandas as pd
import numpy
#import pylab
import unittest
import scipy.stats
import pickle




bm_file = "Days_BloodMeal.csv"
bc_file = "Days_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"

bm_data = pd.read_csv(bm_file,index_col=0)
bm_time = numpy.array([int(x) for x in bm_data.columns])
bm_data = bm_data.as_matrix()

bc_data = pd.read_csv(bc_file,index_col=0)
bc_time = numpy.array([int(x) for x in bc_data.columns])
bc_data = bc_data.as_matrix()
birdnames = pd.read_csv(bc_file,index_col=0).index


bc_index = numpy.zeros(len(bm_time))
for j in range(len(bm_time)):
	test = bc_time - bm_time[j]
	bc_index[j] = numpy.argmin(numpy.abs(test))

FI = numpy.zeros(numpy.shape(bm_data))
for i in range(len(bm_time)):
	for j in range(len(birdnames)-1):
		bm_ratio = bm_data[j,i]/(numpy.sum(bm_data[:,i])-bm_data[j,i])
		bc_ratio = bc_data[j,bc_index[i]]/(numpy.sum(bc_data[:,bc_index[i]])-bc_data[j,bc_index[i]])
		if bc_ratio == 0:
			FI[i,j] = numpy.nan
		else:
			FI[i,j] = bm_ratio/bc_ratio 