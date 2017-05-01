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

#sigma_vals =pickle.load(open('sigma_vals.pkl','rb'))
sigma_vals = 0

splines = prob_spline.HostSpline(bc_file,sigma=sigma_vals)

splines.plot()
