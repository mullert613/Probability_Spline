#!/usr/bin/python3
'''
Test the module with an example.
'''

from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn

import prob_spline
import test_common
import pandas as pd

bm_file = "Days_BloodMeal.csv"

# sigma_vals = .199
sigma_vals = 0

splines = prob_spline.BloodmealSpline(bm_file,sigma=sigma_vals)

splines.plot()
