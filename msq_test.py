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

msq_file = "Vector_Data(NoZeros).csv"

sigma_vals = 0

curve = prob_spline.MosCurve(msq_file,sigma_vals)

curve.plot()
