from matplotlib import pyplot
import numpy
import scipy.stats
import seaborn
import pickle

import prob_spline
import test_common
import pandas as pd
import pickle

bc_splines = pickle.load(open('host_splines_sample.pkl', 'rb'))
bm_splines = pickle.load(open('vectors_splines_sample.pkl', 'rb'))
mos_curve = pickle.load(open('mos_curve_sample.pkl', 'rb'))

tstart = prob_spline.time_transform(90)
tend = prob_spline.time_transform(270)
x = numpy.linspace(tstart,tend,1001)

N = len(bm_splines.splines)

ODE = prob_spline.Seasonal_Spline_ODE(bc_splines,bm_splines,mos_curve,tstart,tend)

ODE.eval_ode_results()