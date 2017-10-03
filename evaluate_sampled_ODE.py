from matplotlib import pyplot
import numpy
import scipy.stats as st
import seaborn
import pickle
from matplotlib.ticker import MultipleLocator

import pylab
import prob_spline
import test_common
import pandas
import pickle
import joblib
from time import gmtime, strftime
import sys
if 'pandas.indexes' not in sys.modules:
	sys.modules['pandas.indexes']=pandas.core.indexes
if 'pandas.core.indexes' not in sys.modules:
	sys.modules['pandas.core.indexes']=pandas.indexes

def unpack_ODE_vals(ODE):
	s_vals,i_vals,r_vals,sv_vals,iv_vals,c_vals,e_vals=[],[],[],[],[],[],[]
	for j in range(len(ODE)):
		s,i,r,sv,iv,c,e = ODE[j].get_SIR_vals(ODE[j].Y)
		s_vals.append(s)
		i_vals.append(i)
		r_vals.append(r)
		sv_vals.append(sv)
		iv_vals.append(iv)
		c_vals.append(c)
		e_vals.append(e)
	return(s_vals,i_vals,r_vals,sv_vals,iv_vals,c_vals,e_vals)

def unpack_vals(ODE):
	res = [ODE_.get_SIR_vals(ODE_.Y) for ODE_ in ODE]
	s,i,r,sv,iv,c,e = map(numpy.array,zip(*res))
	return(s,i,r,sv,iv,c,e)

def get_confidence_interval(vals,alpha=.95,method='numpy'):
	mean = numpy.mean(vals,axis=0)
	if method=='numpy':
		lower,upper = numpy.percentile(vals,((1-alpha)/2*100,100*(1-(1-alpha)/2)),axis=0)
	else:
		lower,upper = st.t.interval(alpha, len(vals)-1, loc=mean, scale=st.sem(vals,axis=0))
	return(lower,mean,upper)


#file_name = 'sampled_ODE_combined_index=[3, 6].pkl'
#file_name = 'corrected_ODE_sampled_combined_index=[6].pkl'
#file_name = 'sampled_ODE_combined_index=[3, 6].pkl'
file_name = 'updated_ sampled_comparison_ODE_combined_index[3, 6].pkl'

def evaluate_sampled_ODE(filename):

	ODE = pickle.load(open(filename,'rb'))

	x = prob_spline.inv_time_transform(numpy.linspace(ODE[0].tstart,ODE[0].tend,1001))
	s,i,r,sv,iv,c,e = unpack_ODE_vals(ODE)
	x_trans = prob_spline.time_transform(x)


	counts = numpy.array([ODE[j].bc_splines(x_trans) for j in range(len(ODE))])
	count_low,count_mean,count_upp = get_confidence_interval(counts)
	alpha_vals = numpy.array([ODE[j].alpha_calc(ODE[j].bm_splines(x_trans),ODE[j].bc_splines(x_trans)) for j in range(len(ODE))])
	alpha_low,alpha_mean,alpha_upp = get_confidence_interval(alpha_vals*counts)

	birdnames = ODE[0].bc_splines.birdnames
	i_low,i_mean,i_upp = get_confidence_interval(i)
	birdnames[-1] = 'Other Birds'
	colors = seaborn.color_palette('Dark2')+['black']
	seaborn.set_palette(colors)

	fig=pylab.plt.figure(1,figsize=(7.5,3.75))
	for j in range(len(birdnames)):
		ax=pylab.subplot(numpy.ceil(len(birdnames)/3),3,j+1)
		ax.yaxis.set_major_locator(MultipleLocator(.1))
		ax.yaxis.set_tick_params(labelsize=8)
		ax.xaxis.set_tick_params(labelsize=8)
		ax.xaxis.set_major_locator(MultipleLocator(30))
		pylab.plot(x,i_low[:,j],color=colors[j])
		pylab.plot(x,i_mean[:,j],color=colors[j])
		pylab.plot(x,i_upp[:,j],color=colors[j])
		pylab.ylim((0,1))
		pylab.fill_between(x,i_low[:,j],i_upp[:,j],color=colors[j],alpha=.5)
		pylab.title(birdnames[j],fontsize=15)
		if j%3==0:
			vals = ax.get_yticks()
			ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
		if j==3:
			pylab.ylabel('Proportion Infected',fontsize=14)
		if j%3==1:
			ax.yaxis.set_visible(False)
		if j%3==2:
			vals = ax.get_yticks()
			ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
			ax.yaxis.tick_right()
		if j<4:
			ax.xaxis.set_visible(False)
		if j==6:
			pylab.xlabel('Time (Days)',fontsize=14)
	fig.tight_layout()

	fig=pylab.plt.figure(2,figsize=(7.5,3.75))

	for j in range(len(birdnames)):
		#pylab.subplot(3,3,j+1)
		pylab.ylim(1,140)
		#pylab.plot(x,count_low[j,:].T,color=colors[j])
		pylab.plot(x,count_mean[j,:],color=colors[j],label=birdnames[j])
		#pylab.plot(x,count_upp[j,:],color=colors[j])
		pylab.fill_between(x,count_low[j,:].T,count_upp[j,:],color=colors[j],alpha=.5)
		pylab.ylabel('Number of Hosts',fontsize=14)
		pylab.xlabel('Time (Days)', fontsize=14)
		pylab.title('Host Populations')
	pylab.legend()
	fig.tight_layout

	fig=pylab.plt.figure(3,figsize=(7.5,3.75))
	ax=pylab.subplot(1,2,1)
	pylab.ylim(0,1)
	pylab.xlim(90,270)
	pylab.stackplot(x,alpha_mean,colors=colors)
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
	pylab.xlabel('Time (Days)', fontsize=14)
	pylab.title('Feeding Preference', fontsize=15)
	ax=pylab.subplot(1,2,2)
	pylab.ylim(0,1)
	pylab.xlim(90,270)
	ax.yaxis.tick_right()
	pylab.stackplot(x,count_mean/numpy.sum(count_mean,axis=0),colors=colors)
	pylab.title('Population', fontsize = 15)
	pylab.legend(birdnames,frameon=1)
	pylab.show()
	c = numpy.array(c)
	e = numpy.array(e)
	return(numpy.mean(c[:,-1]/e[:,-1]))



