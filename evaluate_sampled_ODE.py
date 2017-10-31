from matplotlib import pyplot
import numpy
import scipy.stats as st
import seaborn
import pickle
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

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

def get_confidence_interval(vals,alpha=.90,method='numpy'):
	mean = numpy.mean(vals,axis=0)
	median = numpy.median(vals,axis=0)
	if method=='numpy':
		lower,upper = numpy.percentile(vals,((1-alpha)/2*100,100*(1-(1-alpha)/2)),axis=0)
	else:
		lower,upper = st.t.interval(alpha, len(vals)-1, loc=mean, scale=st.sem(vals,axis=0))
	return(lower,median,upper)

def color_dictionary():
	data_file = "Days_BirdCounts.csv"
	count_data = pandas.read_csv(data_file,index_col=0)
	birdnames = list(count_data.index)
	birdnames[-1] = 'Other Birds'
	colors = seaborn.color_palette('Dark2')+['grey']
	d={}
	for j in range(len(birdnames)):
		d[birdnames[j]]=colors[j]
	return(d)

	



def generate_file_name(index):
	return('sampled_comparison_ODE_combined_index%s.pkl' %index)

f_1 = 'sampled_ODE_combined_index=[3, 6].pkl'
f_2 = 'corrected_ODE_sampled_combined_index=[6].pkl'
f_3 = 'updated_ sampled_comparison_ODE_combined_index[3, 6].pkl'

def evaluate_sampled_ODE(index):
	file_name = generate_file_name(index)
	ODE = pickle.load(open(file_name,'rb'))

	x = prob_spline.inv_time_transform(numpy.linspace(ODE[0].tstart,ODE[0].tend,1001))
	s,i,r,sv,iv,c,e = unpack_ODE_vals(ODE)
	x_trans = prob_spline.time_transform(x)


	counts = numpy.array([ODE[j].bc_splines(x_trans) for j in range(len(ODE))])
	count_low,count_mid,count_upp = get_confidence_interval(counts)
	alpha_vals = numpy.array([ODE[j].alpha_calc(ODE[j].bm_splines(x_trans),ODE[j].bc_splines(x_trans)) for j in range(len(ODE))])
	alpha_low,alpha_mid,alpha_upp = get_confidence_interval(alpha_vals*counts)

	birdnames = ODE[0].bc_splines.birdnames
	i_low,i_mean,i_upp = get_confidence_interval(i)
	i_max = numpy.max(i_upp)
	birdnames[-1] = 'Other Birds'
	colors = seaborn.color_palette('Dark2')+['grey']
	seaborn.set_palette(colors)
	bird_colors = color_dictionary()
	color_val=[]
	for j in range(len(birdnames)):
		color_val.append(bird_colors[birdnames[j]])

	rows = numpy.ceil(len(birdnames)/3)
	fig,axes = matplotlib.pyplot.subplots(numpy.int(rows),3,sharex=True,sharey=True,figsize=(7.5,3.75))
	for j in range(len(birdnames)):
		ax = axes[numpy.int(numpy.floor(j/(rows+1)))][j%3]
		ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
		ax.yaxis.set_major_locator(MultipleLocator(i_max/5))
		ax.yaxis.set_tick_params(labelsize=8)
		ax.xaxis.set_tick_params(labelsize=8)
		ax.xaxis.set_major_locator(MultipleLocator(30))
		ax.plot(x,i_low[:,j],color=color_val[j])
		ax.plot(x,i_mean[:,j],color=color_val[j])
		ax.plot(x,i_upp[:,j],color=color_val[j])
		ax.fill_between(x,i_low[:,j],i_upp[:,j],color=color_val[j],alpha=.5)
		ax.set_ylim([0,min(i_max+i_max/5,1)])
		ax.set_title(birdnames[j],fontsize=12)
		if j%3==0:
			vals = ax.get_yticks()
			ax.set_yticklabels(['{:.2f}%'.format(x*100) for x in vals])
			ax.set_ylabel('Proportion Infected',fontsize=10)
		if j%3==2:
			vals = ax.get_yticks()
			ax.set_yticklabels(['{:.2f}%'.format(x*100) for x in vals])
			ax.yaxis.tick_right()
		if len(birdnames)%3==0:
			if j == len(birdnames)-2:
				ax.set_xlabel('Time (Days)',fontsize=10)
		else:
			if j == range(len(birdnames))[-1]:
				ax.set_xlabel('Time (Days)',fontsize=10)			
	fig.tight_layout()
	pylab.savefig('SavedFigures/%s_infections.png' %index)

	fig=pylab.plt.figure(2,figsize=(7.5,3.75))

	for j in range(len(birdnames)):
		#pylab.subplot(3,3,j+1)
		pylab.ylim(1,numpy.max(count_upp)+10)
		pylab.xlim(90,270)
		#pylab.plot(x,count_low[j,:].T,color=colors[j])
		pylab.plot(x,count_mid[j,:],color=color_val[j],label=birdnames[j])
		#pylab.plot(x,count_upp[j,:],color=colors[j])
		pylab.fill_between(x,count_low[j,:].T,count_upp[j,:],color=color_val[j],alpha=.5)
		pylab.ylabel('Number of Hosts',fontsize=14)
		pylab.xlabel('Time (Days)', fontsize=14)
		pylab.title('Host Populations')
	pylab.legend(bbox_to_anchor=(1,0.85))
	fig.tight_layout()
	pylab.savefig('SavedFigures/%s_populations.png' %index)

	fig=pylab.plt.figure(3,figsize=(7.5,3.75))
	ax=pylab.subplot(1,2,1)
	pylab.ylim(0,1)
	pylab.xlim(90,270)
	pylab.stackplot(x,alpha_mid,colors=color_val)
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:g}%'.format(x*100) for x in vals])
	pylab.xlabel('Time (Days)', fontsize=14)
	pylab.title('Feeding Preference', fontsize=15)
	ax=pylab.subplot(1,2,2)
	pylab.ylim(0,1)
	pylab.xlim(90,270)
	ax.set_yticklabels(['{:g}%'.format(x*100) for x in vals])
	ax.yaxis.tick_right()
	pylab.stackplot(x,count_mid/numpy.sum(count_mid,axis=0),colors=color_val)
	pylab.title('Population', fontsize = 15)
	ax.legend(ax.collections[::-1],birdnames[::-1],frameon=1)
	fig.tight_layout()
	pylab.savefig('SavedFigures/%s_feedingindex.png' %index)
	c = numpy.array(c)
	e = numpy.array(e)
	val = numpy.mean(c[:,-1]/e[:,-1])
	matplotlib.pyplot.close('all')
	return(val)



