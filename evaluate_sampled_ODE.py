from matplotlib import pyplot
import numpy
import scipy.stats as st
import seaborn
import pickle
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

import statsmodels.nonparametric
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

def unpack_ODE_vals(ODE):  # Unpacks the values of a single ODE class
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

def unpack_vals(ODE):	# Returns arrays of the unpacked ODE values for each ODE sampled
	res = [ODE_.get_SIR_vals(ODE_.Y) for ODE_ in ODE]
	s,i,r,sv,iv,c,e = map(numpy.array,zip(*res))
	return(s,i,r,sv,iv,c,e)

def get_confidence_interval(vals,alpha,method):
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

def lowess(x,y,frac,**kargs):
	smoothed_values = statsmodels.nonparametric.smoothers_lowess.lowess(y,x,frac=frac,is_sorted=True,return_sorted=False,**kargs)
	return(smoothed_values)	



def generate_file_name(combine_index=[],remove_index=[]):
	if remove_index != []:
		return('sampled_comparison_ODE_remove_index=%s.pkl' %remove_index)
	elif combine_index != []:
		return('sampled_comparison_ODE_combined_index%s.pkl' %combine_index)
	else:
		print('Incorrect generate_file_name Input')
		raise ValueError('Both combine and remove index may not be blank')

f_1 = 'sampled_ODE_combined_index=[3, 6].pkl'
f_2 = 'corrected_ODE_sampled_combined_index=[6].pkl'
f_3 = 'updated_ sampled_comparison_ODE_combined_index[3, 6].pkl'


def CI_vals(ODE,alpha,method,remove_index=[],combine_index=[],generate=1):
	if generate==1:
		x = prob_spline.inv_time_transform(numpy.linspace(ODE[0].tstart,ODE[0].tend,1001)) 
		s,i,r,sv,iv,c,e = unpack_ODE_vals(ODE)
		x_trans = prob_spline.time_transform(x)
		counts = numpy.array([ODE[j].bc_splines(x_trans) for j in range(len(ODE))])
		count_low,count_mid,count_upp = get_confidence_interval(counts,alpha,method)
		alpha_vals = numpy.array([ODE[j].alpha_calc(ODE[j].bm_splines(x_trans),ODE[j].bc_splines(x_trans)) for j in range(len(ODE))])
		alpha_low,alpha_mid,alpha_upp = get_confidence_interval(alpha_vals*counts,alpha,method)
		i_low,i_mean,i_upp = get_confidence_interval(i,alpha,method)
		i_max = numpy.max(i_upp)
		if remove_index!=[]:
			with open('Pickle_Files/remove_index=%s_counts' %remove_index,'wb') as output:
				pickle.dump(counts,output)
			with open('Pickle_Files/remove_index=%s_count_low' %remove_index,'wb') as output:
				pickle.dump(count_low,output)
			with open('Pickle_Files/remove_index=%s_count_mid' %remove_index,'wb') as output:
				pickle.dump(count_mid,output)
			with open('Pickle_Files/remove_index=%s_count_upp' %remove_index,'wb') as output:
				pickle.dump(count_upp,output)
			with open('Pickle_Files/remove_index=%s_alpha_vals' %remove_index,'wb') as output:
				pickle.dump(alpha_vals,output)
			with open('Pickle_Files/remove_index=%s_alpha_low' %remove_index,'wb') as output:
				pickle.dump(alpha_low,output)
			with open('Pickle_Files/remove_index=%s_alpha_mid' %remove_index,'wb') as output:
				pickle.dump(alpha_mid,output)
			with open('Pickle_Files/remove_index=%s_alpha_upp' %remove_index,'wb') as output:
				pickle.dump(alpha_upp,output)
			with open('Pickle_Files/remove_index=%s_i_low' %remove_index,'wb') as output:
				pickle.dump(i_low,output)
			with open('Pickle_Files/remove_index=%s_i_mean' %remove_index,'wb') as output:
				pickle.dump(i_mean,output)
			with open('Pickle_Files/remove_index=%s_i_upp' %remove_index,'wb') as output:
				pickle.dump(i_upp,output)	
		if combine_index!=[]:
			with open('Pickle_Files/combine_index=%s_counts' %combine_index,'wb') as output:
				pickle.dump(counts,output)
			with open('Pickle_Files/combine_index=%s_count_low' %combine_index,'wb') as output:
				pickle.dump(count_low,output)
			with open('Pickle_Files/combine_index=%s_count_mid' %combine_index,'wb') as output:
				pickle.dump(count_mid,output)
			with open('Pickle_Files/combine_index=%s_count_upp' %combine_index,'wb') as output:
				pickle.dump(count_upp,output)
			with open('Pickle_Files/combine_index=%s_alpha_vals' %combine_index,'wb') as output:
				pickle.dump(alpha_vals,output)
			with open('Pickle_Files/combine_index=%s_alpha_low' %combine_index,'wb') as output:
				pickle.dump(alpha_low,output)
			with open('Pickle_Files/combine_index=%s_alpha_mid' %combine_index,'wb') as output:
				pickle.dump(alpha_mid,output)
			with open('Pickle_Files/combine_index=%s_alpha_upp' %combine_index,'wb') as output:
				pickle.dump(alpha_upp,output)
			with open('Pickle_Files/combine_index=%s_i_low' %combine_index,'wb') as output:
				pickle.dump(i_low,output)
			with open('Pickle_Files/combine_index=%s_i_mean' %combine_index,'wb') as output:
				pickle.dump(i_mean,output)
			with open('Pickle_Files/combine_index=%s_i_upp' %combine_index,'wb') as output:
				pickle.dump(i_upp,output)
	if generate==0:
		if remove_index!=[]:
			with open('Pickle_Files/remove_index=%s_counts' %remove_index,'rb') as output:
				counts=pickle.load(output)
			with open('Pickle_Files/remove_index=%s_count_low' %remove_index,'rb') as output:
				count_low=pickle.load(output)
			with open('Pickle_Files/remove_index=%s_count_mid' %remove_index,'rb') as output:
				count_mid = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_count_upp' %remove_index,'rb') as output:
				count_upp = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_alpha_vals' %remove_index,'rb') as output:
				alpha_vals = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_alpha_low' %remove_index,'rb') as output:
				alpha_low = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_alpha_mid' %remove_index,'rb') as output:
				alpha_mid = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_alpha_upp' %remove_index,'rb') as output:
				alpha_upp = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_i_low' %remove_index,'rb') as output:
				i_low = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_i_mean' %remove_index,'rb') as output:
				i_mean = pickle.load(output)
			with open('Pickle_Files/remove_index=%s_i_upp' %remove_index,'rb') as output:
				i_upp = pickle.load(output)
		if combine_index!=[]:
			with open('Pickle_Files/combine_index=%s_counts' %combine_index,'rb') as output:
				counts=pickle.load(output)
			with open('Pickle_Files/combine_index=%s_count_low' %combine_index,'rb') as output:
				count_low=pickle.load(output)
			with open('Pickle_Files/combine_index=%s_count_mid' %combine_index,'rb') as output:
				count_mid = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_count_upp' %combine_index,'rb') as output:
				count_upp = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_alpha_vals' %combine_index,'rb') as output:
				alpha_vals = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_alpha_low' %combine_index,'rb') as output:
				alpha_low = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_alpha_mid' %combine_index,'rb') as output:
				alpha_mid = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_alpha_upp' %combine_index,'rb') as output:
				alpha_upp = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_i_low' %combine_index,'rb') as output:
				i_low = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_i_mean' %combine_index,'rb') as output:
				i_mean = pickle.load(output)
			with open('Pickle_Files/combine_index=%s_i_upp' %combine_index,'rb') as output:
				i_upp = pickle.load(output)
	return(counts,count_low,count_mid,count_upp,alpha_vals,alpha_low,alpha_mid,alpha_upp,i_low,i_mean,i_upp)

# joblib.memory / memoization




def evaluate_sampled_ODE(file_name=[],combine_index=[],remove_index=[],generate=1,save_fig=1,CI_generate=0,
	alpha=.85,method='numpy',plot='smoothed',frac=.1,**kargs):
	if generate==0:
		file_name = file_name
	else:
		file_name = generate_file_name(combine_index=combine_index,remove_index=remove_index)
	ODE = pickle.load(open(file_name,'rb'))

	x = prob_spline.inv_time_transform(numpy.linspace(ODE[0].tstart,ODE[0].tend,1001))
	s,i,r,sv,iv,c,e = unpack_ODE_vals(ODE)
	x_trans = prob_spline.time_transform(x)

	counts,count_low,count_mid,count_upp,alpha_vals,alpha_low,alpha_mid,alpha_upp,i_low,i_mean,i_upp = CI_vals(ODE,
		remove_index=remove_index,combine_index=combine_index,generate=CI_generate,alpha=alpha,method=method)

	i_max = numpy.max(i_upp)

	birdnames = ODE[0].bc_splines.birdnames

	if remove_index!=6:		# Look at this
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
		if plot=='smoothed':
			ax.plot(x,lowess(x,i_mean[:,j],frac,**kargs),color=color_val[j])
			ax.fill_between(x,lowess(x,i_low[:,j],frac,**kargs),lowess(x,i_upp[:,j],frac,**kargs),color=color_val[j],alpha=.5)
		else:	
		#ax.plot(x,i_low[:,j],color=color_val[j])
			ax.plot(x,i_mean[:,j],color=color_val[j])
		#ax.plot(x,i_upp[:,j],color=color_val[j])
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
	if save_fig==1:
		if remove_index!=[]:
			pylab.savefig('SavedFigures/remove_index=%s_lowess_frac=%s_infections.png' %(remove_index,frac))
		if combine_index!=[]:
			pylab.savefig('SavedFigures/combine_index=%s_infections.png' %combine_index)

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
#	fig.tight_layout()
	if save_fig==1:
		if remove_index!=[]:
			pylab.savefig('SavedFigures/remove_index=%s_populations.png' %remove_index)
		if combine_index!=[]:
			pylab.savefig('SavedFigures/combine_index=%s_populations.png' %combine_index)

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
	if save_fig==1:
		if remove_index!=[]:
			pylab.savefig('SavedFigures/remove_index=%s_feeding_index.png' %remove_index)
		if combine_index!=[]:
			pylab.savefig('SavedFigures/combine_index=%s_feeding_index.png' %combine_index)
	c = numpy.array(c)
	e = numpy.array(e)
	val = numpy.mean(c[:,-1]/e[:,-1])
	#matplotlib.pyplot.close('all')
	return(val)



