import pandas as pd
import numpy
#import pylab
import unittest
import scipy.stats
import joblib
import BirdCount
import MCMC
import bloodmeal

def F(coeff_mat,time):            #Returns the negative piece of the birdcount derivative, or 0.
	return(-numpy.clip(birdcounts_derivative(coeff_mat,time),-numpy.inf,0))	

def birdcounts(splines,time):   #the splines are a list of the calculated splines, time is the time to be differentiated
	return(numpy.array([splines[i](time) for i in range(len(splines))]))

def birdcounts_derivative(splines,time,sign="+"):
	vals = numpy.array([splines[i].derivative(i) for i in range(len(splines))])
	if sign!="+":
		return(numpy.array([numpy.min((vals[i],0)) for i in range(len(vals))]))
	else:
		return(numpy.array([numpy.max((vals[i],0)) for i in range(len(vals))]))



def alpha_calc(bm,counts):   #If 0 bm returns 0, atm if 0 counts returns inf
	bm_rat = bm / numpy.sum(bm)
	count_rat = counts / numpy.sum(counts)
	with numpy.errstate(divide="raise"):
		alpha = bm_rat/count_rat
		weight = numpy.dot(alpha,counts)
		return(alpha/weight)


#change bc,bm to respective splines	

def rhs_count(Y,t, beta1, beta2, gammab, v, b, d, dv, dEEE, bird_counts,bloodmeals,mospop,mosprime,mosin,fun,mos_coeff,p):
	# Consider adding epsilon term , for proportion infected entering population eps = .001
	eps = .001
	s=Y[0:p]
	i=Y[p:2*p]
	r=Y[2*p:3*p]
	sv=Y[-2]
	iv=Y[-1]
	alpha_val = alpha_calc(bloodmeals(t),bird_counts(t))
	N=bird_counts(t)
	denom = numpy.dot(N,alpha_val)
	lambdab = beta1*v*iv*numpy.array(alpha_val)/denom
	lambdav = v*(numpy.dot(beta2*i,alpha_val))/denom
	# Bird Mortality
	ds = birdcounts_derivative(bird_counts,t)*(1-eps)-lambdab*s-birdcounts_derivative(bird_counts,t,sign="-")*s/N
	di = birdcounts_derivative(bird_counts,t))*eps+lambdab*s-gammab*i-birdcounts_derivative(bird_counts,t,sign="-")*i/N
	dr = gammab*i-birdcounts_derivative(bird_counts,t,sign="-")*r/N
	dsv = -lambdav*sv  #Need to be updated
	div = lambdav*sv           # Need to be updated
	#dSv = mospop(t) - dIv
	dY = numpy.hstack((ds,di,dr,dsv,div))
	return dY

def rhs_prop(Y,t, beta1_rho,beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_coeff_mat,mospop,mosprime,mosin,fun,mos_coeff,p):
	# Here beta1 = beta1/rho_v, where Nv = theta_v/rho_v
	eps = .001
	s=Y[0:p]
	i=Y[p:2*p]
	r=Y[2*p:3*p]
	sv=Y[-2]
	iv=Y[-1]

	alpha_val = alpha_calc(bloodmeal.bloodmeal_function(bm_coeff_mat,t),BirdCount.birdcounts_function(bc_coeff_mat,t))
	theta=BirdCount.birdcounts_function(bc_coeff_mat,t)		# Theta is the vector of the MCMC sample coeffieicents
	theta_v = bloodmeal.vector_pop(mos_coeff,t)
	denom = numpy.dot(theta,alpha_val)
	lambdab = beta1_rho*v*iv*numpy.array(alpha_val)*theta_v/denom
	lambdav = v*(numpy.dot(beta2*i*theta,alpha_val))/denom
	# Bird Mortality

	ds = BirdCount.Xi(bc_coeff_mat,t)*((1-eps)-s)-lambdab*s
	di = BirdCount.Xi(bc_coeff_mat,t)*(eps-i)+lambdab*s-gammab*i
	dr = gammab*i - r*BirdCount.Xi(bc_coeff_mat,t)
	dsv = bloodmeal.Xi(mos_coeff,t)*iv - lambdav*sv
	div = lambdav*sv - bloodmeal.Xi(mos_coeff,t)*iv

	dY = 365./2*numpy.hstack((ds,di,dr,dsv,div))  # the 365/2 is the rate of change of the time transform
	return dY


def ode_test_rhs(t,Y, args):
	return(rhs_prop(Y,t,*args))

def ode_solver(Y0,t,args=()):
	solver = scipy.integrate.ode(ode_test_rhs)
	solver.set_f_params(args)
	solver.set_integrator('lsoda')
	solver.set_initial_value(Y0,t[0])
	Y = numpy.zeros((len(t),len(Y0)))
	Y[0] = Y0
	for i in range(1,len(t)):
		assert solver.successful()
		solver.integrate(t[i])
		solver.set_initial_value(numpy.clip(solver.y,0,numpy.inf),solver.t)
		Y[i] = solver.y
	return(Y)

def run_ode_solver(beta1,rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,flag):
	# Set up to run the ODE using the ode_solver function, which manually integrates the ode for each iteration 

	p = len(bm_coeff_mat[:,0])+1
	beta2 = 1
	gammab = .1*numpy.ones(p)
	v=.14			# Biting Rate of Vectors on Hosts
	b=0			# Bird "Recruitment" Rate
	d=0	
			# Bird "Death" Rate
	dv=.10			# Mosquito Mortality Rate
	dEEE= 0	
	 # Run for ~ 6 Months
	
	T = scipy.linspace(tstart,tend,1001)
	if flag==0:
		Sv= 1*bloodmeal.vector_pop(mos_coeff,tstart)
		Iv= 0*bloodmeal.vector_pop(mos_coeff,tstart)
		S0 = 1*BirdCount.birdcounts_function(bc_coeff_mat,tstart)	
		I0 = 0*BirdCount.birdcounts_function(bc_coeff_mat,tstart)	
		R0 = 0*BirdCount.birdcounts_function(bc_coeff_mat,tstart)	
	elif flag==1:
		Sv = .99
		Iv = .01
		S0 = .99*numpy.ones(p)
		I0 = .01*numpy.ones(p)
		R0 = 0*numpy.ones(p)
	Y0 = numpy.hstack((S0, I0, R0, Sv, Iv))
	#Y = scipy.integrate.odeint(rhs_func,Y0,T,args = (beta1, beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_amean,bm_bmean,bloodmeal.vector_pop,bloodmeal.vector_derivative,bloodmeal.vector_in,bloodmeal.fun,mos_coeff,p))
	Y = ode_solver(Y0,T,args = (beta1, beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_coeff_mat,bloodmeal.vector_pop,bloodmeal.vector_derivative,bloodmeal.vector_in,bloodmeal.fun,mos_coeff,p))
	return(Y)

def run_ode(beta1,rhs_func,bloodmeals,bird_counts,mosquitos,tstart,tend,flag):
	p = len(bird_counts)
	beta2 = 1
	gammab = .1*numpy.ones(p)
	v=.14			# Biting Rate of Vectors on Hosts
	b=0			# Bird "Recruitment" Rate
	d=0	
			# Bird "Death" Rate
	dv=.10			# Mosquito Mortality Rate
	dEEE= 0	
	 # Run for ~ 6 Months
	
	T = scipy.linspace(tstart,tend,1001)
	if flag==0:
		Sv= 1*bloodmeal.vector_pop(mos_coeff,tstart)
		Iv= 0*bloodmeal.vector_pop(mos_coeff,tstart)
		S0 = 1*BirdCount.birdcounts_function(bc_coeff_mat,tstart)	
		I0 = 0*BirdCount.birdcounts_function(bc_coeff_mat,tstart)	
		R0 = 0*BirdCount.birdcounts_function(bc_coeff_mat,tstart)
		Y0 = numpy.hstack((S0, I0, R0, Sv, Iv))	
	elif flag==1:
		Sv = .99
		Iv = .01
		S0 = .99*numpy.ones(p)
		I0 = .01*numpy.ones(p)
		R0 = 0*numpy.ones(p)
		Y0 = numpy.hstack((S0, I0, R0, Sv, Iv))
	elif flag==2:
		print("hi")
		Sv = numpy.log(.99)
		Iv = numpy.log(.01)
		S0 = numpy.log(.99)*numpy.ones(p)
		I0 = numpy.log(.01)*numpy.ones(p)
		Y0 = numpy.hstack((S0, I0, Sv, Iv))
	Y = scipy.integrate.odeint(rhs_func,Y0,T,args = (beta1, beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_coeff_mat,bloodmeal.vector_pop,bloodmeal.vector_derivative,bloodmeal.vector_in,bloodmeal.fun,mos_coeff,p))
	return(Y)
	
def get_SIR_vals(Y,p):		# Takes the values from scipy.integrate.odeint and returns the SIR vals
	S=Y[:,0:p]
	I=Y[:,p:2*p]
	R=Y[:,2*p:3*p]
	sv=Y[:,-2]
	iv=Y[:,-1]
	return(S,I,R,sv,iv)

def get_SI_vals(Y,p):  # for the log function that doesn't calculate R
	S=Y[:,0:p]
	I=Y[:,p:2*p]
	sv=Y[:,-2]
	iv=Y[:,-1]
	return(S,I,sv,iv)

def eval_log_results(Y,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,bird_data_file):
	import pylab
	birdnames = pd.read_csv(bird_data_file,index_col=0).index
	name_list = list(birdnames)
	name_list.append('Vector')
	T = scipy.linspace(tstart,tend,1001)
	p = len(bm_coeff_mat[:,0])+1
	s,i,sv,iv = get_SI_vals(Y,p)
	bc = numpy.zeros((p,len(T)))
	bm = numpy.zeros((p,len(T)))
	alpha_val = numpy.zeros((p,len(T)))
	mos_pop = numpy.zeros(len(T))
	for j in range(len(T)):
		bc[:,j] = BirdCount.birdcounts_function(bc_coeff_mat,T[j])
		bm[:,j] = bloodmeal.bloodmeal_function(bm_coeff_mat,T[j])
		mos_pop[j] = bloodmeal.vector_pop(mos_coeff,T[j])
		alpha_val[:,j] = alpha_calc(bm[:,j],bc[:,j])
	sym = ['b','g','r','c','m','y','k','--','g--']
	pylab.figure(1)
	for k in range(p+1):
		if k==p:
			pylab.plot(T,mos_pop,sym[k])
		else:
			pylab.plot(T,bc[k],sym[k])
	pylab.title("Populations")
	pylab.legend(name_list)
	pylab.figure(2)
	for k in range(p):
		temp=numpy.exp(i[:,k])
		pylab.plot(T,temp,sym[k])	
	pylab.legend(birdnames)
	pylab.title("Infected Birds")
	pylab.figure(3)
	for k in range(p):
		pylab.plot(T,alpha_val[k])
	pylab.legend(birdnames)
	pylab.title("Feeding Index Values")
	return()

def eval_ode_results(Y,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,bird_data_file,flag,alpha=1):	
	import pylab
	birdnames = pd.read_csv(bird_data_file,index_col=0).index
	name_list = list(birdnames)
	name_list.append('Vector')
	T = scipy.linspace(tstart,tend,1001)
	p = len(bm_coeff_mat[:,0])+1
	s,i,r,sv,iv = get_SIR_vals(Y,p)
	bc = numpy.zeros((p,len(T)))
	bm = numpy.zeros((p,len(T)))
	alpha_val = numpy.zeros((p,len(T)))
	mos_pop = numpy.zeros(len(T))
	for j in range(len(T)):
		bc[:,j] = BirdCount.birdcounts_function(bc_coeff_mat,T[j])
		bm[:,j] = bloodmeal.bloodmeal_function(bm_coeff_mat,T[j])
		alpha_val[:,j] = alpha_calc(bm[:,j],bc[:,j])
		mos_pop[j] = bloodmeal.vector_pop(mos_coeff,T[j])	
	sym = ['b','g','r','c','m','y','k','--','g--']
	pylab.figure(1)
	for k in range(p):
		#if k==p:
		#	pylab.plot(T,mos_pop,sym[k])
		#else:
		pylab.plot(T,bc[k],sym[k],alpha=alpha)
	pylab.title("Populations")
	pylab.legend(name_list)
	N=s+i+r
	N=numpy.clip(N,0,numpy.inf)
	pylab.figure(2)
	for k in range(p):
		if flag==0:
			temp=numpy.divide(i[:,k],N[:,k])
		elif flag==1:
			temp=i[:,k]
		elif flag==2:
			temp=numpy.exp(i[:,k])
		pylab.plot(numpy.linspace(90,270,1001),temp,sym[k],alpha=alpha)	
		#pylab.plot(T,temp,sym[k],alpha=alpha)	
	pylab.legend(birdnames)
	pylab.title("Infected Birds")
	pylab.figure(3)
	for k in range(p):
		numpy.linspace(90,270,1001)
		pylab.plot(numpy.linspace(90,270,1001),alpha_val[k],alpha=alpha)
		#pylab.plot(T,alpha_val[k],alpha=alpha)
	pylab.legend(birdnames)
	pylab.title("Feeding Index Values")
	return()

def findbeta(beta1,rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,flag,ODE_flag):  
	print beta1
	p = len(bm_coeff_mat[:,0])+1
	Y = run_ode(beta1,rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,flag)
	s,i,r,sv,iv = get_SIR_vals(Y,p)
	N=s+i+r
	if ODE_flag==0:
		finalrec=numpy.ma.divide(r[-1].sum()+i[-1].sum(),N[-1].sum())
		final=finalrec-.13
	else:
		N=BirdCount.birdcounts_function(bc_coeff_mat,tend)
		r = r[-1]*N
		i = i[-1]*N
		finalrec = numpy.ma.divide(r.sum()+i.sum(),N.sum())
		final = finalrec-.13
	print(numpy.abs(final))
	return numpy.abs(final)

def debug_fun(Y,T, beta1_rho,beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_coeff_mat,mospop,mosprime,mosin,fun,mos_coeff,p):
	ds = []
	di = []
	dsv = []
	div = []
	lbdv = []
	eps = .001
	s=Y[0:p]
	i=Y[p:2*p]
	sv=Y[-2]
	iv=Y[-1]
	dt = T[1]-T[0]
	for t in T:
		alpha_val = alpha_calc(bloodmeal.bloodmeal_function(bm_coeff_mat,t),BirdCount.birdcounts_function(bc_coeff_mat,t))
		theta=BirdCount.birdcounts_function(bc_coeff_mat,t)		# Theta is the vector of the MCMC sample coeffieicents
		theta_v = bloodmeal.vector_pop(mos_coeff,t)
		denom = numpy.dot(theta,alpha_val)
		lambdab = beta1_rho*v*numpy.exp(iv)*numpy.array(alpha_val)*theta_v/denom
		lambdav = v*(numpy.dot(beta2*numpy.exp(i)*theta,alpha_val))/denom
		# Bird Mortality

		ds.append(BirdCount.Xi(bc_coeff_mat,t)*((1-eps)*numpy.exp(-s)-1)-lambdab)
		di.append(BirdCount.Xi(bc_coeff_mat,t)*(eps*numpy.exp(-i)-1)+lambdab*numpy.exp(s-i)-gammab)
		#dr = gammab*numpy.exp(i-r) - BirdCount.Xi(bc_coeff_mat,t)
		dsv.append(bloodmeal.Xi(mos_coeff,t)*numpy.exp(iv-sv) - lambdav)
		div.append(lambdav*numpy.exp(sv-iv) - bloodmeal.Xi(mos_coeff,t))
		s += ds[-1]*dt
		i += di[-1]*dt
		sv += dsv[-1]*dt
		iv += div[-1]*dt
		lbdv.append(lambdav)


	return(ds,di,dsv,div,lbdv)

def get_ODE_vals(Y,T, beta1_rho,beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_coeff_mat,mospop,mosprime,mosin,fun,mos_coeff,p):
	ds = []
	di = []
	dsv = []
	div = []
	lbdv = []
	alpha = []
	eps = .001
	s=Y[0:p]
	i=Y[p:2*p]
	sv=Y[-2]
	iv=Y[-1]
	dt = T[1]-T[0]
	for t in T:
		alpha_val = alpha_calc(bloodmeal.bloodmeal_function(bm_coeff_mat,t),BirdCount.birdcounts_function(bc_coeff_mat,t))
		theta=BirdCount.birdcounts_function(bc_coeff_mat,t)		# Theta is the vector of the MCMC sample coeffieicents
		theta_v = bloodmeal.vector_pop(mos_coeff,t)
		denom = numpy.dot(theta,alpha_val)
		lambdab = beta1_rho*v*iv*numpy.array(alpha_val)*theta_v/denom
		lambdav = v*(numpy.dot(beta2*i*theta,alpha_val))/denom
		# Bird Mortality

		ds.append(BirdCount.Xi(bc_coeff_mat,t)*((1-eps)-s)-lambdab*s)
		di.append(BirdCount.Xi(bc_coeff_mat,t)*(eps-i)+lambdab*s-gammab*i)
		#dr = gammab*numpy.exp(i-r) - BirdCount.Xi(bc_coeff_mat,t)
		dsv.append(bloodmeal.Xi(mos_coeff,t)*iv - lambdav*sv)
		div.append(lambdav*sv - bloodmeal.Xi(mos_coeff,t)*iv)
		alpha.append(alpha_val)
		s += ds[-1]*dt
		i += di[-1]*dt
		sv += dsv[-1]*dt
		iv += div[-1]*dt
		lbdv.append(lambdav)


	return(ds,di,dsv,div,lbdv,alpha)


	