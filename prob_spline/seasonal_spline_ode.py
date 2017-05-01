import abc
import numbers

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation
import pandas as pd
from . import base
import prob_spline
import matplotlib.pyplot as pyplot
import scipy.stats

class Seasonal_Spline_ODE():

	def __init__(self, bc_splines, bm_splines, mos_curve,tstart,tend,beta_1=1):
		self.tstart = tstart
		self.tend = tend
		self.beta_1 = beta_1
		self.Y = self.run_ode(beta_1,bm_splines,bc_splines,mos_curve)

	def alpha_calc(self,bm,counts):   #If 0 bm returns 0, atm if 0 counts returns inf
		bm_rat = bm / numpy.sum(bm)
		count_rat = counts / numpy.sum(counts)
		with numpy.errstate(divide="ignore"):
			alpha = numpy.where(count_rat>0,bm_rat/count_rat,0)
			weight = numpy.sum(alpha*counts,axis=0)
			return(alpha/weight)

	def rhs(self,Y,t, bc_splines, bm_splines, mos_curve):
		# Consider adding epsilon term , for proportion infected entering population eps = .001
		p=self.p
		eps = .001
		s=Y[0:p]
		i=Y[p:2*p]
		r=Y[2*p:3*p]
		sv=Y[-2]
		iv=Y[-1]
		alpha_val = self.alpha_calc(bm_splines(t),bc_splines(t))
		N=bc_splines(t)
		N_v = mos_curve(t)
		denom = numpy.dot(N,alpha_val)
		lambdab = self.beta1*self.v*iv*numpy.array(alpha_val)*N_v/denom
		lambdav = self.v*(numpy.dot(self.beta2*i*N,alpha_val))/denom

		ds = bc_splines.pos_der(t)*(1-eps) - lambdab*s
		di = bc_splines.pos_der(t)*eps + lambdab*s - self.gammab*i
		dr = self.gammab*i - bc_splines.pos_der(t)
		dsv = mos_curve.pos_der(t)*iv-lambdav*sv  
		div = lambdav*sv - mos_curve.pos_der(t)*iv          

		dY = 365./2*numpy.hstack((ds,di,dr,dsv,div))  # the 365/2 is the rate of change of the time transform
		return dY

	def run_ode(self,beta1,bm_splines,bc_splines,mos_curve):
		self.p = len(bc_splines(0))
		self.beta2 = 1
		self.gammab = .1*numpy.ones(self.p)
		self.v=.14			# Biting Rate of Vectors on Hosts
		self.b=0			# Bird "Recruitment" Rate
		self.d=0			# Bird "Death" Rate
		self.dv=.10			# Mosquito Mortality Rate
		self.dEEE= 0	
		self.beta1= beta1
		 # Run for ~ 6 Months
		
		T = scipy.linspace(self.tstart,self.tend,1001)
		Sv = .99
		Iv = .01
		S0 = .99*numpy.ones(self.p)
		I0 = .01*numpy.ones(self.p)
		R0 = 0*numpy.ones(self.p)
		Y0 = numpy.hstack((S0, I0, R0, Sv, Iv))
		Y = scipy.integrate.odeint(self.rhs,Y0,T,args = (bc_splines,bm_splines,mos_curve))
		return(Y)
		
	def get_SIR_vals(self,Y):		# Takes the values from scipy.integrate.odeint and returns the SIR vals
		p=self.p
		S=Y[:,0:p]
		I=Y[:,p:2*p]
		R=Y[:,2*p:3*p]
		sv=Y[:,-2]
		iv=Y[:,-1]
		return(S,I,R,sv,iv)

	def eval_ode_results(self,Y,bm_splines,bc_splines,mos_curve,alpha=1):	
		import pylab
		self.birdnames = bc_splines.birdnames
		name_list = list(self.birdnames)
		name_list.append('Vector')
		T = scipy.linspace(self.tstart,self.tend,1001)
		p = self.p
		s,i,r,sv,iv = self.get_SIR_vals(Y)
		bc = numpy.zeros((p,len(T)))
		bm = numpy.zeros((p,len(T)))
		alpha_val = numpy.zeros((p,len(T)))
		mos_pop = numpy.zeros(len(T))
		bc = bc_splines(T)
		bm = bm_splines(T)
		alpha_val = self.alpha_calc(bm_splines(T),bc_splines(T))
		mos_pop = mos_curve(T)	
		sym = ['b','g','r','c','m','y','k','--','g--']
		pylab.figure(1)
		for k in range(self.p):
			pylab.plot(prob_spline.inv_time_transform(T),bc[k],sym[k],alpha=alpha)
		pylab.title("Populations")
		pylab.legend(name_list)
		N=s+i+r
		N=numpy.clip(N,0,numpy.inf)
		pylab.figure(2)
		for k in range(self.p):
			temp=i[:,k]
			pylab.plot(prob_spline.inv_time_transform(T),temp,sym[k],alpha=alpha)	
		pylab.legend(self.birdnames)
		pylab.title("Infected Birds")
		pylab.figure(3)
		for k in range(self.p):
			pylab.plot(prob_spline.inv_time_transform(T),alpha_val[k],alpha=alpha)
		pylab.legend(self.birdnames)
		pylab.title("Feeding Index Values")
		return()

	def findbeta(beta1,rhs_func,bm_splines,bc_splines,mos_curve):  
		print(beta1)
		Y = run_ode(beta1,self.rhs,bm_splines,bc_splines,mos_curve)
		s,i,r,sv,iv = get_SIR_vals(Y,self.p)
		N=s+i+r
		N=bc_splines(self.tend)
		r = r[-1]*N
		i = i[-1]*N
		finalrec = numpy.ma.divide(r.sum()+i.sum(),N.sum())
		final = finalrec-.13
		print(numpy.abs(final))
		return numpy.abs(final)

			