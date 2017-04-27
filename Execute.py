#Calculate optimal sigma values from cv initially
#Resample from the original data to generate new samples,
#For each sample data, find the spline with given optimal sigma values
#Store each of the splines
#For these sampled splines (N=1000, etc) run the ODE
#For each of these ODE results, reoptimize the beta value.
#For each of these optimized betas, evaluate the ODE to generate the CI's, 

#Multinomial, be careful of proportion vs number
#Sample the total bloodmeal counts (Poisson), then sample the proportions (Multinomial)

#Cs. Melanura biting rate data
