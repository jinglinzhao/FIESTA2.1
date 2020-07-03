# FIESTA #

import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


NUNBER_OF_SECTION = np.arange(10,16)				# Test which number of section works best
bic = np.zeros(len(NUNBER_OF_SECTION))				# Bayesian information criterion
n_bic = 0
cv_scores=[]

global V, idx_ccf, v
V 			= (np.arange(401)-200)/10				# CCF Velocity grid [-20,20] km/s
idx_ccf		= abs(V) <= 10							# only study velocities in [-10,10] km/s
v 			= V[idx_ccf]


def power_phase_freq(file):
	'''
	Obtain the power spectrum
	, phase spectrum
	, the frequency grid 
	and line centroid 
	of the input file. 

	'''
	hdulist     = fits.open(file)
	CCF     	= 1 - hdulist[0].data 					# flip the line profile 
	ccf 		= CCF[idx_ccf]
	popt, pcov 	= curve_fit(gaussian, v, ccf)
	rv_gauss 	= popt[1] * 1000						# line centroild (m/s)
	power, phase, freq = ft(ccf, 0.1)					# 0.1 is the spacing in velocity space

	return power, phase, freq, rv_gauss


'''
For now, choose the first file as the template. 
All radial velocities calculated are relative to the first file. 
'''
FILE 							= sorted(glob.glob('./fits/*.fits'))
N_file 							= len(FILE)
power_tpl, phase_tpl, freq, _ 	= power_phase_freq(file=FILE[0])


def freq_of_each_section(power_tpl, freq, number_of_section):
	'''
	Choose a proper frequency range [-upper_freq, upper_freq] where power is concentrated.
	Divide this range into number_of_sections to study RV shifts in each frequency section. 
	Each section has the same integrated power, so that the forthcoming derived RVs are equally weighted. 
	Sections are seperated by freq_i[i] , i = 0,1,2...

	'''
	upper_freq 	= 0.145						# higher limit of frequency range. Hard-coded for now. 
	idx 		= (freq <= upper_freq)
	n_idx 		= len(freq[idx])

	integrated_power 	= np.zeros(n_idx)
	for i in range(n_idx):					# calculate the accumulated (integrated) power
		integrated_power[i] = np.trapz(power_tpl[:i+1], x=freq[:i+1])

	power_per_section = integrated_power[n_idx-1]/number_of_section
	freq_i 	= np.zeros(number_of_section+1)

	# Section 1: [0, freq_i[1]; Section 2: [freq_i[1], freq_i[2]]...
	for i in range(number_of_section):
		freq_i[i] = max(freq[idx][integrated_power<=power_per_section*i])
	freq_i[number_of_section] = upper_freq

	return freq_i


for number_of_section in NUNBER_OF_SECTION:				

	print('\n%d' %number_of_section)

	RV_gauss 	= np.zeros(N_file)							# RV derived from a Gaussian fit
	RV_FT 		= np.zeros((N_file, number_of_section))		# RV derived from the FIESTA method
	delta_RV 	= np.zeros((N_file, number_of_section))		# RV shift
	freq_i 		= freq_of_each_section(power_tpl, freq, number_of_section)

	# plt.rcParams.update({'font.size': 12})
	# fig, axes 	= plt.subplots(figsize=(12, 12))

	for n in range(N_file):

		power, phase, freq, RV_gauss[n] = power_phase_freq(FILE[n])
		RV_gauss[n]	= RV_gauss[n] - RV_gauss[0]

		for i in range(number_of_section):
			RV_FT[n, i] 	= rv_ft(freq_i[i], freq_i[i+1], freq, phase-phase_tpl, power)
			delta_RV[n, i] 	= RV_FT[n, i] - RV_gauss[n]

	# 	plt.plot(freq, phase-phase_tpl)
	# plt.show()	

	#---------------------------#
	# Multiple Regression Model # 
	#---------------------------#

	from sklearn import linear_model
	regr = linear_model.LinearRegression()
	from sklearn.model_selection import cross_val_score
	import seaborn as sns

	from sklearn.model_selection import train_test_split
	train_x, test_x, train_y, test_y = train_test_split(delta_RV, RV_gauss, test_size=0.5, random_state=42)

	# train_x = delta_RV
	# train_y = RV_gauss
	regr.fit (train_x, train_y)
	# The coefficients
	print ('Coefficients: ', regr.coef_)
	print ('Intercept: ', regr.intercept_)


	# cross_val_score returns a list of the scores, which we can visualize
	# to get a reasonable estimate of our classifier's performance
	cv_scores.append(cross_val_score(regr, delta_RV, RV_gauss, cv =10))

	# Plot the results
	# sb.distplot(cv_scores)
	# plt.title('Average score: {}'.format(np.mean(cv_scores)))	
	# plt.show()

	#------------#
	# Prediction # 
	#------------#

	if 0:
		TEST 		= sorted(glob.glob('./test/*.fits'))
		RV_gauss 	= np.zeros(len(TEST))
		RV_FT 		= np.zeros((len(TEST), number_of_section))
		delta_RV 	= np.zeros((len(TEST), number_of_section))

		for n in range(len(TEST)):
			hdulist = fits.open(TEST[n])
			CCF     = 1 - hdulist[0].data 					# flip the line profile

			from scipy.interpolate import interp1d
			from random import random
			f_cubic_spline = interp1d(V, CCF, kind='cubic')
			delta_v = (random()-0.5)/20
			ccf_shift = f_cubic_spline(v + delta_v)
			# ccf_shift = f_cubic_spline(v + 0)

			popt, pcov 	= curve_fit(gaussian, v, ccf_shift)
			RV_gauss[n] = popt[1] * 1000
			RV_gauss[n] = RV_gauss[n] - RV_gauss[0]
			power, phase, freq = ft(ccf_shift, 0.1)

			for i in range(number_of_section):
				RV_FT[n, i] = rv_ft(freq_i[i], freq_i[i+1], freq, phase-phase_tpl, power)
				delta_RV[n, i] = RV_FT[n, i] - RV_gauss[n]


		# 	plt.plot(freq[idx], (phase-phase_tpl)[idx])
		# plt.show()		

		test_x = delta_RV
		test_y = RV_gauss




	# correlation plot of the "features"
	if 0:
		import pandas as pd
		data = delta_RV
		data=pd.DataFrame(data)
		sns.pairplot(data) 
		plt.show()

	# 
	if 0:
		plt.plot(test_y, RV_FT, '.', test_y, y_hat, 'x'); plt.show()

	y_hat= regr.predict(test_x)
	rss = np.mean((y_hat - test_y) ** 2)
	print('rms: %.2f' % rss**0.5)
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(test_x, test_y))

	if 0:
		# #############################################################################
		# Compute paths

		n_alphas = 200
		alphas = np.logspace(-10, -2, n_alphas)

		coefs = []
		for a in alphas:
		    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
		    ridge.fit(train_x, train_y)
		    coefs.append(ridge.coef_)

		# #############################################################################
		# Display results

		ax = plt.gca()

		ax.plot(alphas, coefs)
		ax.set_xscale('log')
		# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
		plt.xlabel('alpha')
		plt.ylabel('weights')
		plt.title('Ridge coefficients as a function of the regularization')
		plt.axis('tight')
		plt.show()


	#PLOT
	fig1 = plt.figure(1)
	frame1=fig1.add_axes((.1,.3,.8,.6))
	plt.title('Section = %d, rms = %.2f' % (number_of_section, rss**0.5))
	plt.plot(test_y, y_hat,'.b', alpha=0.5) #Noisy data
	plt.ylabel('Prediction')
	frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
	plt.grid()

	#Residual plot
	frame2=fig1.add_axes((.1,.1,.8,.2))        
	plt.plot(test_y, y_hat-test_y,'r.',alpha=0.5)
	plt.xlabel('Test data')
	plt.ylabel('Residual')
	plt.grid()
	plt.savefig('./outputs/Redidual%d' % number_of_section)
	plt.close('all')


	#--------------------------------#
	# Bayesian information criterion # 
	#--------------------------------#
	# https://en.wikipedia.org/wiki/Bayesian_information_criterion
	n = len(test_y)
	k = number_of_section + 1
	bic[n_bic] = n*np.log(error_variance(test_y, y_hat)) + k*np.log(n)
	n_bic += 1

plt.plot(NUNBER_OF_SECTION, bic)
plt.xlabel('# number_of_sectiontions')
plt.ylabel('bic')
plt.savefig('./outputs/bic.png')
# plt.show()


# plt.savefig('./outputs/Overview.png')

if 0:
	# ---------------------------- #
	# Radial velocity correlations #
	# ---------------------------- #
	RV_gauss 	= (RV_gauss - RV_gauss[0]) * 1000 			# all radial velocities are relative to the first ccf
	fig, axes 	= plt.subplots(figsize=(15, 5))
	plt.subplots_adjust(wspace=wspace)
	plot_correlation(RV_gauss, RV, RV_L, RV_H)
	plt.savefig('./outputs/RV_FT.png')
	plt.close('all')


# plt.plot(RV_FT); plt.show()


# plt.plot(jitter, RV_FT, '.'); plt.show()