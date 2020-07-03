# FIESTA #

import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

Section = np.arange(25,26)
bic = np.zeros(len(Section))
n_bic = 0


freq_HN 	= 0.1563									# higher limit of frequency range

# jitter 		= np.loadtxt('/Volumes/DataSSD/SOAP_2/outputs/02.01/RV.dat')
# jitter 		= (jitter - np.mean(jitter))

FILE 		= sorted(glob.glob('./fits/*.fits'))
global power_tpl, idx_ccf, phase_tpl, freq_LH
def templating(file):
	hdulist     = fits.open(file)
	CCF_tpl     = 1 - hdulist[0].data 						# flip the line profile
	V 			= (np.arange(401)-200)/10					# CCF Velocity grid
	idx_ccf		= (abs(V) <= 10)
	v 			= V[idx_ccf]
	ccf_tpl 	= CCF_tpl[idx_ccf]
	power_tpl, phase_tpl, freq = ft(ccf_tpl, 0.1)

	idx 		= (freq <= freq_HN)

	n_idx 		= len(freq[idx])
	power_int 	= np.zeros(n_idx) 							# integrated power
	for i in range(n_idx):
		power_int[i] = np.trapz(power_tpl[:i+1], x=freq[:i+1])


	for N_section in Section:											# equally divide power spectrum into #N_section
		print('\n%d' %N_section)

		per_portion = power_int[n_idx-1]/N_section
		freq_LH 	= np.zeros(N_section+1)

		# Section 1: [0, freq_LH[1]; section 2: [freq_LH[1], freq_LH[2]]...
		for i in range(N_section):
			freq_LH[i] = max(freq[idx][power_int<=per_portion*i])
		freq_LH[N_section] = freq_HN

#------------------#
# Line deformation #
#------------------#
def line_deformation(N_section,FILE):

	N_file = len(FILE)
	RV_gauss = np.zeros(N_file)	
	RV_FT = np.zeros((N_file, N_section))
	delta_RV = np.zeros((N_file, N_section))
# plt.rcParams.update({'font.size': 12})
# fig, axes 	= plt.subplots(figsize=(12, 12))
	for n in range(N_file):
		hdulist     = fits.open(FILE[n])
		CCF         = 1 - hdulist[0].data 					# flip the line profile
		ccf 		= CCF[idx_ccf]
		popt, pcov 	= curve_fit(gaussian, v, ccf)
		RV_gauss[n] = popt[1] * 1000

		power, phase, freq = ft(ccf, 0.1)
		for i in range(N_section):
			RV_FT[n, i] = rv_ft(freq_LH[i], freq_LH[i+1], freq, phase-phase_tpl, power_tpl)
			delta_RV[n, i] = RV_FT[n, i] - RV_gauss[n]
	return RV_gauss, RV_FT, delta_RV


	# mean_Gaussian = np.mean(RV_gauss)
	# mean_delta_RV = np.mean(delta_RV)
	# RV_gauss = (RV_gauss - np.mean(RV_gauss))*1000
	# delta_RV = delta_RV - np.mean(delta_RV)
	# RV derived from a Gaussian fit
 		
 	
#---------------------------#
# Multiple Regression Model # 
#---------------------------#
templating(FILE[0])
from sklearn import linear_model
regr = linear_model.LinearRegression()
from sklearn.model_selection import cross_val_score
import seaborn as sb

train_x = delta_RV
train_y = RV_gauss
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


TEST 		= sorted(glob.glob('./test/*.fits'))
RV_gauss 	= np.zeros(len(TEST))
RV_FT 		= np.zeros((len(TEST), N_section))
delta_RV 	= np.zeros((len(TEST), N_section))


for n in range(len(TEST)):
	hdulist     = fits.open(TEST[n])
	CCF         = 1 - hdulist[0].data 					# flip the line profile
	ccf 		= CCF[idx_ccf]


	from scipy.interpolate import interp1d
	from random import random
	f_cubic_spline = interp1d(V, CCF, kind='cubic')
	ccf_shift = f_cubic_spline(v + (random()-0.5)/10)


	popt, pcov 	= curve_fit(gaussian, v, ccf)
	RV_gauss[n] = popt[1]
	power, phase, freq = ft(ccf_shift, 0.1)
	for i in range(N_section):
		RV_FT[n, i] = rv_ft(freq_LH[i], freq_LH[i+1], freq, phase-phase_tpl, power_tpl)
		delta_RV[n, i] = RV_FT[n, i] - RV_gauss[n] * 1000
RV_gauss = (RV_gauss - mean_Gaussian)*1000
delta_RV = delta_RV - mean_delta_RV
test_x = delta_RV
test_y = RV_gauss

y_hat= regr.predict(delta_RV)
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
plt.title('Section = %d, rms = %.2f' % (N_section, rss**0.5))
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
plt.savefig('./outputs/Redidual%d' % N_section)
plt.close('all')


#--------------------------------#
# Bayesian information criterion # 
#--------------------------------#
# https://en.wikipedia.org/wiki/Bayesian_information_criterion
n = len(test_y)
k = N_section + 1
bic[n_bic] = n*np.log(error_variance(test_y, y_hat)) + k*np.log(n)
n_bic += 1

plt.plot(Section, bic)
plt.xlabel('# sections')
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