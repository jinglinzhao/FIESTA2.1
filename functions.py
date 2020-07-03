# Fourier transform using numpy.fft.rfft # 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

alpha 	=	0.5

####################################################################

def ft(signal, spacing):
	oversample 	= 6 										# oversample folds; to be experiemented further
	n 			= 2**(int(np.log(signal.size)/np.log(2))+1 + oversample)
	fourier 	= np.fft.rfft(signal, n)
	freq 		= np.fft.rfftfreq(n, d=spacing)
	power 		= np.abs(fourier)**2
	phase 		= np.angle(fourier)

	return [power, phase, freq]

####################################################################

def gaussian(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

####################################################################

def plot_overview(x, ccf, power, phase, phase_tpl, freq, freq_HL, freq_HN):

	idx 	= (freq <= freq_HN)
	idx_L 	= (freq < freq_HL)
	idx_H 	= (freq >= freq_HL) & (freq < freq_HN)

	# Singal # 
	plt.subplot(221)
	plt.plot(x, ccf, 'k', alpha=alpha)
	plt.title('Signal (CCF)')
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Normalized intensity')
	plt.grid(True)

	# power spectrum # 
	plt.subplot(222)
	plt.plot(freq[idx], power[idx], 'k', alpha=alpha)
	plt.title('Power spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel('Power')
	plt.grid(True)

	# differential phase spectrum 
	plt.subplot(223)
	# diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl)
	diff_phase = phase - phase_tpl
	plt.plot(freq[idx], diff_phase[idx], 'k', alpha=alpha)
	plt.title('Differential phase spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel(r'$\Delta \phi$ [radian]')
	plt.grid(True)

	# shift spectrum # 
	plt.subplot(224)
	rv = -np.gradient(diff_phase, np.mean(np.diff(freq))) / (2*np.pi)
	plt.plot(freq[idx], rv[idx] * 1000, 'k', alpha=alpha)
	plt.title('Shift spectrum')
	plt.xlabel(r'$\xi$ [s/km]')	
	plt.ylabel('RV [m/s]')
	plt.grid(True)



####################################################################

# calculate the "averaged" radial veflocity shift between freq1 and freq2 in Fourier space

def rv_ft(freq1, freq2, freq, diff_phase, power):

	idx 	= (freq >= freq1) & (freq <= freq2)
	coeff 	= np.polyfit(freq[idx], diff_phase[idx], 1, w=power[idx]**0.5)
	RV_FT 	= -coeff[0] / (2*np.pi) * 1000

	return RV_FT

####################################################################

def plot_correlation(RV_gauss, RV, RV_L, RV_H):

	plt.subplot(131)
	plt.plot(RV_gauss, RV, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT}$ [m/s]')

	plt.subplot(132)
	plt.plot(RV_gauss, RV_L, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV_L, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV_L)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT,L}$ [m/s]')

	plt.subplot(133)
	plt.plot(RV_gauss, RV_H, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV_H, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV_H)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT,H}$ [m/s]')	

####################################################################

#  unbiased form of approximating the error variance (therefore len(x)-1)

def error_variance(y_test, y_pred):
	return 1/(len(y_test)-1) * sum((y_test-y_pred)**2)

