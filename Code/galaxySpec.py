########################################################
##################### Imports ##########################
########################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from uncertainties import ufloat
from astropy.cosmology import FlatLambdaCDM
np.random.seed(3_141_592_653)

########################################################
################# Define the redshift ##################
########################################################

z = 0.077985 # +- 1.917881e-5 redshift/corrimiento al rojo
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
lumDist= cosmo.luminosity_distance(z).to('cm').value

def redshifted(intrinsic,z):
    return intrinsic*(z+1)

def redCorrected(observed,z):
    return observed/(z+1)

########################################################
######### Funciones modeladoras de cosas################
########################################################

# Define the function to fit (a zeroth order polynomial)
def recta(x,m):
    return m*x

# Define another function to fit (one Gaussian curve)
def gaussian(x, mu, sigma, A, c):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + c

# Define another function to fit (two Gaussian curves)
def doubleGaussian(x, mu1, sigma1, A1, mu2, sigma2, A2, c):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + c

# Define another function to fit (three Gaussian curves)
def tripleGaussian(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, c):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + A3*np.exp(-(x-mu3)**2/(2*sigma3**2)) + c

########################################################
############## Seting up the data ######################
########################################################

# Open the file with the flux
spec = fits.open('/Users/mphstph/Documents/TTB/DatosTTB/J2215+0002/J2215+0002_coadd_tellcorr.fits')
#print(spec[1].header)

# Data of the .fits file
wave = spec[1].data['wave']/(z+1)
wave_grid_mid = spec[1].data['wave_grid_mid']
flux = spec[1].data['flux'] * 1e-17
ivar = spec[1].data['ivar']
mask = spec[1].data['mask']
telluric = spec[1].data['telluric']
obj_model = spec[1].data['obj_model']


# Select a subset of the data
# Deleting first noise part
subset = (wave > 8800)
# Cleaning some noisy points
eliminar = np.where(1/ivar[subset] > 10000) # con N = 10000, se elimina el 10% de los puntos
error = np.delete(1/ivar[subset],eliminar)
#eliminar = []
waveClean = np.delete(wave[subset],eliminar) # redshift corrected
fluxClean = np.delete(flux[subset],eliminar)



########################################################
############## Modeling of the data ####################
########################################################

######################
# Paschen Alpha 4 -> 3
subset_alpha = (waveClean > 18700) & (waveClean < 18800)
wave_alpha = waveClean[subset_alpha]
flux_alpha = fluxClean[subset_alpha]

# 1 Gaussian
# Guess initial values for the parameters (1 Gaussian)
p1_alpha = [18750, 2.83, 9.508e-16,1.085e-17]

# Fit the data
popt1_alpha, pcov1_alpha = curve_fit(gaussian, wave_alpha, flux_alpha, p1_alpha)
mu1_alpha, sigma1_alpha, A1_alpha, c1_alpha = popt1_alpha

# 2 Gaussians
# Guess initial values for the parameters (2 Gaussians)
p2_alpha = [18750, 2.99, 6.33e-16, 18751, 1.45, 4.56e-16, 9.16e-18]

# Fit the data
popt2_alpha, pcov2_alpha = curve_fit(doubleGaussian, wave_alpha, flux_alpha, p2_alpha)
mu21_alpha, sigma21_alpha, A21_alpha, mu22_alpha, sigma22_alpha, A22_alpha, c2_alpha = popt2_alpha

plt.clf()
plt.plot(waveClean,fluxClean, label='data')
plt.plot(wave_alpha,gaussian(wave_alpha, mu21_alpha, sigma21_alpha, A21_alpha, c2_alpha))
plt.plot(wave_alpha,gaussian(wave_alpha, mu22_alpha, sigma22_alpha, A22_alpha, c2_alpha))
plt.plot(wave_alpha, doubleGaussian(wave_alpha, *popt2_alpha), 'r-', label='Pa alpha')

############################
# Linea 12821 Paschen 5 -> 3

subset_12821 = (waveClean > 12760) & (waveClean < 12880)
wave_12821 = waveClean[subset_12821]
flux_12821 = fluxClean[subset_12821]

# Fit the data
p12821 = [12818, 8, 1e-17,0]
popt12821, pcov12821 = curve_fit(gaussian, wave_12821, flux_12821, p12821)
mu12821, sigma12821, A12821, c12821 = popt12821
flux12821 = A12821*sigma12821*np.sqrt(2*np.pi)


# Linea 10941 Paschen 6 -> 3

subset_10941 = (waveClean > 10910) & (waveClean < 10960)
wave_10941 = waveClean[subset_10941]
flux_10941 = fluxClean[subset_10941]

# Fit the data
p10941 = [10938, 10, 1e-17,0]
popt10941, pcov10941 = curve_fit(gaussian, wave_10941, flux_10941, p10941)
mu10941, sigma10941, A10941, c10941 = popt10941
flux10941 = A10941*sigma10941*np.sqrt(2*np.pi)




# Linea 10052 Paschen 7 -> 3

subset_10052 = (waveClean > 10000) & (waveClean < 10080)
wave_10052 = waveClean[subset_10052]
flux_10052 = fluxClean[subset_10052]

# Fit the data
p10052 = [10047, 3, 1e-17,0]
popt10052, pcov10052 = curve_fit(gaussian, wave_10052, flux_10052, p10052)
mu10052, sigma10052, A10052, c10052 = popt10052
flux10052 = A10052*sigma10052*np.sqrt(2*np.pi)



# Line at lamba_obs = 10260
# Guess initial values for the parameters
#subset_DD = (wave > 10250) & (wave < 10285)
#wave_DD = wave[subset_DD]
#flux_DD = flux[subset_DD]
#pDD = [10275, 5, 1e-17,0]

# Fit the data
#poptDD, pcovDD = curve_fit(gaussian, wave_DD, flux_DD, pDD)

# Line at lamba_obs = 10290
# Guess initial values for the parameters
#subset_OD = (wave > 10280) & (wave < 10300)
#wave_OD = wave[subset_OD]
#flux_OD = flux[subset_OD]
#pOD = [10290, 2, 1e-18,0]

# Fit the data
#poptOD, pcovOD = curve_fit(gaussian, wave_OD, flux_OD, pOD)

# Line at lamba_obs = 11790
# Guess initial values for the parameters
#subset_AD = (wave > 11775) & (wave < 11804)
#wave_AD = wave[subset_AD]
#flux_AD = flux[subset_AD]
#pAD = [11790, 5, 1e-17,0]

# Fit the data
#poptAD, pcovAD = curve_fit(gaussian, wave_AD, flux_AD, pAD)

# Plot the data and the fitted function
#bars = plt.errorbar(waveClean, fluxClean,yerr=error,ls='dotted', label='data')
#bars[-1][0].set_color('red')
#plt.plot(wave_alpha, flux_alpha, 'b.', label='data')
#plt.plot(wave_alpha, gaussian(wave_alpha, *popt1_alpha), label='Pa alpha')
#plt.plot(wave_12821,gaussian(wave_12821, *popt12821), label='Pa beta')
#plt.plot(wave_10941,gaussian(wave_10941, *popt10941), label='Pa gamma')
#plt.plot(wave_10052,gaussian(wave_10052, *popt10052), label='Pa delta')
#plt.plot(wave_DD, gaussian(wave_DD, *poptDD), label='gaussian fit')
#plt.plot(wave_OD, gaussian(wave_OD, *poptOD), label='gaussian fit')
#plt.plot(wave_AD, gaussian(wave_AD, *poptAD), label='gaussian fit')
#plt.ylim(-0.1e-15,1.5e-15)
#plt.xlim(9600,12000)
plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
plt.legend()
plt.show()
