'''
I only believe in data.
'''
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
############### Modelos de Extincion ###################
########################################################

R_v = ufloat(4.05,0.8)
Pa_alpha_long = 1.8751 # micrometer
Pa_gamma_long = 1.0938 # micrometer
Pa_delta_long = 1.0049 # micrometer

def calzetti(long):
    '''
    For lambda in (0.63,2.2) micrometers
    '''
    return 2.659*(-1.857+1.04/long) + R_v

def shivaei(long):
    '''
    For lambda in (0.6,2.85) micrometers
    '''
    x = 1/long
    return -2.672-0.01*x+1.532*x**2-0.412*x**3+2.505

########################################################
######### Funciones modeladoras de cosas################
########################################################

# Define the function to fit (a zeroth order polynomial)
def zeroth(x,m):
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

#############################
# Open the file with the flux
spec = fits.open('/Users/mphstph/Documents/TTB/DatosTTB/J2215+0002/J2215+0002_coadd_tellcorr.fits')
#print(spec[1].header)

########################
# Data of the .fits file
wave = spec[1].data['wave']/(z+1)
wave_grid_mid = spec[1].data['wave_grid_mid']
flux = spec[1].data['flux'] * 1e-17
ivar = spec[1].data['ivar']
mask = spec[1].data['mask']
telluric = spec[1].data['telluric']
obj_model = spec[1].data['obj_model']

######################################
# Select a specific subset of the data

# Deleting first noise part
subset = (wave > 8800)
# Cleaning some noisy points
eliminar = np.where(1/ivar[subset] > 10000) # con N = 10000, se elimina el 10% de los puntos
error = np.delete(1/ivar[subset],eliminar)
waveClean = np.delete(wave[subset],eliminar) # redshift corrected
fluxClean = np.delete(flux[subset],eliminar)



########################################################
############## Modeling of the data ####################
########################################################

class emisionLine:
    '''
    Creation of a class for emision lines. a emision line is a set
    which includes a both wavelengths and fluxes inside some defined
    range between a wavelength min and max.
    ''' 
    def __init__(self, bordeIzq=8800, bordeDer=23385):
        self.waveMin = bordeIzq
        self.waveMax = bordeDer
        self.subset = (waveClean > bordeIzq) & (waveClean < bordeDer)
        self.wave = waveClean[self.subset]
        self.flux = fluxClean[self.subset]

    def plotSpectrum(self):
        w = self.wave
        f = self.flux
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(w, f, label = 'full spectrum')
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()

    def plotLine(self):
        f = self.flux
        w = self.wave
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(w, f, label = 'data')
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()

    def fit1Gauss(self, best1Fit, N=500):
        '''
        best1Fit is an numpy array containing the 3 parameters
        corresponding to 1 gaussian.
        '''
        mu, sigma, A = best1Fit
        p1 = [mu, sigma, A, 0]
        popt1, pcov1 = curve_fit(gaussian, self.wave, self.flux, p1)
        w = np.linspace(self.waveMin, self.waveMax, N)
        f = gaussian(w, *popt1)
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave,self.flux, label='data')
        ax.plot(w, f, label='1 Gaussian model')
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()

    def fit2Gauss(self, best2Fit, N=500):
        '''
        best2Fit is an numpy array containing the 6 parameters
        corresponding to the 2 gaussians.
        '''
        mu1, sigma1, A1, mu2, sigma2, A2 = best2Fit
        p2 = [mu1, sigma1, A1, mu2, sigma2, A2, 0]
        popt2, pcov2 = curve_fit(doubleGaussian, self.wave, self.flux, p2)
        w = np.linspace(self.waveMin, self.waveMax, N)
        f = doubleGaussian(w, *popt2)
        f1 = gaussian(w, popt2[0],popt2[1],popt2[2],0)
        f2 = gaussian(w, popt2[3],popt2[4],popt2[5],0)
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave,self.flux, label='data')
        ax.plot(w, f,label='2 Gaussian model')
        ax.plot(w,f1, label='1st Gaussian')
        ax.plot(w,f2, label='2nd Gaussian')
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()

    def fit3Gauss(self, best3Fit, N=500):
        '''
        best3Fit is an numpy array containing the 9 parameters
        corresponding to the 3 gaussians.
        '''
        mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3 = best3Fit
        p3 = [mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, 0]
        popt3, pcov3 = curve_fit(tripleGaussian, self.wave, self.flux, p3)
        w = np.linspace(self.waveMin, self.waveMax, N)
        f = tripleGaussian(w, *popt3)
        f1 = gaussian(w, popt3[0],popt3[1],popt3[2],popt3[9])
        f2 = gaussian(w, popt3[3],popt3[4],popt3[5],popt3[9])
        f3 = gaussian(w, popt3[6],popt3[7],popt3[8],popt3[9])
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave,self.flux, label='data')
        ax.plot(w, f,label='3 Gaussian model')
        ax.plot(w,f1, label='1st Gaussian')
        ax.plot(w,f2, label='2nd Gaussian')
        ax.plot(w,f3, label='3rd Gaussian')
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()


Alpha1Gauss = np.array([18750.11, 2.83, 9.508e-16])
Alpha2Gauss = np.array([18750, 2.83, 1e-17, 18750, 2.83, 1e-17])
Alpha3Gauss = np.array([18749, 3.237, 1e-16, 18751, 1.550 ,1e-16 , 18751, 2.550 , 1e-17])

paAlpha = emisionLine(18700,18800)

gamma1Gauss = np.array([10938,2,3e-16])
gamma2Gauss = np.array([10938,1,1.5e-16,10938,1,1.5e-16])

paGamma = emisionLine(10910,10960)






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

#plt.clf()
#plt.plot(waveClean,fluxClean, label='data')
#plt.plot(wave_alpha,gaussian(wave_alpha, mu21_alpha, sigma21_alpha, A21_alpha, c2_alpha))
#plt.plot(wave_alpha,gaussian(wave_alpha, mu22_alpha, sigma22_alpha, A22_alpha, c2_alpha))
#plt.plot(wave_alpha, doubleGaussian(wave_alpha, *popt2_alpha), 'r-', label='Pa alpha')

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
#plt.xlabel(r'Wavelength ($\AA$)')
#plt.ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
#plt.legend()
#plt.show()
