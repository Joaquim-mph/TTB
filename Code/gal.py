'''
We only believe in data
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from uncertainties import ufloat
from astropy.cosmology import FlatLambdaCDM
np.random.seed(3_141_592_653)


# Define the redshift
z = 0.077985 # +- 1.917881e-5 redshif
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
#lumDist= cosmo.luminosity_distance(z).to('cm').value

# some functions

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
# Open the file with the flux
spec = fits.open('/Users/mphstph/Documents/TTB/DatosTTB/J2215+0002/J2215+0002_coadd_tellcorr.fits')
#print(spec[1].header)

########################
# Data of the .fits file
waveShifted = spec[1].data['wave']
wave = spec[1].data['wave']/(z+1)
wave_grid_mid = spec[1].data['wave_grid_mid']
flux = spec[1].data['flux'] * 1e-17
ivar = spec[1].data['ivar'] * 1e17
mask = spec[1].data['mask']
telluric = spec[1].data['telluric']
obj_model = spec[1].data['obj_model']

######################################
# Select a specific subset of the data

# Deleting first noise part
subset = (wave > 8800)
# Cleaning some noisy points
#eliminar = np.where(1/ivar[subset] > 10000) 
# con N = 10000, se elimina el 10% de los puntos
#eliminar=[0]
#error = 1/ivar[subset]
# np.delete(wave[subset], someIndices)
waveClean = wave[subset] 
fluxClean = flux[subset]
ivar = ivar[subset]


G1 = np.array(['Center', 'Sigma', 'Amplitude',
        	   'Continuum'])
G2 = np.array(['Center 1', 'Sigma 1', 'Amplitude 1',
               'Center 2', 'Sigma 2', 'Amplitude 2',
               'Continuum'])
G3 = np.array(['Center 1', 'Sigma 1', 'Amplitude 1',
               'Center 2', 'Sigma 2', 'Amplitude 2',
               'Center 3', 'Sigma 3', 'Amplitude 3',
               'Continuum'])

names = [G1,G2,G3]

class emisionLine:
    '''
    Creation of a class for emision lines. a emision line is a set
    which includes a both wavelengths and fluxes inside some defined
    range between a wavelength min and max.
    ''' 
    def __init__(self, bordeIzq=8800, bordeDer=23385, bestFit=[np.zeros(4),np.zeros(7),np.zeros(10)]):
        self.waveMin = bordeIzq
        self.waveMax = bordeDer
        self.subset = (waveClean > bordeIzq) & (waveClean < bordeDer)
        self.wave = waveClean[self.subset]
        self.flux = fluxClean[self.subset]
        self.std = 1/ivar[self.subset]
        self.bestFit = bestFit
        self.plotPoints = len(self.flux)*10

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

    def fitGauss(self,numberOfGaussians):
        if numberOfGaussians == 1:
            popt, pcov = curve_fit(gaussian, self.wave, self.flux, self.bestFit[0])
        if numberOfGaussians == 2:
            popt, pcov = curve_fit(doubleGaussian, self.wave, self.flux, self.bestFit[1])
        if numberOfGaussians == 3:
            popt, pcov = curve_fit(tripleGaussian, self.wave, self.flux, self.bestFit[2])
        return popt

    def plotGauss(self,numberOfGaussians):
        w = np.linspace(self.waveMin, self.waveMax, self.plotPoints)
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave,self.flux, label='data')
        if numberOfGaussians == 1:
            popt = self.fitGauss(1)
            f = gaussian(w, *popt)
            ax.plot(w, f, label='1 Gaussian model')
        if numberOfGaussians == 2:
            popt = self.fitGauss(2)
            f = doubleGaussian(w, *popt)
            f1 = gaussian(w, popt[0],popt[1],popt[2],0)
            f2 = gaussian(w, popt[3],popt[4],popt[5],0)
            ax.plot(w, f, label='2 Gaussian model')
            ax.plot(w,f1, label='1st Gaussian')
            ax.plot(w,f2, label='2nd Gaussian')
        if numberOfGaussians == 3:
            popt = self.fitGauss(3)
            f = tripleGaussian(w, *popt)
            f1 = gaussian(w, popt[0],popt[1],popt[2],0)
            f2 = gaussian(w, popt[3],popt[4],popt[5],0)
            f3 = gaussian(w, popt[6],popt[7],popt[8],0)
            ax.plot(w, f, label='3 Gaussian model')
            ax.plot(w,f1, label='1st Gaussian')
            ax.plot(w,f2, label='2nd Gaussian')
            ax.plot(w,f3, label='3rd Gaussian')
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()


    def fitGaussN(self,numberOfGaussians, N=100):
        l = len(self.std)
        fluxes = np.zeros((l, N))
        for i in range(l):
            random_set = np.random.uniform(self.flux[i]-self.std[i], self.flux[i]+self.std[i], N)
            fluxes[i][:] = random_set
        if numberOfGaussians == 1:
            datos = np.zeros((N,4))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(gaussian, self.wave, flujo, self.bestFit[0])
                datos[i][:] = popt
        elif numberOfGaussians == 2:
            datos = np.zeros((N,7))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(doubleGaussian, self.wave, flujo, self.bestFit[1])
                datos[i][:] = popt
        elif numberOfGaussians == 3:
            datos = np.zeros((N,10))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(tripleGaussian, self.wave, flujo, self.bestFit[2])
                datos[i][:] = popt
        return datos

    def plotGaussN(self, numberOfGaussians, N=100):
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave,self.flux, label='data')
        wave = np.linspace(self.waveMin,self.waveMax,self.plotPoints)
        l = len(self.std)
        fluxes = np.zeros((l, N))
        for i in range(l):
            random_set = np.random.uniform(self.flux[i]-self.std[i], self.flux[i]+self.std[i], N)
            fluxes[i][:] = random_set
        if numberOfGaussians == 1:
            datos = np.zeros((N,4))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(gaussian, self.wave, flujo, self.bestFit[0])
                datos[i][:] = popt
                ax.plot(wave, gaussian(wave, *popt))
        elif numberOfGaussians == 2:
            datos = np.zeros((N,7))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(doubleGaussian, self.wave, flujo, self.bestFit[1])
                datos[i][:] = popt
                ax.plot(wave, gaussian(wave, popt[0], popt[1], popt[2], 0))
                ax.plot(wave, gaussian(wave, popt[3], popt[4], popt[5], 0))
                ax.plot(wave, doubleGaussian(wave, *popt))
        elif numberOfGaussians == 3:
            datos = np.zeros((N,10))
            for i in range(N):
                flujo = fluxes[:,i]
                popt, pcov = curve_fit(tripleGaussian, self.wave, flujo, self.bestFit[2])
                datos[i][:] = popt
                ax.plot(wave, gaussian(wave, popt[0], popt[1], popt[2], 0))
                ax.plot(wave, gaussian(wave, popt[3], popt[4], popt[5], 0))
                ax.plot(wave, gaussian(wave, popt[6], popt[7], popt[8], 0))
                ax.plot(wave, tripleGaussian(wave, *popt))
        ax.set_xlim(self.waveMin, self.waveMax)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$ergs / s \cdot \AA \cdot cm^2}$')
        ax.legend()
        return datos

    def printGaussN(self, numberOfGaussians, N=100):
        datos = self.plotGaussN(numberOfGaussians, N)
        for i in range(len(self.bestFit[numberOfGaussians-1])):
            print(names[numberOfGaussians-1][i],ufloat(np.mean(datos[:,i]),np.std(datos[:,i])))

    def bestGauss(self, numberOfGaussians, N = 100):
        datos = self.fitGaussN(numberOfGaussians, N)
        parameters = []
        for i in range(len(self.bestFit[numberOfGaussians-1])):
            parameters.append(ufloat(np.mean(datos[:,i]),np.std(datos[:,i])))
        return parameters

    def obsFluxGauss(self,numberOfGaussians, N=100):
        parameters = self.bestGauss(numberOfGaussians, N)
        flux=0
        for i in range(len(parameters)):
            if i%3 == 1:
                flux += np.sqrt(2*np.pi)*parameters[i]*parameters[i+1]
        return flux

    def inFluxGauss(self,numberOfGaussians,N=100):
        obsFlux = self.obsFluxGauss(numberOfGaussians, N)
        lumDist= cosmo.luminosity_distance(z).to('cm').value
        return 4*np.pi*lumDist**2*obsFlux




fullSpec = emisionLine()

Alpha1Gauss = np.array([1.87508271e+04, 2.62915039, 9.50990078e-16,
                        1.05962901e-17])
Alpha2Gauss = np.array([1.87500256e+04, 2.99371843, 6.33286577e-16,
                        1.87518558e+04, 1.44845869, 4.56140366e-16, 
                        9.16371519e-18])
Alpha3Gauss = np.array([1.87519660e+04, 1.53029701e+00, 5.78783331e-16,
                        1.87494558e+04, 2.43436494e+00, 4.84955554e-16,
                        1.87506246e+04, 5.06434611e+00, 1.10599006e-16,
                        7.36607411e-18])

alpha = [Alpha1Gauss,Alpha2Gauss,Alpha3Gauss]

paAlpha = emisionLine(18700,18800,alpha)
