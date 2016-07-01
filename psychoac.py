import numpy as np
from window import *
#import matplotlib.pyplot as plt
from mdct import *
from quantize import *


def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    return np.maximum(96 + 10*np.log10(intensity),-30,) # TO REPLACE WITH YOUR CODE

def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    return 10**((spl-96)/10) # TO REPLACE WITH YOUR CODE

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""

    return    (3.64*((f/1000.)**(-0.8))) \
            - (6.5*np.exp((-0.6*(((f/1000.)-3.3)**2)))) \
            + ((10**(-3))*((f/1000.)**4)) # TO REPLACE WITH YOUR CODE

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    return 13*np.arctan(0.76*f/1000.) + 3.5*np.arctan((f/7500.)**2) # TO REPLACE WITH YOUR CODE

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        if isTonal:
            self.drop = 14.5 + 0.5
        else:
            self.drop = 5.5
        self.z = Bark(f)
        self.SPL = SPL
        self.f = f
        pass # TO REPLACE WITH YOUR CODE

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        return  self.IntensityAtBark(Bark(freq))# TO REPLACE WITH YOUR CODE

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        dz = z - self.z
        if abs(dz) <= 0.5:
            intensity = Intensity(self.SPL-self.drop)
        elif dz > 0.5:
            intensity = Intensity(self.SPL-self.drop \
                        + (-27 + 0.37*np.maximum(self.SPL-40,0))*(dz-0.5))
        else:
            intensity = Intensity(self.SPL-self.drop \
                        + -27*abs(dz+0.5))
        return  intensity# TO REPLACE WITH YOUR CODE

    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        dzVec = zVec - self.z
        sign = dzVec > 0.5
        mag = np.abs(dzVec) > 0.5

        iVec = Intensity(self.SPL-self.drop \
                + -27*(np.abs(dzVec)-0.5)*mag \
                + 0.37*np.maximum(self.SPL-40,0)*(np.abs(dzVec)-0.5)*mag*sign)

        return iVec # TO REPLACE WITH YOUR CODE


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100, 200, 300, 400, 510, 630,  770,  920,  1080,\
                1270,1480,1720,2000,2320,2700, 3150, 3700,\
                4400,5300,6400,7700,9500,12000,15500,24000]  # TO REPLACE WITH THE APPROPRIATE VALUES

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    n = np.array(range(nMDCTLines))
    MDCTFreq = (n+0.5)*((float(sampleRate)/nMDCTLines)/2.)
    binnedLines = np.zeros(len(flimit))
    i = 0
    j = 0
    while i < len(flimit)-1:
        while MDCTFreq[j] < flimit[i] and j < len(MDCTFreq):
            binnedLines[i] += 1
            j += 1
        i += 1
    binnedLines[i] = nMDCTLines - sum(binnedLines)

    return binnedLines # TO REPLACE WITH YOUR CODE

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        self.nBands = len(nLines)
        # self.lowerLine = np.dot(np.tril(np.ones((self.nBands,self.nBands)),-1)\
        #                         ,np.transpose(nLines))
        self.lowerLine = np.cumsum(np.append([0],nLines[0:self.nBands-1]),dtype = int)
        self.upperLine = np.cumsum(np.transpose(nLines),dtype = int)-1

        # self.upperLine = np.dot(np.tril(np.ones((self.nBands,self.nBands)),0)\
        #                         ,np.transpose(nLines))-1
        self.nLines = self.upperLine - self.lowerLine + 1
        pass # TO REPLACE WITH YOUR CODE


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    # Calculate the MDCT lines
    n = np.array(range(len(MDCTdata)))
    MDCTFreq = (n+0.5)*((float(sampleRate)/len(MDCTdata))/2.)

    N = len(data)
    # Window the data
    xWin = HanningWindow(data)
    # FFT the windowed data
    X = np.fft.fft(xWin)
    # Convert frequency information to Intensity
    XI = 4.*(np.abs(X)**2.)/((N**2.)*(3./8.))
    # Convert intensity to SPL
    XSPL = SPL(XI)
    # Initialize the mask threshold with the quiet threshold for MDCT lines
    totalMask = Intensity(Thresh(MDCTFreq))
    
    # Find peaks
    XI0 = XI[0]
    XI1 = XI[1]
    for i in range(2,N/2-100):
        XI2 = XI[i]
        if XI1 > XI0 and XI1 > XI2:
            # Create tonal masker objects for each peak
            XMask = SPL(XI0 + XI1 + XI2)
            nMask = (sampleRate/N)*(n[i-2]*XI0 + n[i-1]*XI1 + n[i]*XI2)/(XI0 + XI1 + XI2)
            Tone = Masker(nMask,XMask) 
            # Find the tonal masker values for the MDCT frequency lines
            totalMask += Tone.vIntensityAtBark(Bark(MDCTFreq))
        XI0 = XI1
        XI1 = XI2
    
    
    return SPL(totalMask) # TO REPLACE WITH YOUR CODE


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands, ms=0, preCalcThresh=0.0):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency lines for the data
                            in data which have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  is an overall scale factor for the set of MDCT
                            frequency lines
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Sums their masking curves with the hearing threshold at each MDCT
                frequency location to the calculate absolute threshold at those
                points. Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    SMR = np.zeros(sfBands.nBands)
    N = len(data)

    if ms==1:
        maskThresh = preCalcThresh
    else:
        maskThresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)

    maskThresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)

    MDCTSPL = SPL(2.*(np.abs(MDCTdata)**2.)/(1./2.)) - 6.*MDCTscale
    
    SMRNotMax = MDCTSPL-maskThresh

    for i in range(sfBands.nBands):
        SMR[i] = np.amax(SMRNotMax[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])

    return SMR # TO REPLACE WITH YOUR CODE

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass