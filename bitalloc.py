import numpy as np

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    total = np.sum(nLines)
    mantBitsPerBand = np.multiply(np.floor(bitBudget/total),np.ones(nBands))
    
    # check for maxed out assignment of bits, making sure none is greater than maxManBits
    maxArray = np.greater(mantBitsPerBand,maxMantBits) * 1.0
    maxedOut = np.sum(maxArray) #total number of maxed out bands
    np.place(mantBitsPerBand, mantBitsPerBand>maxMantBits, maxMantBits)
    
    #how many bits we've used so far
    used = np.multiply(nLines,mantBitsPerBand)
    totUsed = np.sum(used)
    
    left = bitBudget - totUsed
    indexer = 0
    maxedOut = 0
    
    while True:
        numBitsBand = nLines[indexer]
        if numBitsBand > left or maxedOut == nBands:
            break
        else:
            if (mantBitsPerBand[indexer] + 1 <= maxMantBits):
                mantBitsPerBand[indexer] += 1
                left = left - numBitsBand
                indexer += 1
            else:
                maxedOut += 1
                indexer += 1
                indexer = np.mod(indexer, nBands)
    
    
    return mantBitsPerBand

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    # find the biggest spl in certain iteration, allocate a bit to that band
    # subtract the total number of bits assigned to that band
    
    total = np.sum(nLines) #total number of frequency lines
    noiseSpl = peakSPL #use this for iteration
    bitsLeft = bitBudget
    maxedOutTotal = 0
    mantBits = np.zeros(nBands)
        
    while bitsLeft > 0:
        indexMax = np.argmax(noiseSpl)
        currentMax = np.amax(noiseSpl)
        if mantBits[indexMax] < maxMantBits and nLines[indexMax] <= bitsLeft:
            mantBits[indexMax] += 1
            bitsLeft -= nLines[indexMax]
            noiseSpl[indexMax] -= 6.0 #subtracting off 6db
        else:
            noiseSpl[indexMax] = -99999999999999999.0
            maxedOutTotal += 1
            if maxedOutTotal == nBands: break
            
    
    return mantBits

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    total = np.sum(nLines) #total number of frequency lines
    noiseSmr = SMR #use this for iteration
    bitsLeft = bitBudget
    maxedOutTotal = 0
    mantBits = np.zeros(nBands)
    
    while bitsLeft > 0:
        indexMax = np.argmax(noiseSmr)
        currentMax = np.amax(noiseSmr)
        if mantBits[indexMax] < maxMantBits and nLines[indexMax] <= bitsLeft:
            mantBits[indexMax] += 1
            bitsLeft -= nLines[indexMax]
            noiseSmr[indexMax] -= 6.0 #subtracting off 6db
        else:
            noiseSmr[indexMax] = -99999999999999999.0
            maxedOutTotal += 1
            if maxedOutTotal == nBands: break
    
    
    return mantBits

# Question 1.c)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Return:
            bits[nBands] is number of bits allocated to each scale factor band

        Logic:
           Maximizing SMR over blook gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).
    """
    total = np.sum(nLines) #total number of frequency lines
    noiseSmr = SMR #use this for iteration
    bitsLeft = bitBudget
    maxedOutTotal = 0
    mantBits = np.zeros(nBands)
    
    while bitsLeft > 0:
        indexMax = np.argmax(noiseSmr)
        currentMax = np.amax(noiseSmr)
        if mantBits[indexMax] < maxMantBits and nLines[indexMax] <= bitsLeft:
            if mantBits[indexMax] == 0:
                mantBits[indexMax] += 2
                bitsLeft -= 2*nLines[indexMax]
                noiseSmr[indexMax] -= 12.0
            else:
                mantBits[indexMax] += 1
                bitsLeft -= nLines[indexMax]
                noiseSmr[indexMax] -= 6.0
        else:
            noiseSmr[indexMax] = -99999999999999999.0
            maxedOutTotal += 1
            if maxedOutTotal == nBands: break
    
    
    return (mantBits, int(bitsLeft))

#-----------------------------------------------------------------------------
#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE
