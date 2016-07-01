"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import math

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    if aNum >= 0.0:
        sign = 0
    else:
        sign = 1
        
    if abs(aNum) >= 1:
        code = pow(2, nBits - 1) - 1
        
    else:
        numToRound = ((pow(2,nBits)-1)*abs(aNum) + 1)/2
        code = np.int64(numToRound)
        
    signShift = sign << (nBits-1)
    aQuantizedNum = signShift+code
    
    
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    ### YOUR CODE STARTS HERE ###
    
    if aQuantizedNum >= pow(2.0,nBits-1):
        sign = -1.0
        code = aQuantizedNum - pow(2.0,nBits-1)
    else:
        sign = 1.0;
        code = aQuantizedNum
        
    aNum = (sign*2.0*code)/(pow(2,nBits) - 1.0)
    
    ### YOUR CODE ENDS HERE ###
    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    
    ### YOUR CODE STARTS HERE ###
    nBits = int(nBits)
    signVec = np.multiply(np.less(aNumVec,0.0), 1.0)
    np.place(signVec, signVec==True, 1<<(nBits-1))
    
    magVec = np.absolute(aNumVec)
    opVec = np.less(magVec,1.0)
    realZeros = np.equal(magVec,0.0) #keeps track of index of actual zeros
    opVecN = np.multiply(opVec, pow(2.0, nBits) - 1)
    opVecN = np.multiply(opVecN, magVec)
    opVecN = np.add(opVecN,1.0)
    opVecN = np.divide(opVecN,2.0)
    opVecN = np.multiply(opVecN,opVec) #brings all clipping numbers back to zero
    np.place(opVecN, opVecN == 0.0, pow(2.0, nBits-1) - 1.0)
    opVecN = np.int64(opVecN)
    
    np.place(opVecN, realZeros, 0.0)
    aQuantizedNumVec = np.add(signVec, opVecN)
    
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """
    ### YOUR CODE STARTS HERE ###
    signVecZero = np.multiply(np.greater_equal(aQuantizedNumVec, pow(2.0,nBits-1)), 1.0)
    signVec = np.multiply(signVecZero, -1.0)
    np.place(signVec,signVec==0.0,1.0)
    
    qCopy = np.multiply(signVecZero,aQuantizedNumVec) #singles out the negative numbers
    qCopy = np.subtract(qCopy, pow(2.0,nBits-1))
    qCopy = np.multiply(signVecZero,qCopy)
    
    signVecOne = np.multiply(np.subtract(signVecZero,1.0),-1.0)
    qCopy2 = np.multiply(signVecOne,aQuantizedNumVec) #singles out the positive numbers
        
    magVec = np.add(qCopy,qCopy2);
    aNumVec = np.divide(np.multiply(np.multiply(signVec,magVec),2.0),pow(2,nBits) - 1.0)
    
    ### YOUR CODE ENDS HERE ###

    return aNumVec

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    nBits = int(pow(2,nScaleBits) - 1 + nMantBits)
           
    quant = QuantizeUniform(aNum,  nBits)
    
    if quant >= pow(2, nBits-1):
        magCode = quant - pow(2, nBits-1)
    else:
        magCode = quant
        
    
    #keeps log from failing at 0
    if magCode == 0:
        index_fromRight = 0
    else:
        index_fromRight = int(math.log(magCode,2)) #index starts at 0
        
    leadingZeros = (nBits-2)-(index_fromRight)
    
    if leadingZeros < (pow(2,nScaleBits) - 1):
        scale = leadingZeros
    else:
        scale = pow(2,nScaleBits) - 1    


    ### YOUR CODE ENDS HERE ###

    return scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """


    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    quant = QuantizeUniform(aNum,  nBits)
    
    #extract the sign of the code, which will now be the most significant bit of mantissa
    if quant >= pow(2, nBits-1):
        magCode = quant - pow(2, nBits-1)
        sign = pow(2,nMantBits-1)
    else:
        magCode = quant
        sign = 0
    
    if scale == (pow(2,nScaleBits) - 1):
        mantissa = sign
        mantissa = mantissa + magCode
    else:
        mantissa = sign
        magMant = magCode >> ((pow(2,nScaleBits) - 1 - 1) - scale)
        magMant = magMant - pow(2,nMantBits-1) #subtract bits for leading one
        mantissa = sign + magMant
       
    
    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    if mantissa >= pow(2, nMantBits-1):
        sign = pow(2, nBits-1)
        mantMag = mantissa - pow(2, nMantBits-1)
    else:
        sign = 0
        mantMag = mantissa
            
    if scale == (pow(2,nScaleBits) - 1):
        quant = sign + mantMag
    else:
        #start by adding the one back
        mantMag = mantMag + pow(2, nMantBits-1)
        
        #shift over left into space according to the spot of the zeros
        shiftNum = (((pow(2,nScaleBits) - 1)) - 1 - scale)
        mantMag = mantMag << shiftNum
        
        #add the one back on the end right after the last bit of the code
        #have check is shiftNum-1 will be negative
        if shiftNum <= 0:
            mantMag = mantMag
        else:
            mantMag = mantMag + pow(2,shiftNum-1)
            
        #then finally add on the sign bit
        quant = sign + mantMag
    
    aNum = DequantizeUniform(quant, nBits)
    
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    quant = QuantizeUniform(aNum,  nBits)
    
    #extract the sign of the code, which will now be the most significant bit of mantissa
    if quant >= pow(2, nBits-1):
        magCode = quant - pow(2, nBits-1)
        sign = pow(2,nMantBits-1)
    else:
        magCode = quant
        sign = 0
    
    if scale == (pow(2,nScaleBits) - 1):
        mantissa = sign
        mantissa = mantissa + magCode
    else:
        mantissa = sign
        magMant = magCode >> ((pow(2,nScaleBits) - 1) - scale) #shifting over to right one than prev
        mantissa = sign + magMant
        
    
    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    if mantissa >= pow(2, nMantBits-1):
        sign = pow(2, nBits-1)
        mantMag = mantissa - pow(2, nMantBits-1)
    else:
        sign = 0
        mantMag = mantissa
            
    if scale == (pow(2,nScaleBits) - 1):
        quant = sign + mantMag
    else:
        #shift over left into space according to the spot of the zeros
        #shift is different than the previous
        #try to shift over left one more (aka get rid of a minus one in shiftNum)
        #we kept the leading one in block so we don't add it again
        
        shiftNum = (((pow(2,nScaleBits) - 1)) - scale)
        mantMag = mantMag << shiftNum
        
        #add the one back on the end right after the last bit of the code
        #have check is shiftNum-1 will be negative
        #NOTE or could lead to some issues not quite sure
        if shiftNum <= 0 or mantMag == 0:
            mantMag = mantMag
        else:
            mantMag = mantMag + pow(2,shiftNum-1)
        
        #then finally add on the sign bit
        quant = sign + mantMag
        
    aNum = DequantizeUniform(quant, nBits)
        
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    quantVec = vQuantizeUniform(aNumVec, nBits)
    
    #extract the sign and then perform the necessary subtraction on the negative values
    signVecNeg = np.multiply(np.greater_equal(quantVec,pow(2, nBits-1)),1.0) #1 means negative
    signVecPos = np.multiply(np.subtract(signVecNeg, 1),-1) #1 means positive
    magVec1 = np.subtract(np.multiply(signVecNeg,quantVec),pow(2, nBits-1)) #computes negatives
    np.place(magVec1, magVec1 < 0, 0)
    magVec2 = np.multiply(signVecPos,quantVec)
    magVec = np.add(magVec1,magVec2)
    
    signVecNeg = np.multiply(signVecNeg,pow(2,nMantBits-1)) #will use this vec to add sign at end
    
    if scale == (pow(2,nScaleBits) - 1):
        mantissaVec = np.add(signVecNeg,magVec)
    else:
        shiftNum = ((pow(2,nScaleBits) - 1) - scale)
        shiftyVec = np.right_shift(np.uint64(magVec), shiftNum)
        mantissaVec = np.add(signVecNeg,shiftyVec)
    
    
    ### YOUR CODE ENDS HERE ###

    return mantissaVec

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBits = pow(2,nScaleBits) - 1 + nMantBits
    signVecNeg = np.multiply(np.greater_equal(mantissaVec,pow(2, nMantBits-1)), 1.0)
    signVecPos = np.multiply(np.subtract(signVecNeg, 1),-1.0)
    
    
    mantMagVec1 = np.subtract(np.multiply(mantissaVec,signVecNeg),np.multiply(signVecNeg,pow(2, nMantBits-1)))
    mantMagVec2 = np.multiply(mantissaVec,signVecPos)
    
    mantMagVec = np.add(mantMagVec1,mantMagVec2)
    signVec = np.multiply(signVecNeg,pow(2, nBits-1)) #becomes the final sign to add
    if scale == (pow(2,nScaleBits) - 1):
        quantVec = np.add(mantMagVec,signVec)
    else:
        shiftNum = (((pow(2,nScaleBits) - 1)) - scale)
        shiftyVec = np.left_shift(np.uint64(mantMagVec), shiftNum)
        if shiftNum > 0:
            trueVec = np.multiply(np.greater(mantMagVec,0),1.0)
            trueVec = np.multiply(trueVec,pow(2,shiftNum-1))
            shiftyVec = np.add(shiftyVec, trueVec)
            
        quantVec = np.add(shiftyVec,signVec)
        
    
    aNumVec = vDequantizeUniform(quantVec,nBits)
    
    ### YOUR CODE ENDS HERE ###

    return aNumVec

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    pass

    ### YOUR TESTING CODE ENDS HERE ###

