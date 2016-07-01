"""
window.py -- Defines functions to window an array of data samples
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = np.size(dataSampleArray)
    nVec = np.linspace(0,N-1,N)
    nVec = np.add(nVec,0.5)
    arg = np.multiply(np.pi/N,nVec)
    window = np.sin(arg)
    dataOut = np.multiply(dataSampleArray,window)    
    return dataOut
    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = np.size(dataSampleArray)
    nVec = np.linspace(0,N-1,N)
    nVec = np.add(nVec,0.5)
    arg = np.multiply((2.0*np.pi)/N,nVec)
    cosVec = np.cos(arg)
    window = np.add(0.5,np.multiply(-0.5,cosVec))
    dataOut = np.multiply(dataSampleArray,window)
    
    return dataOut
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following pp. 108-109 and pp. 117-118 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = np.size(dataSampleArray)
    M = N/2.0 #for 50 percent overlap
    
    besselnVec = np.linspace(0,M,M+1)
    denom = np.i0(np.pi*alpha)
    besselnVec = np.subtract(besselnVec,M/2.0)
    besselnVec = np.square(np.divide(besselnVec,M/2.0))
    rad = np.sqrt(np.subtract(1.0,besselnVec))
    arg = np.multiply(np.pi*alpha,rad)
    num = np.i0(arg)
    
    window = np.divide(num,denom)
    
    #now perform window normalization on 'window' for 50% overlap
    win_sq = np.square(window)
    
    #nVecWindow = np.linspace(0,N-1,N)
    win_sq_top = win_sq[0:np.size(win_sq)-1]
    win_sq_bot = win_sq[1:np.size(win_sq)]
    
    #for the top portion of the guide
    #from n=0,...,N/2 - 1s
    nVecTop = np.linspace(0,(N/2.0) - 1, (N/2.0))
    topSize = np.size(nVecTop)
    denom = np.sum(win_sq)
    onesy = np.ones((topSize,topSize))
    lowTri = np.tril(onesy) #will be used to compute the summing of top
    nVecTopOut = np.dot(lowTri,win_sq_top)
    
    nVecTopOut = np.sqrt(np.divide(nVecTopOut,denom))
    
    #now do the bottom of the piecewise
    #note the triangularity of the matrix is reversed, now we want to use upper triangular
    nVecBot = np.linspace((N/2.0),N-1,(N/2.0))
    botSize = np.size(nVecBot)
    onesy = np.ones((botSize,botSize))
    upTri = np.triu(onesy)
    nVecBotOut = np.dot(upTri,win_sq_bot)
    nVecBotOut = np.sqrt(np.divide(nVecBotOut,denom))
    
    #concatentate top and bottom
    norm_windowOut = np.concatenate((nVecTopOut,nVecBotOut))
    dataOut = np.multiply(dataSampleArray,norm_windowOut)
    
    return dataOut
    ### YOUR CODE ENDS HERE ###
"""----------------------------------------EDIT--------------------------------------------------"""
def TransitionWindow(dataSampleArray,a,b):
    """
    Reutrns a copy of the dataSampleArray transition windowed by 
    the window WindowDef. The AC-2A trasition window utilized is 
    defined following pp. 135-136 of Bosi & Goldberg, 
    "Introduction to Digital Audio..." book
    """

    # Window the first 'a' samples with a window of length 2*a
    aExtension = np.zeros(a)
    leftWindowed = KBDWindow(np.append(dataSampleArray[:a],aExtension))
    # Window the following 'b' samples with a window of length 2*b
    bExtension = np.zeros(b)
    rightWindowed = KBDWindow(np.append(bExtension,dataSampleArray[a:]))
    # Combine the windowed samples into one signal
    windowedData = np.append(leftWindowed[:a],rightWindowed[b:])

    return windowedData
"""------------------------------------END-EDIT--------------------------------------------------"""

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    pass # THIS DOES NOTHING

    ### YOUR TESTING CODE ENDS HERE ###

