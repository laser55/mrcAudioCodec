"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import math
# import mdct_ as soln
import time

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    n0 = (b + 1.0)/2.0

    if(not isInverse):
        X = np.zeros(N/2)

        for k in range(0, N/2):
            n = range(0,N)
            cosine = np.cos( (2.0*np.pi/N) * (np.add(n,n0)) * (k + 1.0/2.0) )
            X[k] = (2.0/N)*np.dot(data,cosine)

            # for n in range(0,N):
            #     # print np.cos( (2.0*np.pi/N) * (n+n0) * (k + 1.0/2) )
            #     X[k] += (2.0/N)*data[n]*np.cos( (2.0*np.pi/N) * (n+n0) * (k + 1.0/2.0) )
            # if(k==1):
            #     print X[k]        

        return X
    else:
        x = np.zeros(N)

        for n in range(0, N):
            for k in range(0,N/2):
                x[n] += 2.0*data[k]*np.cos( (2.0*np.pi/N) * (n + n0) * (k + 1.0/2.0) )

        return x
    # return np.zeros_like( data ) # CHANGE THIS
    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    if(isInverse == False):
        N = a+b
        n = range(0,N)
        n0 = (b+1.0)/2.0
        pre_twiddle_factor = np.exp(np.multiply(n,-1j*np.pi/N))
        pre_twiddled_data = np.multiply(pre_twiddle_factor,data)
        transformed_data = np.fft.fft(pre_twiddled_data, N)
        k = range(0,N/2)
        k = np.add(k,1.0/2.0)
        # get post twiddle factor
        post_twiddle_factor = np.exp(np.multiply(k, -1j*2.0*np.pi*n0/N )) 
        # then get the first 0 to N/2-1 and take real part, multiply by factor
        post_twiddled_data = (transformed_data[0:N/2])
        post_twiddled_data = (2.0/N)*np.real(np.multiply(post_twiddle_factor,post_twiddled_data))
        
    else:
        X = np.zeros(a+b)
        # print X
        N = a+b
        X[0:N/2] = data
        X[N/2:] = -1*data[::-1]
        n0 = (b + 1)/2.0
        # print X 
        k = range(0,N)
        n = range(0,N)
        pre_twiddle_factor = np.exp(np.multiply( k ,1j*2*np.pi*n0/N))
        pre_twiddled_data = np.multiply(pre_twiddle_factor,X)
        inverse_transformed_data = np.fft.ifft(pre_twiddled_data,N)
        post_twiddle_factor = np.multiply(np.add(n, n0), (1j*2*np.pi/(2.0*N)))
        post_twiddle_factor = np.exp(post_twiddle_factor)
        post_twiddled_data = N*np.real(np.multiply(inverse_transformed_data,post_twiddle_factor))
    
    return post_twiddled_data
    ### YOUR CODE ENDS HERE ###

def IMDCT(data,a,b):

    ### YOUR CODE STARTS HERE ###
    X = np.zeros(a+b)
    # print X
    N = a+b
    X[0:N/2] = data
    X[N/2:] = -1*data[::-1]
    n0 = (b + 1)/2.0

    # print X 
    k = range(0,N)
    n = range(0,N)
    pre_twiddle_factor = np.exp(np.multiply( k ,1j*2*np.pi*n0/N))
    pre_twiddled_data = np.multiply(pre_twiddle_factor,X)
    inverse_transformed_data = np.fft.ifft(pre_twiddled_data,N)
    # print 'inverse transform:'
    # print inverse_transformed_data
    post_twiddle_factor = np.multiply(np.add(n, n0), (1j*2*np.pi/(2.0*N)))
    post_twiddle_factor = np.exp(post_twiddle_factor)
    post_twiddled_data = N*np.real(np.multiply(inverse_transformed_data,post_twiddle_factor))

    return post_twiddled_data
    # return np.zeros_like( data ) # CHANGE THIS
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    print '\n-------------TESTING MDCTSLOW ON SIZE 8 ARRAY-------------'
    x = np.array([3,3,3,3,2,0,-2,-4,-1,0,1,2]);
    print 'original: ' + str(x)

    N_half_block = 4
    a = 4
    b = 4
    N = a + b
    zeros = np.zeros(N_half_block)
    prev_imdct = np.zeros(N)
    num_blocks = int(np.ceil(len(x)/N_half_block)) + 1
    output = np.zeros(0)
    for nblock in range(0, num_blocks):
        current_block = x[(nblock)*N_half_block:(nblock)*N_half_block+N_half_block]
        if(nblock == 0):
            block_to_transform = np.concatenate([zeros,current_block])
        elif(nblock == num_blocks-1):
            previous_block = x[(nblock-1)*N_half_block:(nblock-1)*N_half_block+N_half_block]
            block_to_transform = np.concatenate([previous_block,zeros])
        else:
            previous_block = x[(nblock-1)*N_half_block:(nblock-1)*N_half_block+N_half_block]
            block_to_transform = np.concatenate([previous_block,current_block])
        
        # get mdct
        mdct = MDCTslow(block_to_transform,a,b)
        mdct_soln = soln.MDCTslow(block_to_transform,a,b)

        # make sure matches the solution
        np.testing.assert_array_almost_equal(mdct,mdct_soln)

        # then get the inverse, scale by 1/2
        imdct = 0.5*MDCTslow(mdct,a,b,1)
        imdct_soln = 0.5*soln.MDCTslow(mdct_soln,a,b,1)

        np.testing.assert_array_almost_equal(imdct,imdct_soln)

        # find overlap, left half of inverse, right half of prev invrse
        overlap = imdct[0:b] + prev_imdct[b:]
        # then concatenate the overlap with the output
        output = np.concatenate([output,overlap])
        print '    ' + str(np.int64(np.rint(output)))
        # set the previous to be the current for the next iteration
        prev_imdct = imdct

        # 0 0 0 0 3 3 3 3 
        #         0 0 0 0 -1 -1 -1 -1
        #                  3  1 -1 -3 0.5  0.5 0.5 0.5
        #                            -1.5 -0.5 0.5 1.5 0 0 0 0
        # ------------------------------------------------------
        # 0 0 0 0 3 3 3 3  2  0 -2 -4  -1   0   1   2  0 0 0 0

    # truncate beginning 4 zeros
    print 'output:' + str(np.int64(np.rint(output[b:])))

    print '\n----------------NOW CONDUCTING TIMING TEST----------------'
    x = range(0,1024)
    N_test = 1024
    
    # print 'fast'
    start = time.time()
    transform = MDCT(x,N_test/2,N_test/2)
    end = time.time()
    # print transform
    print 'MDCT took ' + str(end-start) + ' s'
    # print 'slow'
    start = time.time()
    slowtransform = MDCTslow(x,N_test/2,N_test/2)
    end = time.time()
    print 'MDCTslow took ' + str(end-start) + ' s'
    np.testing.assert_array_almost_equal(transform,slowtransform)

    print '----------------'
    start = time.time()
    fastIMDCT = IMDCT(transform,N_test/2,N_test/2)
    end = time.time()
    print 'IMDCT took ' + str(end-start) + ' s'
    start = time.time()
    slowIMDCT = MDCTslow(transform,N_test/2,N_test/2,1)
    end = time.time()
    print 'IMDCTslow took ' + str(end-start) + ' s'
    np.testing.assert_array_almost_equal(slowIMDCT,fastIMDCT)

    print ''
    ### YOUR TESTING CODE ENDS HERE ###

