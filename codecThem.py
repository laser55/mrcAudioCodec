# coding: utf-8

"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import KBDWindow,TransitionWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)
from ms_stereo import *

# used only by Encode
from psychoac import CalcSMRs,getMaskedThreshold,Bark  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs
import time

# things meredith imported:
import os
from glob import glob
import pickle


def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    """----------------------------------------EDIT--------------------------------------------------"""
    # compute the actual window and half window lengths
    N = codingParams.a + codingParams.b
    halfN = (codingParams.a + codingParams.b)/2.
    """------------------------------------END-EDIT--------------------------------------------------"""
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level


    # IMDCT and window the data for this channel
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send IMDCT and TransitionWindow the actual block lengths
    data = TransitionWindow(IMDCT(mdctLine,codingParams.a,codingParams.b),
        codingParams.a,codingParams.b)  # takes in halfN MDCT coeffs
    """------------------------------------END-EDIT--------------------------------------------------"""

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data
    
def JointDecode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,ms_switch):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""
    
    # might have to know which channel in order to keep the current structure the same

    ### rescale is now different, 4 different rescale levels
    rescaleLevelLeft = 1.*(1<<overallScaleFactor[0])
    rescaleLevelMid = 1.*(1<<overallScaleFactor[2])
    rescaleLevelRight = 1.*(1<<overallScaleFactor[1])
    rescaleLevelSide = 1.*(1<<overallScaleFactor[3])
    ###
    
    """----------------------------------------EDIT--------------------------------------------------"""
    # compute the actual window and half window lengths
    N = codingParams.a + codingParams.b
    halfN = (codingParams.a + codingParams.b)/2.
    """------------------------------------END-EDIT--------------------------------------------------"""

    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine1 = np.zeros(halfN,dtype=np.float64)
    mdctLine2 = np.zeros(halfN,dtype=np.float64)
        
    iMant1 = 0
    iMant2 = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[0][iBand]:
            mdctLine1[iMant1:(iMant1+nLines)]=vDequantize(scaleFactor[0][iBand], mantissa[0][iMant1:(iMant1+nLines)],codingParams.nScaleBits, bitAlloc[0][iBand])
            if ms_switch[iBand] == 1:
                mdctLine1[iMant1:(iMant1+nLines)] /= rescaleLevelMid
            else:
                mdctLine1[iMant1:(iMant1+nLines)] /= rescaleLevelLeft
        
        iMant1 += nLines
        
        if bitAlloc[1][iBand]:
            mdctLine2[iMant2:(iMant2+nLines)]=vDequantize(scaleFactor[1][iBand], mantissa[1][iMant2:(iMant2+nLines)],codingParams.nScaleBits, bitAlloc[1][iBand])
            if ms_switch[iBand] == 1:
                mdctLine2[iMant2:(iMant2+nLines)] /= rescaleLevelSide
            else:
                mdctLine2[iMant2:(iMant2+nLines)] /= rescaleLevelRight
        
        iMant2 += nLines
    

    # it looks like our MDCT lines are off, they are only size 25, which is concerni
    #want to convert M/S subband information to L/R before the IMDCT
    #note that now this will no longer be here, we are reconstucting in the time domain??
    (mdctLeft, mdctRight) = ReconstructLR(mdctLine1, mdctLine2, codingParams.sfBands, ms_switch)
    
    # IMDCT and window the data for this channel
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send IMDCT and TransitionWindow the actual block lengths
    dataLeft = TransitionWindow(IMDCT(mdctLeft,codingParams.a,codingParams.b),
        codingParams.a,codingParams.b)  # takes in halfN MDCT coeffs
    dataRight = TransitionWindow(IMDCT(mdctRight,codingParams.a,codingParams.b),
        codingParams.a,codingParams.b)
    """------------------------------------END-EDIT--------------------------------------------------"""
    
    data = []
    data.append(dataLeft)
    data.append(dataRight)

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data

def calculateHuffmanGain(mantissa, bitAlloc, codingParams):
    file_path = "./training_data/"
    pickles = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*table.pkl'))]
    # print pickles
    # calculate the bits for the mantissa
    sum_mantissa = 0
    for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
        if bitAlloc[iBand]:
            # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
            sum_mantissa += bitAlloc[iBand]*codingParams.sfBands.nLines[iBand] 
    
    current_min = sum_mantissa 

    table_to_use = 15
    # iterate through all the huffman tables
    for i in range(0,len(pickles)):
        dill_pkl = open(pickles[i], 'rb')
        current_pickle = pickle.load(dill_pkl)
        escape_value = current_pickle[1]
        current_table = current_pickle[0]
        
        # loop through the mantissa and see if using this current huff table 
        # leads to saving bits
        sum_huffman = 0
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
            
            if(sum_huffman > sum_mantissa):
                break 
            if bitAlloc[iBand]:
                # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                for j in range(int(codingParams.sfBands.nLines[iBand])):
                    # code = mantissa[iCh][iMant]
                    if(not (current_table.has_key(mantissa[iMant])) and (not mantissa[iMant] == escape_value)):
                        sum_huffman = sum_huffman + bitAlloc[iBand] + current_table.get(escape_value)[1]
                    else:
                        sum_huffman = sum_huffman + current_table.get(mantissa[iMant])[1]
                    iMant += 1 
        # after iterating, is it better than just encoding it regularly?
        if(sum_huffman < current_min):
            # if yes then keep track of this info
            current_min = sum_huffman
            table_to_use = i
        dill_pkl.close()

    # now recode the mantissas
    if(table_to_use == 15):
        mant_codes_to_use = mantissa 
    else:
        dill_pkl = open(pickles[table_to_use], 'rb')
        current_pickle = pickle.load(dill_pkl)
        escape_value = current_pickle[1]
        current_table = current_pickle[0]
        current_band = 0
        mant_codes_to_use = ["" for x in range(len(mantissa))]
        for k in range(0,len(mantissa)):
            if(k >= codingParams.sfBands.upperLine[current_band]):
                current_band += 1 
            if(current_table.has_key(mantissa[k]) and (not(mantissa[k] == escape_value))):                
                mant_codes_to_use[k] = str(current_table.get(mantissa[k])[0])
            else:
                # concatenate the escape code with the original mantissa
                escape_str = (current_table.get(escape_value))[0]
                escape_str_with_mant = escape_str +"/"+ str(mantissa[k])
                mant_codes_to_use[k] = str(escape_str_with_mant)

    bits_saved = sum_mantissa - current_min
    return (table_to_use, mant_codes_to_use, bits_saved)

def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    huffTable = []
    table_to_use = []
    bits_saved = 0 

    bit_reservoir = 0
    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        """----------------------------------------EDIT--------------------------------------------------"""
        # see if huffman coding would help
        (table_to_use, new_m, bits_saved) = calculateHuffmanGain(m,b,codingParams)
        codingParams.bitReservoir += bits_saved
        huffTable.append(table_to_use)
        mantissa.append(new_m)
        """--------------------------------------END-EDIT--------------------------------------------------"""
        overallScaleFactor.append(o)
    # return results bundled over channels

    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,huffTable)

# function primarily used for huffman training tables, 
def EncodeNoHuff(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    huffTable = []
    table_to_use = []
    bits_saved = 0 

    bit_reservoir = 0
    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        

        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        # see if huffman coding would help
        # (table_to_use, new_m, bits_saved) = calculateHuffmanGain(m,b,codingParams,blockNum)
        # codingParams.bitReservoir += bits_saved
        huffTable.append(15)
        mantissa.append(m)

        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,huffTable)

def JointEncode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    dataLeft = data[0]
    dataRight = data[1]
    
    new_mantissa = []
    huffTable = []
    bits_saved = 0

    (scaleFactor, bitAlloc, mantissa, overallScaleFactor, ms_switch) = JointEncodeChannels(dataLeft,dataRight,codingParams)
    for iCh in range(codingParams.nChannels):
        (table_to_use, new_m, bits_saved) = calculateHuffmanGain(mantissa[iCh],bitAlloc[iCh],codingParams)
        codingParams.bitReservoir += bits_saved
        huffTable.append(table_to_use)
        new_mantissa.append(new_m)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,new_mantissa,overallScaleFactor, ms_switch,huffTable)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    """----------------------------------------EDIT--------------------------------------------------"""
    # compute the actual window and half window lengths
    N = codingParams.a + codingParams.b
    halfN = (codingParams.a + codingParams.b)/2.
    """------------------------------------END-EDIT--------------------------------------------------"""

    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    """----------------------------------------EDIT--------------------------------------------------"""
    # account for (binary) block length bits
    bitBudget -= codingParams.blkswBitA
    bitBudget -= codingParams.blkswBitB
    # add bit reservoir
    bitBudget += codingParams.bitReservoir
    """------------------------------------END-EDIT--------------------------------------------------"""

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send MDCT and TransitionWindow the actual block lengths
    mdctTimeSamples = TransitionWindow(data,codingParams.a,codingParams.b)
    mdctLines = MDCT(mdctTimeSamples,codingParams.a,codingParams.b)[:halfN]
    """------------------------------------END-EDIT--------------------------------------------------"""

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    (bitAlloc, remaining_bits) = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
    bitAlloc = bitAlloc.astype(int)
    """----------------------------------------EDIT--------------------------------------------------"""
    codingParams.bitReservoir = int(remaining_bits)
    """--------------------------------------END-EDIT--------------------------------------------------"""

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)



###
def JointEncodeChannels(dataLeft,dataRight,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    #calculate the mid channel and the side channel
    dataMid = (dataLeft + dataRight)/2.0
    dataSide = (dataLeft - dataRight)/2.0
    
    
    """----------------------------------------EDIT--------------------------------------------------"""
    # compute the actual window and half window lengths
    N = codingParams.a + codingParams.b
    halfN = (codingParams.a + codingParams.b)/2.
    """------------------------------------END-EDIT--------------------------------------------------"""

    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    
    
    ### need to also factor in subtracting the scale factor bands for a bit determining m/s code
    bitBudget += bitBudget
    bitBudget -= sfBands.nBands
    bitBudget -= nScaleBits*4

    bitBudget += codingParams.bitReservoir
    """----------------------------------------EDIT--------------------------------------------------"""
    # account for (binary) block length bits
    bitBudget -= codingParams.blkswBitA
    bitBudget -= codingParams.blkswBitB
    """------------------------------------END-EDIT--------------------------------------------------"""
    


    ### Compute all of the MDCTs For the Left, Right, Mid, Side
    # window data for side chain FFT and also window and compute MDCT
    timeSamples_Left = dataLeft
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send MDCT and TransitionWindow the actual block lengths
    mdctTimeSamples_Left = TransitionWindow(dataLeft,codingParams.a,codingParams.b)
    mdctLinesLeft = MDCT(mdctTimeSamples_Left,codingParams.a,codingParams.b)[:halfN]
    """------------------------------------END-EDIT--------------------------------------------------"""

    timeSamples_Right = dataRight
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send MDCT and TransitionWindow the actual block lengths
    mdctTimeSamples_Right = TransitionWindow(dataRight,codingParams.a,codingParams.b)
    mdctLinesRight = MDCT(mdctTimeSamples_Right,codingParams.a,codingParams.b)[:halfN]
    """------------------------------------END-EDIT--------------------------------------------------"""
    
    timeSamples_Mid = dataMid
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send MDCT and TransitionWindow the actual block lengths
    mdctTimeSamples_Mid = TransitionWindow(dataMid,codingParams.a,codingParams.b)
    mdctLinesMid = MDCT(mdctTimeSamples_Mid,codingParams.a,codingParams.b)[:halfN]
    """------------------------------------END-EDIT--------------------------------------------------"""
    
    timeSamples_Side = dataSide
    """----------------------------------------EDIT--------------------------------------------------"""
    # use TransitionWindow instead of SineWindow
    # send MDCT and TransitionWindow the actual block lengths
    mdctTimeSamples_Side = TransitionWindow(dataSide,codingParams.a,codingParams.b)
    mdctLinesSide = MDCT(mdctTimeSamples_Side,codingParams.a,codingParams.b)[:halfN]
    """------------------------------------END-EDIT--------------------------------------------------"""
    ###
    
    # determine which bands to enable ms coding
    ms_switch = MSSwitchSFBands(mdctLinesLeft, mdctLinesRight, sfBands)
    
    ### compute scale factors for lr ms: this is the normalization stuff
    # compute overall scale factor for this block and boost mdctLines using it
    maxLine_Left = np.max( np.abs(mdctLinesLeft) )
    overallScale_Left = ScaleFactor(maxLine_Left,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLinesLeft *= (1<<overallScale_Left)
    
    maxLine_Right = np.max( np.abs(mdctLinesRight) )
    overallScale_Right = ScaleFactor(maxLine_Right,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLinesRight *= (1<<overallScale_Right)
    
    maxLine_Mid = np.max( np.abs(mdctLinesMid) )
    overallScale_Mid = ScaleFactor(maxLine_Mid,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLinesMid *= (1<<overallScale_Mid)
    
    maxLine_Side = np.max( np.abs(mdctLinesSide) )
    overallScale_Side = ScaleFactor(maxLine_Side,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLinesSide *= (1<<overallScale_Side)
    
    scaleFactorsPack = []
    scaleFactorsPack.append(overallScale_Left)
    scaleFactorsPack.append(overallScale_Right)
    scaleFactorsPack.append(overallScale_Mid)
    scaleFactorsPack.append(overallScale_Side)
    
    ###
    
    ### Calculate the new thresholds for mid and side with the masking factors
    midThresh = getMaskedThreshold(timeSamples_Mid, mdctLinesMid, overallScale_Mid, codingParams.sampleRate, sfBands)
    sideThresh = getMaskedThreshold(timeSamples_Side, mdctLinesSide, overallScale_Side, codingParams.sampleRate, sfBands)
    
    lin = np.linspace(0,N-1,N)
    freq = np.add(lin,0.5)
    freq = np.multiply(freq,(codingParams.sampleRate)/N)
    freqtest = freq[0:N/2]
    zVec = Bark(freqtest)
    
    new_threshs = StereoMaskingFactor(midThresh,sideThresh,sfBands,zVec)
    new_midThresh = new_threshs[0]
    new_sideThresh = new_threshs[1]
    
    
    
    ### Calculate all of the SMRs for all 
    SMRs_Left = CalcSMRs(timeSamples_Left, mdctLinesLeft, overallScale_Left, codingParams.sampleRate, sfBands)
    SMRs_Right = CalcSMRs(timeSamples_Right, mdctLinesRight, overallScale_Right, codingParams.sampleRate, sfBands)
    SMRs_Mid = CalcSMRs(timeSamples_Mid, mdctLinesMid, overallScale_Mid, codingParams.sampleRate, sfBands, 1, new_midThresh)
    SMRs_Side = CalcSMRs(timeSamples_Side, mdctLinesSide, overallScale_Side, codingParams.sampleRate, sfBands, 1, new_sideThresh)    
    (SMR1, SMR2) = OverallSMRs(SMRs_Left, SMRs_Right, SMRs_Mid, SMRs_Side, sfBands, ms_switch)
    
    ###
    
    # perform bit allocation using SMR results
    #need to concatenate together, then split appart once the allocations have been made back into 2 channels
    nLines1 = sfBands.nLines
    nLinesPass = np.append(nLines1, nLines1)
    
    SMRsConcat = SMR1
    SMRsPass = np.append(SMRsConcat, SMR2)
    
    
    (bitAlloc, remaining_bits) = BitAlloc(bitBudget, maxMantBits, 2*sfBands.nBands, nLinesPass, SMRsPass)
    bitAlloc = bitAlloc.astype(int)

    bitAlloc1 = bitAlloc[0:sfBands.nBands]
    bitAlloc2 = bitAlloc[sfBands.nBands:]
    codingParams.bitReservoir = int(remaining_bits)
    
    
    

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor1 = np.empty(sfBands.nBands,dtype=np.int32)
    scaleFactor2 = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
        
    #for the first bit allocation of the left and mid signals
    for iBand in range(sfBands.nBands):
        if not bitAlloc1[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa1=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        ms_band = ms_switch[iBand]
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]
        
        if ms_band == 1:
            mdctLines = mdctLinesMid
        else:
            mdctLines = mdctLinesLeft
        
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor1[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc1[iBand])
        if bitAlloc1[iBand]:
            mantissa1[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor1[iBand], nScaleBits, bitAlloc1[iBand])
            iMant += nLines
    # end of loop over scale factor bands
    
    nMant=halfN
    #for the second bit allocation of the right and side signals
    for iBand in range(sfBands.nBands):
        if not bitAlloc2[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa2=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        ms_band = ms_switch[iBand]
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        
        if ms_band == 1:
            mdctLines = mdctLinesSide
        else:
            mdctLines = mdctLinesRight
        
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor2[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc2[iBand])
                
        if bitAlloc2[iBand]:
            mantissa2[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor2[iBand], nScaleBits, bitAlloc2[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    
    scaleFactors = []
    bitAlloc = []
    mantissa = []
    
    scaleFactors.append(scaleFactor1)
    scaleFactors.append(scaleFactor2)
    bitAlloc.append(bitAlloc1)
    bitAlloc.append(bitAlloc2)
    mantissa.append(mantissa1)
    mantissa.append(mantissa2)
    
    return (scaleFactors, bitAlloc, mantissa, scaleFactorsPack, ms_switch)


