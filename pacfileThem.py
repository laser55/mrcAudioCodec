# coding: utf-8

"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codecThem as codec   # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
import sys

import numpy as np  # to allow conversion of data blocks to numpy's array object
from scipy import signal
MAX16BITS = 32767

# things meredith imported:
import os
from glob import glob
import pickle
from huffman import HuffmanNode

class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag='PAC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag=self.fp.read(4)
        if tag!=self.tag: raise "Tried to read a non-PAC file into a PACFile object"
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLHH',self.fp.read(calcsize('<LHLLHH')))
        nBands = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines=  unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        sfBands=ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBands =sfBands
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append( np.zeros(nMDCTLines, dtype=np.float64) )
        myParams.overlapAndAdd=overlapAndAdd
        return myParams


    def ReadDataBlock(self, codingParams, blockNum):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        """----------------------------------EDIT----------------------------------------"""
        file_path = "./training_data/"
        pickles = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*tree.pkl'))]
        """--------------------------------END-EDIT----------------------------------------"""
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))
            
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0] # read it as a little-endian unsigned long
                        
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) ) # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:  raise "Only read a partial block of coded PACFile data"

            """----------------------------------EDIT----------------------------------------"""
            # extract the data from the PackedBits object
            nHuffBits = 4
            huffTable = pb.ReadBits(nHuffBits)

            # load transient data:
            # initialize the blksw variable
            blksw = np.zeros(codingParams.blkswBitA + codingParams.blkswBitB,np.int32)
            # load the (binary) block states (short or long)
            blksw[0] = pb.ReadBits(codingParams.blkswBitA)
            blksw[1] = pb.ReadBits(codingParams.blkswBitB)
            # convert the block states to block lengths
            codingParams.a = (1 - blksw[0])*codingParams.nMDCTLines + blksw[0]*128
            codingParams.b = (1 - blksw[1])*codingParams.nMDCTLines + blksw[1]*128

            # check the size of the combined blocks
            if codingParams.a + codingParams.b == 2*codingParams.nMDCTLines:
                # if two long blocks are present, process normally
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                    (codingParams.a+codingParams.b)/2,codingParams.sampleRate))
            else:
                # for anything else, use fewer scale factor bands
                freqLimits = [300,630,1080,1720,2700,4400,7700,15500,24000]
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                    (codingParams.a+codingParams.b)/2,codingParams.sampleRate,freqLimits))
            """------------------------------END-EDIT----------------------------------------"""

            # extract the data from the PackedBits object
            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            if(huffTable == 15):
                scaleFactor=[]
                bitAlloc=[]
                mantissa=np.zeros(codingParams.nMDCTLines,np.int32)  # start w/ all mantissas zero
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                    bitAlloc.append(ba)  # bit allocation for this band
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                    if bitAlloc[iBand]:
                        # if bits allocated, extract those mantissas and put in correct location in matnissa array
                        m=np.empty(codingParams.sfBands.nLines[iBand],np.int32)
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            m[j]=pb.ReadBits(bitAlloc[iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                        mantissa[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = m
            # """----------------------------------EDIT----------------------------------------"""
            else:
                dill_pkl = open(pickles[huffTable], 'rb')
                current_pickle = pickle.load(dill_pkl)
                dill_pkl.close()

                huff_root = current_pickle[0]

                all_rev_tables = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*_table.revpkl'))]
                rev_pkl = open(all_rev_tables[huffTable])
                
                rev_pickle = pickle.load(rev_pkl)
                rev_pkl.close()

                escape_str = rev_pickle[1]
                rev_table = rev_pickle[0]

                scaleFactor=[]
                bitAlloc=[]

                mantissa=np.zeros(codingParams.nMDCTLines,np.int32)  # start w/ all mantissas zero
                
                iMant = 0
                all_codes = []
                for iBand in range(codingParams.sfBands.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                    bitAlloc.append(ba)  # bit allocation for this band
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                    # bits_already_read += codingParams.nMantSizeBits + codingParams.nScaleBits
                    if bitAlloc[iBand]:
                        for j in range(codingParams.sfBands.nLines[iBand]):
                            # for each mantissa value, traverse the tree
                            current_node = huff_root
                            code_str = ""
                            
                            while(current_node[0] is not None):
                                my_bit = pb.ReadBits(1)
                                # bits_already_read += 1
                                if(my_bit == 1):
                                    code_str += "1"
                                    if isinstance(current_node[0].right[0], HuffmanNode):
                                        current_node = current_node[0].right
                                    else:
                                        break
                                else:
                                    code_str += "0"
                                    if isinstance(current_node[0].left[0], HuffmanNode):
                                        current_node = current_node[0].left
                                    else:
                                        break
                            if(rev_table.has_key(code_str)):
                                if(code_str == escape_str):
                                    mantissa[iMant]=pb.ReadBits(bitAlloc[iBand])
                                    
                                    all_codes.append(str(str(code_str) + "/" + str(mantissa[iMant])))
                                else:
                                    mantissa[iMant] = rev_table.get(code_str)
                                    all_codes.append(str(code_str))
                            else:
                                print "Something has gone horribly wrong..."
                            iMant += 1
                    else:
                        iMant += codingParams.sfBands.nLines[iBand]
            """--------------------------------END-EDIT----------------------------------------"""
            # done unpacking data (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can unpack any custom data passed in the nBytes of data >

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)
            """----------------------------------EDIT----------------------------------------"""
            # concatenate the first block of samples instead of halfN samples
            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[:codingParams.a]) ) )  # data[iCh] is overlap-and-added data
            # store the second block of samples instead of halfN samples
            codingParams.overlapAndAdd[iCh] = decodedData[codingParams.a:]  # save other half for next pass
            """------------------------------END-EDIT----------------------------------------"""


        # end loop over channels, return signed-fraction samples for this block
        return data

    def JointReadDataBlock(self, codingParams, blockNum):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        scaleFactor1=[]
        scaleFactor2=[]
        bitAlloc1=[]
        bitAlloc2=[]
        overallScaleFactor = []
        ms_switch = []
        mantissa1=np.zeros(codingParams.nMDCTLines,np.int32)  # start w/ all mantissas zero
        mantissa2=np.zeros(codingParams.nMDCTLines,np.int32)
        huffTable = 15 

        file_path = "./training_data/"
        pickles = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*tree.pkl'))]
        
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))            
            
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0]
            
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) )
            
            # read the nBytes of data into a PackedBits object to unpack
            # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            
            
            if pb.nBytes < nBytes:  raise Exception("Only read a partial block of coded PACFile data")
            
            # extract the data from the PackedBits object
            nHuffBits = 4
            huffTable = pb.ReadBits(nHuffBits)
            
            """----------------------------------EDIT----------------------------------------"""
            # load transient data:
            # initialize the blksw variable
            blksw = np.zeros(codingParams.blkswBitA + codingParams.blkswBitB,np.int32)
            # load the (binary) block states (short or long)
            blksw[0] = pb.ReadBits(codingParams.blkswBitA)
            blksw[1] = pb.ReadBits(codingParams.blkswBitB)
            # convert the block states to block lengths
            codingParams.a = (1 - blksw[0])*codingParams.nMDCTLines + blksw[0]*128
            codingParams.b = (1 - blksw[1])*codingParams.nMDCTLines + blksw[1]*128

            # check the size of the combined blocks
            if codingParams.a + codingParams.b == 2*codingParams.nMDCTLines:
                # if two long blocks are present, process normally
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                    (codingParams.a+codingParams.b)/2,codingParams.sampleRate))
            else:
                # for anything else, use fewer scale factor bands
                freqLimits = [300,630,1080,1720,2700,4400,7700,15500,24000]
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                    (codingParams.a+codingParams.b)/2,codingParams.sampleRate,freqLimits))
            """------------------------------END-EDIT----------------------------------------"""
            
        # extract the data from the PackedBits object
        ### note that this is now the pack of all of the 4 mdct overall scale factors
            if iCh == 0:
                overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                for iBand in range(codingParams.sfBands.nBands):
                    ms_switch.append(pb.ReadBits(1))

            
            if iCh==0:
                if(huffTable == 15):
                    for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba+=1  
                        bitAlloc1.append(ba)
                        scaleFactor1.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                        
                        if bitAlloc1[iBand]:
                            # if bits allocated, extract those mantissas and put in correct location in matnissa array
                            m=np.empty(codingParams.sfBands.nLines[iBand],np.int32)
                            for j in range(int(codingParams.sfBands.nLines[iBand])):
                                testHere = pb.ReadBits(bitAlloc1[iBand])
                                
                                m[j]=testHere     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                            mantissa1[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = m
                    # done unpacking data (end loop over scale factor bands)
                else:
                    dill_pkl = open(pickles[huffTable], 'rb')
                    current_pickle = pickle.load(dill_pkl)
                    dill_pkl.close()

                    huff_root = current_pickle[0]

                    all_rev_tables = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*_table.revpkl'))]
                    rev_pkl = open(all_rev_tables[huffTable])
                    rev_pickle = pickle.load(rev_pkl)
                    rev_pkl.close()

                    escape_str = rev_pickle[1]
                    rev_table = rev_pickle[0]
                    
                    iMant = 0
                    all_codes = []
                    for iBand in range(codingParams.sfBands.nBands):
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                        bitAlloc1.append(ba)  # bit allocation for this band
                        scaleFactor1.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                        
                        if bitAlloc1[iBand]:
                            for j in range(codingParams.sfBands.nLines[iBand]):
                                # for each mantissa1 value, traverse the tree
                                current_node = huff_root
                                code_str = ""
                                
                                while(current_node[0] is not None):
                                    my_bit = pb.ReadBits(1)
                                    
                                    if(my_bit == 1):
                                        code_str += "1"
                                        if isinstance(current_node[0].right[0], HuffmanNode):
                                            current_node = current_node[0].right
                                        else:
                                            break
                                    else:
                                        code_str += "0"
                                        if isinstance(current_node[0].left[0], HuffmanNode):
                                            current_node = current_node[0].left
                                        else:
                                            break
                                
                                if(rev_table.has_key(code_str)):
                                    if(code_str == escape_str):
                                        mantissa1[iMant]=pb.ReadBits(bitAlloc1[iBand])
                                        
                                        all_codes.append(str(str(code_str) + "/" + str(mantissa1[iMant])))
                                    else:
                                        mantissa1[iMant] = rev_table.get(code_str)
                                        
                                        all_codes.append(str(code_str))
                                else:
                                    print "Something has gone horribly wrong..."
                                iMant += 1
                        else:
                            iMant += codingParams.sfBands.nLines[iBand]

            else:
                if(huffTable == 15):
                    for iBand in range(codingParams.sfBands.nBands):
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba+=1
                        bitAlloc2.append(ba)
                        scaleFactor2.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                        if bitAlloc2[iBand]:
                            # if bits allocated, extract those mantissas and put in correct location in matnissa array
                            m=np.empty(codingParams.sfBands.nLines[iBand],np.int32)
                            for j in range(int(codingParams.sfBands.nLines[iBand])):
                                m[j]=pb.ReadBits(bitAlloc2[iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                            mantissa2[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = m
                else:
                    dill_pkl = open(pickles[huffTable], 'rb')
                    current_pickle = pickle.load(dill_pkl)
                    dill_pkl.close()

                    huff_root = current_pickle[0]

                    all_rev_tables = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*_table.revpkl'))]
                    rev_pkl = open(all_rev_tables[huffTable])
                    rev_pickle = pickle.load(rev_pkl)
                    rev_pkl.close()

                    escape_str = rev_pickle[1]
                    rev_table = rev_pickle[0]
                    
                    iMant = 0
                    all_codes = []
                    for iBand in range(codingParams.sfBands.nBands):
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                        bitAlloc2.append(ba)  # bit allocation for this band
                        scaleFactor2.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                        if bitAlloc2[iBand]:
                            for j in range(codingParams.sfBands.nLines[iBand]):
                                # for each mantissa1 value, traverse the tree
                                current_node = huff_root
                                code_str = ""
                                # print "entering while (k = " + str(k) + "/" + str(len(mantissa1)) + "): " + str(bits_already_read)
                                while(current_node[0] is not None):
                                    my_bit = pb.ReadBits(1)
                                    # bits_already_read += 1
                                    if(my_bit == 1):
                                        code_str += "1"
                                        if isinstance(current_node[0].right[0], HuffmanNode):
                                            current_node = current_node[0].right
                                        else:
                                            break
                                    else:
                                        code_str += "0"
                                        if isinstance(current_node[0].left[0], HuffmanNode):
                                            current_node = current_node[0].left
                                        else:
                                            break
                                
                                if(rev_table.has_key(code_str)):
                                    if(code_str == escape_str):
                                        mantissa2[iMant]=pb.ReadBits(bitAlloc2[iBand])
                                        
                                        all_codes.append(str(str(code_str) + "/" + str(mantissa2[iMant])))
                                    else:
                                        mantissa2[iMant] = rev_table.get(code_str)
                                        
                                        all_codes.append(str(code_str))
                                else:
                                    print "Something has gone horribly wrong..."
                                iMant += 1
                        else:
                            iMant += codingParams.sfBands.nLines[iBand]
                    
                
                    
        
        bitAlloc = []
        mantissa = []
        scaleFactor = []
        
        
        bitAlloc.append(bitAlloc1)
        bitAlloc.append(bitAlloc2)
        
        mantissa.append(mantissa1)
        mantissa.append(mantissa2)
        
        scaleFactor.append(scaleFactor1)
        scaleFactor.append(scaleFactor2)
        
        decodedData = self.JointDecode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,ms_switch)
        
        
        ### now that i have this decoded data, we need to iterate through channels again to do overlap add
        for iCh in range(codingParams.nChannels):
            """----------------------------------EDIT----------------------------------------"""
            # concatenate the first block of samples instead of halfN samples
            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[iCh][:codingParams.a]) ) )  # data[iCh] is overlap-and-added data
            # store the second block of samples instead of halfN samples
            codingParams.overlapAndAdd[iCh] = decodedData[iCh][codingParams.a:]  # save other half for next pass
            """------------------------------END-EDIT----------------------------------------"""

        # end loop over channels, return signed-fraction samples for this block
        
        return data
    def WriteFileHeader(self,codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples%codingParams.nMDCTLines:
            codingParams.numSamples += (codingParams.nMDCTLines
                        - codingParams.numSamples%codingParams.nMDCTLines) # zero padding for partial final PCM block

        # # also add in the delay block for the second pass w/ the last half-block (JH: I don't think we need this, in fact it generates a click at the end)
        # codingParams.numSamples+= codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block

        # write the coded file attributes
        self.fp.write(pack('<LHLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLines,
            codingParams.nScaleBits, codingParams.nMantSizeBits  ))
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                                codingParams.sampleRate)
                                )
        codingParams.sfBands=sfBands
        self.fp.write(pack('<L',sfBands.nBands))
        self.fp.write(pack('<'+str(sfBands.nBands)+'H',*(sfBands.nLines.tolist()) ))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines,dtype=np.float64) )
        codingParams.priorBlock = priorBlock
        return


    def WriteDataBlock(self,data, codingParams, blockNum=0):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData=[]
        for iCh in range(codingParams.nChannels):
            fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        """----------------------------------EDIT----------------------------------------"""
        file_path = "./training_data/"
        pickles = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*table.pkl'))]
        # check the size of the combined blocks
        if codingParams.a + codingParams.b == 2*codingParams.nMDCTLines:
            # if two long blocks are present, process normally
            codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                (codingParams.a+codingParams.b)/2,codingParams.sampleRate))
        else:
            # for anything else, use fewer scale factor bands
            freqLimits = [300,630,1080,1720,2700,4400,7700,15500,24000]
            codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                (codingParams.a+codingParams.b)/2,codingParams.sampleRate,freqLimits))
        """------------------------------END-EDIT----------------------------------------"""

        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor,huffTable) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):

            # determine the size of this channel's data block and write it to the output file
            nBytes = codingParams.nScaleBits  # bits for overall scale factor

            """----------------------------------EDIT----------------------------------------"""
            nHuffBits = 4
            nBytes += nHuffBits
            """------------------------------END-EDIT----------------------------------------"""
            
            if(huffTable[iCh] == 15):
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                    nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                    if bitAlloc[iCh][iBand]:
                        # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                        nBytes += bitAlloc[iCh][iBand]*codingParams.sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
            # """----------------------------------EDIT----------------------------------------"""
            else:
                # open up hufftable and add it up for each mantissa. length for the whole mantissa
                # instead of going band by band since they now have nothing to do with the bands
                # BUT IF THERES AN ESCAPE CODE THEN YOU WILL HAVE TO KEEP TRACK OF THE BITALLOC INFORMATION
                dill_pkl = open(pickles[huffTable[iCh]], 'rb')
                current_pickle = pickle.load(dill_pkl)
                dill_pkl.close()
                escape_value = current_pickle[1]
                current_table = current_pickle[0]
                
                escape_code = current_table.get(escape_value)[0]
                iMant=0 
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                    nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                    if bitAlloc[iCh][iBand]:
                        # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            code = mantissa[iCh][iMant]
                            length = len(str(code).split("/")[0])
                            if(str(code)[0:length] == escape_code):
                                nBytes += length
                                nBytes += bitAlloc[iCh][iBand]
                            else:
                                nBytes += length
                            iMant += 1 
            """------------------------------END-EDIT----------------------------------------"""
            # end computing bits needed for this channel's data

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>
            """----------------------------------EDIT----------------------------------------"""
            # add two bits for blksw
            nBytes += codingParams.blkswBitA
            nBytes += codingParams.blkswBitB
            """------------------------------END-EDIT----------------------------------------"""

            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes%BYTESIZE==0:  nBytes /= BYTESIZE
            else: nBytes = nBytes/BYTESIZE + 1
                        
            self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long
            

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            """----------------------------------EDIT----------------------------------------"""
            # write the huffman table
            pb.WriteBits(huffTable[iCh],nHuffBits)
            # Write the blksw bits FIRST! They have to be used to recreate sfBands
            pb.WriteBits(1-codingParams.a/codingParams.nMDCTLines,codingParams.blkswBitA)
            pb.WriteBits(1-codingParams.b/codingParams.nMDCTLines,codingParams.blkswBitB)
            """------------------------------END-EDIT----------------------------------------"""

            # now pack the nBytes of data into the PackedBits object
            pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
            bitsWritten = codingParams.nScaleBits
            if(huffTable[iCh] == 15):
                iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                    pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                    bitsWritten += codingParams.nMantSizeBits
                    pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                    bitsWritten += codingParams.nScaleBits
                    if bitAlloc[iCh][iBand]:
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                            bitsWritten += bitAlloc[iCh][iBand]
                        iMant += codingParams.sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
            # """----------------------------------EDIT----------------------------------------"""
            else:
                dill_pkl = open(pickles[huffTable[iCh]], 'rb')
                current_pickle = pickle.load(dill_pkl)
                dill_pkl.close()

                escape_value = current_pickle[1]
                current_table = current_pickle[0]
                escape_code = current_table.get(escape_value)[0]
                iMant = 0
                for iBand in range(codingParams.sfBands.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                    pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                    pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                    # bits_written += codingParams.nMantSizeBits + codingParams.nScaleBits
                    if bitAlloc[iCh][iBand]:
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            # now we want to write the mantissa
                            code = mantissa[iCh][iMant]
                            length = len(str(code).split("/")[0])

                            if(str(code)[0:length] == escape_code):
                                # write bit by bit
                                # then write the mantissa
                                original_mant = str(code).split("/")[1]
                                code = str(code).split("/")[0]
                                # for loop through the length
                                # print "escaped, write mantissa code: " + str(code)
                                for n in range(0,length):
                                    pb.WriteBits(np.int32(str(code)[n]),1)
                                    # bits_written += 1
                                # print "escaping a value: " + str(original_mant)
                                pb.WriteBits(np.int32(original_mant),bitAlloc[iCh][iBand])
                                # bits_written += bitAlloc[iCh][iBand]
                            else:
                                for n in range(0,length):
                                    pb.WriteBits(np.int32(str(code)[n]),1)
                                    # bits_written += 1
                            iMant += 1 
                """--------------------------------END-EDIT----------------------------------------"""
            # done packing (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>
            
            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return


    def JointWriteDataBlock(self,data, codingParams, blockNum):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData=[]
        for iCh in range(codingParams.nChannels):
            fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        file_path = "./training_data/"
        pickles = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*table.pkl'))]
        """----------------------------------EDIT----------------------------------------"""
        # check the size of the combined blocks
        if codingParams.a + codingParams.b == 2*codingParams.nMDCTLines:
            # if two long blocks are present, process normally
            codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                (codingParams.a+codingParams.b)/2,codingParams.sampleRate))
        else:
            # for anything else, use fewer scale factor bands
            freqLimits = [300,630,1080,1720,2700,4400,7700,15500,24000]
            codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(
                (codingParams.a+codingParams.b)/2,codingParams.sampleRate,freqLimits))
        """------------------------------END-EDIT----------------------------------------"""

        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor, ms_switch,huffTable) = self.JointEncode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header
        
        
        # for each channel, write the data to the output file
        # needs to be no loop in packing the bits
        for iCh in range(codingParams.nChannels):
            nBytes = 0

            if iCh == 0:
                nBytes += codingParams.sfBands.nBands
                nBytes += 4*codingParams.nScaleBits

            nHuffBits = 4
            nBytes += nHuffBits
            
            if(huffTable[iCh] == 15):
                # determine the size of this channel's data block and write it to the output file
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                    nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits   # mantissa bit allocation and scale factor for that sf band
                    if bitAlloc[iCh][iBand]:
                        # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                        nBytes += bitAlloc[iCh][iBand]*codingParams.sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
                # end computing bits needed for this channel's data
            else:
                # open up hufftable and add it up for each mantissa. length for the whole mantissa
                # instead of going band by band since they now have nothing to do with the bands
                # BUT IF THERES AN ESCAPE CODE THEN YOU WILL HAVE TO KEEP TRACK OF THE BITALLOC INFORMATION
                dill_pkl = open(pickles[huffTable[iCh]], 'rb')
                current_pickle = pickle.load(dill_pkl)
                dill_pkl.close()
                escape_value = current_pickle[1]
                current_table = current_pickle[0]
                
                escape_code = current_table.get(escape_value)[0]
                iMant=0 
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                    nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                    if bitAlloc[iCh][iBand]:
                        # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            code = mantissa[iCh][iMant]
                            length = len(str(code).split("/")[0])
                            # print "length: " + str(length) 
                            if(str(code)[0:length] == escape_code):
                                nBytes += length
                                nBytes += bitAlloc[iCh][iBand]
                            else:
                                nBytes += length
                            iMant += 1 

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>
            """----------------------------------EDIT----------------------------------------"""
            # add two bits for blksw
            nBytes += codingParams.blkswBitA
            nBytes += codingParams.blkswBitB
            """------------------------------END-EDIT----------------------------------------"""
            
            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes%BYTESIZE==0:  nBytes /= BYTESIZE
            else: nBytes = nBytes/BYTESIZE + 1
                        
            self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long
            
            # print "ENCODE SIDE"
            # print int(nBytes)
            # print ""
        
            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # now pack the nBytes of data into the PackedBits object
            #pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
            pb.WriteBits(huffTable[iCh],nHuffBits)
            """----------------------------------EDIT----------------------------------------"""
            # Write the blksw bits FIRST! They have to be used to recreate sfBands
            pb.WriteBits(1-codingParams.a/codingParams.nMDCTLines,codingParams.blkswBitA)
            pb.WriteBits(1-codingParams.b/codingParams.nMDCTLines,codingParams.blkswBitB)
            """------------------------------END-EDIT----------------------------------------"""
            
            if iCh == 0:
                pb.WriteBits(overallScaleFactor[0],codingParams.nScaleBits)
                pb.WriteBits(overallScaleFactor[1],codingParams.nScaleBits)
                pb.WriteBits(overallScaleFactor[2],codingParams.nScaleBits)
                pb.WriteBits(overallScaleFactor[3],codingParams.nScaleBits)
                for iBand in range(codingParams.sfBands.nBands):
                    pb.WriteBits(ms_switch[iBand], 1)

            
            if(huffTable[iCh] == 15):
                ###
                iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
                for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                    
                    pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                    pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                ###
                    if bitAlloc[iCh][iBand]:
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                        iMant += codingParams.sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
            else: 
                
                dill_pkl = open(pickles[huffTable[iCh]], 'rb')
                current_pickle = pickle.load(dill_pkl)
                dill_pkl.close()

                escape_value = current_pickle[1]
                current_table = current_pickle[0]
                escape_code = current_table.get(escape_value)[0]
                iMant = 0
                for iBand in range(codingParams.sfBands.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                    pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                    pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                    # bits_written += codingParams.nMantSizeBits + codingParams.nScaleBits
                    if bitAlloc[iCh][iBand]:
                        for j in range(int(codingParams.sfBands.nLines[iBand])):
                            # now we want to write the mantissa
                            code = mantissa[iCh][iMant]
                            length = len(str(code).split("/")[0])

                            if(str(code)[0:length] == escape_code):
                                # write bit by bit
                                # then write the mantissa
                                original_mant = str(code).split("/")[1]
                                code = str(code).split("/")[0]
                                # for loop through the length
                                
                                for n in range(0,length):
                                    pb.WriteBits(np.int32(str(code)[n]),1)
                                    # bits_written += 1
                                
                                pb.WriteBits(np.int32(original_mant),bitAlloc[iCh][iBand])
                                # bits_written += bitAlloc[iCh][iBand]
                            else:
                                for n in range(0,length):
                                    pb.WriteBits(np.int32(str(code)[n]),1)
                                    # bits_written += 1
                            iMant += 1 
            # done packing (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return
    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [ np.zeros(codingParams.nMDCTLines,dtype=np.float),
                     np.zeros(codingParams.nMDCTLines,dtype=np.float) ]
            self.WriteDataBlock(data, codingParams)
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)
        
    def JointEncode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.JointEncode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)
        
    def JointDecode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,ms_switch):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.JointDecode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,ms_switch)




"""----------------------------------EDIT----------------------------------------"""
def TransientDetector(data,codingParams,sos,T):
    """
    Returns a vector of variable length containing the locations of all transients in the passed 
    data. Input variables are described below
    data: a matrix of data nChannels by nSamplesPerBlock
    codingParams: a set of info about the current encoding method
    sos: a HPF separated into second order sections
    T: a vector of thresholds used to detect transients
    """
    
    blksw = np.array([])
    for iCh in range(codingParams.nChannels):
        # filter the newest set of data
        dataFilt = signal.sosfilt(sos,data[iCh])
        # peak detection for each subsegment 
        for i in range(codingParams.nSamplesPerBlock/codingParams.nSamplesShort):
            codingParams.P[iCh][i+1] = np.amax(np.abs(dataFilt[i*codingParams.nSamplesShort
                :(i+1)*codingParams.nSamplesShort]))
            
        # threshold comparison
        if np.amax(np.abs(dataFilt)) > T[0]:
            for i in range(codingParams.nSamplesPerBlock/codingParams.nSamplesShort):
                if codingParams.P[iCh][i+1]*T[1] > codingParams.P[iCh][i]:
                    blksw = np.append(blksw,i + 1)

    # update prior peaks
    codingParams.P[:,0] = codingParams.P[:,codingParams.nSamplesPerBlock/codingParams.nSamplesShort]

    # only return the unique, nonzero values of transient locations
    blksw = np.unique(blksw[np.nonzero(blksw)])

    return blksw
"""------------------------------END-EDIT----------------------------------------"""



#-----------------------------------------------------------------------------

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__=="__main__":

    import sys
    import time
    from pcmfile import * # to get access to WAV file handling

    input_filename = "harpRef.wav"
    coded_filename = "harp128.pac"
    output_filename = "harp128.wav"

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        coded_filename = sys.argv[1][:-4] + ".pac"
        output_filename = sys.argv[1][:-4] + "_decoded.wav"


    print "\nRunning the PAC coder ({} -> {} -> {}):".format(input_filename, coded_filename, output_filename)
    elapsed = time.time()
    blocky = 0
    blockBreak = -2
    for Direction in ("Encode", "Decode"):
    # for Direction in ("Decode"):

        # create the audio file objects
        if Direction == "Encode":
            print "\n\tEncoding PCM file ({}) ...".format(input_filename),
            inFile= PCMFile(input_filename)
            outFile = PACFile(coded_filename)
        else: # "Decode"
            print "\n\tDecoding PAC file ({}) ...".format(coded_filename),
            inFile = PACFile(coded_filename)
            outFile= PCMFile(output_filename)
        # only difference is file names and type of AudioFile object

        # open input file
        codingParams=inFile.OpenForReading()  # (includes reading header)

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = 1024
            codingParams.nScaleBits = 4 #was 3
            codingParams.nMantSizeBits = 4
            codingParams.targetBitsPerSample = 2.86 #was 1.9
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
            """----------------------------------EDIT----------------------------------------"""
            # huffman table bit reservoir
            codingParams.bitReservoir = 0
            # short block length
            codingParams.nSamplesShort = 128
            # previous and current block lengths
            codingParams.a = codingParams.nMDCTLines
            codingParams.b = codingParams.nMDCTLines
            # bits to keep track of block lengths
            codingParams.blkswBitA = 1
            codingParams.blkswBitB = 1
            """------------------------------END-EDIT----------------------------------------"""
        else: # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
            """----------------------------------EDIT----------------------------------------"""
            # short block length
            codingParams.nSamplesShort = 128
            # previous and current block lengths
            codingParams.a = codingParams.nMDCTLines
            codingParams.b = codingParams.nMDCTLines
            # bits to keep track of block lengths
            codingParams.blkswBitA = 1
            codingParams.blkswBitB = 1
            """------------------------------END-EDIT----------------------------------------"""
        # only difference is in setting up the output file parameters


        # open the output file
        outFile.OpenForWriting(codingParams) # (includes writing header)

        """----------------------------------EDIT----------------------------------------"""
        # Transient Detection Variables:
        # Create HPF
        # b,a = signal.butter(20,9000./codingParams.sampleRate,'high')
        b,a = signal.cheby2(20,40,9000./codingParams.sampleRate,'high')
        sos = signal.tf2sos(b,a)
        # block switching flag
        # blksw = np.zeros(codingParams.nChannels)
        # block peak matrix
        codingParams.P = np.zeros((codingParams.nChannels,
            1 + codingParams.nSamplesPerBlock/codingParams.nSamplesShort))
        # Threshold values: 1st is threshold for block comparison, the second block ratio threshold
        T = np.array([0.1,0.075])
        """------------------------------END-EDIT----------------------------------------"""

        # Read the input file and pass its data to the output file to be written
        firstBlock = True  # when de-coding, we won't write the first block to the PCM file. This flag signifies that
        while True:
            if blockBreak == blocky and Direction == "Decode":
                data=inFile.ReadDataBlock(codingParams,blocky)
            else:
                data=inFile.JointReadDataBlock(codingParams,blocky)
                
            if not data: 
                blockBreak = blocky-1
                blocky = 0
                break# we hit the end of the input file
            """----------------------------------EDIT----------------------------------------"""
            # convert list of arrays to matrix
            data = np.vstack(data)
            """------------------------------END-EDIT----------------------------------------"""

            # don't write the first PCM block (it corresponds to the half-block delay introduced by the MDCT)
            if firstBlock and Direction == "Decode":
                firstBlock = False
                continue

            """----------------------------------EDIT----------------------------------------"""
            # detect transients while encoding wav
            if Direction == 'Encode':
                blksw = TransientDetector(data,codingParams,sos,T)

            # incur a one block 'look ahead' delay
            if Direction == 'Encode' and firstBlock:
                dataMem = data
                blkswMem = blksw
                firstBlock = False
                continue

            #  If a transient is detected, process many smaller blocks instead of one large block
            if np.sum(blkswMem) > 1 or any(blksw == 1):
                for i in range(codingParams.nSamplesPerBlock/codingParams.nSamplesShort):
                    # current block length
                    codingParams.b = codingParams.nSamplesShort
                    # process current block
                    outFile.JointWriteDataBlock(dataMem[:,codingParams.b*i:codingParams.b*(i+1)]
                        ,codingParams, blocky)
                    # store current block length for next iteration
                    codingParams.a = codingParams.b
                    blocky += 1
                
            else:
                # current block length
                codingParams.b = codingParams.nSamplesPerBlock
                # process current block
                outFile.JointWriteDataBlock(dataMem,codingParams,blocky)
                # store current block length for next iteration
                codingParams.a = codingParams.b
                blocky += 1

            # store current data and transient info for next iteration
            dataMem = data
            blkswMem = blksw
            """------------------------------END-EDIT----------------------------------------"""

            #outFile.JointWriteDataBlock(data,codingParams,blocky)
            #outFile.WriteDataBlock(data,codingParams,blocky)
            
            sys.stdout.write(".")  # just to signal how far we've gotten to user
            sys.stdout.flush()
        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode

    elapsed = time.time()-elapsed
    print "\nDone with Encode/Decode test\n"
    print elapsed ," seconds elapsed"
