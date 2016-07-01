# from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
# import codec as codec   # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
# from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
# import sys
from pacfile import *
from quantize import *
from huffman import *
from glob import glob
import os
import csv
import pickle, pprint

import sys
import time
from pcmfile import * # to get access to WAV file handling

if __name__=="__main__":

    # create the audio file objects of the appropriate audioFile type
    huff_num = 'silence'
    file_path = "./training_data/" + huff_num
    test_files = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*.wav'))]
    print test_files
    pickles = [y for x in os.walk('.') for y in glob(os.path.join(x[0], '*.pkl'))]
    # print pickles

    
    freq_table = dict()
    elapsed = time.time()
    for file_idx in range(0,len(test_files)):
        print "\nLooking at " + str(test_files[file_idx])
        inFile = PCMFile(test_files[file_idx])
        coded_filename = test_files[file_idx][0:-4] + '_coded.pac'
        outFile = PACFile(coded_filename)

        codingParams=inFile.OpenForReading()  # (includes reading header)

        codingParams.nMDCTLines = 1024
        codingParams.nScaleBits = 3 #was 3
        codingParams.nMantSizeBits = 5
        codingParams.targetBitsPerSample = 2.27 #was 1.9
        # tell the PCM file how large the block size is
        codingParams.nSamplesPerBlock = codingParams.nMDCTLines
        codingParams.bitReservoir = 0

        outFile.OpenForWriting(codingParams) 

        # Read the input file and pass its data to the output file to be written
        firstBlock = True  # when de-coding, we won't write the first block to the PCM file. This flag signifies that
        while True:
            data=inFile.ReadDataBlock(codingParams,0)
            if not data: break  # we hit the end of the input file

            fullBlockData=[]
            for iCh in range(codingParams.nChannels):
                fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
            codingParams.priorBlock = data  # current pass's data is next pass's prior block data
            # (ENCODE HERE) Encode the full block of multi=channel data
            (scaleFactor,bitAlloc,mantissaVec,overallScaleFactor,huffTable) = outFile.EncodeNoHuff(fullBlockData,codingParams) 
            for iCh in range(len(mantissaVec)):
                freq_table = calculateFrequencies(freq_table,mantissaVec[iCh])

            sys.stdout.write(".")  # just to signal how far we've gotten to user
            sys.stdout.flush()
        
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode


    elapsed = time.time()-elapsed
    print "\nDone with Encode/Decode test\n"
    print elapsed ," seconds elapsed"
    # print freq_table


    print "Done! Building huffman tree and code table..."

    (root, escape_value) = createTree(freq_table,10)
    
    lookup = root[0].createCodesArray(dict())
    print lookup
    huffman_lookup_table = [lookup, escape_value]
    huffman_tree = [root, lookup.get(escape_value)]
    print lookup.get(escape_value)

    for k in huffman_lookup_table[0].keys():
        print str(k) + ": " + str(huffman_lookup_table[0].get(k))

    # print "\nEscape value: " + str(escape_value)

    huff_pickle = open(huff_num + '_tree.pkl', 'wb')
    pickle.dump(huffman_tree,huff_pickle)
    huff_pickle.close()

    huff_pickle = open(huff_num + '_table.pkl', 'wb')
    pickle.dump(huffman_lookup_table,huff_pickle)
    huff_pickle.close()

    reverse = dict()
    for k in huffman_lookup_table[0].keys():
        mantissa_value = k 
        code_tuple = huffman_lookup_table[0].get(k)
        code = code_tuple[0]
        # make code into a string
        code_str = ""
        for i in range(0,len(code)):
            code_str += str(code[i])
        # code = code_tuple[0]
        # length = code_tuple[1]
        reverse[code_str] = mantissa_value

    escape_code_arr = lookup.get(escape_value)[0]
    escape_str = ""
    for i in range(0,len(escape_code_arr)):
        escape_str += str(escape_code_arr[i])

    reverse = [reverse, escape_str]
    huff_pickle_rev = open(huff_num + '_table.revpkl', 'wb')
    pickle.dump(reverse,huff_pickle_rev)
    huff_pickle_rev.close()

    for k in reverse[0].keys():
        print str(k) + ": " + str(reverse[0].get(k))
    # reverse = {v: k for k, v in huffman_lookup_table.iteritems()}
    # print reverse

# if __name__=="__main__":
    
#     # create the audio file objects of the appropriate audioFile type
#     file_path = "./training_data"
#     test_files = [y for x in os.walk(file_path) for y in glob(os.path.join(x[0], '*.wav'))]

#     freq_table = dict()
#     for file_idx in range(0,len(test_files)):
#         # inFile= PCMFile("input.wav")
#         inFile = PCMFile(test_files[file_idx])
#         print str(test_files[file_idx]) + "..."
#         # outFile = PCMFile("output_huff_test.wav")
#         # print "outfile: " + str(test_files[file_idx][0:-4]+"_output.wav")
#         # outFile = PCMFile(test_files[file_idx][0:-4]+"_output.wav")

#         # open input file and get its coding parameters
#         codingParams = inFile.OpenForReading()

#         # set additional coding parameters that are needed for encoding/decoding
#         codingParams.nSamplesPerBlock = 1024

#         # open the output file for writing, passing needed format/data parameters
#         # outFile.OpenForWriting(codingParams)

#         # initialize the frequency table, which measure the frequency with which
#         # certain mantissa values appear in the file
        
#         # Read the input file and pass its data to the output file to be written
#         while True:
#             data = inFile.ReadDataBlock(codingParams)
#             # The three lines below (aka the for-loop) is for adding quantization noise
#             for iCh in range(codingParams.nChannels):
#                 if not data: break 

#                 aNumVec = data[iCh]
#                 max_sample = np.amax(np.fabs(aNumVec))
#                 scale = ScaleFactor(max_sample)
#                 mantissaVec = vMantissa(aNumVec, scale)
#                 freq_table = calculateFrequencies(freq_table,mantissaVec)
#                 data[iCh] = vDequantize(scale,mantissaVec)

#             if not data: break  # we hit the end of the input file
#             # outFile.WriteDataBlock(data,codingParams)
#         # end loop over reading/writing the blocks

#         # close the files
#         inFile.Close(codingParams)
#         # outFile.Close(codingParams)

#     print "Done! Building huffman tree and code table..."
#     for k in freq_table.keys():
#         print str(k) + ": " + str(freq_table.get(k))

#     root = createTree(freq_table)
#     # print "\nwalking the hufftree..."
#     huffman_lookup_table = dict()
#     huffman_lookup_table = root[0].createCodesArray(dict())

#     for k in huffman_lookup_table.keys():
#         print str(k) + ": " + str(huffman_lookup_table.get(k))
    
#     # print "\n"
#     huffman_lookup_table = dict()
#     huffman_lookup_table = root[0].createCodes(dict())

#     # for k in huffman_lookup_table.keys():
#         # print str(k) + ": " + str(huffman_lookup_table.get(k))

#     huff_pickle = open('huffman_table0.pkl', 'wb')
#     pickle.dump(huffman_lookup_table,huff_pickle)
#     huff_pickle.close()

#     # pkl_file = open('data.pkl', 'rb')

#     # data1 = pickle.load(pkl_file)
#     # print "PRINTING THE PICKLE THING!"
#     # # pprint.pprint(data1)
#     # for k in data1.keys():
#     #     print str(k) + ": " + str(huffman_lookup_table.get(k))

#     # pkl_file.close()