
import numpy as np  # used for arrays
import Queue as queue
import operator

class HuffmanNode(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    def children(self):
        return((self.left, self.right))
    def createCodesArray(self,huff_table,path=None,length=1):
        # save instead of printing, and then bit shift over
        # the length of the path array before adding
        if path is None:
            # path = []
            path = ''
        if self.left is not None:
            if isinstance(self.left[0], HuffmanNode):
                # newpath = (path << 1) + 0
                # self.left[0].createCodesArray(huff_table,path+['0'],length+1)
                self.left[0].createCodesArray(huff_table,path+'0',length+1)
            else:
                # huff_table[self.left[0]] = (path+['0'],length)
                huff_table[self.left[0]] = (path+'0',length)
        if self.right is not None:
            if isinstance(self.right[0], HuffmanNode):
                # self.right[0].createCodesArray(huff_table,path+['1'],length+1)
                self.right[0].createCodesArray(huff_table,path+'1',length+1)
            else:
                # huff_table[self.right[0]] = (path+['1'],length)
                huff_table[self.right[0]] = (path+'1',length)
        return huff_table

    def createCodes(self,huff_table,path=None,length=1):
        # save instead of printing, and then bit shift over
        # the length of the path array before adding
        if path is None:
            path = 0
        if self.left is not None:
            if isinstance(self.left[0], HuffmanNode):
                newpath = (path << 1) + 0
                self.left[0].createCodes(huff_table,newpath,length+1)
            else:
                newpath = (path << 1) + 0
                huff_table[self.left[0]] = (newpath,length)
        if self.right is not None:
            if isinstance(self.right[0], HuffmanNode):
                newpath = (path << 1) + 1
                self.right[0].createCodes(huff_table,newpath,length+1)
            else:
                newpath = (path << 1) + 1
                huff_table[self.right[0]] = (newpath,length)
        return huff_table

def calculateFrequencies(table, data):
    current_max = -1
    # total_freq = 0
    # print data
    for i in range(0,len(data)):
        key = data[i]
        if(table.has_key(key)):
            table[key] = table.get(key) + 1
        else:
            # print key
            for k in range(current_max+1,key):
                table[k] = 0
            table[key] = 1
            current_max = key
        # total_freq = total_freq + 1
    return table

def createTree(table, numEntries):
    # create the escape code
    # val_sorted = sorted(table.items(), key=operator.itemgetter(0))
    freq_sorted = sorted(table.items(), key=operator.itemgetter(1), reverse=True)
    
    # get the top Threshold values in terms of frequency of appearance
    num_top_freqs = numEntries #pow(2,6)-1
    top_freqs = freq_sorted[0:num_top_freqs]
    # max value within top N
    # cutoff_idx = max(lis,key=lambda item:item[1])[0]
    cutoff_idx = num_top_freqs
    # print len(freq_sorted)
    for k in range(cutoff_idx+1,len(freq_sorted)):
        freq_sorted[cutoff_idx] = (freq_sorted[cutoff_idx][0], freq_sorted[cutoff_idx][1] + freq_sorted[k][1])

    # print freq_sorted[cutoff_idx]
    escape_value = freq_sorted[cutoff_idx][0]
    print escape_value
    # print freq_sorted
    del freq_sorted[cutoff_idx+1:]


    # print freq_sorted
    sorted_table = sorted(freq_sorted, key=operator.itemgetter(1))
    print "\n\nSorted by value:"
    print sorted(freq_sorted, key=operator.itemgetter(0))

    print "\n\nSorted by frequency:"
    print sorted_table
    print "\n\n"
    while(len(sorted_table) > 1):
        l = sorted_table[0]
        r = sorted_table[1]
        node = HuffmanNode(l,r)
        sorted_table[0] = (node,l[1] + r[1])
        del sorted_table[1]
        sorted_table = sorted(sorted_table,key=operator.itemgetter(1)) 
    return (sorted_table[0], escape_value)




#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    print '\n-------------GETTING FREQ COUNT-------------'
    x = np.array([[3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,0], \
                  [3,4,5,6,6,6,6,6,7,7,1,1,1,1,1,1,10]]);

    nCriticalBands = 2
    # cb_freq_table = [dict()]*nCriticalBands;
    freq_table = dict()
    for i in range(0,nCriticalBands):
        # freq_table[i] = calculateFrequencies(dict(),x[i])
        freq_table = calculateFrequencies(freq_table,x[i])

    # print '\n-------------CONSTRUCT HUFF TREE-------------'
    (root, escape_value) = createTree(freq_table)
    

    print "\nwalking the hufftree..."
    huffman_lookup_table = dict()
    huffman_lookup_table = root[0].createCodes(dict())

    for k in huffman_lookup_table.keys():
        print huffman_lookup_table.get(k)
    
    print "\n"
    huffman_lookup_table = dict()
    huffman_lookup_table = root[0].createCodes2(dict())

    for k in huffman_lookup_table.keys():
        print huffman_lookup_table.get(k)
    


