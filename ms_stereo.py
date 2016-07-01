import numpy as np


#### Calculates if the ms encoding should be enabled for the each scale factor band
def MSSwitchSFBands(mdct_left, mdct_right, sfBands):
    lr_diff = np.square(mdct_left) - np.square(mdct_right)
    lr_sum = np.square(mdct_left) + np.square(mdct_right)
    ms_switch = []
    
    for i in range(0, sfBands.nBands):
        lowerLimit = sfBands.lowerLine[i]
        upperLimit = sfBands.upperLine[i] + 1
        
        #sum_dif = np.abs(np.sum(lr_diff[lowerLimit:upperLimit]))
        #sum_sum = np.abs(np.sum(lr_sum[lowerLimit:upperLimit]))

        sum_dif = np.sum(np.abs(lr_diff[lowerLimit:upperLimit]))
        sum_sum = np.sum(np.abs(lr_sum[lowerLimit:upperLimit]))
        
        if sum_dif < 0.8*sum_sum:
            ms_switch.append(1)
        else:
            ms_switch.append(0)
            
    
    
    return ms_switch




### Rebuilds data for ms coding in decoder
def ReconstructLR(mdct1, mdct2, sfBands, ms_switch):
    mdct_left = []
    mdct_right = []
    
    for i in range(0, sfBands.nBands):
        lowerLimit = sfBands.lowerLine[i]
        upperLimit = sfBands.upperLine[i] + 1
        
        if ms_switch[i] == 1:
            mdct_left = np.append(mdct_left, mdct1[lowerLimit:upperLimit] + mdct2[lowerLimit:upperLimit])
            mdct_right = np.append(mdct_right, mdct1[lowerLimit:upperLimit] - mdct2[lowerLimit:upperLimit])
        else:
            mdct_left = np.append(mdct_left, mdct1[lowerLimit:upperLimit])
            mdct_right = np.append(mdct_right, mdct2[lowerLimit:upperLimit])
        
    
    return (mdct_left, mdct_right)
    


def StereoMaskingFactor(midThresh, sideThresh, sfBands, zVec):
    MLD = np.power(10.0, 1.25*(1-np.cos((np.pi/15.5)*np.minimum(zVec, 15.5*np.ones(np.size(zVec)))) - 2.5))
    
    new_midThresh = MLD*midThresh
    new_sideThresh = MLD*sideThresh
    
    final_midThresh = np.maximum(midThresh, np.minimum(sideThresh, new_sideThresh))
    final_sideThresh = np.maximum(sideThresh, np.minimum(midThresh, new_midThresh))
    
    #pack the data and send back
    dataPack = []
    dataPack.append(final_midThresh)
    dataPack.append(final_sideThresh)
    
    return dataPack
    

def OverallSMRs(SMR_l, SMR_r, SMR_m, SMR_s, sfBands, ms_switch):
    SMR_final1 = []
    SMR_final2 = []
    for i in range(0, sfBands.nBands):
        if ms_switch[i] == 1:
            SMR_final1.append(SMR_m[i])
            SMR_final2.append(SMR_s[i])
        else:
            SMR_final1.append(SMR_l[i])
            SMR_final2.append(SMR_r[i])
    
    return (SMR_final1, SMR_final2)
    
