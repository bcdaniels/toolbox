# informationDecomposition.py
#
# Bryan Daniels
# 2021/6/18
#
# Implement Williams and Beer pairwise mutual information decomposition.
#
# Branched from CAS-503-Collectives/neural (2022.3.25)
# Branched from mutualInfo.py (4.2.2012)

import numpy as np
import warnings
from . import mutualInfo as mi

def discreteMutualInfo(data1,data2,maxVal1=None,maxVal2=None,**kwargs):
    """
    Using data sampled simultaneously from two discrete distributions,
    return an estimate in bits of the mutual information between the two
    distributions.
    
    data1 and data2 should have the same length.
    
    maxVals are passed to the discreteInfo function, and other kwargs
    are passed to the mutualInfo function.
    """
    assert(len(data1)==len(data2))
    info1 = mi.discreteInfo(data1,maxVal=maxVal1)
    info2 = mi.discreteInfo(data2,maxVal=maxVal2)
    return mi.mutualInfo(info1,info2,**kwargs)
    
def discreteJointInfo(data1,data2,data3,maxVal1=None,maxVal2=None,
    maxVal3=None,**kwargs):
    """
    Using data sampled simultaneously from three discrete distributions,
    return an estimate in bits of the mutual information between the
    (single-valued) distribution of the first variable and the
    joint distribution of the last two variables.  That is:
    
    mutualInfo( data1 | data2 , data3 )
    
    data1, data2, and data3 should have the same length.
    
    maxVals are passed to the discreteInfo function, and other kwargs
    are passed to the mutualInfo function.
    """
    assert(len(data1)==len(data2))
    assert(len(data1)==len(data3))
    
    info1 = mi.discreteInfo(data1,maxVal=maxVal1)
    info2 = mi.discreteInfo(data2,maxVal=maxVal2)
    info3 = mi.discreteInfo(data3,maxVal=maxVal3)
    
    return mi.mutualInfo(info1,jointInfo(info2,info3),**kwargs)

def specificInfo(infoContainerY,infoContainerX,stateIndexY,naive=True):
    """
    As defined in Timme et al. 2014, equation (29).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discreteInfo infoContainers.
    """
    pY = infoContainerY.nVec/float(sum(infoContainerY.nVec))
    pStateY = pY[stateIndexY]
    
    # pXgivenStateY is indexed by X states
    infoXgivenStateY = mi.conditionalInfo(infoContainerX,infoContainerY,stateIndexY)
    pXgivenStateY = infoXgivenStateY.nVec/float(sum(infoXgivenStateY.nVec))
    
    # pStateYgivenX is also indexed by X states
    pStateYgivenX = []
    possibleStateIndicesX = range(infoContainerX.maxVal)
    for stateIndexX in possibleStateIndicesX:
        infoYgivenStateX = mi.conditionalInfo(infoContainerY,infoContainerX,stateIndexX)
        pYgivenStateX = infoYgivenStateX.nVec/float(sum(infoYgivenStateX.nVec))
        pStateYgivenStateX = pYgivenStateX[stateIndexY]
        pStateYgivenX.append(pStateYgivenStateX)
    
    if naive:
        with warnings.catch_warnings():
            # ignore "divide by zero" and "invalid value" warnings
            warnings.simplefilter("ignore")
            si = np.sum( np.nan_to_num(
                pXgivenStateY * ( - np.log2(pStateY) + np.log2(pStateYgivenX) ) ) )
        return (si,0.)
    else:
        raise(NotImplementedError("NSB entropy estimation not yet implemented"))
   
def redundancy(dataY,dataX1,dataX2,naive=True):
    """
    Calculate the redundant info. as given in Timme et al. 2014, equation (31).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = mi.discreteInfo(dataY)
    infoContainerX1 = mi.discreteInfo(dataX1)
    infoContainerX2 = mi.discreteInfo(dataX2)
    
    return redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2,
                               naive=naive)

def redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2,naive=True):
    """
    Calculate the redundant info. as given in Timme et al. 2014, equation (31).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete infoContainers.
    """
    pY = infoContainerY.nVec/float(sum(infoContainerY.nVec))
    Imin,IminStd = 0.,0.
    possibleStateIndicesY = range(infoContainerY.maxVal)
    for stateIndexY in possibleStateIndicesY:
        specificInfoX1 = specificInfo(infoContainerY,infoContainerX1,stateIndexY,
                                      naive=naive)
        specificInfoX2 = specificInfo(infoContainerY,infoContainerX2,stateIndexY,
                                      naive=naive)
        if naive:
            Imin += pY[stateIndexY] * min(specificInfoX1[0],specificInfoX2[0])
        else:
            raise(NotImplementedError("NSB entropy estimation not yet implemented"))
    return Imin,IminStd

def unique(dataY,dataX1,dataX2,naive=True):
    """
    Calculate the unique info. given by X1 and X2, as given in
    Timme et al. 2014, equation (33-34).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = mi.discreteInfo(dataY)
    infoContainerX1 = mi.discreteInfo(dataX1)
    infoContainerX2 = mi.discreteInfo(dataX2)
    
    R = redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2,
                            naive=naive)
    U1 = mi.mutualInfo(infoContainerY,infoContainerX1,naive=naive) - R
    U2 = mi.mutualInfo(infoContainerY,infoContainerX2,naive=naive) - R
    return U1,U2

def synergy(dataY,dataX1,dataX2,naive=True):
    """
    Calculate the synergistic info. as given in
    Timme et al. 2014, equation (32).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = mi.discreteInfo(dataY)
    infoContainerX1 = mi.discreteInfo(dataX1)
    infoContainerX2 = mi.discreteInfo(dataX2)
    
    joint = mi.mutualInfo(infoContainerY,
                          mi.jointInfo(infoContainerX1,infoContainerX2),
                          naive=naive)
    MI1 = mi.mutualInfo(infoContainerY,infoContainerX1,naive=naive)
    MI2 = mi.mutualInfo(infoContainerY,infoContainerX2,naive=naive)
    R = redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2,
                            naive=naive)
    if naive:
        synStd = 0.
    else:
        raise(NotImplementedError("NSB entropy estimation not yet implemented"))
    return joint[0] - MI1[0] - MI2[0] + R[0], synStd
