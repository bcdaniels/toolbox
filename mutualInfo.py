# mutualInfo.py
#
# Bryan Daniels
# 2022/1/7
#
# Branched from neural/mutualInfo.py
#
#

from . import EntropyEstimates as ent
import numpy as np
import pylab

def find(arr):
    """
    A replacement for the old pylab.find
    """
    #return np.where(arr)[0]
    return np.asarray(arr).nonzero()[0]

# 7.20.2012
class infoContainer():

    def __init__(self,):
        raise notImplementedError
        
    def _calculateNvec(self,possibleValues=None):
        if possibleValues is not None:
            # ensure each state is counted once, then remove one from each at the end
            trialVals = list(self.trialValues) + list(possibleValues)
        else:
            trialVals = self.trialValues
    
        # taken from EntropyEstimates.fights2kxnx "efficient version"
        tvSorted = np.sort(trialVals)
        diffJumpLocs = find( (tvSorted[1:]-tvSorted[:-1])>0. )
        nList = list( diffJumpLocs[1:]-diffJumpLocs[:-1] )
        if len(diffJumpLocs) > 0:
            # add for beginning and end
            nList.insert(0,diffJumpLocs[0]+1)
            nList.append(len(trialVals)-diffJumpLocs[-1]-1)
        else: # all values are equal
            nList.append(len(trialVals))
            
        if possibleValues is not None:
            nList = np.array(nList) - 1
        self.nVec = np.array(nList)
        
    def _setupEmpty(self,):
        self.trialValues = []
        self.numTrials = 0
        self.numDimensions = 0
        self.maxVal = 0
        self._calculateNvec()
        
    def calculateEntropy(self,naive=None,save=True):
        """
        naive (None)        : True uses naive entropy estimation,
                              fast but potentially biased (and
                              gives no estimate of its error).
                              False uses NSB entropy estimation.
                              The default (None) uses
                              the naive method when there are at
                              least 10 samples for every possibility.
        """
        if self.numTrials == 0:
            return (np.nan,np.nan)
        if naive is None:
            naive = np.all(self.nVec > 10)
        if naive:
            if hasattr(self,'savedEntropyNaive'):
                return self.savedEntropyNaive
            entropy = (ent.naiveEntropy(self.nVec/float(sum(self.nVec))),0.)
            if save: self.savedEntropyNaive = entropy
        else:
            if hasattr(self,'savedEntropy'):
                return self.savedEntropy
            entropy = ent.meanAndStdevEntropyNem(self.nVec,K=self.maxVal)
            if save: self.savedEntropy = entropy
        return entropy


# 7.20.2012
class discreteInfo(infoContainer):
    def __init__(self,discreteData,maxVal=None):
        """
        discreteData        : length = #trials
        maxVal (None)       : Defaults to
                              len(np.unique(discreteData))
        """
        if np.prod(np.shape(discreteData)) == 0:
            self._setupEmpty()
            return
        if len(np.shape(discreteData)) > 1:
            raise(Exception, "discreteData should have "        \
                "1 dimension, not "                             \
                +str(len(np.shape(discreteData))))
        self.discreteValues,self.trialValues =                  \
            np.unique(discreteData,return_inverse=True)
        if maxVal is None: self.maxVal = len(self.discreteValues)
        else: self.maxVal = maxVal
        self.numTrials = len(discreteData)
        self._calculateNvec()

# As of 2022/3/25, I'm not sure of the status of this old code implementing
# binning for continuous-valued data.  May come in handy in the future?
## 9.7.2012
#class continuousInfo(infoContainer):
#    def __init__(self,continuousData,numBins):
#        """
#        continuousData      : length = #trials
#        numBins             : Data is binned into numBins bins of
#                              equal width.  The first bin has left
#                              edge at the minimum value in the data;
#                              the last bin has right edge at the
#                              maximum value in the data.
#        """
#        if np.prod(np.shape(continuousData)) == 0:
#            self._setupEmpty()
#            return
#        if len(np.shape(continuousData)) > 1:
#            raise Exception("continuousData should have 1 dimension, not "\
#                +str(len(np.shape(continuousData))))
#        mn,mx = min(continuousData),max(continuousData)
#        binEdges = np.linspace(mn,mx,numBins+1)
#        d = np.digitize(continuousData,binEdges)
#        # digitize maps max to next bin; fix:
#        d[find(d==numBins+1)] = numBins
#        self.trialValues = d - 1
#        self.maxVal = numBins
#        self.numTrials = len(continuousData)
#        self._calculateNvec()

# 7.20.2012
class jointInfo(infoContainer):
    def __init__(self,infoContainer1,infoContainer2):
        IC1, IC2 = infoContainer1, infoContainer2
        IC1vals, IC2vals = IC1.trialValues, IC2.trialValues
        if len(IC1vals) != len(IC2vals):
            raise(Exception, "infoContainers must have "        \
                             "equal numbers of trials")
        self.maxVal = IC1.maxVal * IC2.maxVal
        
        self.trialValues =                                      \
            IC1.maxVal*IC2.trialValues + IC1.trialValues
        self.numTrials = len(self.trialValues)
        self._calculateNvec()

# 2021/6/18
class conditionalInfo(infoContainer):
    def __init__(self,infoContainerX,infoContainerY,stateIndexY):
        """
        Conditional distribution over X given Y = the state referred to
        by stateIndexY.
        
        (Has currently only been tested with discreteValues infoContainers.)
        """
        ICX, ICY = infoContainerX, infoContainerY
        ICXvals, ICYvals = ICX.trialValues, ICY.trialValues
        if len(ICXvals) != len(ICYvals):
            raise Exception("infoContainers must have equal numbers of trials")
        stateYtrials = find(ICYvals == stateIndexY)
        
        self.maxVal = ICX.maxVal
        self.trialValues = ICX.trialValues[stateYtrials]
        
        self.numTrials = len(self.trialValues)
        self._calculateNvec(possibleValues=range(ICX.maxVal))

# 7.20.2012
def mutualInfo(infoContainer1,infoContainer2,verbose=True,
    returnStds=True,**kwargs):
    """
    returnStds (True)       : Also return estimates of the
                              standard deviations for
                              S1, S2, and S12.
    """
    ICboth = jointInfo(infoContainer1,infoContainer2)
    S1 = infoContainer1.calculateEntropy(**kwargs)
    if verbose: print("S1,stdS1 =",S1)
    S2 = infoContainer2.calculateEntropy(**kwargs)
    if verbose: print("S2,stdS2 =",S2)
    S12 = ICboth.calculateEntropy(**kwargs)
    if verbose: print("S12,stdS12 =",S12)
    if returnStds:
        return S1[0] + S2[0] - S12[0], (S1[1],S2[1],S12[1])
    else:
        return S1[0] + S2[0] - S12[0]

