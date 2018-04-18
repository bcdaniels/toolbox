# Sparseness.py
#
# Bryan Daniels
# 7.28.2010
#
# 10.20.2015 Starting the transition to standalone code.
#

# See Olshausen and Field, Nature 381, 607 (1996).

import scipy
import os
from simplePickle import save,load # currently coming from Dropbox/Research/Conflict
import pylab
import scipy.linalg,scipy.optimize
import scipy.weave
try:
    import scipy.stats # for bootstrappedPval
except ValueError: # for bizarre error on sidious
    import scipy.stats
import copy

from SparsenessTools import * 

#import EntropyEstimates
#reload(EntropyEstimates)

dot = scipy.dot
inv = scipy.linalg.inv
sum = scipy.sum


def SExp(x):
    return 1.-scipy.exp(-x*x) # the 1. so minimum is zero
    
def SExpPrime(x):
    return 2.*x*scipy.exp(-x*x)
    
def SExpDPrime(x):
    return (2. - 4.*x*x)*scipy.exp(-x*x)
    
def SLog(x):
    return scipy.log(1 + x*x)
    
def SLogPrime(x):
    return 2.*x / (1 + x*x)
    
def SLogDPrime(x):
    x2 = x*x
    onePlusx2 = 1. + x2
    return -2.*(x2 - 1) / (onePlusx2*onePlusx2)
    
# 12.9.2010
def S2sqrt(x):
    return scipy.sqrt(x*x + 1.)

def S2(x):
    return S2sqrt(x) - 1.
    
def S2Prime(x):
    return x/S2sqrt(x)
    
def S2DPrime(x):
    return 1./S2sqrt(x)*(1. - x*x/(x*x + 1.))
    
def SAbs(x):
    return scipy.abs(x)
    
# 2.27.2011
def squash(x):
    return scipy.tanh(x)
    
def squashPrime(x):
    return 1./(scipy.cosh(x)**2)
    
def squashDPrime(x):
    return -2.*scipy.tanh(x)/(scipy.cosh(x)**2)

# 4.18.2011
def squashShifted(x):
    return 0.5*(scipy.tanh(x) + 1.)
    
def squashShiftedPrime(x):
    return 0.5*1./(scipy.cosh(x)**2)
    
def squashShiftedDPrime(x):
    return 0.5*-2.*scipy.tanh(x)/(scipy.cosh(x)**2)
    
def identity(x):
    return x
    
def identityPrime(x):
    return 1.
    
def identityDPrime(x):
    return 0.
    
def cost(a,phi,image):
    """
    Measures how well the model fits the image (data).
    (unsummed)
    """
    return (scipy.dot(a,phi) - image)**2



def timeSeriesDataMatrix(origDataMatrix,n,doShuffle=False,          \
    shuffleLast=False,seed=0):
    """
    Returns a matrix consisting of length-n contiguous subsets
    of the rows of the original matrix.
    
    doShuffle       : shuffles entire bouts in time
    """
    m = len(origDataMatrix)
    #rowLists = [ origDataMatrix[i:m-(n-1)+i] for i in range(n) ]
    if doShuffle:
        scipy.random.seed(seed)
        rowLists = [ shuffle(origDataMatrix[i:m-(n-1)+i])           \
                     for i in range(n) ]
    else:
        scipy.random.seed(seed)
        rowLists = [ origDataMatrix[i:m-(n-1)+i] for i in range(n) ]
    if shuffleLast:
        rowLists[-1] = shuffle(rowLists[-1])
    return scipy.array( [ scipy.concatenate( z )                    \
        for z in zip(*rowLists) ] )



# 8.10.2010
def PCAcoefficients(images,basis):
    N,m = scipy.shape(images)
    basis = scipy.array(basis)
    
    # subtract off mean to get shifted matrix b (copied from PCA)
    mn = scipy.mean(images,axis=0)
    mnMat = scipy.dot(scipy.ones(N).reshape(N,1),[mn])
    b = images - mnMat
    
    aBest = dot(b,basis.T)
    
    sp = SparsenessProblem(images,'SExp',0.,1.,len(basis),          \
        constrainRows=False,constrainCols=False,lmbda2=0.)
    aBest = [ sp.minimizeE(basis,image) for image in b ]
    
    return aBest

# 8.10.2010
def PCAavgCost(images,basis,noiseSigma=0.,seed=0):
    N,m = scipy.shape(images)
    basis = scipy.array(basis)
    
    # 11.2.2010 copied from analyzeSparsenessProblem.avgCostsOutOfSample
    scipy.random.seed(seed)
    basisSigma = scipy.sqrt(scipy.var(basis))
    noiseMat = noiseSigma*basisSigma*                               \
        scipy.random.normal(0.,1.,scipy.shape(basis))
    basis = basis + noiseMat
    
    # subtract off mean to get shifted matrix b (copied from PCA)
    mn = scipy.mean(images,axis=0)
    mnMat = scipy.dot(scipy.ones(N).reshape(N,1),[mn])
    b = images - mnMat

    aBest = PCAcoefficients(images,basis)

    variance = scipy.mean(scipy.var(images,axis=0))

    costs = [ (bImage - scipy.dot(a,basis))**2                      \
        for bImage,a in zip(b,aBest) ]
    return scipy.mean(costs)/variance



# 8.10.2010
# 9.29.2010 modified to more simply use PCA eigenvalues
def PCAavgCostShuffleMany(images,timeWindow,numRepeats=1,           \
    verbose=True,doTimeShuffle=False,doBoutShuffle=False,           \
    doOldWay=False):
    
    #scipy.random.seed(seed)
    avgCostList = []
    for i in range(numRepeats):
        if verbose:
            print "PCAavgCostShuffle: shuffle",i+1,"of",numRepeats
        if doBoutShuffle:
            images = shuffleParticipants(images,seed=i)
        dynamicMatrix = timeSeriesDataMatrix(images,timeWindow,     \
            doShuffle=doTimeShuffle,seed=i)
        svals,svecs = PCA(dynamicMatrix)
        variance = scipy.sum( scipy.var(dynamicMatrix,axis=0) )
        if doOldWay:
            avgCost = [ PCAavgCost(dynamicMatrix,svecs[:i])         \
                for i in range(1,len(svecs)) ]
        else:
            avgCost = 1. - scipy.cumsum(svals)/variance
        avgCostList.append(avgCost)
     
    return scipy.array(avgCostList)
    #return scipy.mean(avgCostList,axis=0),                          \
    #       scipy.var(avgCostList,axis=0)

# 9.30.2010
def plotPCAexcessError(images,timeWindow,numRepeats=100,            \
    verbose=False,plotTwo=False,newFigure=True,lineFormat='b-',     \
    plotAllShuffles=True,label='',doBoutShuffle=False):
    """
    Plots the excess error for bout data shuffled in time 
    (or individuals shuffled among bouts -- use doBoutShuffle)
    """
    
    avgCostList = PCAavgCostShuffleMany(images,timeWindow,          \
        numRepeats=1,verbose=verbose,doTimeShuffle=False,           \
        doBoutShuffle=False)
    avgCostListShuf = PCAavgCostShuffleMany(images,timeWindow,      \
        numRepeats=numRepeats,verbose=verbose,                      \
        doTimeShuffle=not doBoutShuffle,doBoutShuffle=doBoutShuffle)
    
    if newFigure:
        pylab.figure()
    if plotAllShuffles:
        [ pylab.plot(range(1,len(avgCostList[0])+1),                \
          avgCostShuf-avgCostList[0],',',color='0.7')               \
          for avgCostShuf in avgCostListShuf ]
    pylab.plot(range(1,len(avgCostList[0])+1),                      \
        scipy.mean(avgCostListShuf,axis=0)-avgCostList[0],          \
        lineFormat,label=label)
    pylab.plot([0.,len(avgCostList[0])+1],[0.,0.],'k-')
    pylab.xlabel('Basis size')
    pylab.ylabel('Excess PCA error on shuffled data')
    if plotTwo:
        if newFigure:
            pylab.figure()
        #[ pylab.plot(avgCostShuf,',',color='0.7')                  \
        #    for avgCostShuf in avgCostListShuf ]
        pylab.plot(range(1,len(avgCostList[0])+1),                  \
            avgCostList[0],'k',lw=2,label="Unshuffled")
        pylab.plot(range(1,len(avgCostList[0])+1),                  \
            scipy.mean(avgCostListShuf,axis=0),'r.--',              \
            label="Avg. shuffled")
        pylab.xlabel('Basis size')
        pylab.ylabel('PCA error')
        pylab.legend()

# 8.24.2010
def PCAmaxCompressibilityVsWindowSize(images,windowSizeRange=(1,10),\
    numRepeats=10,verbose=True,filePrefix=None):
    
    maxCompList,maxCompListStdErr,indexList = [],[],[]
    for ws in range(windowSizeRange[0],windowSizeRange[1]):
      avgCostList = PCAavgCostShuffleMany(images,ws,doShuffle=False,\
        verbose=verbose)
      avgCostListShuffle = PCAavgCostShuffleMany(images,ws,         \
        doShuffle=True,numRepeats=numRepeats,verbose=verbose)
      excessError = scipy.mean(avgCostListShuffle,axis=0) -         \
        scipy.array(avgCostList[0])
      excessErrorStd = scipy.std(avgCostListShuffle,axis=0)
      maxIndex = scipy.argsort(excessError)[-1]
      maxCompList.append(excessError[maxIndex])
      maxCompListStdErr.append(                                     \
        excessErrorStd[maxIndex]/scipy.sqrt(numRepeats))
      indexList.append(maxIndex)
      if filePrefix is not None:
        save(maxCompList,filePrefix+'_maxCompList.data')
        save(maxCompListStdErr,filePrefix+'_maxCompListStdErr.data')
        save(indexList,filePrefix+'_maxIndexList.data')
    
    return maxCompList,maxCompListStdErr,indexList
      
# 10.5.2010
def generateRandomBouts(fightList,num=1,seed=0):
    """
    Generates a set of random fight bouts with the same probability
    for each individual to appear as in the data.
    """
    scipy.random.seed(seed)
    frequencyData = scipy.sum(fightList,axis=0)
    probs = frequencyData/scipy.sum(frequencyData)
    m = len(probs)
    return [ scipy.ones(m)*(scipy.rand(m)<probs) for i in range(num) ]
   
# 11.17.2010
def fakeData(basis,basisProbs,num,seed=0):
    """
    basis           : list of basis vectors
    basisProbs      : list of probability of seeing each basis vector
    num             : length of output data
    
    Output data is constrained to be only zeros and ones.
    """
    data = []
    scipy.random.seed(seed)
    while len(data) < num:
      newFight = scipy.zeros(scipy.shape(basis)[-1])
      for j in range(len(basis)):
        if scipy.rand() < basisProbs[j]:
          newFight += basis[j]
      # make anything nonzero one
      newFight = scipy.nan_to_num( newFight/newFight )
      if sum(newFight) > 0: # only return nonzero data
        data.append(newFight)
    return scipy.array(data)
   
# 10.5.2010
def shuffleParticipants(fightList,seed=0):
    """
    Shuffles the fight participants among fights, leaving the
    frequency of each individual unchanged.
    """
    scipy.random.seed(seed)
    shuffledTranspose = [ shuffle(col) for col in scipy.transpose(fightList) ]
    return scipy.transpose(shuffledTranspose)

# 8.11.2010
# 10.20.2015 don't think this is used anymore
#def thresholdedRules(basis,threshold=1.,m=48,nameDict=matureNameDict):
#    
#    # 3.1.2011 check that nameDict size is correct
#    if len(nameDict.items()) != len(basis[0]):
#        raise Exception, "thresholdedRules exception:"                              \
#             +"Length of nameDict not equal to basis dimension."
#    
#    # perform thresholding
#    thresholdedBasis = thresholdMatrix(basis,threshold)
#        
#    # convert to human-readable rules
#    ruleList = []
#    for vec in thresholdedBasis:
#        #thisRule = scipy.repeat([],len(basis[0])/48)
#        thisRule = []
#        for k in range(len(thresholdedBasis[0])/m):
#            thisRule.append([])
#        for i in range(len(vec)):
#            if nameDict is None:
#                name = str(i%m)
#            else:
#                name = nameDict[i%m]
#            if vec[i] > 0.:
#                thisRule[i/m].append(name)
#            if vec[i] < 0.:
#                thisRule[i/m].append("-"+name)
#        ruleList.append(thisRule)
#    return ruleList



# 3.14.2011
#def thresholdMatrixNumNonzero(mat,numNonzero):
#    if numNonzero is not None:
#        thresh = thresholdFromNumNonzero(mat,numNonzero)
#        return thresholdMatrix(mat,thresh)
#    else:
#        return mat



# 8.27.2010
def entropyFreqListNaive(freqList):
    """
    In bits.
    """
    return -scipy.sum(freqList*scipy.nan_to_num(scipy.log2(freqList)))
    #ent = 0.
    #for freq in freqList:
    #    if freq != 0.:
    #        ent += freq*scipy.log2(freq)
    #return -ent
    


    
# 10.26.2010
# 10.20.2015 don't think this is used anymore
#def entropyNem(fightList,returnErrorBar=True):
#    """
#    In bits.
#    """
#    freqData = freqDataFromFightList(fightList)
#    mean = EntropyEstimates.meanEntropyNem(freqData)
#    log2,e = scipy.log2,scipy.e
#    if returnErrorBar:
#        s2s0 = EntropyEstimates.s2s0Nem(freqData)
#        errorBar = scipy.sqrt( s2s0 - mean**2 )
#        return mean*log2(e), errorBar*log2(e)
#    else:
#        return mean*log2(e)

# 9.3.2010
# 10.20.2015 don't think this is used anymore
#def PCAentropy(fightList):
#    """
#    Finds entropy assuming multidimensional Gaussian distribution,
#    in bits.
#    (prop. to the log of the product of the PCA eigenvalues)
#    """
#    n = scipy.shape(fightList)[1]
#    log2,pi,e = scipy.log2,scipy.pi,scipy.e
#    
#    vals,vecs = PCA(fightList)
#    return 0.5*( n*log2(2.*pi*e) + scipy.sum(log2(vals)) )

# 9.6.2010
# 10.20.2015 don't think this is used anymore
#def excessPCAentropy(origFightList,windowSize,numShuffles=10,       \
#    shuffleLast=True,doBoutShuffle=False):
#    fightList = timeSeriesDataMatrix(origFightList,windowSize)
#    unshuffledEntropy = PCAentropy(fightList)
#    excessEntropyList = []
#    for i in range(numShuffles):
#      if shuffleLast:
#        fightListShuf = timeSeriesDataMatrix(origFightList,         \
#                      windowSize,shuffleLast=True,                  \
#                      seed=i) #doShuffle=True
#      elif doBoutShuffle:
#        fightListShuf = shuffleParticipants(origFightList,seed=i)
#      else:
#        print "excessPCAentropy error: no shuffle type defined."
#        return 0
#      shuffledEntropy = PCAentropy(fightListShuf)
#      excessEntropyList.append(shuffledEntropy - unshuffledEntropy)
#    return excessEntropyList

# 9.6.2010
# 11.2.2011 updated
# 10.20.2015 don't think this is used anymore
#def excessEntropy(origFightList,windowSize,numShuffles=10,useNemEnt=False):
#    if useNemEnt: entropy = lambda fights: entropyNem(fights,False)
#    else: entropy = entropyNaive
#    fightList = timeSeriesDataMatrix(origFightList,windowSize)
#    unshuffledEntropy = entropy(fightList)
#    excessEntropyList = []
#    for i in range(numShuffles):
#      fightListShuf = timeSeriesDataMatrix(origFightList,           \
#                      windowSize,shuffleLast=True,seed=i) #doShuffle=True
#      shuffledEntropy = entropy(fightListShuf)
#      excessEntropyList.append(shuffledEntropy - unshuffledEntropy)
#    return excessEntropyList



# 8.1.2011
# 10.20.2015 don't think this is used anymore
#def randomCombinationBasis(uniqueGroupsList,uglFreqs,n,seed=None,   \
#    maxiter=1000):
#    """
#    Creates a basis consisting of vectors chosen from uniqueGroupsList
#    with frequency given by uglFreqs.  The basis is constrained
#    to have n nonzero elements (by throwing away bases that don't
#    have this property).
#    """
#    if seed is not None:
#        scipy.random.seed(seed)
#    if len(scipy.shape(uniqueGroupsList[0])) == 2:
#        uniqueGroupsList = [ scipy.mean(groups,axis=0)              \
#                             for groups in uniqueGroupsList ]
#    numNonzero = -1
#    i = 0
#    while numNonzero != n:
#      i += 1
#      trialBasis = []
#      for group,freq in zip(uniqueGroupsList,uglFreqs):
#        if scipy.random.random() < freq:
#          trialBasis.append(group)
#      numNonzero = scipy.sum(abs(scipy.array(trialBasis))>0.)
#      #print numNonzero
#      if i > maxiter:
#        raise Exception, "maxiter ("+str(maxiter)+") exceeded"
#    return trialBasis



class SparsenessProblem:

    def __init__(self,images,SfuncName,                             \
        lmbda,sigma,N,verbose=False,veryVerbose=False,              \
        veryveryVerbose=False,seed=0,c=1.,                          \
        tol=1.e-3,maxIter=100,epsCost=1.e-20,                       \
        gradDescStepSize=1.e-2,constrainRows=True,                  \
        constrainCols=False,lmbda2=None,sigma2=1.,                  \
        phiStart=None,aStart=None,randomStart=True,maxiter=None,    \
        randomAMult=1.,randomPhiMult=1.,avextol=1.e-9,              \
        orthogStart=False,squashFuncName='identity'):
        """
        images          : array of one-dimensional vectors
        SfuncName       : string giving the name of the function to
                          calculate the sparseness penalty.
                          SfuncName+'Prime' and SfuncName+'DPrime'
                          should give, respectively, the first and
                          second derivative of Sfunc.
        lmbda           : sparseness parameter
                          (rescaled to variance in data 9.20.10)
                          (rescaled with N 9.21.10)
                          (NOT rescaled with N with lmbda2 11.18.10)
        N               : number of basis functions
        c (1)           : maximum squared norm of basis vectors
                          (rescaled to variance in data 9.20.10;
                           rescaled to shape of phi 10.11.10)
        tol (1e-3)      : minimization stops when the change in 
                          average cost is less than tol*cost
        maxIter (100)   : maximum number of minimization iterations
        epsCost (1e-20) : minimization stops if cost is less than
                          epsCost
        gradDescStepSize: step size for gradient descent
                  (1e-2)
        constrainRows   : If True, constrain squared norm of rows 
                  (True)  of phi (using Lagrange multiplier method).
        constrainCols   : If True, constrain squared norm of columns
                          of phi (using projected gradient descent).
        lmbda2          : If not None, impose sparseness on both
                          elements of coefficient and basis matrices
        maxiter         : maximum number of optimization iterations.
                          (as of 12.22.10, only implemented in
                          'double-sparseness' case)
        randomAMult     : 12.23.2010 with randomStart=True, starting
                          basis elements will be drawn from a uniform
                          distribution in range 
                          [-randomAMult*sigma, +randomAMult*sigma]
        avextol (1e-9)
        orthogStart     : if True (and randomStart is True), start
                (False)   from a random orthonormal basis 
                          (and if N==ell, start phi such that it 
                          reconstructs the data exactly) 2.22.2011
        squashFuncName  : used to 'squash' the output of coeffs*basis.
                          As of 2.27.11, only implemented for
                          double sparseness.
        """
        self.images = images
        self.variance = scipy.mean( scipy.var(                          \
            self.images,axis=1) ) # should this be axis 1 or 0?
                                  # prob. depends on whether I'm doing
                                  # it 'forward' or 'backward'
        self.Sfunc = eval(SfuncName)
        self.SfuncPrime = eval(SfuncName+'Prime')
        self.SfuncDPrime = eval(SfuncName+'DPrime')
        self.N = N
        self.m = len(images[0])
        self.ell = len(images)
        # changed scaling 2.28.2011
        #self.lmbda = self.variance*self.ell*self.m*lmbda
        #if lmbda2 is None:
        #    self.lmbda = self.lmbda/N
        #    self.lmbda2 = None
        #else:
        #    self.lmbda2 = self.variance*self.ell*self.m*lmbda2/self.m
        self.lmbda = self.m*lmbda
        self.lmbda2 = lmbda2
        self.sigma = sigma
        # 10.29.10 need to change scaling depending on whether you
        # take the transpose of the data (whether rows or columns
        # contain the images in self.images)
        self.sigma2 = sigma2
        self.verbose = verbose
        self.veryVerbose = veryVerbose
        self.veryveryVerbose = veryveryVerbose
        self.seed = seed
        self.tol = tol
        self.maxIter = maxIter
        self.gradDescStepSize = gradDescStepSize
        self.constrainRows = constrainRows
        self.constrainCols = constrainCols
        self.phiStart = phiStart
        self.aStart = aStart
        self.randomStart = randomStart
        self.maxiter = maxiter
        self.randomAMult = randomAMult
        self.randomPhiMult = randomPhiMult
        self.avextol = avextol
        self.orthogStart = orthogStart
        self.squashFunc = eval(squashFuncName)
        self.squashFuncPrime = eval(squashFuncName+'Prime')
        self.squashFuncDPrime = eval(squashFuncName+'DPrime')
        
        if self.constrainRows:
            self.c = self.variance*c*self.m
        else:
            self.c = self.variance*c*self.N
        
        self.epsCost = epsCost
        
        self.frequencyData = sum(self.images,axis=0)
        self.probabilityFreqData = self.frequencyData/len(self.images)
        

    def E(self,phi,image,a):
        lmbda,sigma = self.lmbda,self.sigma
        return scipy.sum( cost(a,phi,image) )                           \
            + lmbda*scipy.sum( self.Sfunc(a/sigma) )
        
    # 7.29.2010
    # for use in the Lagrange multiplier method in LeeBatRai06
    def D(self,lmbdaVec,A,images):
        X = images.T
        S = A.T
        tr = scipy.trace
        Lmbda = scipy.diagflat(lmbdaVec)
        return                                                          \
              - tr( dot(inv( dot(S,S.T) + Lmbda ),                      \
                         dot( dot(X,S.T).T, dot(X,S.T) ) ) )            \
              - tr( self.c*Lmbda )
              # + tr( dot(X.T,X) )
        #return                                                         \
        #      - tr( dot( dot(X,S.T),                                   \
        #                 dot( inv( dot(S,S.T) + Lmbda ),               \
        #                      dot(X,S.T).T ) ) )                       \
        #      - tr( self.c*Lmbda )
        #      #+ tr( dot(X.T,X) ) 
              
    def Dprime(self,lmbdaVec,A,images):
        X = images.T
        S = A.T
        Lmbda = scipy.diagflat(lmbdaVec)
        return                                                          \
         sum( dot( dot(X,S.T), inv( dot(S,S.T) + Lmbda ) )**2, axis=0 ) \
         - self.c
        
    def Dhess(self,lmbdaVec,A,images):
        X = images.T
        S = A.T
        Lmbda = scipy.diagflat(lmbdaVec)
        return -2.*dot(                                                 \
            dot( inv( dot(S,S.T) + Lmbda ), dot(X,S.T).T ),             \
            dot( dot(X,S.T), inv( dot(S,S.T) + Lmbda ) ) ) *            \
          inv( dot(S,S.T) + Lmbda )
        
    def minimizeE(self,phi,image,aStart=None):
        """
        Returns the best a given phi and image.  Minimizes using 
        conjugate gradient.
        
        Derivative assuming sums of squares for cost.
        """
        # 11.9.2010
        a0 = aStart
        if aStart is None:
            a0 = scipy.zeros(self.N)
        # these two are actually about the same speed
        #minimizer = scipy.optimize.fmin_cg 
        minimizer = scipy.optimize.fmin_ncg
        Efunc = lambda a: self.E(phi,image,a)
        Eprime = lambda a: -2.*scipy.dot(phi,image-scipy.dot(a,phi))    \
            + self.lmbda/self.sigma*self.SfuncPrime(a/self.sigma)
        Ehess = lambda a: 2.*dot(phi,phi.T)                             \
            + self.lmbda/(self.sigma*self.sigma)                        \
             *scipy.diagflat(self.SfuncDPrime(a/self.sigma)) # 8.4.10
        a0 = scipy.zeros(self.N)
        #*** (look out for finite difference problems when checking SfuncPrime)
        #print "check",scipy.optimize.check_grad(Efunc,Eprime,a0)
        #phiTestDir = a0[:]
        #phiTestDir[3] = 100.
        #eps = 1e-5
        #phiTestDirNormed = phiTestDir/scipy.sqrt(scipy.sum(phiTestDir*phiTestDir))
        #deriv = (Eprime(a0+eps*phiTestDirNormed)-Eprime(a0-eps*phiTestDirNormed))/(2.*eps)
        #derivHess = scipy.dot(Ehess(a0),phiTestDirNormed)
        #print "check hess",scipy.sum((deriv-derivHess)**2)
        #***
        return minimizer(Efunc,a0,fprime=Eprime,fhess=Ehess,            \
            disp=self.veryveryVerbose)
        
    def phiMinimizer(self,aBest,phiStart=None,testImages=None,          \
        returnEVal=False):
        """
        Uses either lagrange multiplier method or gradient 
        descent method, depending on self.constrainRows.
        
        returnEVal currently only supported for double sparseness.
        (lmbda2 != None)
        """
        phi = phiStart
        
        if testImages is None:
            testImages = self.images
        m = scipy.shape(testImages)[-1]
        
        # 2.27.2011 for back-compatibility
        if not hasattr(self,'squashFunc'):
            squashFuncName = 'identity'
            self.squashFunc = eval(squashFuncName)
            self.squashFuncPrime = eval(squashFuncName+'Prime')
            self.squashFuncDPrime = eval(squashFuncName+'DPrime')
        
        if phiStart is None:
            # Note: doesn't matter if self.constrainRows = True
            # 11.9.2010 below: possibly temporary change to phi = zero
            #           when lmbda2 is not None
            phi = self.randomPhi(m=m)
        
        # use lagrange multiplier method found in LeeBatRai06
        if self.constrainRows:
            #minimizer = scipy.optimize.fmin_cg
            minimizer = scipy.optimize.fmin_ncg  
            
            S = aBest.T
            X = testImages.T
            Dfunc =  lambda lmbdaVec: -self.D(lmbdaVec,aBest,testImages)
            Dprime = lambda lmbdaVec: -self.Dprime(lmbdaVec,aBest,testImages)
            Dhess = lambda lmbdaVec: -self.Dhess(lmbdaVec,aBest,testImages)
            def Dhess_p(lmbdaVec,p):
                if scipy.log10(max(abs(p))) > 10.:
                    raise Exception,                                    \
                        "phiMinimizer error: parameter evaporation?"
                return scipy.dot(Dhess(lmbdaVec),p)
            lmbdaVec0 = scipy.zeros(self.N) # 11.4.2010 +1e-9
            lmbdaVecMax = minimizer(Dfunc,lmbdaVec0,fprime=Dprime,      \
                fhess=None,disp=self.veryVerbose,fhess_p=Dhess_p)
            LmbdaMax = scipy.diagflat(lmbdaVecMax)
            phi = dot( inv( dot(S,S.T) + LmbdaMax ), dot(X,S.T).T ) 
            # (if using lagrange multiplier method, we should check
            #  to make sure the constraint is actually satisfied)
            meanPhiNorm =                                               \
                scipy.mean(scipy.array([normSq(p) for p in phi]))
            if meanPhiNorm/self.c > 2:
                print "phiMinimizer: Warning! "                         \
                    +"Mean basis vector L2 norm =",meanPhiNorm 
        
        # 9.13.2010 use brute-force gradient descent with projection
        # (puts constraint on columns of phi instead of rows)
        elif self.constrainCols:
          for j in range(5000):
              oldPhi = copy.copy(phi)
              
              deltaPhi = self.gradDescStepSize / len(testImages) *         \
                ( scipy.dot(aBest.T,testImages - scipy.dot(aBest,phi)) )   \
                # Divide by len(images) for average.
              phi += deltaPhi
              
              # project onto space with constant size for columns of phi
              normPhi = scipy.sqrt( scipy.sum(phi*phi,axis=0) )
              normPhiMat = \
                scipy.repeat( normPhi, self.N ).reshape(m,self.N)
              phi = scipy.sqrt(self.c) * phi / normPhiMat.T
              
              # ** debug
              if j%500 == 0:
                actualDeltaPhi = phi - oldPhi
                print j, "norm deltaPhi =", scipy.sqrt(normSq(actualDeltaPhi))
              #normPhiNew = scipy.sqrt( scipy.sum(phi*phi,axis=0) )
              #print "Average norm phi =", scipy.mean(normPhiNew)
                    
        elif self.lmbda2 is not None:
          # in self.minimizeE, we loop over images, performing a fit
          # of coefficients for each image.  here, we want to loop
          # over 'individuals', performing a fit of basis vector
          # components for each individual.  Thus we reverse the roles
          # of phi and a. (see notes 10.28.2010)
          for i in range(m): 
            # 11.9.2010 possibly temporary change to phi = zero
            #phi = scipy.zeros_like(phi)
    
            minimizer = scipy.optimize.fmin_ncg
            image = testImages[:,i].reshape(self.ell,1)
            def Efunc(phiCol):
              if scipy.isfinite(phiCol).all():
                phiCol = phiCol.reshape(self.N,1)
                SQaphiMinusI = self.squashFunc(dot(aBest,phiCol))-image
                # *****************
                #print "phiCol =",phiCol[:,0]
                #print "E =",scipy.sum( SQaphiMinusI*SQaphiMinusI )               \
                #      + self.lmbda2*scipy.sum( self.Sfunc(phiCol/self.sigma2) )
                # *****************
                return scipy.sum( SQaphiMinusI*SQaphiMinusI )               \
                  + self.lmbda2*scipy.sum( self.Sfunc(phiCol/self.sigma2) )
              else:
                raise Exception
            #self.E(aBest,image,phiCol,useLmbda2=True)
            def Eprime(phiCol):  
                #image-scipy.dot(aBest,phiCol.reshape(self.N,1))
                phiCol = phiCol.reshape(self.N,1)
                SQaphiMinusI = self.squashFunc(dot(aBest,phiCol))-image
                SQPaphi = self.squashFuncPrime(dot(aBest,phiCol))
                return (2.*scipy.dot(aBest.T,SQaphiMinusI*SQPaphi)
                + self.lmbda2/self.sigma2*                                  \
                    self.SfuncPrime(phiCol/self.sigma2))                    \
                  .reshape(self.N)
            def Ehess(phiCol): 
                phiCol = phiCol.reshape(self.N,1)
                SQaphiMinusI = self.squashFunc(dot(aBest,phiCol))-image
                SQPaphi = self.squashFuncPrime(dot(aBest,phiCol))
                SQDPaphi = self.squashFuncDPrime(dot(aBest,phiCol))
                SQPcrossTerm = SQPaphi*SQPaphi + SQaphiMinusI*SQDPaphi
                return 2.*dot(aBest.T,aBest*SQPcrossTerm)                   \
                + self.lmbda2/(self.sigma2*self.sigma2)                     \
                 *scipy.diagflat(self.SfuncDPrime(phiCol/self.sigma2)) 
            
            #*** (look out for finite difference problems when checking SfuncPrime)
            #print "lmbda2",self.lmbda2
            #print "check",scipy.optimize.check_grad(Efunc,Eprime,phiCol0)
            #phiTestDir = phiCol0[:]
            #phiTestDir[3] = 100.
            #eps = 1e-6
            #phiTestDirNormed = phiTestDir/scipy.sqrt(scipy.sum(phiTestDir*phiTestDir))
            #deriv = (Eprime(phiCol0+eps*phiTestDirNormed)-Eprime(phiCol0-eps*phiTestDirNormed))/(2.*eps)
            #erivHess = scipy.dot(Ehess(phiCol0),phiTestDirNormed)
            #print "check hess",scipy.sum((deriv-derivHess)**2)
            #die
            #***
            phiCol0 = phi[:,i] #.reshape(self.N,1)
            foundSolution = False
            # 9.24.2012 ************************
            minimizer = scipy.optimize.fmin_cg
            # **********************************
            while foundSolution is False:
                try:
                  if minimizer is scipy.optimize.fmin_cg: # 9.24.2012
                    minimizerOutput = minimizer(Efunc,phiCol0,fprime=Eprime,    \
                      disp=self.veryveryVerbose,full_output=1,                  \
                      gtol=self.avextol)
                  else:
                    minimizerOutput = minimizer(Efunc,phiCol0,fprime=Eprime,    \
                      fhess=Ehess,disp=self.veryveryVerbose,full_output=1,      \
                      avextol=self.avextol)
                  phi[:,i] = minimizerOutput[0]
                  EVal = minimizerOutput[1]
                  foundSolution = True
                except KeyboardInterrupt:
                  raise
                except TypeError:
                  #raise # for debugging 
                  if False: # new 9.24.2012
                    print "phiMinimizer: Minimization problem.  Trying a different minimizer (fmin_cg)."
                    minimizer = scipy.optimize.fmin_cg
                  else:
                    print "phiMinimizer: Minimization problem.  Trying random phiStart."
                    phiCol0 = 2.*self.sigma2*(scipy.random.random(scipy.shape(phiCol0)) - 0.5)
                except:
                  raise
                  print "phiMinimizer Warning: minimizer error!  Returning zeros"
                  phi = scipy.zeros_like(phi)
            
                    
        else:
            raise Exception, "sparsenessProblem.phiMinimizer error: "       \
                            +"no constraint defined."
        if returnEVal:
            return phi,EVal
        else:
            return phi
        
    # 11.3.2010
    def randomPhi(self,N=None,m=None):
        if N is None:
            N = self.N
        if m is None:
            m = self.m
        scipy.random.seed(self.seed)
        if not hasattr(self,'sigma2'): # for back-compatibility
          self.sigma2 = 1.
        phi = 2.*self.sigma2*(scipy.rand(N,m) - 0.5)
        
        # ********************* testing 2.20.2011
        #phi = 2.*(scipy.rand(N,m) - 0.5)
        # **********************************************
        
        # rescale to make each basis vector squared norm equal c
        # removed 11.9.2010
        #for i in range(len(phi)):
        #  if normSq(phi[i]) != 0.:
        #    phi[i] = scipy.sqrt(self.c)                                 \
        #            * phi[i] / scipy.sqrt(normSq(phi[i]))
        #  else:
        #    phi[i] = scipy.sqrt(self.c/self.m)*scipy.ones(self.m)
            
        return phi
    
    def unpack(self,x):
        return scipy.reshape(x[:self.N*self.m],[self.N,self.m]),    \
               scipy.reshape(x[self.N*self.m:],[self.ell,self.N])
    def pack(self,phi,a):
        return scipy.concatenate(                                   \
            [scipy.reshape(phi,[self.N*self.m]),                    \
             scipy.reshape(a,[self.ell*self.N])] )
    def Efunc(self,x):
        phi,a = self.unpack(x)
        return scipy.sum( (scipy.dot(a,phi)-self.images)**2 )       \
        + self.lmbda *scipy.sum( self.Sfunc(a/self.sigma) )         \
        + self.lmbda2*scipy.sum( self.Sfunc(phi/self.sigma2) )
    #self.E(aBest,image,phiCol,useLmbda2=True)
    def Eprime(self,x):
        phi,a = self.unpack(x)
        dEdPhi =                                                    \
          2.*scipy.dot(a.T,scipy.dot(a,phi)-self.images)            \
        + self.lmbda2/self.sigma2 * self.SfuncPrime(phi/self.sigma2)
        dEda =                                                      \
          2.*scipy.dot(scipy.dot(a,phi)-self.images,phi.T)          \
        + self.lmbda/self.sigma * self.SfuncPrime(a/self.sigma)
        return self.pack(dEdPhi,dEda)
    def phiInd(self,a,b):
        return a*self.m + b
    def aInd(self,c,d):
        return self.N*self.m + c*self.N + d
    def Ehess(self,x):
        phi,a = self.unpack(x)
        n = self.N*(self.m+self.ell)
        hess = scipy.zeros([n,n])
        aTa = scipy.dot(a.T,a)
        phiphiT = scipy.dot(phi,phi.T)
        aPhi = scipy.dot(a,phi)
        N,m,ell = self.N,self.m,self.ell
        images = self.images
        lmbda,lmbda2 = self.lmbda,self.lmbda2
        sigma,sigma2 = self.sigma,self.sigma2
        SDPrimePhi = self.SfuncDPrime(phi/self.sigma2)
        SDPrimeA = self.SfuncDPrime(a/self.sigma)
        
        aMat = a
        
        code = """
        int index1,index2;
        float lmbdaF = float (lmbda);
        float lmbda2F = float (lmbda2);
        for (int A=0; A<N; A++){
          for (int c=0; c<N; c++){
            for (int b=0; b<m; b++){
              double phiphi = 2.*aTa(A,c);
              if (A == c)
                phiphi += lmbda2F/(sigma2*sigma2)*SDPrimePhi(A,b);
              index1 = c*m + b;
              index2 = A*m + b;
              hess(index1,index2) = phiphi;
            }
            for (int b=0; b<ell; b++){ 
              double aa = 2.*phiphiT(A,c);
              if (A == c)
                aa += lmbdaF/(sigma*sigma)*SDPrimeA(b,A);
              index1 = N*m + b*N + A;
              index2 = N*m + b*N + c;
              hess(index1,index2) = aa;
            }
          }
        }
        for (int A=0; A<N; A++){
          for (int b=0; b<m; b++){
            for (int c=0; c<ell; c++){
              for (int d=0; d<N; d++){
                double crossterm = 2.*aMat(c,A)*phi(d,b);
                if (A == d)
                  crossterm += 2.*(aPhi(c,b)-images(c,b));
                index1 = A*m + b;
                index2 = N*m + c*N + d;
                hess(index1,index2) = crossterm;
                hess(index2,index1) = crossterm;
              }
            }
          }
        }
        """
        
        if False:
         # phi-phi, a-a
         for A in range(self.N):
          for c in range(self.N):
            for b in range(self.m): # phi-phi
              phiphi = 2.*aTa[A,c]
              if A == c:
                phiphi += self.lmbda2/(self.sigma2**2)              \
                  *self.SfuncDPrime(phi[A,b]/self.sigma2)
              hess[phiInd(c,b),phiInd(A,b)] = phiphi
            for b in range(self.ell): # a-a
              aa = 2.*phiphiT[A,c]
              if A == c:
                aa += self.lmbda/(self.sigma**2)                    \
                  *self.SfuncDPrime(a[b,A]/self.sigma)
              hess[aInd(b,A),aInd(b,c)] = aa
         # phi-a, a-phi
         for A in range(self.N):
          for b in range(self.m):
            for c in range(self.ell):
              for d in range(self.N):
                crossterm = 2.*a[c,A]*phi[d,b]
                if A == d:
                  crossterm += 2.*(aPhi[c,b]-self.images[c,b])
                hess[phiInd(A,b),aInd(c,d)] = crossterm
                hess[aInd(c,d),phiInd(A,b)] = crossterm
        
        if True:
          err = scipy.weave.inline(code,                              \
            ['hess','aMat','phi','aTa','phiphiT','aPhi','N','m','ell', \
            'images','lmbda','lmbda2','sigma','sigma2',             \
            'SDPrimePhi','SDPrimeA'],                               \
            type_converters = scipy.weave.converters.blitz)
        return hess
    
    # 11.18.2010
    def Ehess_p(self,x,p):
        phi,a = self.unpack(x)
        xphi,xa = self.unpack(p)
        lmbda,lmbda2 = self.lmbda,self.lmbda2
        sigma,sigma2 = self.sigma,self.sigma2
        
        aphiMinusImages = dot(a,phi)-self.images
        
        phiphiphi = 2.*dot(dot(a.T,a),xphi)                                 \
            + lmbda2/(sigma2**2)*self.SfuncDPrime(phi/sigma2)*xphi
        phiaa = 2.*dot(dot(a.T,xa),phi)                                     \
              + 2.*dot(xa.T,aphiMinusImages)
        aphiphi = 2.*dot(dot(a,xphi),phi.T)                                 \
                + 2.*dot(aphiMinusImages,xphi.T)
        aaa = 2.*dot(xa,dot(phi,phi.T))                                     \
            + lmbda/(sigma**2)*self.SfuncDPrime(a/sigma)*xa
            
        return self.pack(phiphiphi+phiaa,aphiphi+aaa)
                
    
    # 2.18.2011
    # 2.21.2011
    def ENormFunc(self,x):
        phi,a = self.unpack(x)
        nrm = scipy.sqrt( sum(a*a) )
        aN = a / nrm
        phiN = phi * nrm
        squash = self.squashFunc
        return sum( (squash(dot(aN,phiN))-self.images)**2 )                 \
        + self.lmbda *sum( self.Sfunc(aN/self.sigma) )                      \
        + self.lmbda2*sum( self.Sfunc(phiN/self.sigma2) )
    # 3.31.2011
    def ENormFunc_unsummed(self,phi,a,images):
        """
        Returns ENormFunc for each fight.
        """
        #phi,a = self.unpack(x)
        nrm = scipy.sqrt( sum(a*a) )
        aN = a / nrm
        phiN = phi * nrm
        squash = self.squashFunc
        return  [ sum( (squash(dot(aN,phiN))-images)**2, axis=0 ),     \
                 self.lmbda *sum( self.Sfunc(aN/self.sigma) ),              \
                 self.lmbda2*sum( self.Sfunc(phiN/self.sigma2) ,axis=0) ]
    def ENormPrime(self,x):  # 2.21.11 checked at one unoptimized location
        phi,a = self.unpack(x)
        nrm = scipy.sqrt( sum(a*a) )
        aN = a / nrm
        phiN = phi * nrm
        squash = self.squashFunc
        squashP = self.squashFuncPrime
        SQaphiMinusI = squash(dot(aN,phiN))-self.images
        SQPaphi = squashP(dot(aN,phiN))
        SprimeAN = self.SfuncPrime(aN/self.sigma)
        SprimePhiN = self.SfuncPrime(phiN/self.sigma2)
        dEdPhi = 2.*dot(a.T,SQaphiMinusI*SQPaphi)                           \
          + self.lmbda2*nrm/self.sigma2 * SprimePhiN
        dEda = 2.*dot(SQaphiMinusI*SQPaphi,phi.T)                           \
          + self.lmbda/self.sigma/nrm * ( SprimeAN - aN*sum(aN*SprimeAN) )  \
          + self.lmbda2/self.sigma2 * aN * sum(phi*SprimePhiN)
        return self.pack(dEdPhi,dEda)
    def ENormHess_p(self,x,p): # 2.21.11 checked in one random direction
        phi,a = self.unpack(x)
        nrm = scipy.sqrt( sum(a*a) )
        aN = a / nrm
        phiN = phi * nrm
        squash = self.squashFunc
        squashP = self.squashFuncPrime
        squashDP = self.squashFuncDPrime
        SQaphiMinusI = squash(dot(aN,phiN))-self.images
        SQPaphi = squashP(dot(aN,phiN))
        SQDPaphi = squashDP(dot(aN,phiN))
        SQPcrossTerm = SQPaphi*SQPaphi + SQaphiMinusI*SQDPaphi
        xphi,xa = self.unpack(p)
        lmbda,lmbda2 = self.lmbda,self.lmbda2
        sigma,sigma2 = self.sigma,self.sigma2
        
        adotPhi = dot(aN,phiN)
        #aphiMinusImages = adotPhi - self.images
        
        sPrimeAN = self.SfuncPrime(aN/sigma)
        sDPrimeAN = self.SfuncDPrime(aN/sigma)
        sPrimePhiN = self.SfuncPrime(phiN/sigma2)
        sDPrimePhiN = self.SfuncDPrime(phiN/sigma2)
        aNxa = sum(aN*xa)
        
        phiphiphi = 2.*dot(a.T,dot(a,xphi)*SQPcrossTerm)                    \
            + lmbda2*nrm*nrm/(sigma2**2)*sDPrimePhiN*xphi
        phiaa = 2. * ( dot(a.T,dot(xa,phi)*SQPcrossTerm)                    \
                     + dot(xa.T,SQaphiMinusI*SQPaphi) )                     \
              + lmbda2/sigma2*aNxa * ( sPrimePhiN                           \
                                     + phiN/sigma2*sDPrimePhiN )
        aphiphi = 2. * ( dot(dot(a,xphi)*SQPcrossTerm,phi.T)                \
                       + dot(SQaphiMinusI*SQPaphi,xphi.T) )                 \
                + lmbda2/sigma2*aN * ( sum(xphi*sPrimePhiN)                 \
                                     + sum(xphi*phiN*sDPrimePhiN)/sigma2 )
        aaa = 2.*dot(dot(xa,phi)*SQPcrossTerm,phi.T)                        \
            + lmbda/(nrm*nrm*sigma) * (                                     \
                    xa*( 1./sigma*sDPrimeAN - sum(aN*sPrimeAN) )            \
                  - aN/sigma*sDPrimeAN*aNxa                                 \
                  - aN/sigma*sum(xa*aN*sDPrimeAN)                           \
                  - aN*sum(xa*sPrimeAN)                                     \
                  - sPrimeAN*aNxa                                           \
                  + aN*sum(3.*aN*sPrimeAN+1./sigma*aN*aN*sDPrimeAN)*aNxa )  \
            + lmbda2/sigma2 * (                                             \
                    1./nrm*(xa - aN*aNxa)*sum(phi*sPrimePhiN)               \
                  + aN*aNxa/sigma2*sum(phi*phi*sDPrimePhiN) )
            
        return self.pack(phiphiphi+phiaa,aphiphi+aaa)
        
    
    def findSparseRepresentation(self,returnBasis=False):
        
        if self.lmbda2 is None:
          # initialize basis vectors
          phi = self.randomPhi()
          #phi = scipy.diagflat(                                            \
          #    scipy.ones(max(self.N,self.m)))[:self.N,:self.m]
        
          aBest = scipy.zeros([self.ell,self.N])

          cost,deltaCost = scipy.inf,scipy.inf
          i = 0
          while (abs(deltaCost) >= self.tol*cost) and (i < self.maxIter)    \
            and (cost > self.epsCost):
            i += 1
            aBestOld = aBest[:]
            phiOld = phi[:]
            costOld = copy.copy(cost)
            
            if self.veryVerbose:
                print "Performing coefficient minimization", i
            aBest = [ self.minimizeE(phi,image,aStart=a)                    \
                for image,a in zip(self.images,aBestOld) ]
            aBest = scipy.array(aBest)
            
            if self.veryVerbose:
                print "Average group size =",self.avgGroupSize(aBest)
                print "Performing basis minimization", i
            phi = self.phiMinimizer(aBest,phiStart=scipy.zeros_like(phiOld)) # *** 11.17.10 phiStart=phiOld
            
            cost = self.avgCost(phi,aBest)
            deltaCost = costOld - cost
            if self.veryVerbose:
                #print "singvals:", scipy.linalg.svd( inv( dot(S,S.T)       \
                #    + LmbdaMax ) )[1]
                #print "Delta phi =", normSq(phiOld-phi)
                print "Average cost =", cost

        # 11.17.2010
        # minimize in the full a,phi space
        elif self.lmbda2 is not None:
            
                        
                          
            #test *********
            #eps = 1e-4
            #xTest = pack(phi-100.,aBest+0.5)
            #xTestDir = xTest[:]
            #xTestDir[3] = 100.
            #xTestDirNormed = xTestDir/scipy.sqrt(scipy.sum(xTestDir*xTestDir))
            #deriv = (Eprime(xTest+eps*xTestDirNormed)-Eprime(xTest-eps*xTestDirNormed))/(2.*eps)
            #derivHess = scipy.dot(Ehess(xTest),xTestDirNormed)
            #derivHessp = Ehess_p(xTest,xTestDirNormed)
            #print scipy.sum((deriv-derivHess)**2)
            #print scipy.sum((deriv-derivHessp)**2)
            #pylab.figure()
            #pylab.plot(deriv)
            #pylab.plot(derivHess)
            #pylab.plot(derivHessp)
            #pylab.figure()
            #pylab.plot(deriv/derivHess)
            #pylab.plot(deriv/derivHessp)
            ##plotMatrix(Ehess(xTest))
            #die
            #
            #def derivSingle(i):
            #    epsVec = scipy.zeros(len(xTest))
            #    epsVec[i] = eps
            #    return (Efunc(xTest+epsVec)-Efunc(xTest-epsVec))/(2.*eps)
            #deriv = [ derivSingle(i) for i in range(len(xTest)) ]
            #deriv2 = scipy.optimize.approx_fprime(xTest,Efunc,eps)
            #pylab.figure()
            #pylab.plot(deriv)
            #pylab.plot(Eprime(xTest))
            #pylab.plot(deriv2)
            #pylab.figure()
            #pylab.plot(deriv-Eprime(xTest))
            #print scipy.sum((deriv-Eprime(xTest))**2)
            #print scipy.sum((deriv2-Eprime(xTest))**2)
            ##print scipy.optimize.check_grad(Efunc,Eprime,xTest)
            #die
            # ************
            
            # try random start for a
            #aBest = 0.1*(scipy.random.random([self.ell,self.N]) - 0.5)
            
            # 11.30.2010 try PCA start (so far only works for N=ell)
            vals,vecs = PCA(self.images.T)
            phi = scipy.dot(scipy.linalg.inv(vecs),self.images)[:self.N]
            aBest = scipy.array(vecs)[:,:self.N]
            if self.N > self.ell: 
                # if we have more than the dimensionality of the data,
                
                # (1) fill in rest with random numbers
                aBestFull = 2.*self.sigma*(scipy.random.random((self.ell,self.N))-0.5)
                phiFull = 2.*self.sigma2*(scipy.random.random((self.N,self.m))-0.5)
                
                # (2) or, fill in aBest with original image data
                aBestFull = copy.deepcopy( 100.*(self.images.T[:self.N]).T )
                
                aBestFull[:self.ell,:self.ell] = aBest
                phiFull[:self.ell,:] = phi
                aBest = copy.deepcopy( aBestFull[:] )
                phi = phiFull[:]
                
            #**************** test
            #plotMatrix(aBestFull)
            #plotMatrix(phi)
            #die
            #print "check pack/unpack phi:",sum((phi-unpack(pack(phi,aBest))[0])**2)
            #print "check pack/unpack aBest:",sum((aBest-unpack(pack(phi,aBest))[1])**2)
            #****************
            
            # 12.8.2010 try starting with diagonal
            #freqData = scipy.sum(self.images,axis=1)
            #phi = scipy.transpose(self.images.T/freqData)
            #aBest = scipy.diagflat(freqData)
            
            # 12.9.2010 try starting with uniform random dist.
            # 12.23.2010 is the starting point for the basis artificially making
            #    the predictive power better?
            if self.randomStart:
                scipy.random.seed(self.seed)
                phi = 2.*self.randomPhiMult*                                        \
                    self.sigma2*(scipy.random.random(scipy.shape(phi)) - 0.5)
                aBest = 2.*self.randomAMult*                                        \
                    self.sigma*(scipy.random.random(scipy.shape(aBest)) - 0.5)
                if self.orthogStart: # 2.22.2011
                    aBest = scipy.transpose(GSOrthogonalize(aBest.T))
                    if self.N == self.ell:
                        phi = dot(aBest.T,self.images)
            
            if self.phiStart is not None:
                phi = self.phiStart
            if self.aStart is not None: 
                aBest = self.aStart
            
            # 12.01.2010 try shuffling
            #scipy.random.seed(self.seed)
            #phi = shuffle(phi)
            #phi = scipy.transpose(shuffle(phi.T))
            #aBest = shuffle(aBest)
            #aBest = scipy.transpose(shuffle(aBest.T))
            
            # 12.01.2010 try alternating zero sparseness
            numIter = 1 #5 #10
            for i in range(numIter):
                # 11.30.2010 try zero-sparseness start
                #phi = scipy.ones_like(phi)
                #aBest = scipy.ones_like(aBest)
                #aBest = 2.*self.lmbda*(scipy.random.random([self.ell,self.N]) - 0.5)
                #phi = 2.*self.lmbda2*(scipy.random.random([self.N,self.m]) - 0.5)
                #if i != 0:
                
                # 2.14.2011 try optimizing kappa 
                # 2.15.2011 moved to before optimization
                if False:
                    kappa = self.optimizeKappa(phi,aBest.T)
                    phi,aBest = phi*kappa,aBest/kappa
                
                x0 = self.pack(phi,aBest)
                minimizer = scipy.optimize.fmin_ncg
                lmbda, lmbda2 = copy.copy(self.lmbda),copy.copy(self.lmbda2)
                
                # 12.8.2010 switching in zero-sparseness minimization
                # certainly adds variability in solutions,
                # but doesn't help predictive power
                #phi,aBest = unpack(x0)
                
                # 12.21.2010 start with zero-sparseness minimization 
                # 12.22.2010 now doing manually in runSparseness.py
                #self.lmbda, self.lmbda2 = 0,0
                #x0 = minimizer(self.Efunc,x0,fprime=self.Eprime,                \
                #    fhess_p=self.Ehess_p,disp=False,avextol=1.e-9) #
                
                # 12.21.2010 ***********
                #self.lmbda, self.lmbda2 = lmbda/2.,lmbda2/2.
                #x0 = minimizer(self.Efunc,x0,fprime=self.Eprime,                \
                #    fhess_p=self.Ehess_p,disp=False,avextol=1.e-9)
                # **********************
                
                #contributions,basis = self.basisVectorContributions(phi,True)
                #aBest = basis.T
                #x0 = pack(phi,aBest)
                self.lmbda, self.lmbda2 = lmbda, lmbda2
                    
                #else:
                #    x0 = pack(phi,aBest)
            
                minimizer = scipy.optimize.fmin_ncg
                if False: # old as of 2.17.2011
                  xNew = minimizer(self.Efunc,x0,fprime=self.Eprime,            \
                    fhess_p=self.Ehess_p,                                       \
                    disp=(self.veryVerbose and i==numIter-1),                   \
                    avextol=self.avextol,maxiter=self.maxiter)
                  phiNew,aBestNew = self.unpack(xNew)
                if True: # new -- 2.18.2011,2.21.2011
                  if self.veryVerbose:
                    def callback(x):
                      a = self.unpack(x)[1]
                      nrm = scipy.sqrt( sum(a*a) )
                      print "Current nrm =",nrm
                      print "Current E =",self.ENormFunc(x)
                      print "Current avgGroupSize =",self.avgGroupSize(a.T/nrm)
                  else:
                    callback = lambda x: 0
                  xNew = minimizer(self.ENormFunc,x0,fprime=self.ENormPrime,    \
                    fhess_p=self.ENormHess_p,                                   \
                    disp=(self.veryVerbose and i==numIter-1),                   \
                    avextol=self.avextol,maxiter=self.maxiter,                  \
                    callback=callback)
                  phiNewUnnormed,aBestNewUnnormed = self.unpack(xNew)
                  nrm = scipy.sqrt( sum(aBestNewUnnormed*aBestNewUnnormed) )
                  aBestNew = aBestNewUnnormed / nrm
                  phiNew = phiNewUnnormed * nrm
                  xNew = self.pack(phiNew,aBestNew)
                #minimizer = scipy.optimize.fmin_tnc
                #xNew = minimizer(Efunc,x0,fprime=Eprime)[0] # 
                
                
                #contributions,basis = self.basisVectorContributions(phi,True)
                #aBest = basis.T
                
                phi,aBest = phiNew,aBestNew
                                                
                if self.veryVerbose:
                    cost = self.avgCost(phiNew,aBestNew)
                    print "Average cost =", cost
                    print "ENormFunc =",self.ENormFunc(xNew)
                    
            phi,aBest = phiNew,aBestNew # (_without_ kappa)

            
        
        if self.verbose:
            if deltaCost < self.tol*cost:
                print "findSparseRepresentation: Successful convergence."
                print "findSparseRepresentation: Average cost =", cost
            elif cost < self.epsCost:
                print "findSparseRepresentation: Cost below epsCost."
                print "findSparseRepresentation: Average cost =", cost
            else:
                print "findSparseRepresentation: Warning! Maximum "     \
                    + "number of iterations (" + str(self.maxIter) +") "\
                    + "exceeded."
        
        # *********** test 12.20.2010
        #hess = self.Ehess(xNew)
        #vals = scipy.linalg.eigvals(hess)
        #pylab.figure()
        #pylab.semilogy(scipy.sort(vals),'o')
        #pylab.semilogy(-scipy.sort(vals),'o')
        # ***********
        
        if returnBasis:
            return phi,aBest.T
        return phi
    
    def Elmbda2(self,phi,a,images,lmbda=None,lmbda2=None):
        if lmbda is None:
            lmbda = self.lmbda
        if lmbda2 is None:
            lmbda2 = self.lmbda2
        return scipy.sum( (scipy.dot(a,phi)-images)**2 )                \
            + lmbda *scipy.sum( self.Sfunc(a/self.sigma) )              \
            + lmbda2*scipy.sum( self.Sfunc(phi/self.sigma2) )
                
    def activityHistogram(self,phi,**kwargs):
        
        allActivity = arrayFlatten(                                     \
          [ self.minimizeE(phi,image) for image in self.images ] )
        binTotals,binLocs,objs =                                        \
            pylab.hist(allActivity,normed=True,**kwargs)
        binLocs = ( binLocs[1:] + binLocs[:-1] ) / 2.
        return binLocs, binTotals
        
    def iterateBout(self,bout0,phi1D,numIterations=1,restart=True):
        """
        See notes 8.4.2010.  Defines a dynamics for simulating bouts.
        """
        m = self.m
        boutFromProbs = \
            lambda probs: scipy.ones(m/2)*(scipy.rand(m/2)<probs)
        currentBout = bout0
        boutList,boutProbsList = [],[]
        scipy.random.seed(self.seed)
        for i in range(numIterations):
            coeffs = self.minimizeE(phi1D[:,:m/2],currentBout)
            currentBoutProbs = dot(coeffs,phi1D)[-m/2:]
            currentBout = boutFromProbs(currentBoutProbs)
            if scipy.sum(currentBout) == 0 and restart:
                # use bout picked from participation prob. dist. 
                #currentBout =                                          \
                #   boutFromProbs(self.probabilityFreqData[:m/2])
                
                # use random pair bout
                currentBout = scipy.zeros(m/2)
                for j in range(2):
                    currentBout[scipy.random.random_integers(0,47)] = 1
                
                #currentBoutProbs = self.probabilityFreqData
                print "restart at iteration number", i
            boutList.append(currentBout)
            boutProbsList.append(currentBoutProbs)
        return boutList,boutProbsList
        
    def longFraction(self,boutsList):
        unnormedLF = [                                                  \
            float(len( pylab.find(scipy.sum(boutsList,axis=1)==i) ))    \
            for i in range(self.m/2) ]
        LF = unnormedLF / scipy.sum(unnormedLF[3:])
        return LF
        
    def avgCost(self,phi,aBest=None,testImages=None):
        """
        Returns average cost per 'pixel', divided by the initial
        variance in the data.
        """
        if testImages is None:
            testImages = self.images
        variance = scipy.mean( scipy.var(testImages,axis=1) ) 
                                  # should this be axis 1 or 0?
                                  # prob. depends on whether I'm doing
                                  # it 'forward' or 'backward'
        if aBest is None:
          aBest = [ self.minimizeE(phi,image) for image in testImages ]
        
        # 2.27.2011
        return scipy.mean( scipy.array(                                 \
            [ (self.squashFunc(dot(a,phi))-image)**2                    \
              for a,image in zip(aBest,testImages) ]) ) / variance
        # Note: We don't have to divide by self.m since the cost
        # is returned unsummed.
        #return scipy.mean( scipy.array([cost(a,phi,image)               \
        #    for a,image in zip(aBest,testImages)]) ) / variance

    # 8.18.2010
    def thresholdedCost(self,phi,threshold,aBest=None):
        """
        Same as avgCost, but the basis functions (aBest; see 
        notes 8.10.2010) are changed to zero if near zero
        using thresholdMatrix.
        """
        if aBest is None:
          aBest = [ self.minimizeE(phi,image) for image in self.images ]
        thresholdedBasis = thresholdMatrix(aBest,threshold)
        return self.avgCost(phi,thresholdedBasis)
    
    # 8.11.2010 (basis and coefficients are switched intentionally...
    # see notes 8.10.2010)
    # 12.8.2010 added aStart
    def basisVectorContributions(self,phi,returnBasis=False,            \
        threshold=0.,aStart=None,givenBasis=None):
        """
        Returns contribution of each basis vector, measured as the 
        increase in error after removing each vector from the set.
        """
        if givenBasis is None:
            if aStart is None:
                aStart = scipy.repeat(None,self.N)
            aBest = scipy.array([ self.minimizeE(phi,image,aStart=aVec)     \
                for image,aVec in zip(self.images,aStart) ])
        else:
            aBest = givenBasis.T
        #initialCost = self.avgCost(phi,aBest)
        initialCost = self.thresholdedCost(phi,threshold,aBest)
        contribution = scipy.zeros(len(aBest[0]))
        for i in range(len(aBest[0])):
            #aSingle = scipy.zeros_like(aBest)
            #aSingle[:,i] = aBest[:,i]
            aRemoved = copy.copy(aBest)
            aRemoved[:,i] = scipy.zeros(len(aBest))
            #costRemoved = self.avgCost(phi,aRemoved)
            costRemoved = self.thresholdedCost(phi,threshold,aRemoved)
            contribution[i] = costRemoved - initialCost
        if returnBasis:
            return contribution,aBest.T
        return contribution
        
    # 8.11.2010
    # modified substantially 3.15.2011
    # 10.20.2015 don't think this is used anymore
#    def sortedRules(self,givenBasis,winSize=1,threshold=None,numNonzero=None,   \
#        includeScores=True,includeIndices=False,includeIncrementalScores=False,
#        scoreByUse=True,phi=None,threshold2=None,includeMagnitudes=False,**kwargs):
#        """
#        winSize (1)         : 'window size'
#        threshold (None)    : any basis component with absolute value
#                              less than threshold is changed to zero
#        scoreByUse          : if True, order basis vectors by number of times
#                              they're used (basis must be given) 1.6.2011
#        """
#        #if threshold is None:
#        #  threshold = self.sigma
#        # 3.16.2011 do thresholding
#        basis = thresholdMatrix(givenBasis,                             \
#            thold=threshold,numNonzero=numNonzero)
#        if threshold2 is None:
#          threshold2 = self.sigma2
#        if phi is None: # 3.16.2011 changed to threshold before calculating phi
#          phi = self.phiMinimizer(basis.T)
#        if scoreByUse: # 1.6.2011
#          contributions = scipy.sum(abs(phi)>threshold2,axis=1)
#          #basis = givenBasis
#          if basis is None:
#            raise Exception, "sortedRules error: no basis given."
#        else:
#          print "sortedRules: WARNING: scoreByUse=False is no longer supported."
#          contributions,basis = self.basisVectorContributions(phi,True, \
#            threshold=threshold,givenBasis=basis)
#        sortOrder = scipy.argsort(contributions)[::-1]
#        sortedBasis = basis[sortOrder]
#        sortedContributions = contributions[sortOrder]
#        rules = thresholdedRules(sortedBasis,0.,                 \
#            m=scipy.shape(basis)[1]/winSize,**kwargs)
#        data = [rules]
#        if includeIncrementalScores:
#            print "sortedRules: WARNING: incrementalScores is no longer supported."
#            sortedPhi = phi[sortOrder]
#            N = len(sortedBasis)
#            incrementalContributions = scipy.zeros(N)
#            basisRemoved = copy.copy(sortedBasis)
#            oldCost = self.thresholdedCost(sortedPhi,                   \
#                threshold,sortedBasis.T)
#            for i in range(N):
#                basisRemoved[i] = scipy.zeros_like(basisRemoved[i])
#                newCost = self.thresholdedCost(sortedPhi,               \
#                    threshold,basisRemoved.T)
#                incrementalContributions[i] = oldCost - newCost
#                oldCost = copy.copy(newCost)
#            data.append(incrementalContributions)
#        if includeScores:
#            data.append(sortedContributions)
#        if includeIndices:
#            data.append(scipy.argsort(contributions)[::-1])
#        # 3.15.2011
#        if includeMagnitudes:
#            tSortedBasis = thresholdMatrix(sortedBasis,0.)
#            magnitudes = [ filter(lambda x: x!=0., group)               \
#                           for group in tSortedBasis ]
#            data.append(magnitudes)
#        return zip(*data)

    def avgGroupSize(self,aBest,threshold=None,N=None,                  \
        countOnlyNonzero=True):
        if scipy.shape(aBest)[0] != self.N:
                print "sparsenessProblem.avgGroupSize Warning: "+       \
                    "given basis size != self.N"
        if threshold is None:
            threshold = self.sigma
        if N is None:
          if countOnlyNonzero:
            N = self.numNonzeroBasisVectors(aBest,threshold)
          else:
            N = self.N
        basis = aBest.T
        #thresholdedBasis = thresholdMatrix(basis,threshold)
        if N == 0:
            return 0
        else:
            return float( len(pylab.find(abs(basis)>threshold)) ) / N
        
    # 12.7.2010
    def numNonzeroBasisVectors(self,aBest,threshold=None):
        if threshold is None:
            threshold = self.sigma
        return len(pylab.find( sum(abs(                                 \
                thresholdMatrix(aBest,threshold)),axis=0)>0 ))
                
    # 2.10.2011
    def optimizeKappa(self,phi,basis,disp=False):
        sum,exp = scipy.sum,scipy.exp
        Sfunc,SfuncPrime = self.Sfunc,self.SfuncPrime
        sigmaB = self.sigma
        sigmaA = self.sigma2
        lmbdaB = self.lmbda
        lmbdaA = self.lmbda2
        Lfunc = lambda lk:                                          \
            lmbdaA*sum( Sfunc(exp( lk)*phi/sigmaA) )                \
          + lmbdaB*sum( Sfunc(exp(-lk)*basis/sigmaB) )
        Lprime = lambda lk:                                         \
            lmbdaA*sum( exp( lk)*phi/sigmaA                         \
                       *SfuncPrime(exp( lk)*phi/sigmaA) )           \
          + lmbdaB*sum(-exp(-lk)*basis/sigmaB                       \
                      * SfuncPrime(exp(-lk)*basis/sigmaB) )
        
        # testing ***
        #lks = scipy.linspace(-50,50,100)
        #pylab.plot(lks,[ Lfunc(lk) for lk in lks ])
        #pylab.plot(lks,[ Lprime(lk) for lk in lks ])
        
        logKappaMin = scipy.optimize.fmin_cg(                       \
            Lfunc,0.,fprime=Lprime,disp=disp )[0]
        return exp(logKappaMin)
        
        

class DynamicCompressibilityProblem:

    def __init__(self,timeSeriesMatrix,SfuncName,                   \
        lmbda,sigma,maxTimeWindow=10,maxN=200,seed=0,c=1.,          \
        xi=0.1,maxIter=100,tol=1.e-3,                               \
        verbose=True,veryVerbose=False,veryveryVerbose=False):
        """
        maxN (200)          : maximum number of basis vectors
        maxTimeWindow (10)  : maximum window of time over which
                              to calculate the compressibility
        xi (.1)             : we require the reconstruction error
                              to be less than xi*(variance in data)
        numIterations (100) : maximum number of minimization iterations
        tol (1e-3)          : stop optimization when average cost
                              changes by less than this fraction
                              
        8.11.2010 changed to swap roles of basis and coefficients
                  (transposed images matrix)
        """
        
        self.lmbda = lmbda
        self.sigma = sigma
        self.maxN = maxN
        self.SfuncName = SfuncName
        self.m = len(timeSeriesMatrix[0])
        self.verbose = verbose
        self.veryVerbose = veryVerbose
        self.veryveryVerbose = veryveryVerbose
        self.seed = seed
        self.c = c
        
        self.timeSeriesMatrix = timeSeriesMatrix
        self.maxTimeWindow = maxTimeWindow
        self.xi = xi
        self.maxIter = maxIter
        self.tol = tol
        
        self.variance = scipy.mean( scipy.var(                      \
            self.timeSeriesMatrix,axis=1) )
        
    def numBasisVectorsVsTimeWindow(self,fileprefix=None):
        numBasisVectorsList = []
        avgErrorListList = []
        for windowSize in range(1,self.maxTimeWindow+1):
          avgErrorListList.append([])
          avgError = scipy.inf
          N = 0 # number of basis functions
          avgErrorList = []
          while (avgError > self.xi) and (N <= self.maxN):
            N += 1
            if True:
                print "Testing N =",N
            images = timeSeriesDataMatrix(self.timeSeriesMatrix,    \
                windowSize).T # .T 8.11.2010
            sp = SparsenessProblem(images,self.SfuncName,           \
                self.lmbda,self.sigma,N,verbose=True,               \
                seed=self.seed,c=self.c,maxIter=self.maxIter,       \
                tol=self.tol,veryVerbose=self.veryVerbose,          \
                veryveryVerbose=self.veryveryVerbose)
            phi = sp.findSparseRepresentation()
            avgError = sp.avgCost(phi)
            avgErrorList.append(avgError)
            avgErrorListList[-1] = avgErrorList
            if fileprefix is not None:
              save(numBasisVectorsList,fileprefix+"_numBasisVectorsList.data")
              save(avgErrorListList,fileprefix+"_avgErrorListList.data")
          numBasisVectorsList.append(N)
          #avgErrorListList.append(avgErrorList)
        return numBasisVectorsList,avgErrorListList
            

# old findSparseRepresentation 7.28.2010
if False:
    for i in range(numIterations):
      aBest = [ self.minimizeE(phi,image)                           \
                for image in self.images ]
      aBest = scipy.array(aBest)
      deltaPhi = self.eta / len(self.images) *                      \
        ( scipy.dot(aBest.T,self.images - scipy.dot(aBest,phi)) )   \
        # Divide by len(images) for average.
      phi += deltaPhi
      
      if False:
          # resize phi to keep a variances at 1
          # (should think more about this)
          aVariances = scipy.var(aBest,axis=0)
          #aMeans = scipy.mean(aBest,axis=0)
          #print 'aMeans',aMeans
          alpha = 0.1
          factors = scipy.exp(alpha*scipy.log(aVariances))
          phi = phi *                                               \
            scipy.repeat(factors.reshape([self.N,1]), self.m, axis=1)
          aBestNew = [ self.minimizeE(phi,image)                    \
                for image in self.images ]
          aVariancesNew = scipy.var(aBestNew,axis=0)
          #print aVariancesNew
                
      if not i%50:
        print i, norm(deltaPhi/self.eta) # debug
        #print 'aVariances',aVariances # debug
        print 'a[0]',aBest[0]
        print 'phi[0]',phi[0]


