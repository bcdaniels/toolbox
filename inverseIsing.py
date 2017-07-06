# inverseIsing.py
#
# Bryan Daniels
# 4.11.2011
#

#from Sparseness import * # removed 3.17.2014
#from generateFightData import * # removed 3.17.2014

import scipy
import pylab
import scipy.weave # for efficient fourth-order matrix calculation
import sys
from isingSample import *
from monteCarloSample import *
from SparsenessTools import aboveDiagFlat,replaceDiag,cooccurranceMatrixFights,zeroDiag,fightSizeDistribution,KLdivergence
from meanFieldIsing import JmeanField
#import dit # for bindingInfo
from EntropyEstimates import nats2bits

def isingE(fight,J):
    """
    J coming from MPF minimization.
    """
    return sum( dot(dot(fight,J),fight) )

# 5.14.2012
# 9.15.2014 changed to assume that off-diagonal elements are
#           counted twice in both representations
# 3.2.2015 fixing for good!
# 7.7.2016 moved from criticalPoint.py
def JplusMinus(JzeroOne):
    """
        This assumes H = - sum_i sum_j<>i x_i J_ij x_j
        """
    # see 3.19.2012
    J,h = zeroDiag(JzeroOne),scipy.diag(JzeroOne).copy()
    #return replaceDiag(0.25*J,sum(0.25*J,axis=0)+0.25*scipy.diag(J))
    return replaceDiag(J/4.,h/2. + sum(J/2.,axis=0))

# 5.22.2014 (may need checking)
# 3.2.2015 fixing for good!
# 7.7.2016 moved from criticalPoint.py
def JzeroOne(JplusMinus):
    """
        This assumes H = - sum_i sum_j<>i x_i J_ij x_j
        """
    Jtilde,htilde = zeroDiag(JplusMinus),scipy.diag(JplusMinus).copy()
    #return replaceDiag(2.*Jtilde,-sum(2.*Jtilde,axis=0)+4.*scipy.diag(Jtilde))
    return replaceDiag(4.*Jtilde,2.*htilde - 4.*sum(Jtilde,axis=0))

# 11.8.2011
def isingEmultiple(fights,J):
    """
    J coming from MPF minimization.
    
    For multiple fights.  'fights' should be #fights x #individs.
    """
    #return scipy.diag( dot(dot(fights,J),fights.T) )
    return scipy.sum( dot(fights,J)*fights ,axis=1)


# 3.8.2011
def findJmatrixMPF(fights,zeroUnseenPairs=False,shiftJDiag=False):
    """
    Use "Maximum Probability Flow" method to solve for Ising model J matrix
    consistent with the given fight data.
    
    See SohBatDeW10.
    """
    ell = len(fights[0])
        
    #conflictDirectory = "/Users/bdaniels/Research/Conflict/"
    conflictDirectory = "./"
    matlabCodeDirectory = conflictDirectory                                 \
        +"Sohl-Dickstein-Minimum-Probability-Flow-Learning-87ad963"
    codeDir1 = matlabCodeDirectory + "/MPF_ising"
    codeDir2 = matlabCodeDirectory + "/3rd_party_code/minFunc"
    scipy.random.seed()
    randomNumber = scipy.random.randint(1e7)
    tmpInputFilename = "temp_MPF_data_"+str(randomNumber)+".dat"
    tmpStdoutFilename = "temp_MPF_stderr_stdout_"+str(randomNumber)+".txt"
    scipy.savetxt(tmpInputFilename,fights)
    
    try:
        callString = "fightData_ising('"+tmpInputFilename+"',"+str(ell)+");"
        Jmatrix = callMATLAB(callString,                                    \
            outputFilename="MPF_ising_Jmatrix_"+tmpInputFilename,           \
            codeDirList=[codeDir1,codeDir2])
        os.remove(tmpInputFilename)
    except:
        os.remove(tmpInputFilename)
        raise
    
    if zeroUnseenPairs:
        Jmatrix = Jmatrix*(dot(scipy.transpose(fights),fights) != 0)
    if shiftJDiag: # 3.25.2011 this may be trivially useless 
        #                      (unless thresholding or limiting numNonzero)
        # 3.8.2011 shift diagonal (see notes)
        maxDiag = max(scipy.diag(Jmatrix))
        eps = 1.e10
        Jmatrix = Jmatrix - (maxDiag+eps)*scipy.diag(scipy.ones(len(Jmatrix)))
    return Jmatrix


def unflatten(flatList,ell,symmetrize=False):
    """
    Inverse of aboveDiagFlat with keepDiag=True.
    """
    mat = scipy.sum([ scipy.diag(                                           \
        flatList[diagFlatIndex(0,j,ell):diagFlatIndex(0,j+1,ell)],k=j )     \
        for j in range(ell)], axis=0)
    if symmetrize:
        return 0.5*(mat + mat.T)
    else:
        return mat

def diagFlatIndex(i,j,ell):
    """
    Should have j>=i...
    """
    D = j-i
    return i + D*ell - D*(D-1)/2

# 4.11.2011
# 1.12.2012 changed to take coocMatDesired instead of dataSamples
def isingDeltaCooc(isingSamples,coocMatDesired):
    isingCooc = cooccurranceMatrixFights(isingSamples,keepDiag=True)
    #dataCooc = cooccurranceMatrixFights(dataSamples,keepDiag=True)
    return aboveDiagFlat(isingCooc-coocMatDesired,keepDiag=True)

# 6.3.2013
def isingDeltaCovTilde(isingSamples,covTildeMatDesired,empiricalFreqs):
    isingCovTilde = covarianceTildeMat(isingSamples,empiricalFreqs)
    return aboveDiagFlat(isingCovTilde-covTildeMatDesired,keepDiag=True)

def samplesFromJ(J,numSamples,startConfig=None,nSkip=None,seed=0,           \
    burnin=None,T=1.,retall=False):
    """
    NOTE!  This is old!  Use metropolisSampleIsing, which is 200x faster!
    """
    print "inverseIsing.samplesFromJ: You probably want to use "            \
        "generateFightData.metropolisSampleIsing instead, which is much faster."
    ell = len(J)
    if nSkip is None: nSkip = ell*10
    if burnin is None: burnin = 10
    isingSamples,isingEnergies,acceptance =                                 \
        monteCarloSample(lambda fight: isingE(fight,J)/T,                   \
        acceptFuncMetropolis,ell,numSamples+burnin,startConfig=startConfig, \
        nSkip=nSkip,seed=seed,retall=True)
    if retall:
        return isingSamples[burnin:],isingEnergies[burnin:],acceptance
    return isingSamples[burnin:]

# 4.11.2011
def coocJacobianOld(samples,histMult=None):
    """
    Derivatives of elements of cooccurrance matrix wrt elements of J.
    
    Replaced by coocJacobian and coocJacobianSecondOrder.
    """
    if histMult is not None:
        Z = scipy.mean(histMult)
        ell = len(samples[0])
        histMultRepeated = scipy.repeat(scipy.transpose([histMult]),ell,axis=1)
        updatedIsingSamples = samples*scipy.sqrt(histMultRepeated)
        coocMat = cooccurranceMatrixFights(updatedIsingSamples,keepDiag=True)/Z
    else:
        coocMat = cooccurranceMatrixFights(samples,keepDiag=True)

    ell = len(coocMat)
    #jdim = (ell+1)*ell/2
    coocMatFlat = aboveDiagFlat(coocMat,keepDiag=True)
    diag = -coocMatFlat*(1.-coocMatFlat)
    jac = scipy.diag(diag)
    for i in range(ell):
      for k in range(i+1,ell):
        coocIndex = diagFlatIndex(i,i,ell)
        Jindex = diagFlatIndex(i,k,ell)
        jac[coocIndex,Jindex] = -coocMat[i,k]*(1.-coocMat[i,i])
    return jac
    
# 2.28.2012
def coocJacobianDiagonal(samples,histMult=None):
    """
    Derivatives of elements of cooccurrance matrix wrt elements of J.
    
    Only includes diagonal entries.
    """
    if histMult is not None:
        Z = scipy.mean(histMult)
        ell = len(samples[0])
        histMultRepeated = scipy.repeat(scipy.transpose([histMult]),ell,axis=1)
        updatedIsingSamples = samples*scipy.sqrt(histMultRepeated)
        coocMat = cooccurranceMatrixFights(updatedIsingSamples,keepDiag=True)/Z
    else:
        coocMat = cooccurranceMatrixFights(samples,keepDiag=True)
    
    ell = len(coocMat)
    #jdim = (ell+1)*ell/2
    coocMatFlat = aboveDiagFlat(coocMat,keepDiag=True)
    # i=m, j=n
    diag = coocMatFlat*(coocMatFlat-1.)
    jac = scipy.diag(diag)
    
    return jac
    
# 2.24.2012
def coocJacobianSecondOrder(samples,histMult=None):
    """
    Derivatives of elements of cooccurrance matrix wrt elements of J.
    
    Only includes entries that can be calculated using averages
    of second order or less.
    """
    if histMult is not None:
        Z = scipy.mean(histMult)
        ell = len(samples[0])
        histMultRepeated = scipy.repeat(scipy.transpose([histMult]),ell,axis=1)
        updatedIsingSamples = samples*scipy.sqrt(histMultRepeated)
        coocMat = cooccurranceMatrixFights(updatedIsingSamples,keepDiag=True)/Z
    else:
        coocMat = cooccurranceMatrixFights(samples,keepDiag=True)
    
    ell = len(coocMat)
    #jdim = (ell+1)*ell/2
    coocMatFlat = aboveDiagFlat(coocMat,keepDiag=True)
    # i=m, j=n
    diag = coocMatFlat*(coocMatFlat-1.)
    jac = scipy.diag(diag)
    for i in range(ell):
        for n in range(i+1,ell):
            # i=j, m=n
            coocIndex = diagFlatIndex(i,i,ell)
            Jindex = diagFlatIndex(n,n,ell)
            entry = coocMat[i,i]*coocMat[n,n]-coocMat[i,n]
            jac[coocIndex,Jindex] = entry
            jac[Jindex,coocIndex] = entry
            
            # i=j=m and i=m=n
            coocIndex = diagFlatIndex(i,i,ell)
            Jindex = diagFlatIndex(i,n,ell)
            entry = coocMat[i,n]*(coocMat[i,i]-1.)
            jac[coocIndex,Jindex] = entry
            jac[Jindex,coocIndex] = entry
    
    return jac

    
# 2.17.2012
def fourthOrderCoocMat(samples,slowMethod=False):
    ell = len(samples[0])
    samples = scipy.array(samples)
    jdim = (ell+1)*ell/2
    f = scipy.zeros((jdim,jdim))
    
    if slowMethod:
        for i in range(ell):
          for j in range(i,ell):
            for m in range(i,ell):
              for n in range(m,ell):
                coocIndex1 = diagFlatIndex(i,j,ell)
                coocIndex2 = diagFlatIndex(m,n,ell)
                cooc = scipy.sum(                                           \
                    samples[:,i]*samples[:,j]*samples[:,m]*samples[:,n])
                f[coocIndex1,coocIndex2] = cooc
                f[coocIndex2,coocIndex1] = cooc
    else:
        code = """
        int coocIndex1,coocIndex2;
        float coocSum;
        for (int i=0; i<ell; i++){
          for (int j=i; j<ell; j++){
            for (int m=i; m<ell; m++){
              for (int n=m; n<ell; n++){
                coocIndex1 = i + (j-i)*ell - (j-i)*(j-i-1)/2;
                coocIndex2 = m + (n-m)*ell - (n-m)*(n-m-1)/2;
                coocSum = 0.;
                for (int k=0; k<numFights; k++){
                  coocSum += samples(k,i)*samples(k,j)*samples(k,m)*samples(k,n);
                }
                f(coocIndex1,coocIndex2) = coocSum;
                f(coocIndex2,coocIndex1) = coocSum;
              }
            }
          }
        }
        """
        numFights = len(samples)
        err = scipy.weave.inline(code,                                      \
            ['f','samples','numFights','ell'],                              \
            type_converters = scipy.weave.converters.blitz)
    return f/float(len(samples))
    
# 2.17.2012
def coocJacobian(samples,histMult=None,cooc4=None):
    """
    Derivatives of elements of cooccurrance matrix wrt elements of J.
    
    cooc4 (None)            : Optionally pass in result from
                              fourthOrderCoocMat to save time
                              (doesn't work with histMult)
    """
    if histMult is not None:
        Z = scipy.mean(histMult)
        ell = len(samples[0])
        histMultRepeated = scipy.repeat(scipy.transpose([histMult]),ell,axis=1)
        updatedIsingSamples = samples*scipy.sqrt(histMultRepeated)
        c = cooccurranceMatrixFights(updatedIsingSamples,keepDiag=True)/Z
        updatedIsingSamples = scipy.sqrt(updatedIsingSamples)
        f = fourthOrderCoocMat(updatedIsingSamples)/Z
    else:
        if cooc4 is None:
            f = fourthOrderCoocMat(samples)
        else:
            f = cooc4
        c = cooccurranceMatrixFights(samples,keepDiag=True)
    cFlat = aboveDiagFlat(c,keepDiag=True)
    cOuter = scipy.outer(cFlat,cFlat)
    return cOuter - f

# 3.11.2016
def unflattenSym(m,ell):
    """
    Like unflatten with symmetrize=True, but put full values on each
    side of the diagonal (instead of half values)
    """
    u = unflatten(m,ell,symmetrize=True)
    return replaceDiag(2.*u,scipy.diag(u))


# 8.26.2013
# 3.11.2016 fixed bug
def chiGrad(samples,cooc4=None):
    """
    Slightly unsure about the sign, but doesn't matter at the moment
    because I only use the square.
    
    This could probably be done more efficiently (removing for loops).
    """
    a = scipy.array
    ell = len(samples[0])

    # first term
    if cooc4 is None:
        cooc4 = fourthOrderCoocMat(samples)
    Np = len(cooc4)
    cooc4reshape = a([ unflattenSym(cooc4[:,beta],ell) for beta in range(Np) ]) # Np x ell x ell
    cooc4reshape = scipy.transpose(cooc4reshape,(1,2,0)) # ell x ell x Np
    
    # second term
    cooc2 = cooccurranceMatrixFights(samples,keepDiag=True)
    freqs = scipy.diag(cooc2)
    cooc2 = replaceDiag(cooc2+cooc2.T,freqs) # 3.11.2016 want a symmetric matrix
    cooc2flat = aboveDiagFlat(cooc2,keepDiag=True) # Np x 1
    cooc2outer = a( [ cooc2flat[alpha]*cooc2 for alpha in range(Np) ] ) # Np x ell x ell
    cooc2outer = scipy.transpose(cooc2outer,(1,2,0)) # ell x ell x Np

    # terms involving one frequency and one cooc factor
    coocj = cooc4[:ell,:] # ell x Np
    term3 = a( [ freqs[i]*coocj for i in range(ell) ] ) # ell x ell x Np
    term4 = a( [ scipy.outer(freqs,coocj[i]) for i in range(ell) ] ) # ell x ell x Np
    
    # term with two frequency and one cooc factor
    outerFreq = scipy.outer(freqs,freqs)
    term5 = a( [ 2.*outerFreq*cooc2flat[alpha] for alpha in range(Np) ] ) # Np x ell x ell
    term5 = scipy.transpose(term5,(1,2,0)) # ell x ell x Np
    
    # m is ell x ell x Np
    m = cooc4reshape - cooc2outer - term3 - term4 + term5

    # grad is Np x 1
    return scipy.sum( scipy.sum(m,axis=1), axis=0 )

# 8.27.2013
def sizeGrad(samples,cooc4=None):
    """
    
    """
    a = scipy.array
    ell = len(samples[0])

    if cooc4 is None:
        cooc4 = fourthOrderCoocMat(samples)
    Np = len(cooc4)
    cooc2 = cooccurranceMatrixFights(samples,keepDiag=True)
    cooc2flat = aboveDiagFlat(cooc2,keepDiag=True) # Np x 1
    freqs = scipy.diag(cooc2)

    indepTerm = scipy.outer(freqs,cooc2flat) # ell x Np
    coocj = cooc4[:ell,:] # ell x Np

    # m is ell x Np
    m = indepTerm - coocj

    # grad is Np x 1
    return scipy.sum(m,axis=0)





# 1.12.2012
def findJmatrixBruteForce(fights,JMPF=None,**kwargs):
    """
    Find J, starting from MPF solution.
    
    If you don't have samples but want to fit a cooccurrence matrix,
    use findJmatrixBruteForce_CoocMat.
    
    See 
    """
    coocMatDesired = cooccurranceMatrixFights(fights,keepDiag=True)
    if JMPF is None:
        print "findJmatrixBruteForce: running findJmatrixMPF..."
        JMPF = findJmatrixMPF(fightsForJMPF,zeroUnseenPairs=False)
        print "findJmatrixBruteForce: done running findJmatrixMPF."
    return findJmatrixBruteForce_CoocMat(coocMatDesired,                    \
        Jinit=JMPF,offDiagMult=2.,**kwargs)

# 2.7.2014
def leastsqCov(func,x0,cov,Dfun=None,full_output=False,args=(),**kwargs):
    """
    Wrapper for scipy.optimize.leastsq that takes a covariance matrix
    specifying transformed coordinates in which to do the minimization,
    for use when errors of func are correlated (``generalized'' nonlinear
    least squares).
    
    cov                 : Should be a symmetric positive-definite matrix
    full_output (False) : Full output is not yet supported.
    kwargs              : Passed on to scipy.optimize.leastsq
    
    Returns : 
        (note this is currently a different return structure than
        scipy.optimize.leastsq)
        x      = fit parameters
        fvec   = the result of applying func in transformed
                 coordinates (a list of z scores)
        numfev = number of function evaluations
        mesg   = convergence message
    """
    if full_output:
        raise Exception, "full_output is not yet supported"
    
    U,s,vT = scipy.linalg.svd(cov)
    
    # make transformed version of func
    funcCov = lambda x: scipy.dot( func(x,*args),U ) / scipy.sqrt(s)

    # make transformed version of Dfun
    if Dfun is None:
        DfunCov = None
    else:
        sqrtCinv = scipy.dot(U.T,scipy.diag(1./scipy.sqrt(s)))
        DfunCov = lambda x: scipy.dot( sqrtCinv,Dfun(x,*args) )
    
    xnew,cov_x,infodict,mesg,ier = \
        scipy.optimize.leastsq(funcCov,x0,Dfun=DfunCov,full_output=1,**kwargs)
    fvec = funcCov(xnew)
    numfev = infodict['nfev']

    return xnew,fvec,numfev,mesg

# 4.11.2011
# 1.12.2012 changed to take a desired cooccurrence matrix
# 2.14.2012 removed bad derivative function
# 2.23.2012 changed definition of stopErrFactor
# 3.1.2012  changed to use meanZSq as stop criterion
def findJmatrixBruteForce_CoocMat(coocMatData,                                  
    maxnumiter=None,numSamples=1e5,nSkip=None,maxfev=0,seed=0,Jinit=None,
    offDiagMult=2.,retall=False,histogramMethod=False,changeSeed=True,
    numProcs=1,thresholdMeanZSq=1.,gradFunc=None,
    maxNumericalGradCalls=0,numFights=None,JdiagMean=None,JdiagSigma=None,
    JoffDiagMean=None,JoffDiagSigma=None,Jseed=100,
    minSize=2,gradDesc=False,minimizeCovariance=False,            
    minimizeIndependent=False,coocCov=None,coocBayesianMean=False,
    priorLmbda=0.,meanFieldInit=False,meanFieldPriorLmbda=None):
    """
    numSamples (1e5)            : To be assured of convergence, should
                                  be >= ~ 10*numFights/thresholdMeanZSq.
    thresholdMeanZSq (1.)       : stop minimization when the cooccurrence
                                  matrix mean squared z score drops 
                                  below thresholdMeanZSq
                                  (unless maxnumiter is 
                                  reached).  To be assured of convergence,
                                  should be >= ~ 10*numFights/numSamples.
    maxNumericalGradCalls (0)   : If gradFunc is None, adds
                                  maxNumericalGradCalls*(ell+1)*ell/2
                                  to the maxfev passed to the 
                                  optimizer
    retall (False)              : If True, also return meanZSq
                                  achieved.
    gradFunc (None)             : A function that calculates the
                                  gradient given samples.  Try
                                  "coocJacobianDiagonal",
                                  "coocJacobian", or "coocJacobianSecondOrder"
    JdiagMean,JdiagSigma,       : If given, the elements of Jinit are chosen
    JoffDiagMean,JoffDiagSigma    from normal distributions with the given
                                  means and standard deviations
    minSize (2)                 : 3.8.2013 Use a modified model in which
                                  fights of size less than minSize are not 
                                  allowed. (5.1.2014 changed from 
                                  removeZerosAndOnes)
    gradDesc (False)            : 5.29.2013 Take a naive gradient descent step
                                  after each LM minimization
    minimizeCovariance (False)  : 6.3.2013 Minimize covariance from emperical
                                  frequencies (see notes); trying to avoid
                                  biases, as inspired by footnote 12 in 
                                  TkaSchBer06
    minimizeIndependent (False) : 2.7.2014 Use this to get old behavior where
                                  each <xi> and <xi xj> residual is treated
                                  as independent
    coocCov (None)              : 2.7.2014 Provide a covariance matrix for
                                  residuals.  Should typically be 
                                  coocSampleCovariance(samples).  Only used
                                  if minimizeCovariance and minimizeIndependent
                                  are False.
    coocBayesianMean (False)    : 2.11.2014 Old way of regularizing zeros in
                                  cooccurrence matrix.  Not using anymore
                                  due to a bias toward criticality.
    priorLmbda (0.)             : 2.11.2014 Strength of noninteracting prior
    meanFieldInit (False)       : 11.20.2014 Use mean field estimate as Jinit.
    """
    
    ell = len(coocMatData) #len(fights[0])
    
    if coocBayesianMean:
        coocMatMean = coocMatBayesianMean(coocMatData,numFights)
    else:
        coocMatMean = coocMatData
    
    if priorLmbda != 0.:
        raise Exception, "11.24.2014 Need to fix prior implementation"
    
    lmbda = priorLmbda / numFights
    
    # 3.1.2012 I'm pretty sure the "repeated" line below should have the
    # transpose, but coocJacobianDiagonal is not sensitive to this.  If you
    # use non-diagonal jacobians in the future and get bad behavior you
    # may want to double-check this.
    if minimizeIndependent:
        coocStdevs = coocStdevsFlat(coocMatData,numFights)
        coocStdevsRepeated = scipy.transpose(                                   \
            coocStdevs*scipy.ones((len(coocStdevs),len(coocStdevs))) )
    elif minimizeCovariance:
        empiricalFreqs = scipy.diag(coocMatData)
        covTildeMean = covarianceTildeMatBayesianMean(coocMatData,numFights)
        covTildeStdevs = covarianceTildeStdevsFlat(coocMatData,numFights,       \
                                                   empiricalFreqs)
        covTildeStdevsRepeated = scipy.transpose(                               \
           covTildeStdevs*scipy.ones((len(covTildeStdevs),len(covTildeStdevs))) )
    else:
        # 2.7.2014
        if coocCov is None: raise Exception
        cov = coocCov # / numFights (can't do this here due to numerical issues)
        # instead include numFights in the calculation of coocMatMeanZSq
    
    #unitErrSq = sum(coocMatDesired*(1.-coocMatDesired))
    #thresholdCoocMatErrSq = stopErrFactor*unitErrSq
    
    if numFights is None:
        print "findJmatrixBruteForce_CoocMat WARNING:"
        print "    numFights not specified.  Assuming numFights = 1000."
    if thresholdMeanZSq*numSamples/numFights < 10:
        print "findJmatrixBruteForce_CoocMat WARNING:"
        print "    thresholdMeanZSq*numSamples/numFights = "+                   \
            str(thresholdMeanZSq*numSamples/numFights)+" < 10."
        print "    Solution may be hard to find (or impossible if < 1)."
    
    # set starting parameters Jinit
    if meanFieldInit:
        if Jinit is not None:
            raise Exception, "Jinit and meanFieldInit arguments cannot be used simultaneously."
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        Jinit = JmeanField(coocMatMean,meanFieldPriorLmbda=meanFieldPriorLmbda,
                           numSamples=numFights)
    if Jinit is None:
        #Jinit = scipy.zeros((ell,ell))
        # 1.17.2012 try starting from frequency model
        if JdiagMean is None:
            freqs = scipy.diag(coocMatMean)
            hList = -scipy.log(freqs/(1.-freqs))
            Jinit = scipy.diag(hList)
        else: # 3.6.2013
            scipy.random.seed(Jseed)
            if JdiagSigma > 0:
                Jdiag = scipy.random.normal(JdiagMean,JdiagSigma,ell)
            else:
                Jdiag = JdiagMean * scipy.ones(ell)
            if JoffDiagSigma > 0:
                JoffDiag = scipy.random.normal(JoffDiagMean,JoffDiagSigma,(ell,ell))
            else:
                JoffDiag = JoffDiagMean * scipy.ones((ell,ell))
            Jinit = replaceDiag(JoffDiag,Jdiag)

    if minimizeCovariance: Jinit = normalJ2tildeJ(Jinit,empiricalFreqs)
    flatJinit = aboveDiagFlat(Jinit,keepDiag=True,offDiagMult=offDiagMult)
    flatJstart = copy.copy(flatJinit)
    Jstart = copy.copy(Jinit)
    lastFight = scipy.zeros(ell)
    flatJnew = copy.copy(flatJinit) # in case we do zero fitting iterations
    
    if changeSeed: seedIter = seedGenerator(seed,1)
    else: seedIter = seedGenerator(seed,0)
    
    def samples(flatJ,lastFight,numProcs):
        # I'm not sure 'lastFight' is actually doing what I intended.
        #   But I'm also not sure it's necessary anyway.
        # 3.9.2013 I removed 'lastFight' functionality completely
        seed = seedIter.next()
        #print seed
        J = unflatten(flatJ,ell,symmetrize=True)
        if minimizeCovariance:
            J = tildeJ2normalJ(J,empiricalFreqs)
        # 1.12.12 changed from samplesFromJ to metropolisSampleIsing
        if numProcs > 1: 
            isingSamples = metropolisSampleIsing_pypar(         \
               numProcs,J,                                      \
               numSamples,startConfig=None,nSkip=nSkip,         \
               seed=seed,minSize=minSize)
        else: 
            isingSamples = metropolisSampleIsing(               \
                 J,                                             \
                 numSamples,startConfig=None,nSkip=nSkip,       \
                 seed=seed,minSize=minSize)#factor=0.1
        isingSamples = scipy.array(isingSamples,dtype=float)
        #lastFight = isingSamples[-1]
        return isingSamples
    
    if maxnumiter is None:
        maxnumiter = scipy.inf
        #if histogramMethod: maxnumiter = scipy.inf
        #else: maxnumiter = 1
    numiter = 0
    coocMatMeanZSq = scipy.inf
    samplesAfter = None
    
    while (numiter < maxnumiter) and (coocMatMeanZSq > thresholdMeanZSq):
        if histogramMethod:
            if minimizeCovariance:
                raise Exception, "minimizeCovariance not yet supported with histogramMethod"
            if not minimizeIndependent:
                raise Exception, "Only minimizeIndependent is currently supported with histogramMethod"
            if samplesAfter is None: # it's our first time through
                isingSamplesStart = samples(flatJstart,lastFight,numProcs)
            else: # use the samples we calculated after the last iteration
                isingSamplesStart = samplesAfter
            def func(flatJ,lastFight,Jold,samplesOld):
                Jnew = unflatten(flatJ,ell,symmetrize=True)
                return isingDeltaCoocApprox(samplesOld,                     \
                    coocMatMean,Jold,Jnew)/coocStdevs
            def Dfun(flatJ,lastFight,Jold,samplesOld): 
                Jnew = unflatten(flatJ,ell,symmetrize=True)
                #coocMat = isingCoocApprox(samplesOld,Jold,Jnew)
                histMult,Z = histogramMultipliers(samplesOld,Jold,Jnew)
                return gradFunc(samplesOld,histMult=histMult)/coocStdevsRepeated
        else:
            isingSamplesStart,Jstart = None,None
            def func(flatJ,lastFight,Jold,samplesOld):
                isingSamples = samples(flatJ,lastFight,numProcs)
                if minimizeIndependent:
                    # 11.21.2014 oops, this bypasses the prior term XXX
                    return isingDeltaCooc(isingSamples,coocMatMean)/coocStdevs
                elif minimizeCovariance:
                    return isingDeltaCovTilde(isingSamples,covTildeMean,    \
                        empiricalFreqs)/covTildeStdevs
                else:
                    dc = isingDeltaCooc(isingSamples,coocMatMean)
                    if lmbda != 0.:
                        # old prior
                        #priorTerm = lmbda * flatJ[ell:]**2
                        
                        # new prior 3.24.2014
                        # 11.21.2014 oops, I think this should be square-rooted XXX
                        freqs = scipy.diag(coocMatMean)
                        factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                        factorFlat = aboveDiagFlat(factor)
                        priorTerm = lmbda * factorFlat * flatJ[ell:]**2
                        
                        dc = scipy.concatenate([dc,priorTerm])
                    return dc

            def Dfun(flatJ,lastFight,Jold,samplesOld): 
                isingSamples = samples(flatJ,lastFight,numProcs)
                #coocMat = cooccurranceMatrixFights(isingSamples,keepDiag=True)
                #print coocMat
                if minimizeIndependent:
                    # 11.21.2014 oops, this should be updated XXX
                    return gradFunc(isingSamples)/coocStdevsRepeated
                elif minimizeCovariance:
                    return covTildeJacobianDiagonal(isingSamples,           \
                        empiricalFreqs)/covTildeStdevsRepeated
                else:
                    g = gradFunc(isingSamples)
                    if lmbda != 0.:
                        # shape(priorTerm) = ell*(ell-1)/2 x ell*(ell-1)/2
                        
                        # old prior
                        #priorTerm = scipy.diag( 2.*lmbda * flatJ[ell:] )
                        
                        # new prior 3.24.2014
                        # 11.21.2014 oops, this should be updated XXX
                        freqs = scipy.diag(coocMatMean)
                        factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                        factorFlat = aboveDiagFlat(factor)
                        priorTerm = scipy.diag( 2.*lmbda*factorFlat * flatJ[ell:] )
                        
                        zero = scipy.zeros((ell*(ell-1)/2,ell))
                        priorg = scipy.concatenate([zero.T,priorTerm.T]).T
                        g = scipy.concatenate([g,priorg])
                    return g
                    
        if (gradFunc is not None) or minimizeCovariance:
            DfunToUse = Dfun
            maxfevToUse = maxfev
        else: 
            DfunToUse = None
            maxfevToUse = maxfev + maxNumericalGradCalls*(ell+1)*ell/2
    
        # calculate meanZSq of initial guess for reference
        if coocMatMeanZSq == scipy.inf:
            fvec = func(flatJstart,lastFight,Jstart,samplesAfter)
            if minimizeCovariance or minimizeIndependent: # old way
                coocMatMeanZSqNoPrior = scipy.mean(fvec**2)
            else:
                m = ell*(ell+1)/2
                coocMatMeanZSqNoPrior = scipy.mean( numFights * fvec[:m]**2 )
                # *****
                print "len(fvec) =",len(fvec)
                print "ell*(ell+1)/2 =",m
                print "ell*(ell-1)/2 =",ell*(ell-1)/2
                # *****
            print "findJmatrixBruteForce_CoocMat: Starting minimization"
            print "    Initial coocMatMeanZSqNoPrior =",coocMatMeanZSqNoPrior
            print "                 thresholdMeanZSq =",thresholdMeanZSq
            print
            sys.stdout.flush()
        
        if minimizeCovariance or minimizeIndependent: # old way
            flatJnew,cov_x,infodict,mesg,ier = scipy.optimize.leastsq(func, \
                flatJstart,args=(lastFight,Jstart,isingSamplesStart),       \
                maxfev=maxfevToUse,full_output=1,Dfun=DfunToUse)
            numfev = infodict['nfev']
            samplesAfter = samples(flatJnew,lastFight,numProcs)
            coocMatMeanZSq = scipy.mean(                                    \
                func(flatJstart,lastFight,Jstart,samplesAfter)**2 )
            sys.stdout.flush()
            if scipy.isnan(coocMatMeanZSq) or scipy.isinf(coocMatMeanZSq):
                raise Exception, "Encountered non-numerical cost."

            # 5.29.2013 take naive gradient descent step
            if gradDesc:
                grad = DfunToUse(flatJnew,lastFight,Jstart,isingSamplesStart)
                resid = func(flatJnew,lastFight,Jstart,isingSamplesStart)
                deltaParams = -scipy.dot(resid,grad)
                flatJstart = flatJnew + deltaParams
                Jstart = unflatten(flatJstart,ell,symmetrize=True)
        else: # new way 2.7.2014
            flatJnew,fvec,numfev,mesg = leastsqCov(
                func,flatJstart,cov,Dfun=DfunToUse,
                args=(lastFight,Jstart,isingSamplesStart),
                maxfev=maxfevToUse)
            # 2.11.2014 (need lmbda \propto 1/numFights)
            coocMatMeanZSq = scipy.mean( numFights * fvec**2 ) 

        print "numiter =",numiter
        print "numfev =",numfev
        print mesg
        print "coocMatMeanZSq/thresholdMeanZSq =",                      \
                coocMatMeanZSq/thresholdMeanZSq
        sys.stdout.flush()

        flatJstart = flatJnew
        Jstart = unflatten(flatJstart,ell,symmetrize=True)
        numiter += 1
        
    if numiter >= maxnumiter:
        print "findJmatrixBruteForce_CoocMat: "+                            \
            "Maximum number of iterations reached ("+str(maxnumiter)+")"
    
    Jnew = unflatten(flatJnew,ell,symmetrize=True)
    if minimizeCovariance:
        Jnew = tildeJ2normalJ(Jnew,empiricalFreqs)
    if retall:
        #coocErr = scipy.sum(infodict['fvec']**2)
        return Jnew,coocMatMeanZSq
    return Jnew

# 11.21.2014 --- Finding best regularized mean field solution ---

def findJmatrixRegMeanField(coocMatData,
    numSamples=1e5,nSkip=None,seed=0,
    changeSeed=False,numProcs=1,
    numFights=None,minSize=2,
    minimizeCovariance=False,minimizeIndependent=False,
    coocCov=None,priorLmbda=0.,verbose=True,bracket=None,
    numGridPoints=200):
    """
    Varies the strength of regularization on the mean field J to
    best fit given cooccurrence data.
    
    numGridPoints (200) : If bracket is given, first test at numGridPoints
                          points evenly spaced in the bracket interval, then give
                          the lowest three points to scipy.optimize.minimize_scalar
    
    See findJmatrixBruteForce_CoocMat for other argument definitions.
    """
    
    if changeSeed: seedIter = seedGenerator(seed,1)
    else: seedIter = seedGenerator(seed,0)
    
    if priorLmbda != 0.:
        raise Exception, "11.24.2014 Need to fix prior implementation"
        lmbda = priorLmbda / numFights

    # 11.21.2014 stuff defining the error model, taken
    #            from findJmatrixBruteForce_CoocMat
    # 3.1.2012 I'm pretty sure the "repeated" line below should have the
    # transpose, but coocJacobianDiagonal is not sensitive to this.  If you
    # use non-diagonal jacobians in the future and get bad behavior you
    # may want to double-check this.
    if minimizeIndependent:
        coocStdevs = coocStdevsFlat(coocMatData,numFights)
        coocStdevsRepeated = scipy.transpose(                                   \
            coocStdevs*scipy.ones((len(coocStdevs),len(coocStdevs))) )
    elif minimizeCovariance:
        empiricalFreqs = scipy.diag(coocMatData)
        covTildeMean = covarianceTildeMatBayesianMean(coocMatData,numFights)
        covTildeStdevs = covarianceTildeStdevsFlat(coocMatData,numFights,       \
            empiricalFreqs)
        covTildeStdevsRepeated = scipy.transpose(                               \
            covTildeStdevs*scipy.ones((len(covTildeStdevs),len(covTildeStdevs))) )
    else:
        # 2.7.2014
        if coocCov is None: raise Exception
        cov = coocCov # / numFights (can't do this here due to numerical issues)
                      # instead include numFights in the calculation of coocMatMeanZSq

    # 11.21.2014 for use in gammaPrime <-> priorLmbda
    freqsList = scipy.diag(coocMatData)
    pmean = scipy.mean(freqsList)
    
    # 11.21.2014 adapted from findJMatrixBruteForce_CoocMat
    def samples(J):
       seed = seedIter.next()
       #print seed
       #J = unflatten(flatJ,ell,symmetrize=True)
       if minimizeCovariance:
           J = tildeJ2normalJ(J,empiricalFreqs)
       if numProcs > 1:
           isingSamples = metropolisSampleIsing_pypar(numProcs,J,
                              numSamples,startConfig=None,nSkip=nSkip,
                              seed=seed,minSize=minSize)
       else:
           isingSamples = metropolisSampleIsing(J,
                            numSamples,startConfig=None,nSkip=nSkip,
                            seed=seed,minSize=minSize)
       isingSamples = scipy.array(isingSamples,dtype=float)
       return isingSamples

    # 11.21.2014 adapted from findJMatrixBruteForce_CoocMat
    def func(meanFieldGammaPrime):
        
        # translate gammaPrime prior strength to lambda prior strength
        meanFieldPriorLmbda = meanFieldGammaPrime / (pmean**2 * (1.-pmean)**2)
        
        # calculate regularized mean field J
        J = JmeanField(coocMatData,meanFieldPriorLmbda=meanFieldPriorLmbda,
                       numSamples=numFights)

        # sample from J
        isingSamples = samples(J)
        
        # calculate residuals, including prior if necessary
        if minimizeIndependent: # Default as of 4.2.2015
            dc = isingDeltaCooc(isingSamples,coocMatData)/coocStdevs
        elif minimizeCovariance:
            dc = isingDeltaCovTilde(isingSamples,covTildeMean,
                                      empiricalFreqs)/covTildeStdevs
        else:
            dc = isingDeltaCooc(isingSamples,coocMatMean)
            if priorLmbda != 0.:
                # new prior 3.24.2014
                # 11.21.2014 oops, I think this should be square-rooted XXX
                # 11.21.2014 oops, should also apply in minimizeIndependent case XXX
                freqs = scipy.diag(coocMatData)
                factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                factorFlat = aboveDiagFlat(factor)
                priorTerm = lmbda * factorFlat * flatJ[ell:]**2
            
            dc = scipy.concatenate([dc,priorTerm])
            
        if verbose:
            print "findJmatrixRegMeanField: Tried "+str(meanFieldGammaPrime)
            print "findJmatrixRegMeanField: sum(dc**2) = "+str(scipy.sum(dc**2))
            
        return scipy.sum(dc**2)

    if bracket is not None:
        gridPoints = scipy.linspace(bracket[0],bracket[1],numGridPoints)
        gridResults = [ func(p) for p in gridPoints ]
        gridBracket = bracket1d(gridPoints,gridResults)
        solution = scipy.optimize.minimize_scalar(func,bracket=gridBracket)
    else:
        solution = scipy.optimize.minimize_scalar(func)

    gammaPrimeMin = solution['x']
    meanFieldPriorLmbdaMin = gammaPrimeMin / (pmean**2 * (1.-pmean)**2)
    J = JmeanField(coocMatData,meanFieldPriorLmbda=meanFieldPriorLmbdaMin,
               numSamples=numFights)

    return J

# 3.18.2016
def bracket1d(xList,funcList):
    """
    *** Assumes xList is monotonically increasing
    
    Get bracketed interval (a,b,c) with a < b < c, and f(b) < f(a) and f(c).
    (Choose b and c to make f(b) and f(c) as small as possible.)
    
    If minimum is at one end, raise error.
    """
    gridMinIndex = scipy.argmin(funcList)
    gridMin = xList[gridMinIndex]
    if (gridMinIndex == 0) or (gridMinIndex == len(xList)-1):
        raise Exception, "Minimum at boundary"
    gridBracket1 = xList[ scipy.argmin(funcList[:gridMinIndex]) ]
    gridBracket2 = xList[ gridMinIndex + 1 + scipy.argmin(funcList[gridMinIndex+1:]) ]
    gridBracket = (gridBracket1,gridMin,gridBracket2)
    return gridBracket


# 6.30.2014
# being lazy and using more general functions I wrote before
def covarianceStdevsFlat(fightData):
    numFights = len(fightData)
    coocMat = cooccurranceMatrixFights(fightData,keepDiag=True)
    freqs = scipy.diag(coocMat)
    return covarianceTildeStdevsFlat(coocMat,numFights,freqs)

# 6.3.2013
def covarianceTildeMat(fightData,empiricalFreqs):
    """
    covTilde_ij = < (x_i - <x_i>_data)(x_j - <x_j>_data) >
    covTilde_ii = < x_i >
    """
    v = fightData - empiricalFreqs*scipy.ones_like(fightData)
    covTilde = scipy.dot(v.T,v)/float(len(fightData))
    covTilde = zeroBelowDiag(covTilde)
    return replaceDiag(covTilde,scipy.mean(fightData,axis=0))

# 6.3.2013
def covarianceTildeMatBayesianMean(coocMat,numFights):
    #coocMat = cooccurranceMatrixFights(empiricalFightData,keepDiag=True)
    empiricalFreqs = scipy.diag(coocMat)
    coocMatB = coocMatBayesianMean(coocMat,numFights)
    freqB = scipy.diag(coocMatB)
    term2 = scipy.outer(freqB,empiricalFreqs)
    term4 = scipy.outer(empiricalFreqs,empiricalFreqs)
    covTildeMatB = zeroBelowDiag(coocMatB - term2 - term2.T + term4)
    return replaceDiag(covTildeMatB,freqB)

# 6.3.2013
def covarianceTildeStdevsFlat(coocMat,numFights,empiricalFreqs):
    """
    Returns a flattened expected standard deviation matrix used
    to divide deltaCovTilde to turn it into z scores.
    """
    coocMatMean = coocMatBayesianMean(coocMat,numFights)
    varianceMat = coocMatMean*(1.-coocMatMean)/numFights
    term2 = scipy.outer(scipy.diag(varianceMat),empiricalFreqs)
    varianceTildeMat = varianceMat + term2 + term2.T
    varianceTildeMat = replaceDiag(varianceTildeMat,scipy.diag(varianceMat))
    return scipy.sqrt(aboveDiagFlat(varianceTildeMat,keepDiag=True))

# 6.3.2013
def covTildeJacobianDiagonal(fightData,empiricalFreqs):
    freqs = scipy.mean(fightData,axis=0)
    v = fightData - empiricalFreqs*scipy.ones_like(fightData)
    covTilde = scipy.dot(v.T,v)/float(len(fightData))
    vsq = v**2
    covTildeSq = scipy.dot(vsq.T,vsq)/float(len(fightData))
    
    jac = covTildeSq - covTilde**2
    jac = replaceDiag(jac,freqs*(1.-freqs))
    jacFlat = aboveDiagFlat(jac,keepDiag=True)
    
    return -scipy.diag(jacFlat)

# 6.3.2013
def normalJ2tildeJ(normalJ,empiricalFreqs):
    h = scipy.diag(normalJ)
    JnoDiag = zeroDiag(normalJ)
    diagTerm = 2.*scipy.dot(JnoDiag,empiricalFreqs)
    return replaceDiag(normalJ,h+diagTerm)

# 6.3.2013
def tildeJ2normalJ(tildeJ,empiricalFreqs):
    hTilde = scipy.diag(tildeJ)
    JnoDiag = zeroDiag(tildeJ)
    diagTerm = 2.*scipy.dot(JnoDiag,empiricalFreqs)
    return replaceDiag(tildeJ,hTilde-diagTerm)

# 2.7.2014
def coocSampleCovariance(samples,bayesianMean=True,includePrior=True):
    """
    includePrior (True)             : Include diagonal component corresponding
                                      to ell*(ell-1)/2 prior residuals for
                                      interaction parameters
    """
    coocs4 = fourthOrderCoocMat(samples)
    if bayesianMean:
        #coocs4mean = coocMatBayesianMean(coocs4,len(samples))
        print "coocSampleCovariance : WARNING : using ad-hoc 'Laplace' correction"
        N = len(samples)
        newDiag = (scipy.diag(coocs4)*N + 1.)/(N + 2.)
        coocs4mean = replaceDiag(coocs4,newDiag)
    else:
        coocs4mean = coocs4
    cov = coocs4mean*(1.-coocs4mean)
    if includePrior:
        ell = len(samples[0])
        one = scipy.ones(ell*(ell-1)/2)
        return scipy.linalg.block_diag( cov, scipy.diag(one) )
    else:
        return cov

# 6.3.2013
def zeroBelowDiag(mat):
    mat2 = copy.copy(mat)
    mat2 *= (1 - scipy.tri(len(mat),k=-1))
    return mat2

# 1.30.2012
def seedGenerator(seedStart,deltaSeed):
    while True:
        seedStart += deltaSeed
        yield seedStart

# 5.30.2013 taken from IsingDegeneracy.ipynb
def coocMatIsing(dataDict,numSamples=1e5):
    samples = dataDict['model'].metropolisSamples(numSamples)[0]
    return cooccurranceMatrixFights(samples,keepDiag=True)

# 5.30.2013 taken from IsingDegeneracy.ipynb
def residualAnalysis(J,fightData=None,makePlots=False,numSamples=1e5,
    numProcs=11,minSize=2,dataDict=None,useBayesianMean=False):
    """
    fightData (None)            : provide either fightData or dataDict 
    dataDict (None)
    """
    
    # take samples
    if dataDict is not None:
      if dataDict.has_key('bruteForceKwargs'):
        b = dataDict['bruteForceKwargs']
        fittingNumSamples = int(b['numSamples'])
        if numSamples < fittingNumSamples:
            print "Warning: numSamples ("+str(int(numSamples))+\
                ") < fittingNumSamples ("+str(fittingNumSamples)+")"
        print "Sqrt( thresholdMeanZSq ) = %1.3f"%scipy.sqrt(b['thresholdMeanZSq'])
    
        if b.has_key('minSize'):
            dataDictMinSize = b['minSize']
        elif b.has_key('removeZerosAndOnes'):
            # back compatibility
            #dataDictRemoveZerosAndOnes = b['removeZerosAndOnes']
            if b['removeZerosAndOnes']: dataDictMinSize = 2
            else: dataDictMinSize = 0
        else:
            # back compatibility
            #dataDictRemoveZerosAndOnes = False
            dataDictMinSize = 0
        if minSize != dataDictMinSize:
            print "Warning: minSize ("+str(minSize)+\
                ") != dataDictMinSize ("+str(dataDictMinSize)+")"

        minCov = False
        if b.has_key('minimizeCovariance'):
            if b['minimizeCovariance']: minCov = True
        print "minimizeCovariance:",minCov
      #J = dataDict['J']
    
    ell = len(J)
    if numProcs > 1:
        samples = metropolisSampleIsing_pypar(numProcs,J,numSamples,
            minSize=minSize)
    else:
        samples = metropolisSampleIsing(J,numSamples,
            minSize=minSize)
    
    # do residual calculations
    c1 = cooccurranceMatrixFights(samples,keepDiag=True)
    if fightData is not None:
        numFights = len(fightData)
        coocMatDesired = cooccurranceMatrixFights(fightData,keepDiag=True)
    elif dataDict is not None:
        numFights = dataDict['numFights']
        coocMatDesired = dataDict['coocMatDesired']
    else:
        raise Exception
    if useBayesianMean:
        cDesiredFlat = aboveDiagFlat(coocMatBayesianMean( coocMatDesired, numFights))
    else:
        cDesiredFlat = aboveDiagFlat(coocMatDesired)
    cDesiredFlatStdevs = coocStdevsFlat( coocMatDesired, numFights)[ell:]
    zVals = (aboveDiagFlat(c1)-cDesiredFlat)/cDesiredFlatStdevs
    cDesiredDiag = scipy.diag(coocMatBayesianMean( coocMatDesired, numFights))
    cDesiredDiagStdevs = coocStdevsFlat( coocMatDesired, numFights)[:ell]
    zValsDiag = (scipy.diag(c1)-cDesiredDiag)/cDesiredDiagStdevs
    print "Off-diagonal z-scores: % 1.3f +/- %1.3f"                                 \
        %(scipy.mean(zVals),scipy.std(zVals))
    print "Diagonal z-scores:     % 1.3f +/- %1.3f"                                 \
        %(scipy.mean(zValsDiag),scipy.std(zValsDiag))
    cost = scipy.mean( scipy.concatenate((zVals**2,zValsDiag**2)) )
    print "Average z^2:           % 1.3f"%cost
    print "Average diag. z^2:     % 1.3f"%scipy.mean(zValsDiag**2)
    print "Average cooc. z^2:     % 1.3f"%scipy.mean(zVals**2)

    # calculate covariance (tilde, with emperical frequencies)
    empiricalFreqs = scipy.diag(coocMatDesired) #scipy.mean(fightData,axis=0)
    covTildeMeanData = covarianceTildeMatBayesianMean(coocMatDesired,numFights)
    covTildeFlatStdevs = covarianceTildeStdevsFlat(coocMatDesired,numFights,       \
                                                   empiricalFreqs)[ell:]
    covTildeMean = covarianceTildeMat(samples,empiricalFreqs)
    zValsCov = (aboveDiagFlat(covTildeMean)-aboveDiagFlat(covTildeMeanData))        \
        /covTildeFlatStdevs
    print "Cov. tilde z-scores:   % 1.3f +/- %1.3f"                                 \
        %(scipy.mean(zValsCov),scipy.std(zValsCov))
    costCov = scipy.mean( scipy.concatenate((zValsCov**2,zValsDiag**2)) )
    print "Average cov. z^2:      % 1.3f"%costCov

    if fightData is not None:
        # do goodness-of-fit calculations
        print ""
        KLmaxSize = 12
        dataSizeDist = fightSizeDistribution(fightData)
        modelSizeDist = fightSizeDistribution(samples)
        KL = KLdivergence(dataSizeDist[:KLmaxSize+1],modelSizeDist[:KLmaxSize+1])
        print "KL divergence:    % 1.3f"%(KL)

        dataFreqs = scipy.mean(fightData,axis=0)
        modelFreqs = scipy.mean(samples,axis=0)
        rFreq,pFreq = scipy.stats.pearsonr(dataFreqs,modelFreqs)
        print "Frequency rho:    % 1.3f (p = %1.3f)"%(rFreq,pFreq)

        rCooc,pCooc = scipy.stats.pearsonr(cDesiredFlat,aboveDiagFlat(c1))
        print "Cooccurrence rho: % 1.3f (p = %1.3f)"%(rCooc,pCooc)

        #cov = covarianceMatrix(samples)
        #covData = covarianceMatrix(fightData)

    if makePlots:
        pylab.figure(figsize=(14,12))
        numCols = 3
        numRows = 3
        
        col = 1
        # diagonal scatter plot
        row = 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.errorbar(cDesiredDiag,scipy.diag(c1),xerr=cDesiredDiagStdevs,marker='o',  \
            color='g',ls='')
        mx = max(max(cDesiredDiag),max(scipy.diag(c1)))
        mx = 1.1*mx
        pylab.plot([0,mx],[0,mx],'k-')
        pylab.xlabel('data')
        pylab.ylabel('model')
        pylab.title('diagonal occurrence frequencies')
        pylab.axis(ymin=0,xmin=0,ymax=mx,xmax=mx)
        
        # diagonal versus frequency
        row += 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.plot(cDesiredDiag,zValsDiag,'go')
        pylab.xlabel('value')
        pylab.ylabel('z score')
        
        # diagonal histogram
        row += 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.hist(zValsDiag,bins=10,color='g');
        pylab.xlabel('z score')
        pylab.axis(xmin=-4,xmax=4)

        col += 1
        # off-diagonal scatter plot
        row = 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.errorbar(cDesiredFlat,aboveDiagFlat(c1),xerr=cDesiredFlatStdevs,marker='o',ls='')
        mx = max(max(cDesiredFlat),max(aboveDiagFlat(c1)))
        mx = 1.1*mx
        pylab.plot([0,mx],[0,mx],'k-')
        pylab.xlabel('data')
        pylab.ylabel('model')
        pylab.title('off-diagonal cooccurrence frequencies')
        pylab.axis(ymin=0,xmin=0,ymax=mx,xmax=mx)
        
        # off-diagonal versus cooc frequency
        row += 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.plot(cDesiredFlat,zVals,'o')
        pylab.xlabel('value')
        pylab.ylabel('z score')
        
        # off-diagonal histogram
        row += 1
        pylab.subplot(numRows,numCols,col+(row-1)*numCols)
        pylab.hist(zVals,bins=100);
        pylab.xlabel('z score')
        pylab.axis(xmin=-4,xmax=4)
        
        if covTildeMean is not None:
            col += 1
            # off-diagonal covariance plot
            row = 1
            pylab.subplot(numRows,numCols,col+(row-1)*numCols)
            pylab.errorbar(aboveDiagFlat(covTildeMeanData),aboveDiagFlat(covTildeMean), \
                xerr=covTildeFlatStdevs,marker='o',ls='',color='r')
            mn = min(min(aboveDiagFlat(covTildeMeanData)),min(aboveDiagFlat(covTildeMean)))
            mx = max(max(aboveDiagFlat(covTildeMeanData)),max(aboveDiagFlat(covTildeMean)))
            mn = mn - 0.1*(mx-mn)
            mx = mx + 0.1*(mx-mn)
            pylab.plot([mn,mx],[mn,mx],'k-')
            pylab.xlabel('data')
            pylab.ylabel('model')
            pylab.title('off-diagonal covariance tilde')
            pylab.axis(ymin=mn,xmin=mn,ymax=mx,xmax=mx)

            # off-diagonal versus cooc frequency
            row += 1
            pylab.subplot(numRows,numCols,col+(row-1)*numCols)
            pylab.plot(aboveDiagFlat(covTildeMeanData),zValsCov,'ro')
            pylab.xlabel('value')
            pylab.ylabel('zscore')

            # off-diagonal histogram
            row += 1
            pylab.subplot(numRows,numCols,col+(row-1)*numCols)
            pylab.hist(zValsCov,bins=100,color='r');
            pylab.xlabel('z score')
            pylab.axis(xmin=-3,xmax=3)



# 5.10.2013
def findHomogeneousJmatrixBruteForce(fHomog,coocHomog,ell,numSamples=1e5,   
    nSkip=None,maxfev=0,seed=0,paramsStart=None,numProcs=1,                      
    minSize=2,verbose=False,changeSeed=True,                  
    fixedh=None):
    """
    Fits homogeneous Ising model with ell individuals to the 
    frequency fHomog and average cooccurrence frequency coocHomog.
    #average covariance covHomog.
    
    paramsStart (None)              : In the form (hinit,Jinit)
    fixedh (None)                   : If given a value for h, it is
                                      fixed and only J is varied
    """
    
    if changeSeed: seedIter = seedGenerator(seed,1)
    else: seedIter = seedGenerator(seed,0)
    
    def samples(h,J):
        seed = seedIter.next()
        Jmat = replaceDiag( J*scipy.ones((ell,ell)), h*scipy.ones(ell) )
        if numProcs > 1:
            isingSamples = metropolisSampleIsing_pypar(                     \
             numProcs,Jmat,numSamples,startConfig=None,nSkip=nSkip,         \
             seed=seed,minSize=minSize)
        else:
            isingSamples = metropolisSampleIsing(                           \
             Jmat,numSamples,startConfig=None,nSkip=nSkip,                  \
             seed=seed,minSize=minSize)
        isingSamples = scipy.array(isingSamples,dtype=float)
        return isingSamples

    def residualFunc(params):
        if (len(params) == 1) and (fixedh is not None):
            # we're using fixedh
            J = params[0]
            h = fixedh
        elif len(params) == 2:
            # default case
            h,J = params
        else:
            raise Exception
        isingSamples = samples(h,J)
        f = scipy.mean(isingSamples)
        coocMat = cooccurranceMatrixFights(isingSamples,keepDiag=False)
        cooc = scipy.mean(aboveDiagFlat(coocMat))
        cov = cooc - f**2
        if verbose:
            print "findHomogeneousJmatrixBruteForce: h,J =",h,",",J
            print "findHomogeneousJmatrixBruteForce: f-fHomog =",f-fHomog
            #print "findHomogeneousJmatrixBruteForce: cov-covHomog =",cov-covHomog
            print "findHomogeneousJmatrixBruteForce: cooc-coocHomog =",cooc-coocHomog
        #return ((f-fHomog)/fHomog,(cov-covHomog)/covHomog*scipy.sqrt((ell-1.)/2.))
        return ((f-fHomog),(cooc-coocHomog)*scipy.sqrt((ell-1.)/2.))

    if paramsStart is None:
      if fixedh is None:
        paramsStart = (0.,0.)
      else:
        paramsStart = (0.,)

    # stepsize for approximation of the derivative
    # (over which we expect parameters to change the output more than
    #  the order of errors on the output)
    epsfcn = 1./scipy.sqrt(numSamples) #2./scipy.sqrt(numSamples)

    #for i in range(10): # <<<< if this works, change to something systematic

    if True:
        paramsNew,cov_x,infodict,mesg,ier =                             \
            scipy.optimize.leastsq(residualFunc,paramsStart,            \
            maxfev=maxfev,epsfcn=epsfcn,full_output=1)
        print mesg
    else: # 5.29.2013 try other minimizers
        costFunc = lambda p: scipy.sum(scipy.array(residualFunc(p))**2)
        paramsNew,fopt,func_calls,grad_calls,warnflag,allvecs =         \
            scipy.optimize.fmin(costFunc,                            \
            paramsStart,maxfun=maxfev,retall=True, full_output=True)

    paramsStart = paramsNew

    # take explicit gradient step


    if fixedh is None:
        return paramsNew
    else:
        return (fixedh,paramsNew[0])
        

# 2.17.2012 taken from old isingCoocApprox
def histogramMultipliers(isingSamples,Jsamples,Jnew):
    """
    Rescaled samples after perturbing J a bit.
    Used for "histogram" sampling.  See BroDudTka07.
    
    Returns updatedIsingSamples,Z
    """
    deltaJ = Jnew - Jsamples
    ell = len(isingSamples[0])
    # can the following line be sped up?
    #sampleMultipliers = [ scipy.exp(-isingE(sample,deltaJ))                 \
    #                     for sample in isingSamples ]
    sampleMultipliers = scipy.exp( -isingEmultiple(isingSamples,deltaJ) )
    Z = scipy.mean(sampleMultipliers)
    #Zfactor = len(isingSamples)/scipy.sum(sampleMultipliers)
    return sampleMultipliers,Z

# 1.27.2012
# 2.17.2012 corrected with sqrt
def isingCoocApprox(isingSamples,Jsamples,Jnew):
    """
    Approximate guess for cooc matrix after perturbing J a bit.
    Used for "histogram" sampling.  See BroDudTka07.
    """
    ell = len(isingSamples[0])
    sampleMultipliers,Z =                                                   \
        histogramMultipliers(isingSamples,Jsamples,Jnew)
    sampleMultipliersRepeated =                                             \
        scipy.repeat(scipy.transpose([sampleMultipliers]),ell,axis=1)
    updatedIsingSamples = isingSamples*scipy.sqrt(sampleMultipliersRepeated)
    return cooccurranceMatrixFights(updatedIsingSamples,keepDiag=True)/Z

# 1.27.2012
def isingDeltaCoocApprox(isingSamples,coocMatDesired,Jsamples,Jnew):
    coocApprox = isingCoocApprox(isingSamples,Jsamples,Jnew)
    return aboveDiagFlat(coocApprox-coocMatDesired,keepDiag=True)

# 2.17.2012 not actually using this
if False:
    def isingFourthApprox(isingSamples,Jsamples,Jnew): 
        ell = len(isingSamples[0])
        sampleMultipliers,Z =                                                   \
            histogramMultipliers(isingSamples,Jsamples,Jnew)
        sampleMultipliersRepeated =                                             \
            scipy.repeat(scipy.transpose([sampleMultipliers]),ell,axis=1)
        updatedIsingSamples = isingSamples*                                     \
            scipy.sqrt(scipy.sqrt(sampleMultipliersRepeated))
        return fourthOrderCoocMat(updatedIsingSamples)/Z

# 3.1.2012
def coocStdevsFlatOld(coocMat,numFights,zeroReplacement=0.5):
    """
    Returns a flattened expected standard deviation matrix used
    to divide deltaCooc to turn it into z scores.
    
    zeroReplacement (0.5)       : Zero cooccurrence frequencies are
                                  replaced by zeroReplacement/numFights
    """
    zr = zeroReplacement
    zeroVarianceVal = zr/numFights*(1.-zr/numFights)/numFights
    varianceMatFlat = aboveDiagFlat(coocMat*(1.-coocMat)/numFights,keepDiag=True)
    varianceMatFlat += (varianceMatFlat==0.)*zeroVarianceVal
    return scipy.sqrt(varianceMatFlat)

# 2.21.2013
def coocStdevsFlat(coocMat,numFights):
    """
    Returns a flattened expected standard deviation matrix used
    to divide deltaCooc to turn it into z scores.
    """
    coocMatMean = coocMatBayesianMean(coocMat,numFights)
    varianceMatFlat = aboveDiagFlat(coocMatMean*(1.-coocMatMean)/numFights,keepDiag=True)
    return scipy.sqrt(varianceMatFlat)

# 3.5.2013
def coocMatBayesianMean(coocMat,numFights):
    """
    Using "Laplace's method"
    """
    return (coocMat*numFights + 1.)/(numFights + 2.)




# 3.3.2011 --- exact Ising stuff ---

# 5.1.2014 changed removeZerosAndOnes to minSize

def fightPossibilities(ell,minSize=0):
    fightNumbers = range(2**ell)
    fp = [ [ int(x) for x in scipy.binary_repr(fN,ell) ]                  \
             for fN in fightNumbers ]
    if minSize > 0:
        fp = scipy.array( filter(lambda x: sum(x)>=minSize, fp) )
    return fp

def unsummedZ(J,hext=0,minSize=0):
    """
    J should have h on the diagonal.
    """
    return scipy.exp( unsummedLogZ(J,hext=hext,minSize=minSize) )

def unsummedLogZ(J,hext=0,minSize=0):
    """
    J should have h on the diagonal.
    """
    ell = len(J)
    h = scipy.diag(J)
    JnoDiag = J - scipy.diag(h)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    return -dot(fp,h-hext)-1.0*scipy.sum(dot(fp,JnoDiag)*fp,axis=1)

def freqExpectations(J,hext=0,minSize=0):
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    Z = unsummedZ(J,hext,minSize)
    return dot(fp.T,Z)/sum(Z)
    
def coocExpectations(J,hext=0,zeroBelowDiag=True,minSize=0):
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    coocp = scipy.array([ scipy.outer(f,f) for f in fp ])
    Z = unsummedZ(J,hext,minSize)
    coocSym = dot(coocp.T,Z)/sum(Z)
    if zeroBelowDiag:
        coocTri = coocSym * scipy.tri(ell).T
        return coocTri
    else:
        return coocSym

# 2.6.2014
def cooc4Expectations(J,hext=0):
    """
    Returns a symmetric matrix with dimensions 
    [ ell*(ell+1)/2 x ell*(ell+1)/2 ]
    
    Probably slower than it needs to be.
    """
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell) )
    cooc4p = scipy.array([ fourthOrderCoocMat([f]) for f in fp ])
    Z = unsummedZ(J,hext)
    cooc4Sym = dot(cooc4p.T,Z)/sum(Z)
    #coocTri = coocSym * scipy.tri(ell).T
    return cooc4Sym
    
# 7.9.2012
def suscExpectation(J,hext=0,minSize=0):
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    sizes = sum(fp,axis=1)
    Z = unsummedZ(J,hext,minSize)
    return (dot(sizes**2,Z)/sum(Z) - (dot(sizes,Z)/sum(Z))**2)/ell

# 7.7.2016
def specificHeatExpectation(J,hext=0,minSize=0):
    """
    In units such that k_B T = 1...
    """
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    energies = scipy.sum( dot(fp,J)*fp, axis=1 )
    Z = unsummedZ(J,hext,minSize)
    return (dot(energies**2,Z)/sum(Z) - (dot(energies,Z)/sum(Z))**2)/ell

# 8.28.2012
def tripletExpectations(J,hext=0,minSize=0):
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    tripletp = scipy.array([ scipy.reshape(scipy.outer(scipy.outer(f,f),f),(ell,ell,ell)) for f in fp ])
    Z = unsummedZ(J,hext,minSize)
    triplet = dot(tripletp.T,Z)/sum(Z)
    return triplet

# 3.30.2017
def sumExpectations(J,hext=0,minSize=0):
    """
    Returns list corresponding to the probability of each sum,
    ranging from 0 to len(J).
    """
    ell = len(J)
    fp = scipy.array( fightPossibilities(ell,minSize) )
    sizes = sum(fp,axis=1)
    Z = unsummedZ(J,hext,minSize)
    sumPossibilities = range(0,ell+1)
    sProbs = [ dot(sizes==s,Z)/sum(Z) for s in sumPossibilities ]
    return scipy.array( sProbs )

# 7.6.2012
def findJmatrixAnalytic_CoocMat(coocMatData,Jinit=None,bayesianMean=False,
    numSamples=None,priorLmbda=0.,minSize=0):
    
    ell = len(coocMatData)
    
    if priorLmbda != 0.:
        lmbda = priorLmbda / numSamples
    else:
        lmbda = 0.
    
    if bayesianMean:
        coocMatDesired = coocMatBayesianMean(coocMatData,numSamples)
    else:
        coocMatDesired = coocMatData
    
    if Jinit is None:
        # 1.17.2012 try starting from frequency model
        freqs = scipy.diag(coocMatDesired)
        hList = -scipy.log(freqs/(1.-freqs))
        Jinit = scipy.diag(hList)
    
    def deltaCooc(Jflat):
        J = unflatten(Jflat,ell)
        cooc = coocExpectations(J,minSize=minSize)
        dCooc = aboveDiagFlat(cooc - coocMatDesired,keepDiag=True)
        if (lmbda > 0.) and (ell > 1):
            freqs = scipy.diag(coocMatDesired)
            factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
            factorFlat = aboveDiagFlat(factor)
            # 3.24.2014 changed from lmbda/2. to lmbda
            priorTerm = lmbda * factorFlat * Jflat[ell:]**2 
            dCooc = scipy.concatenate([dCooc,priorTerm])
        return dCooc
    
    JinitFlat = aboveDiagFlat(Jinit,keepDiag=True)
    Jflat = scipy.optimize.leastsq(deltaCooc,JinitFlat)[0]
    Jnew = unflatten(Jflat,ell,symmetrize=True)
    
    return Jnew

# 2.18.2014
def analyticEntropy(J):
    """
    In nats.
    """
    Z = unsummedZ(J)
    p = Z / scipy.sum(Z)
    return - scipy.sum( p * scipy.log(p) )

# 12.12.2014
def independentEntropy(J):
    """
    Entropy of noninteracting model with individual frequencies given
    by the Ising model corresponding to J.  Equal to the first term 
    in the total correlation.
    
    In nats.
    """
    freqs = freqExpectations(J)
    hList = -scipy.log(freqs/(1.-freqs))
    Jindep = scipy.diag(hList)
    return analyticEntropy(Jindep)

# 1.2015
def totalCorrelation(J,normed=True):
    H1 = independentEntropy(J)
    Hn = analyticEntropy(J)
    if normed: return (H1 - Hn)/Hn
    else: return H1 - Hn

# 1.27.2015
def ditDist(J,minSize=0):
    ell = len(J)
    outcomes = [ ''.join([str(i) for i in fight]) \
                for fight in fightPossibilities(ell) ]
    logZ = unsummedLogZ(J,minSize=minSize)
    logpmf = logZ - scipy.log(sum(scipy.exp(logZ)))
    d = dit.Distribution(outcomes,logpmf,base=scipy.e)
    return d

# 1.27.2015
def bindingInfo(J,minSize=0,bits=False):
    """
    Binding information of Ising model, calculated by dit.
    
    Measured in nats.
    """
    d = ditDist(J,minSize=minSize)
    nats = dit.multivariate.binding_information(d)
    if bits: return nats2bits(nats)
    else: return nats

# 1.27.2015
def coInfo(J,minSize=0,bits=False):
    """
    Coinformation of Ising model, calculated by dit.
    
    Measured in nats.
    """
    d = ditDist(J,minSize=minSize)
    nats = dit.multivariate.coinformation(d)
    if bits: return nats2bits(nats)
    else: return nats
