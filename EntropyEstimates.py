# EntropyEstimates.py
#
# Bryan Daniels
# 10.21.2010
#
# from WolWol95 and NemShaBia08 

from .SparsenessTools import *
import scipy
from scipy.special import gamma,gammaln,polygamma
from scipy.integrate import quad

Phi = lambda n,z: polygamma(n-1,z)
deltaPhi = lambda n,z1,z2: Phi(n,z1) - Phi(n,z2)

# 1.6.2012
def naiveEntropy(dist):
    """
    In bits.
    """
    eps = 1.e-6
    if abs(1. - sum(dist)) > eps:
        raise Exception("Distribution is not normalized.")
    return -scipy.sum( scipy.nan_to_num(dist*scipy.log2(dist)) )

def meanEntropy(nVec,beta=1,m=None):
    """
    Measured in nats (I think)
    
    m (None)            : If m=None, assume nVec lists all 
                          possibilities ( len(nVec) = m ).
                          If a number is given, it's used 
                          and assumed that all other bins
                          have zero counts. 
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    numBins = float( len(nVec) )
    if m is None:
        m = len(nVec) # aka K
    # assuming we can use result from WolWol95 and replace
    # n_i + 1 --> n_i + beta
    # m --> beta*m
    # (I think this is correct, based on the definition of beta
    #  and p.33 of WolWol94.)
    return Phi(1,N+beta*m+1) +                                  \
           sum( - (nVec+beta)/(N+beta*m) * Phi(1,nVec+beta+1) ) \
           - (m-numBins) * beta/(N+beta*m) * Phi(1,beta+1)
    
    return sum( - (nVec+beta)/(N+beta*m)                        \
                * deltaPhi(1,nVec+beta+1,N+beta*m+1) )          \
                - (m-numBins) * beta/(N+beta*m)                 \
                * deltaPhi(1,beta+1,N+beta*m+1)
    
# 10.25.2010
def s2s0Slow(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    total = 0.
    for i in range(m):
      for j in range(m):
        if i == j:
          total += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+2,N+beta*m+2)**2          \
            + deltaPhi(2,nVec[i]+beta+2,N+beta*m+2))
        else:
          total += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+2)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+2)             \
             -Phi(2,N+beta*m+2))
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010 version made to match with NemShaBia08
# (turns out to be identical)
def s2s0NemBetaOld(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    totalOffDiag,totalDiag = 0.,0.
    for i in range(m):
      for j in range(m):
        if i == j:
          totalDiag += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+1)**2          \
            + deltaPhi(2,nVec[i]+beta+1,N+beta*m+1))
        else:
          totalOffDiag += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+1)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+1)             \
             -Phi(2,N+beta*m+1))
    print("Phi(2,N+beta*m+1) =",Phi(2,N+beta*m+1))
    print("totalOffDiag =",totalOffDiag)
    print("totalDiag =",totalDiag)
    total = totalDiag + totalOffDiag
    return total / ( (N+beta*m)*(N+beta*m+1) )
    
# 10.25.2010 version made to match with NemShaBia08
# (turns out to be identical)
# 6.21.2011 (turns out to be identical?)
def s2s0NemBetaNew(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    totalOffDiag,totalDiag = 0.,0.
    for i in range(m):
      for j in range(m):
        if i == j:
          totalDiag += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+2,N+beta*m+2)**2          \
            + deltaPhi(2,nVec[i]+beta+2,N+beta*m+2))
        else:
          totalOffDiag += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+2)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+2)             \
             -Phi(2,N+beta*m+2))
    print("Phi(2,N+beta*m+1) =",Phi(2,N+beta*m+2))
    print("totalOffDiag =",totalOffDiag)
    print("totalDiag =",totalDiag)
    total = totalDiag + totalOffDiag
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010
# 6.21.2011 fixed bugs
def s2s0(nVec,beta=1,m=None,useLessMemory=False):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    numBins = len(nVec)
    if m is None:
        m = numBins # aka K
    factor1 = nVec+beta
    factor2 = deltaPhi(1,nVec+beta+1,N+beta*m+2)
    factor1zero = beta
    factor2zero = deltaPhi(1,beta+1,N+beta*m+2)
    if (m > 1e4) or useLessMemory or m!=numBins:
        # use different (slower?) method that uses
        # less memory
        total = 0
        p2 = Phi(2,N+beta*m+2)
        for i in range(numBins):
            sumVec = scipy.dot(factor1[i],factor1)*             \
                   ( scipy.dot(factor2[i],factor2) - p2 )
            # remove diagonal
            sumVec[i] = 0.
            total += scipy.sum(sumVec)
            # 7.19.2011 take into account extra zeros
            sumVecZeros = 2.*(m-numBins)*factor1[i]*factor1zero*\
                          (factor2[i]*factor2zero - p2)
            total += sumVecZeros
        # add zero-zero elements (except diagonal)
        total += (m-numBins)*((m-numBins)-1)*                   \
                 factor1zero*factor1zero*                       \
                (factor2zero*factor2zero - p2)
        # add zero-zero diagonal elements
        total += (m-numBins)*factor1zero*(factor1zero+1)*       \
                 ( deltaPhi(1,beta+2,N+beta*m+2)**2             \
                + deltaPhi(2,beta+2,N+beta*m+2) )
                
    else:
        sumMatrix = scipy.outer(factor1,factor1)*               \
            ( scipy.outer(factor2,factor2)-Phi(2,N+beta*m+2) )
        # remove diagonal
        sumMatrix = sumMatrix - scipy.diag(scipy.diag(sumMatrix))
        total = scipy.sum(sumMatrix)
    
    diagonal = factor1*(factor1+1)*                             \
        ( deltaPhi(1,nVec+beta+2,N+beta*m+2)**2                 \
        + deltaPhi(2,nVec+beta+2,N+beta*m+2) )
    #sumMatrix += scipy.diag(diagonal)
    total += scipy.sum(diagonal)
        
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010
def varianceEntropy(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    """
    return s2s0(nVec,beta) - meanEntropy(nVec,beta)**2
    
# 10.25.2010
def varianceEntropyNemZero(beta,K):
    beta,K = float(beta),float(K)
    return (beta+1)/(beta*K+1)*Phi(2,beta+1) - Phi(2,beta*K+1)
    
def xiFromBeta(beta,K):
    """
    Modified to be an odd function, such that finding 
    the inverse is easier.
    """
    oddMultiplier = (beta > 0.)*2 - 1
    beta = abs(beta)
    kappa = K*beta
    xi = polygamma(0,kappa+1) - polygamma(0,beta+1)
    return oddMultiplier * xi
    
def xiFromBetaPrime(beta,K):
    """
    (Using the modified xi -- see xiFromBeta)
    """
    beta = abs(beta)
    kappa = K*beta
    xiDeriv = K*polygamma(1,kappa+1) - polygamma(1,beta+1)
    return xiDeriv
    
def betaFromXi(xi,K):
    xtol = 1e-20
    maxiter = 10000
    betaMin,betaMax = 0.,100. 
    while xiFromBeta(betaMax,K) < xi:
        betaMax *= 10
    
    #betaStart = 0.
    #beta = scipy.optimize.newton(                               \
    #    lambda beta:(xiFromBeta(beta,K) - xi)/xi, betaStart,    \
    #    fprime=lambda beta:xiFromBetaPrime(beta,K)/xi,             \
    #    maxiter=maxiter,tol=tol)
    beta = scipy.optimize.brentq(                               \
        lambda beta:(xiFromBeta(beta,K) - xi)/xi, betaMin, betaMax,    \
        maxiter=maxiter,xtol=xtol)
    if abs((xiFromBeta(beta,K) - xi)/xi) > 0.01:
        print("found beta =",beta)
        print("xi desired =",xi)
        print("xi found =",xiFromBeta(beta,K))
        raise Exception("Loss of precision in betaFromXi.")
    return beta
        
def lnXiDistrib(xi,nVec,K=None):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    if K is None:
        K = len(nVec) # aka m
    beta = betaFromXi(xi,K)
    kappa = K*beta
    prod,exp = scipy.prod,scipy.exp
    #return gamma(kappa)/gamma(N+kappa)                         \
    #     * scipy.prod( gamma(nVec+beta)/gamma(beta) )
    #return exp( gammaln(kappa) - gammaln(N+kappa) )            \
    #    * prod( exp( gammaln(nVec+beta) - gammaln(beta) ) )
    return gammaln(kappa) - gammaln(N+kappa)                    \
        + sum( gammaln(nVec+beta) - gammaln(beta) ) 
         
def integrateOverXi(func,nVec,maxScaledXi=0.9999,               \
    constMult=True,range=None,K=None,verbose=False):
    """
    maxScaledXi used to stay away from the problematic 
    endpoint at log(K)...
    
    constMult       : Multiply integrand by a constant that 
                      depends only on nVec [currently 
                      e^(-max(lnXiDistrib)); 
                      used to be e^(-<lnXiDistrib>)].  Designed to remove 
                      difficulties with extremely small xiDistrib.
    """
    if K is None:
        K = len(nVec)
    if range is None:
        min,max = 0.,maxScaledXi*scipy.log(K)
    else:
        min,max = range
    exp = scipy.exp
    if constMult:
        # use average LnXi
        #fn = lambda xi: lnXiDistrib(xi,nVec)
        #avgLnXi = quad(fn,min,max)[0]/(max-min)
        #lnConst = -avgLnXi
        
        # 3.30.2011 use max LnXi
        def fn(xi): 
            #if xi < max: return -lnXiDistrib(xi,nVec,K=K)
            #else: return -lnXiDistrib(max,nVec,K=K)
            if xi >= max: return -lnXiDistrib(max,nVec,K=K)
            elif xi <= min: return -lnXiDistrib(min,nVec,K=K) 
            else: return -lnXiDistrib(xi,nVec,K=K)
        xiMax =                                                 \
            scipy.optimize.fmin(fn,(max+min)/2.,maxiter=100,    \
                disp=verbose)[0]
        if xiMax > max: xiMax = max
        if xiMax < min: xiMax = min
        lnConst = fn(xiMax)
       
        if verbose:
            print("lnConst =",lnConst)
    else:
        lnConst = 0.
    integrand = lambda xi:                                      \
        exp(lnConst+lnXiDistrib(xi,nVec,K=K))*func(xi)
    if verbose:
        print("maxBeta = ",betaFromXi(max,K))
        print("integrating over",min,"< xi <",max)
    return quad(integrand,min,max,epsrel=1e-10)
    
def meanEntropyNem(nVec,K=None,verbose=False,**kwargs):
    """
    Flat prior on beta.
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    if K is None:
        K = len(nVec) # aka m
    entropyFunc =                                               \
        lambda xi: meanEntropy(nVec,beta=betaFromXi(xi,K),m=K)
    numInt = integrateOverXi(entropyFunc,nVec,K=K,              \
        verbose=verbose,**kwargs)
    denInt = integrateOverXi(lambda x:1,nVec,K=K,               \
        verbose=verbose,**kwargs)
    num,den = numInt[0],denInt[0]
    if verbose:
        print("num =",num,", den =",den)
        print("numAbsErr =",numInt[1],", denAbsErr =",denInt[1])
        print("s1s0 =",num/den)
    return num/den
    
def s2s0Nem(nVec,K=None,verbose=False,**kwargs):
    """
    Flat prior on beta.
    """
    nVec = scipy.array(nVec)
    N = sum(nVec)
    if K is None:
        K = len(nVec) # aka m
    s2s0Func =                                                  \
        lambda xi: s2s0(nVec,beta=betaFromXi(xi,K),m=K)
    numInt = integrateOverXi(s2s0Func,nVec,K=K,                 \
        verbose=verbose,**kwargs)
    denInt = integrateOverXi(lambda x:1,nVec,K=K,               \
        verbose=verbose,**kwargs)
    num,den = numInt[0],denInt[0]
    if verbose:
        print("num =",num,", den =",den)
        print("numAbsErr =",numInt[1],", denAbsErr =",denInt[1])
        print("s2s0 =",num/den)
    return num/den
    
    
# 3.29.2011
def meanAndStdevEntropyNem(freqData,bits=True,**kwargs):
    mean = meanEntropyNem(freqData,**kwargs)
    s2s0 = s2s0Nem(freqData,**kwargs)
    stdev = scipy.sqrt(s2s0-mean*mean)
    if bits:
        mean = nats2bits( mean )
        stdev = nats2bits( stdev )
    return mean, stdev
    
def varianceEntropyNem(nVec,**kwargs):
    """
    Flat prior on beta.
    """
    print("10.26.2010 Not sure if this is correct.")
    # do we have to integrate s2s0 separately?
    nVec = scipy.array(nVec)
    N = sum(nVec)
    K = len(nVec) # aka m
    varianceFunc =                                              \
        lambda xi: varianceEntropy(nVec,beta=betaFromXi(xi,K))
    num = integrateOverXi(varianceFunc,nVec,**kwargs)[0]
    den = integrateOverXi(lambda x:1,nVec,**kwargs)[0]
    return num/den

# 3.29.2011
def nats2bits(nats):
    return nats * scipy.log2(scipy.e)
    
# 7.18.2011 for use with nsbEntropyMATLAB
def fights2kxnx(fights,useEfficientVersion=True):
    nList = []
    fightsCopy = list( copy.copy(fights) )
    
    if useEfficientVersion:
        fightNumberList = fight2number(fights)
        fightNumbersSorted = scipy.sort(fightNumberList)
        #print "fightNumbersSorted =",fightNumbersSorted
        diffJumpLocs = pylab.find(                                  \
            (fightNumbersSorted[1:]-fightNumbersSorted[:-1])>0.)
        #print "diffJumpLocs =",diffJumpLocs
        nList = list(diffJumpLocs[1:]-diffJumpLocs[:-1])
        # add for beginning and end
        nList.append(diffJumpLocs[0]+1)
        nList.append(len(fights)-diffJumpLocs[-1]-1)
    else:
        while len(fightsCopy)>0:
          fight1 = fightsCopy[0]
          fightsCopy.remove(fight1)
          n = 1
          for fight2 in fightsCopy:
            if fight1 == fight2:
              n += 1
              fightsCopy.remove(fight2)
          nList.append(n)
    return scipy.ones_like(nList),scipy.array(nList)

    
# 7.18.2011
def nsbEntropyMATLAB(fights,bits=True,precision=1e-5,K=None):
    
    codeDir = "./nsb-entropy-oct_1.11_BCD"
    outputFilename =                                            \
        "find_nsb_entropy_python_interface_output.txt"
    kxFilename = "kx_temp.data"
    nxFilename = "nx_temp.data"
    
    if K is None:
        K = 2**len(fights[0])
    kx,nx = fights2kxnx(fights)
    
    scipy.savetxt(kxFilename,kx)
    scipy.savetxt(nxFilename,nx)
    
    callStr = "find_nsb_entropy_python_interface('"             \
        +kxFilename+"','"+nxFilename+"',"+str(K)+","            \
        +str(precision)+");"
    try:
        S,dS,errcode = callMATLAB(callStr,outputFilename,       \
            codeDirList=[codeDir])
        os.remove(kxFilename)
        os.remove(nxFilename)
    except:
        os.remove(kxFilename)
        os.remove(nxFilename)
        raise
        
    if bits:
        S,dS = nats2bits(S),nats2bits(dS)
    
    return S,dS


# 9.20.2011
def entropyFightSizeExpansion(fightList):
    ell = len(fightList[0])
    numFights = len(fightList)
    meanSlist,stdSlist,fightSizeNVals = [],[],[]
    for fightSize in range(ell):
        fightsFiltered = filter(lambda x: sum(x)==fightSize,fightList)
        fightSizeNVals.append(len(fightsFiltered))
        fightPatternDict = {}
        for fight in fightsFiltered:
            fight = tuple(fight)
            if fight in fightPatternDict.keys():
                fightPatternDict[fight] += 1
            else:
                fightPatternDict[fight] = 1
        nVals = fightPatternDict.values() # how many of each fight pattern were seen
        K = scipy.misc.comb(ell,fightSize) # how many fight patterns are possible
        
        if sum(nVals)>0: # if there are a nonzero number of samples
            meanS,stdS = meanAndStdevEntropyNem(nVals,K=K)
        else:
            meanS,stdS = 0,0
        meanSlist.append(meanS)
        stdSlist.append(stdS)
        
        print(fightSize, float(sum(nVals))/scipy.sqrt(K), meanS, stdS)
        
    fightSizeEnt,stdFightSizeEnt = meanAndStdevEntropyNem(fightSizeNVals)
        
    fightSizeDist = scipy.array(fightSizeNVals)/float(numFights)
    return meanSlist*fightSizeDist,stdSlist*fightSizeDist,fightSizeEnt,stdFightSizeEnt


















if False:
    # 6.20.2011 TEST
    def sigmaNemBeta(nVec,beta=1):
        s2 = s2s0(nVec,beta=beta)
        s1 = meanEntropy(nVec,beta=beta)
        return scipy.sqrt(s2-s1*s1)

    # 6.20.2011 TEST
    def sigmaNem(nVec,**kwargs):
        """
        Flat prior on beta.
        """
        nVec = scipy.array(nVec)
        N = sum(nVec)
        K = len(nVec) # aka m
        sigmaFunc =                                                  \
            lambda xi: sigmaNemBeta(nVec,beta=betaFromXi(xi,K))
        num = integrateOverXi(sigmaFunc,nVec,**kwargs)[0]
        den = integrateOverXi(lambda x:1,nVec,**kwargs)[0]
        return num/den
