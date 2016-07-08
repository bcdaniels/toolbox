# meanFieldIsing.py
#
# Bryan Daniels
# 3.7.3012
#

import scipy
import scipy.optimize
import pylab
from SparsenessTools import replaceDiag,zeroDiag

exp,cosh = scipy.exp,scipy.cosh
dot = scipy.dot

def m(h,J,ell,T):
    """
    Careful if T is small for loss of precision?
    """
    func = lambda mm: mm - 1./(1.+exp((h+2.*(ell-1.)*mm*J)/T))
    #dfunc = lambda m: 1. - (ell-1.)*J/T /                   \
    #    ( cosh((h-(ell-1.)*m*J)/(2.*T)) )**2
    #***
    #ms = scipy.linspace(-0.1,1.1,100)
    #pylab.plot(ms,[func(m) for m in ms])
    #***
    #mRoot0 = 0.5
    
    mRoot = scipy.optimize.brentq(func,-0.,1.)
    return mRoot
    
def avgE(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    
    return -mr*hloc
    
def dmdT(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    
    return hloc/(4.*T**2) / cosh(-hloc/(2.*T))**2 / (1.+(ell-1)*J/(4.*T)*cosh(-hloc/(2.*T))**-2)
    
def specificHeat(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h
    denomFactor = 4.*T/(ell-1)/J * cosh(-hloc/(2.*T))**2
    
    return hloc*(2*(ell-1)*mr*J - h)
    """
      /                      \
        ( T*(ell-1)*J*                                      \
             ( 1. + denomFactor ) * \
             ( 1. + 1./denomFactor ) )
    """

# 6.23.2014
def susc(h,J,ell,T):
    mr = m(h,J,ell,T)
    hloc = (ell-1)*mr*J - h

    return 1./(T**2 * (exp(-hloc) + exp(+hloc) + 2.))

# 3.27.2014 moved from selectiveClusterExpansion.py
def coocCluster(coocMat,cluster):
    #orderedIndices = scipy.sort(cluster)
    orderedIndices = cluster
    newMat = scipy.array(coocMat)[:]
    newMat = newMat[orderedIndices,:]
    newMat = newMat[:,orderedIndices]
    return newMat

# 3.27.2014 moved from selectiveClusterExpansion.py
def JfullFromCluster(Jcluster,cluster,N):
    """
    There is perhaps a faster way of doing this?
    """
    J = scipy.zeros((N,N))
    for i,iFull in enumerate(cluster):
        for j,jFull in enumerate(cluster):
            J[iFull,jFull] = Jcluster[i,j]
    return J

# 3.27.2014 moved from selectiveClusterExpansion.py
def symmetrizeUsingUpper(mat):
    if len(mat) != len(mat[0]): raise Exception
    d = scipy.diag(mat)
    matTri = (1.-scipy.tri(len(mat)))*mat
    matSym = replaceDiag(matTri+matTri.T,d)
    return matSym

# 3.27.2014 moved from selectiveClusterExpansion.py
# Eqs. 30-33
def SmeanField(cluster,coocMat,meanFieldPriorLmbda=0.,
    numSamples=None,indTerm=True,alternateEnt=False,
    useRegularizedEq=True):
    """
    meanFieldPriorLmbda (0.): 3.23.2014
    indTerm (True)          : As of 2.19.2014, I'm not
                              sure whether this term should
                              be included, but I think so
    alternateEnt (False)    : Explicitly calculate entropy
                              using the full partition function
    useRegularizedEq (True) : Use regularized form of equation
                              even when meanFieldPriorLmbda = 0.
    """
    
    coocMatCluster = coocCluster(coocMat,cluster)
    # in case we're given an upper-triangular coocMat:
    coocMatCluster = symmetrizeUsingUpper(coocMatCluster)
    
    outer = scipy.outer
    N = len(cluster)
    
    freqs = scipy.diag(coocMatCluster)
    c = coocMatCluster - outer(freqs,freqs)
    
    Mdenom = scipy.sqrt( outer(freqs*(1.-freqs),freqs*(1-freqs)) )
    M = c / Mdenom
    
    if indTerm:
        Sinds = -freqs*scipy.log(freqs)             \
            -(1.-freqs)*scipy.log(1.-freqs)
        Sind = scipy.sum(Sinds)
    else:
        Sind = 0.
    
    # calculate off-diagonal (J) parameters
    if (meanFieldPriorLmbda != 0.) or useRegularizedEq:
        # 3.22.2014
        if meanFieldPriorLmbda != 0.:
            gamma = meanFieldPriorLmbda / numSamples
        else:
            gamma = 0.
        mq,vq = scipy.linalg.eig(M)
        mqhat = 0.5*( mq-gamma +                        \
                scipy.sqrt((mq-gamma)**2 + 4.*gamma) )
        jq = 1./mqhat #1. - 1./mqhat
        Jprime = scipy.real_if_close(                   \
                dot( vq , dot(scipy.diag(jq),vq.T) ) )
        JMF = zeroDiag( Jprime / Mdenom )
        
        ent = scipy.real_if_close(                      \
                Sind + 0.5*scipy.sum( scipy.log(mqhat)  \
                + 1. - mqhat ) )
    else:
        # use non-regularized equations
        Minv = scipy.linalg.inv(M)
        JMF = zeroDiag( Minv/Mdenom )
        
        logMvals = scipy.log( scipy.linalg.svdvals(M) )
        ent = Sind + 0.5*scipy.sum(logMvals)
    
    # calculate diagonal (h) parameters
    piFactor = scipy.repeat( [(freqs-0.5)/(freqs*(1.-freqs))],
                            N, axis=0).T
    pjFactor = scipy.repeat( [freqs], N, axis=0 )
    factor2 = c*piFactor - pjFactor
    hMF = scipy.diag( scipy.dot( JMF, factor2.T  ) ).copy()
    if indTerm:
        hMF -= scipy.log(freqs/(1.-freqs))
    
    J = replaceDiag( 0.5*JMF, hMF )
    
    if alternateEnt:
        ent = analyticEntropy(J)
    
    # make 'full' version of J (of size NfullxNfull)
    Nfull = len(coocMat)
    Jfull = JfullFromCluster(J,cluster,Nfull)
    
    return ent,Jfull


# 11.20.2014 for convenience
def JmeanField(coocMat,**kwargs):
    """
    See SmeanField for important optional arguments,
    including noninteracting prior weighting.
    """
    ell = len(coocMat)
    S,JMF = SmeanField(range(ell),coocMat,**kwargs)
    return JMF


# 11.21.2014
def meanFieldStability(J,freqs):
    # 6.26.2013
    #freqs = scipy.mean(samples,axis=0)
    f = scipy.repeat([freqs],len(freqs),axis=0)
    m = -2.*zeroDiag(J)*f*(1.-f)
    stabilityValue = max(abs( scipy.linalg.eigvals(m) ))
    return stabilityValue


# 3.6.2015
# exact form of log(cosh(x)) that doesn't die at large x
def logCosh(x):
    return abs(x) + scipy.log(1. + scipy.exp(-2.*abs(x))) - scipy.log(2.)

# 3.6.2015
# see notes 3.4.2015
def FHomogeneous(h,J,N,m):
    """
    Use Hubbard-Stratonovich (auxiliary field) to calculate the
    (free energy?) of a homogeneous system as a function of the
    field m (m equals the mean field as N -> infinity?).
    """
    Jbar = N*J
    s = scipy.sqrt(scipy.pi/(N*Jbar))
    L = Jbar * m*m - scipy.log(2.) - logCosh(2.*Jbar*m + h)
    return N*L + scipy.log(s)

# 3.6.2015
def dFdT(h,J,N,m):
    Jbar = N*J
    return -N*Jbar*m*m + N*(2.*Jbar*m + h)*scipy.tanh(2.*Jbar*m + h) + 0.5

# 3.6.2015
def SHomogeneous(h,J,N):
    """
    Use Hubbard-Stratonovich (auxiliary field) to numerically 
    calculate entropy of a homogeneous system.
    """
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]

    dFdTFunc = lambda m: dFdT(h,J,N,m) * scipy.exp(-FHomogeneous(h,J,N,m))
    avgdFdT = scipy.integrate.quad(dFdTFunc,-scipy.inf,scipy.inf)[0] / Z

    return scipy.log(Z) - avgdFdT

# 3.6.2015
def avgmHomogeneous(h,J,N):
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]
    
    mFunc = lambda m: m * scipy.exp(-FHomogeneous(h,J,N,m))
    avgm = scipy.integrate.quad(mFunc,-scipy.inf,scipy.inf)[0] / Z

    return avgm

# 3.6.2015
def avgxHomogeneous(h,J,N):
    Zfunc = lambda m: scipy.exp(-FHomogeneous(h,J,N,m))
    Z = scipy.integrate.quad(Zfunc,-scipy.inf,scipy.inf)[0]
    
    Jbar = N*J
    xFunc = lambda m: scipy.tanh(2.*Jbar*m+h) * scipy.exp(-FHomogeneous(h,J,N,m))
    avgx = scipy.integrate.quad(xFunc,-scipy.inf,scipy.inf)[0] / Z
    
    return avgx

# 3.6.2015
def multiInfoHomogeneous(h,J,N):
    Sind = independentEntropyHomogeneous(h,J,N)
    S = SHomogeneous(h,J,N)
    return Sind - S

# 3.6.2015
def independentEntropyHomogeneous(h,J,N):
    avgx = avgxHomogeneous(h,J,N)
    S1 = - (1.+avgx)/2. * scipy.log((1.+avgx)/2.) \
        - (1.-avgx)/2. * scipy.log((1.-avgx)/2.)
    return N*S1

# 3.6.2015
def independentEntropyHomogeneous2(h,J,N):
    avgx = avgxHomogeneous(h,J,N)
    heff = scipy.arctanh(avgx)
    return N*(scipy.log(2.) + scipy.log(scipy.cosh(heff)) - avgx*heff)


