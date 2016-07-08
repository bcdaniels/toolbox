# monteCarloSample.py
#
# Bryan Daniels
# 11.20.2013 branched from generateFightData.py
#

import scipy
import copy
import pylab # for find, filter
import os # for getpid
from isingSample import metropolisSampleIsing
from subprocess import call, STDOUT
from simplePickle import load,save


    
# 4.1.2011
def randomIndivMoveFunc(oldState,rand):
    randomIndiv = scipy.floor( rand*len(oldState) )
    #newState = copy.copy(oldState)
    newState = oldState.copy()
    newState[randomIndiv] = (newState[randomIndiv] + 1)%2
    return newState
    
# 4.1.2011
# as of 4.7.2011, not yet used
def basisMoveFunc(basis,numNonzero):
    """
    Thresholds given basis to get correlated individuals.  Steps
    are taken that flip all individuals in a given group.  Groups
    consisting of each individual individually are also included.
    
    Note: Doesn't allow for individuals of opposite sign in basis.
    """
    binaryBasis = abs(thresholdMatrix(basis,numNonzero=numNonzero))>0.
    binaryBasis = pylab.filter(lambda v: sum(v)>0, binaryBasis)
    # we still want to include individual bit flips
    binaryBasis = scipy.concatenate(                                        \
        [binaryBasis, scipy.diag(scipy.ones(len(binaryBasis[0])))] )
    def moveFunc(oldState):
        randomIndex = scipy.random.randint(len(binaryBasis))
        #*****
        if randomIndex > 47:
            print "randomIndex =",randomIndex
        #*****
        newState = copy.copy(oldState)
        newState = (newState + binaryBasis[randomIndex])%2
        return newState
    return moveFunc

# 11.15.2013 taken from SloppyCell.Ensembles._sampling_matrix
def hessian2samplingMatrix(hess,singValCutoff=0.):
    """
    singValCutoff (0.)      : Any directions of the hessian with
                              singular value smaller than 
                              singValCutoff*max(singVals) will
                              be truncated to have singular value
                              equal to singValCutoff*max(singVals)
    """
    u,vals,vh = scipy.linalg.svd(0.5*hess)
    
    cutoffVal = singValCutoff*max(vals)
    invValsCut = 1./scipy.maximum(vals, cutoffVal)

    samplingMat = scipy.transpose(vh) * scipy.sqrt(invValsCut)

    # rescale sampling matrix appropriately
    cutVals = scipy.compress(vals < cutoffVal, vals)
    if len(cutVals) > 0:
        scale = scipy.sqrt(len(vals) - len(cutVals)
                           + scipy.sum(cutVals)/cutoffVal)
    else:
        scale = scipy.sqrt(len(vals))

    samplingMat /= scale
    
    return samplingMat

# 11.19.2013
def samplingMatrixMoveFunc(oldState,samplingMat,gaussianRand=None):
    """
    samplingMat             : Sampling matrix
    gaussianRand (None)     : Optionally pass a list of same length
                              as oldState consisting of gaussian
                              random numbers (for speed)
    """
    if gaussianRand is None:
        gaussianRand = scipy.random.randn(len(oldState))
    
    deltaParams = scipy.dot(samplingMat,gaussianRand)
    
    newState = oldState.copy()
    newState = newState + deltaParams

    return newState



# 3.24.2011
def monteCarloSample(Efunc,acceptFunc,ell,numSamples,T=1.,              \
    bathEStart=0.,maxBathE=scipy.inf,seed=0,retall=False,nSkip=1,       \
    startConfig=None,moveFunc=randomIndivMoveFunc,                      \
    filename=None,saveEvery=10,acceptRand=None,newStateRand=None):
    """
    Monte carlo sampling.  (By default on (0,1)^ell)
    
    Change acceptFunc to use different algorithms:
        acceptFuncMetropolis
        acceptFuncCreutz
    
    Efunc(currentState)
    moveFunc(oldState,newStateRand[i])
    acceptFunc(deltaE,T,newBathE,maxBathE,acceptRand[i])
    
    Output should be filtered to remove effects of 'burn-in'
    and dependence.
    
    With retall=True, returns
        ( samples, energies, acceptance ratio )
        
    filename (None)         : If given, save result to the
                              filename every saveEvery samples.
                              (Result is what would be returned
                              with retall=True)
    """
    if seed is not None: scipy.random.seed(seed)
    
    #randomIndiv =                                                       \
    #    scipy.floor( scipy.random.random(nSkip*numSamples)*ell )
    #randomNum = scipy.random.random(numSamples)
    
    if acceptRand is None: # for back-compatibility
        acceptRand = scipy.random.random(nSkip*numSamples)
    if newStateRand is None:
        newStateRand = scipy.random.random(nSkip*numSamples)
    
    if startConfig is None:
        #currentState = scipy.zeros(ell)
        currentState = scipy.random.randint(0,2,ell)
    else:
        currentState = startConfig
    currentE = Efunc(currentState)
    currentBathE = bathEStart
    
    samplesList,EList = [],[]
    movesAccepted = 0
    
    #sumFlat = lambda x: scipy.sum(x) #scipy.sum(pylab.flatten(x))
    
    for i in range(int(nSkip*numSamples)):
        #oldState = copy.copy(currentState)
        #oldE = copy.copy(currentE)
        #oldBathE = copy.copy(currentBathE)
        
        #newState = copy.copy(oldState)
        newState = moveFunc(currentState,newStateRand[i])
        newE = Efunc(newState)
        
        #deltaE = sumFlat(newE) - sumFlat(oldE)
        deltaE = scipy.sum(newE) - scipy.sum(currentE)
        
        newBathE = currentBathE - deltaE
        
        if acceptFunc(deltaE,T,newBathE,maxBathE,acceptRand[i]):
            currentState = newState
            currentE = newE
            currentBathE = newBathE
            movesAccepted += 1
        #else:
        #    currentState = oldState
        #    currentE = oldE
        #    currentBathE = oldBathE
        
        if i%nSkip == 0:
            samplesList.append(currentState)
            EList.append(currentE)

            # 11.15.2013 save to file
            if (filename is not None) and \
               (len(samplesList)%saveEvery == 0):
                a = float(movesAccepted)/(i)
                d = scipy.array(samplesList),scipy.array(EList),a
                try:
                    save(d,filename)
                except:
                    print "monteCarloSample: Error saving to "+str(filename)
    
    samplesList,EList = scipy.array(samplesList),scipy.array(EList)
    if retall:
        return samplesList,EList,float(movesAccepted)/(nSkip*numSamples)
    return samplesList
    
# 9.18.2012
def metropolisSampleIsing_pypar(numprocs,J,numSamples,**kwargs):
    """
    See doc string for metropolisSample_pypar and 
    isingSample.metropolisSampleIsing.
    """
    return metropolisSample_pypar('Ising',numprocs,numSamples,J=J,**kwargs)
    
# 9.18.2012
def metropolisSampleSparse_pypar(numprocs,sparseBasis,sp,                   \
    numSamples,T,**kwargs):
    """
    See doc string for metropolisSample_pypar and 
    isingSample.metropolisSampleSparse.
    """
    return metropolisSample_pypar('Sparse',numprocs,numSamples,             \
        sparseBasis=sparseBasis,sp=sp,T=T,**kwargs)
    
# 1.23.2012
# 9.18.2012 changed from Ising-only to more general implementation  
def metropolisSample_pypar(type,numprocs,numSamples,retall=False,           \
                           seed=0,tempDir='/tmp',**kwargs):
    """
    """
    codePath = os.path.dirname(os.path.abspath(__file__))
    
    kwargs['type'] = type
    kwargs['numSamples'] = numSamples
    kwargs['seed'] = seed
    
    scipy.random.seed()
    prefix = tempDir+"/temporary_" + str(os.getpid())                       \
        + "_metropolisSampleIsing_pypar_"
    paramsDictFilename = prefix + "paramsDict.data"
    outputFilename = prefix + "output.data"
    kwargs['outputFilename'] = outputFilename
    save(kwargs,paramsDictFilename)
    
    # call mpi
    stdoutFile = open(prefix+"stdout.txt",'w')
    call([ "mpirun","-np",str(numprocs),"python",
          codePath+"/sampleParallel.py",paramsDictFilename ],
          stderr=stdoutFile,stdout=stdoutFile)
    stdoutFile.close()
    os.remove(paramsDictFilename)
    
    try:
        samples,energies,a = load(outputFilename)
        os.remove(outputFilename)
        os.remove(prefix+"stdout.txt")
    except IOError:
        print "metropolisSampleIsing_pypar error:"
        stdoutFile = open(prefix+"stdout.txt")
        stdout = stdoutFile.read()
        print stdout
        os.remove(prefix+"stdout.txt")
        raise Exception, "metropolisSampleIsing_pypar:"                     \
            + " error in sampleParallel.py"
    
    if retall:
        return samples,energies,a
    return samples

    
# 11.21.2011
def metropolisSampleSparse(sparseBasis,sp,numSamples,T,hext=0.,             \
    seed=0,retall=False,nSkip=None,startConfig=None,burnin=None,            \
    hextIndivids=None):
    """
    Monte carlo Metropolis sampling on {0,1}^ell for Sparse model.
    
    Designed to look like metropolisSampleIsing.
    Used in criticalPoint.averageEnergy.
    
    With retall=True, returns
        ( samples, energies, acceptance ratio )
        
    sp                  : SparsenessProblem instance
    hext (0)            : 9.18.2012 external field coupling to all
                          individuals
    hextIndivids (None) : List of indices of individuals to which
                          to apply the external field.  Defaults
                          to all individuals.
    """
    
    if seed is not None:  scipy.random.seed(seed)
    ell = len(sparseBasis[0])
    T = float(T)
    
    if nSkip is None: nSkip = ell*10
    if burnin is None: burnin = 10
    numSamples += burnin
    nSkip = int(nSkip)
    numSamples = int(numSamples)
    
    acceptFunc = acceptFuncMetropolis
    phiStart = scipy.zeros((len(sparseBasis),1)) #None
    Efunc = sparsenessEfunc(sp,sparseBasis,phiStart=phiStart,hext=hext, \
                            hextIndivids=hextIndivids)

    samplesList,EList,a = monteCarloSample(Efunc,acceptFunc,ell,        \
        numSamples,T=T,seed=seed,retall=True,nSkip=nSkip,               \
        startConfig=startConfig)
        
    if retall:
        return samplesList[burnin:],EList[burnin:],a
    return samplesList[burnin:]
    
    
# 3.31.2011
def acceptFuncMetropolis(deltaE,T,bathE,maxBathE,rand):
    return deltaE < 0. or scipy.exp(-deltaE/T) > rand
    
# 3.31.2011
def acceptFuncCreutz(deltaE,T,bathE,maxBathE):
    return bathE > 0. and bathE < maxBathE
    
# 3.24.2011
def sparsenessEfunc(sparsenessProblem,basis,phiStart=None,hext=0.,      \
    unsummed=False,hextIndivids=None):
    """
    hextIndivids (None) : List of indices of individuals to which
                          to apply the external field.  Defaults
                          to all individuals.
    """
    sp = sparsenessProblem
    if hextIndivids is None: 
        ell = len(basis[0])
        hextIndivids = range(ell)
    def Efunc(fight):
      image = scipy.array([fight,]).T
      Efield = -hext*scipy.sum(scipy.array(fight)[hextIndivids])
      try:
        phi,E = sp.phiMinimizer(basis.T,testImages=image,                   \
            phiStart=copy.copy(phiStart),returnEVal=True)
      except TypeError: # 11.22.2011 weird error I don't know what to do with
        raise # for debugging 
        N,m = len(basis),1
        scipy.random.seed()
        # from Sparseness.randomPhi
        phiStartNew = 2.*sp.sigma2*(scipy.rand(N,m) - 0.5) 
        print "sparsenessEfunc: Weird TypeError.  Trying different phiStart."
        return sparsenessEfunc(sp,basis,phiStart=phiStartNew,hext=hext,     \
            unsummed=unsummed,hextIndivids=hextIndivids)(fight)
      e1,e2,e3 = sp.ENormFunc_unsummed(phi,basis.T,image)
      if unsummed: return e1[0]+Efield,e2,e3[0]
      else: return e1[0]+Efield+e2+e3[0]
    return Efunc
    

