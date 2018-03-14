# SparsenessTools.py
#
# Bryan Daniels
# 4.2.2012 moved from Sparseness.py
#
# Things that were defined in Sparseness.py that may
# be useful outside of Sparseness.py.
#
# 10.20.2015 Starting the transition to standalone code.
#


import scipy
import os,sys
import pylab
import scipy.linalg,scipy.optimize
import copy
from decimal import *
try:
    import scipy.stats # for bootstrappedPval
except ValueError: # for bizarre error on sidious
    import scipy.stats
if os.uname()[1][:8] != 'vader': # not supported on SFI's vader
    import mpl_toolkits.mplot3d.axes3d as p3 # for 3D plots
try:
    from pygraphviz import *
except ImportError:
    print "SparsenessTools: Could not load pygraphviz."

# 5.14.2014 moved from LoadFlackTimeSeriesData2011.py
# 3.4.2011 moved from Sparseness.py
def shuffle(lst):
    lstShuffled = copy.copy(lst)
    scipy.random.shuffle(lstShuffled)
    return lstShuffled

# 9.15.2014 updated for new scipy.diag behavior
# 1.17.2013 moved from criticalPoint.py
# 1.31.2012
def replaceDiag(mat,lst):
    if len(scipy.shape(lst)) > 1:
        raise Exception, "Lst should be 1-dimensional"
    if scipy.shape(mat) != (len(lst),len(lst)):
        raise Exception, "Incorrect dimensions."+                   \
            "  shape(mat) = "+str(scipy.shape(mat))+                \
            ", len(lst) = "+str(len(lst))
    return mat - scipy.diag(scipy.diag(mat).copy()).copy()          \
        + scipy.diag(lst).copy()

# 2.15.2013 moved from branchingProcess.py
# 2.11.2013
def zeroDiag(mat):
    return replaceDiag(mat,scipy.zeros(len(mat)))

def arrayFlatten(a):
    return scipy.resize(a,scipy.prod(scipy.shape(a)))
    
def normSq(vec):
    return scipy.sum( vec*vec )

def plotMatrix(mat,cmap=pylab.cm.gray,colorbar=True,X=None,Y=None,          \
    interp='nearest',plot3D=False,plotContour=False,ax=None,                \
    contours=None,filled=True,outlineColor=None,autoLimits=False,**kwargs):
    """
    some popular cmaps:
        pylab.cm.gray
        pylab.cm.copper
        pylab.cm.bone
        pylab.cm.jet
        
    Can also use kwargs for pylab.imshow, including
        vmin,vmax: Set the range of values for the color bar.
        
    If 1D arrays for X and Y are given, axes are drawn.
    (unknown behavior if X and Y values are not equally spaced)
    Reminders:
        Y axis corresponds to 0 axis of matrix.
        X axis corresponds to 1 axis of matrix.
        Image will be flipped vertically compared to version w/o X and Y.
        
    For plot3D, useful kwargs:
        linewidth
    """
    if len(scipy.shape(mat)) == 1:
        mat = [mat]
    #minVal,maxVal = max(arrayFlatten(mat)),min(arrayFlatten(mat))
    if (ax is None) and plot3D:
      axNew = p3.Axes3D(pylab.figure())
    elif ax is None:
      pylab.figure()
      axNew = pylab.axes()
    else:
      pylab.axes(ax)
      axNew = ax
    
    if (plot3D or plotContour) and X is None: X = range(scipy.shape(mat)[1])
    if (plot3D or plotContour) and Y is None: Y = range(scipy.shape(mat)[0])
    
    if X is not None and Y is not None:
        X,Y = scipy.array(X),scipy.array(Y)
        
        # shift axes so they align with middle of each component
        if len(X) > 1:  deltaX = X[1] - X[0]
        else: deltaX = 1 
        if len(Y) > 1:  deltaY = Y[1] - Y[0]
        else: deltaY = 1
        if scipy.any(abs(X[1:]-X[:-1] - deltaX) > 1e-5) or                  \
           scipy.any(abs(Y[1:]-Y[:-1] - deltaY) > 1e-5):
           print "plotMatrix WARNING: X and/or Y values are not equally "+  \
                 "spaced.  May produce strange behavior."
          
        if plot3D or plotContour:
          
          Xmesh,Ymesh = scipy.meshgrid(X,Y)
          Z = scipy.array(mat)
          if plotContour:
            if filled:
              contourFn = axNew.contourf3D
            else:
              contourFn = axNew.contour3D
            if contours is None:
              contourLevels = contourFn(Xmesh,Ymesh,Z,extend3D=True,stride=1,**kwargs)
            else:
              contourLevels = contourFn(Xmesh,Ymesh,Z,contours,**kwargs)
          else:
            if filled:
              axNew.plot_surface(Xmesh,Ymesh,Z,rstride=1,cstride=1,         \
                cmap=cmap,**kwargs)
            else:
              axNew.plot_wireframe(Xmesh,Ymesh,Z,rstride=1,cstride=1,       \
                **kwargs)
        else:
          Xshifted = scipy.concatenate([[X[0]-deltaX],X]) + deltaX/2.
          Yshifted = scipy.concatenate([[Y[0]-deltaY],Y]) + deltaY/2.
          newAxis = [Xshifted[0],Xshifted[-1],Yshifted[0],Yshifted[-1]]
          if (ax is not None) and not autoLimits:
            pylab.axes(ax)
            oldAxis = pylab.axis()
            newAxis = [min(newAxis[0],oldAxis[0]),                          \
                       max(newAxis[1],oldAxis[1]),                          \
                       min(newAxis[2],oldAxis[2]),                          \
                       max(newAxis[3],oldAxis[3])]             
          pylab.pcolor(Xshifted, Yshifted, mat, cmap=cmap, **kwargs)
          pylab.axis(newAxis)
          #axNew = pylab.axes()
          if outlineColor is not None:
            rect = pylab.Rectangle([Xshifted[0],Yshifted[0]],
                            Xshifted[-1]-Xshifted[0],
                            Yshifted[-1]-Yshifted[0],fill=False,
                            ec=outlineColor,lw=0.25)
            axNew.add_patch(rect)
    else:
        #if ax is None:
        #  pylab.figure()
        #else:
        #  pylab.axes(ax)
        pylab.axes(axNew)
        pylab.imshow(mat, interpolation=interp, cmap=cmap, **kwargs)
        #axNew = pylab.axes()
        axNew.get_xaxis().set_visible(False)
        axNew.get_yaxis().set_visible(False)
        pylab.ylabel('')
        #ax = pylab.subplot(111)
        #ax.axis["right"].set_visible(False)
        #ax.axis["top"].set_visible(False)
        #grid(True)
    if colorbar and plotContour:
        pylab.colorbar(contourLevels,ax=axNew)
    elif colorbar:
        pylab.colorbar(ax=axNew)
        
    return axNew
    

# 8.9.2010
def PCA(mat):
    """
    Performs principal component analysis on the set of vectors
    in the input matrix mat.  mat is (N x m), where N is the
    number of vectors and m is the length of each vector.
    (singular value decomposition on the covariance matrix)
    """
    N,m = scipy.shape(mat)
    
    # subtract off mean to get shifted matrix b
    mn = scipy.mean(mat,axis=0)
    mnMat = scipy.dot(scipy.ones(N).reshape(N,1),[mn])
    b = mat - mnMat
    
    # DEBUG 10/1/10 REMOVE FOR USUAL PCA
    #b = mat
    
    # find the covariance matrix c
    c = scipy.dot(b.T,b)/N
    
    # perform singular value decomposition
    U,s,Vh = scipy.linalg.svd(c)
    
    # sort the singular values and vectors
    sortedList = sorted( zip(s,U.T), key=lambda tup: -tup[0] )
    sortedSvals = [ sortedSval for (sortedSval,sortedSvec) in sortedList]
    sortedSvecs = [ sortedSvec for (sortedSval,sortedSvec) in sortedList]
    
    return sortedSvals,sortedSvecs

# 9.8.2010
def MarkovProcess(transitionMatrix,numIter=10,s0=0,seed=0):
    """
    transitionMatrix            : NxN matrix.  Rows should
                                  sum to 1.
    """
    stateList = [s0]
    N = len(transitionMatrix)
    M = scipy.transpose(transitionMatrix)
    scipy.random.seed(seed)
    s = s0
    for i in range(numIter):
        stateVec = scipy.zeros((N,1))
        stateVec[s,0] = 1.
        sNewProbs = scipy.dot(M,stateVec)
        probSum = scipy.cumsum(sNewProbs)
        r = scipy.rand()
        newS = pylab.find(probSum>r)[0]
        stateList.append(newS)
        s = newS
    return stateList
        

# 8.31.2010 
def MarkovSteadyState(transitionMatrix,tol=1.e-10):
    """
    transitionMatrix            : NxN matrix, where N = 2^i for 
                                  some integer i.  Rows should
                                  sum to 1.
    """
    # find the steady-state probabilities p0 by finding the
    # eigenvector with eigenvalue 1.
    M = scipy.transpose(transitionMatrix)
    valsM,vecsM = scipy.linalg.eig(M)
    valsM = scipy.real_if_close(valsM)
    indices = pylab.find(abs(valsM-1.)<tol)
    if len(indices) != 1:
        raise Exception, "MarkovPCA: No unique steady-state solution.  " +  \
            "Check form of transition matrix."
    p0 = vecsM[:,indices[0]].T
    p0 = p0/sum(p0)
    
    return p0

# 9.8.2010
def MarkovEntropy(transitionMatrix):
    p0 = MarkovSteadyState(transitionMatrix)
    sum,log2 = scipy.sum,lambda x: scipy.nan_to_num(scipy.log2(x))
    return -scipy.real_if_close( sum(p0*log2(p0)) )

# 9.8.2010
def MarkovMutualInfo(transitionMatrix):
    p0 = MarkovSteadyState(transitionMatrix)
    #M = scipy.transpose(transitionMatrix)
    M = transitionMatrix #***8testing
    sum,dot,log2 = scipy.sum,scipy.dot,lambda x: scipy.nan_to_num(scipy.log2(x))
    return scipy.real_if_close( sum(dot(p0,M*log2(M))) - sum(p0*log2(p0)) )

# 8.27.2010
# 6.15.2014 updated to use longs (int64)
def fight2number(fightList,useDecimals=False):
    """
    Returns the base-10 number corresponding to the 'base-2 fights'.
    """
    ell = len(fightList[0])
    if (2**ell > sys.maxint) or useDecimals:
        print "fight2number: Using decimals..."
        #decimalNums = [ 2**Decimal(a) for a in range(ell) ]
        nums = [ [f[i]*(2**Decimal(i)) for i in range(ell) ]                \
            for f in fightList ]
        return scipy.array([ sum(n) for n in nums ])
    return scipy.sum(                                                       \
        2**scipy.arange(ell,dtype=long)[::-1]                               \
        * scipy.array(fightList,dtype=long), axis=1 )
    

# 9.8.2010
def number2fight(stateList,numIndividuals):
    """
    Returns the 'base-2 fight' corresponding to the base-10 number.
    """
    return [ [int(bit) for bit in scipy.binary_repr(state,numIndividuals)]  \
        for state in stateList ]

# 2.28.2011
def numDistinctFights(fightList):
    fightNumbers = scipy.sort(fight2number(fightList))
    return 1+sum(fightNumbers[:-1]-fightNumbers[1:]<0)

    
# 10.26.2010
# 7.19.2011 see also EntropyEstimates.fights2kxnx
def freqDataFromFightList(fightList):
    if len(scipy.shape(fightList)) == 1: 
        # 4.2.2012 single individual 'fights'
        fightNumberList = fightList
        histData = scipy.histogram(fightNumberList,                     \
            bins=range(0,3))[0]
    else: #elif len(fightList[0])>1: # changed 6.5.2012
        fightNumberList = fight2number(fightList)
        histData = scipy.histogram(fightNumberList,                     \
            bins=range(0,2**len(fightList[0])+1))[0]
    #else: # 4.2.2012 not sure what this part is supposed to do...
    #    fightNumberList = fightList
    #    histData = scipy.histogram(fightNumberList,                     \
    #        bins=range(0,max(fightNumberList)+1))[0]
    
    #freqData = histData/float(sum(histData))
    freqData = histData
    return freqData
    
# 8.27.2010
def entropyNaive(fightList,newMethod=True):
    """
    In bits.
    """
    if newMethod: # 4.6.2011
      count = 0
      freqData = []
      fightNums = scipy.sort(fight2number(fightList))
      diffs = scipy.concatenate([fightNums[:-1]-fightNums[1:],[-1]])
      for diff in diffs:
        count += 1
        if diff < 0: # it's a new fight
            freqData.append(count)
            count = 0
      freqData = scipy.array(freqData)
    else:
      freqData = freqDataFromFightList(fightList)
    return entropyFreqListNaive(freqData/float(sum(freqData)))

# 2.22.2011
def GSOrthogonalize(mat):
    dot = scipy.dot
    OMat = []
    for v in mat:
        vOutsideOMat = scipy.sum([u * dot(v,u)/dot(u,u) for u in OMat],axis=0)
        u = v - vOutsideOMat
        OMat.append(u/scipy.sqrt(sum(u*u)))
    return OMat

# 9.1.2011
def bootstrappedPVal(dataNumAppear,dataTotalNum,fNull):
    """
    Assuming binomial distribution, prob. that the 
    number of appearances matches with fNull. 
    """
    fData = dataNumAppear/float(dataTotalNum)
    diff = fData - fNull
    
    if diff >= 0.:
        p = scipy.stats.binom.cdf(dataTotalNum-1-dataNumAppear,dataTotalNum,1.-fNull)
        #p = 1.-p
    else:
        p = scipy.stats.binom.cdf(dataNumAppear,dataTotalNum,fNull)
        
        
    return diff,p

def thresholdMatrix(mat,thold=None,numNonzero=None,useAbs=True,**kwargs):
    """
    Set any elements with absolute value less than thold to zero.
    
    numNonzero overrides any given threshold.
    
    useAbs (True)      : 7.12.2012 Set false to not take absolute value.
    """
    if thold is None and numNonzero is None:
        return mat
    if numNonzero is not None:
        thold = thresholdFromNumNonzero(mat,numNonzero,useAbs=useAbs,**kwargs)
    thresholded = copy.copy(mat)
    for i in range(len(thresholded)):
        for j in range(len(thresholded[i])):
            if useAbs: tij = abs(thresholded[i][j])
            else: tij = thresholded[i][j]
            if tij < thold:
                thresholded[i][j] = 0.
    return thresholded

# 3.1.2011
def thresholdFromNumNonzero(mat,numNonzero,sym=False,useAbs=True,aboveDiagOnly=False):
    """
    Things get complicated if the matrix elements are not all distinct...
    
    sym:        If True, treat the matrix as symmetric, and count only 
                nonzero elements at or below the diagonal.
    """
    if sym:
        mat = scipy.tri(len(mat))*mat
    if useAbs: absMat = abs(mat)
    else: absMat = mat
    if not aboveDiagOnly:
        flatAbsMat = scipy.sort(arrayFlatten(absMat))[::-1]
    else:
        flatAbsMat = scipy.sort(aboveDiagFlat(absMat))[::-1]
    if numNonzero < 1:
        return scipy.inf
    elif numNonzero == len(flatAbsMat):
        if useAbs: return 0.
        else: return flatAbsMat[-1]
    elif numNonzero > len(flatAbsMat):
        raise Exception, "Desired numNonzero > number of matrix elements."
    return scipy.mean([flatAbsMat[numNonzero],flatAbsMat[numNonzero-1]])

# 4.8.2011
# 8.16.2012 moved from generateFightData.py
def aboveDiagFlat(mat,keepDiag=False,offDiagMult=None):
    """
    Return a flattened list of all elements of the 
    matrix above the diagonal.
    
    Use offDiagMult = 2 for symmetric J matrix.
    """
    m = copy.copy(mat)
    if offDiagMult is not None:
        m *= offDiagMult*(1.-scipy.tri(len(m)))+scipy.diag(scipy.ones(len(m))) 
    if keepDiag: begin=0
    else: begin=1
    return scipy.concatenate([ scipy.diagonal(m,i)                          \
                              for i in range(begin,len(m)) ])

# 3.17.2014 copied from SparsenessPrediction.py
# 3.10.2011 taken out of checkPredictions
def covarianceMatrix(fights):
    """
    Equivalent to scipy.cov(fights,rowvar=0,ddof=0)
    """
    fights = scipy.array(fights)
    meanFight = scipy.mean(fights,axis=0)
    # 1.27.2011 make sure PCA is doing what I think it is
    meanMat = ( meanFight*scipy.ones_like(fights) ).T
    c = scipy.dot(fights.T-meanMat,(fights.T-meanMat).T) / len(fights)
    return c

# 3.17.2014 copied from generateFightData.py
# 4.1.2011
def fightSizeDistribution(fightList,normed=True,removeZeros=True,           \
                          confIntP=None,removeOnes=True,maxSize=None):
    """
    removeZeros (True)      : Doesn't count as a fight if there are no
    participants!
    removeOnes (True)       : Doesn't count as a fight if there is only
    one participant
    confIntP (None)         : Given a confidence interval percentage,
    also return confidence interval (length 2)
    for each fight size.
    maxSize (None)          : (Only used if fightList is a list of fight
    sizes.)  Bins range from 0 to maxSize.
    
    """
    if len(scipy.shape(fightList)) == 2:
        ell = len(fightList[0])
        fightSizes = scipy.sum(fightList,axis=1)
    elif len(scipy.shape(fightList)) == 1:
        # 10.30.2012 allow to directly pass fight sizes
        fightSizes = fightList
        if maxSize is None:
            ell = int(max(fightList))
        else:
            ell = int(maxSize)
    else:
        raise Exception, "Incorrect shape of fightList"
    if removeZeros:
        fightSizes = filter(lambda x: x>0, fightSizes)
    if removeOnes:
        fightSizes = filter(lambda x: x!=1, fightSizes)
    hist = scipy.histogram(fightSizes,bins=range(ell+2),normed=False)[0]
    N = sum(hist)
    if confIntP is not None: # calculate confidence intervals
        confIntList = scipy.array(                                          \
            [ binomialConfidenceIntervalFreq(n,N,confIntP) for n in hist ] )
        if normed:
            #confIntList = confIntList/N
            hist = hist/float(N)
        return hist,confIntList
    else:
        if normed: hist = hist/float(N)
        return hist


# 3.17.2014 copied from generateFightData.py
def binomialConfidenceIntervalFreq(Nsuccess,Nsamples,percent=0.95):
    """
    Gives the 95% confidence interval for the frequency parameter of a binomial
    distribution given Nsuccess successes in Nsamples samples.
    """
    betaincinv = scipy.special.betaincinv
    a = 1 + Nsuccess
    b = 1 - Nsuccess + Nsamples
    lower = betaincinv(a,b,0.5-percent/2.)
    upper = betaincinv(a,b,0.5+percent/2.)
    return lower,upper


# 3.17.2014 copied from generateFightData.py
# 4.1.2011
def cooccurranceMatrixFights(fightList,normed=True,keepDiag=False):
    """
    normed (True)           : Normalize such that the sum of the matrix = 1.
    : 4.11.2011 changed to divide by number of fights
    """
    fightList = scipy.array(fightList,dtype=float)
    mat = scipy.dot(fightList.T,fightList)
    if keepDiag: k=-1
    else: k=0
    mat *= (1 - scipy.tri(len(mat),k=k)) # only above diagonal
    if normed: mat /= float(len(fightList)) # mat /= scipy.sum(mat)
    return mat

# 3.19.2014 copied from generateFightData.py
# 6.30.2011
# could probably be more efficient
def KLdivergence(pList,qList,skipQzeros=False):
    """
        In bits.
        
        skipQzeros (False)      : If qList has zeros where pList doesn't,
        the KL divergence isn't defined (at least
        according to Wikipedia).  Set this flag
        True to instead skip these values in the
        calculation.
        """
    eps = 1e-5
    if (abs(sum(pList)-1.)>eps) or (abs(sum(qList)-1.)>eps):
        print "KLdivergence: WARNING: Check normalization of distributions."
        print "KLdivergence: sum(pList) =",sum(pList)
        print "KLdivergence: sum(qList) =",sum(qList)
    if len(pList) != len(qList):
        raise Exception, "pList and qList have unequal length."

    div = 0.
    for p,q in zip(pList,qList):
        if p == 0.:
            div += 0.
        elif (q == 0.) and skipQzeros:
            div += 0.
        elif (q == 0.) and not skipQzeros:
            return scipy.nan
        else:
            div += p*scipy.log2(p/q)
    return div
#return scipy.sum(pList*scipy.log2(pList/qList))


# 4.14.2014 moved from analyzeSparsenessProblem.py
# 3.15.2011
def makeGroupsFigure(groupsList,nameDict,fontsize=12,                   \
                     cmap=pylab.cm.Oranges,fontColor='bw',sortByMag=True,edgecolor=None, \
                     linewidth=0.75,numBases=None,offset=[0.,0.],newFigure=True,         \
                     includeTimesUsed=False,maxAbsMag=None,eraseNegatives=True,sp=None,  \
                     numNonzero=None,threshold=None,swapxy=False,relatedMatrix=None):
    """
        eraseNegatives  : Gets rid of any minus sign at the beginning of
        a name.  (This doesn't matter if each basis
        vector has a single sign, which is true of the
        bases I've been looking at.)
        
        numBases        : Number of bases that includes each group, in the
        original (unsorted) order.  The order is taken
        from the indices listed in the given rules.
        
        sp              : If given sparsenessProblem, use given
        groupsList as basis and order by number of
        times used to reconstruct the data.
        
        relatedMatrix   : If given a relatedMatrix
        (see biologicallyMeaningfulGroups.py), resort
        and draw rectangles around related groups
        """
    
    padHorz = 0.1 #0.1
    padVert = 0.3
    
    relatedLineColor = 'k'
    relatedLinewidth = 4.*linewidth
    padRelated = 0. #padHorz/2.
    
    if newFigure: pylab.figure()
    axes = pylab.subplot(111)
    
    groupsThresh = thresholdMatrix(groupsList,thold=threshold,          \
                                   numNonzero=numNonzero)
    
    xVals,yVals = [],[]
    if maxAbsMag is None:
        #maxAbsMag = max(pylab.flatten([ rule[-1] for rule in rules ]))
        maxAbsMag = max(pylab.flatten(abs(groupsThresh)))
    
    if numBases is not None or includeTimesUsed:
        offset[0] += 1 + padHorz
    
    if sp is not None: # 4.25.2011
        rules = sp.sortedRules(groupsList,numNonzero=numNonzero,        \
                               includeIndices=True,includeMagnitudes=True,nameDict=nameDict)
        indices = [ rule[2] for rule in rules ]
        groupsThresh = groupsThresh[indices]
    #groupsThresh = filteredBasis(groupsThresh[indices],0)
    #print "len(groupsThresh) =",len(groupsThresh)
    
    if eraseNegatives:
        for group in groupsThresh:
            filteredGroup = scipy.sort( filter(lambda x:abs(x)>0.,group) )
            if len(filteredGroup) > 0.:
                allSameSign = 0.5*(1.+                                        \
                                   scipy.sign(filteredGroup[0])*scipy.sign(filteredGroup[-1]))
                if not allSameSign:
                    print "makeGroupsFigure: WARNING: using eraseNegatives, "   \
                        "but group doesn't have all equal sign: group =",     \
                        filteredGroup
        groupsThresh = abs(groupsThresh)
    
    i = 0
    #for rule in rules:
    for groupIndex,group in enumerate(groupsThresh):
        #mags = abs(scipy.array(rule[-1]))
        mags = filter(lambda x: abs(x)>0, group)
        
        if len(mags) > 0:
            magsR = mags/maxAbsMag*cmap.N
            #names = scipy.array(rule[0][0])
            nonzeroIndices = scipy.array( filter(lambda x: x>0,             \
                                                 (scipy.arange(len(group))+1)*(abs(group)>0)) ) - 1
            names = scipy.array([ nameDict[ind] for ind in nonzeroIndices ])
            srt = scipy.argsort(-magsR)
            if sortByMag:
                magsR = magsR[srt]
                names = names[srt]
                nonzeroIndices = nonzeroIndices[srt]
            
            # 4.20.2012 corral related individuals
            if relatedMatrix is not None:
                srt2 = []
                relatedRects = []
                currentPos = 0
                for k in range(len(nonzeroIndices)):
                    numRelated = 0
                    if k not in srt2: # we're starting a new group
                        srt2.append(k)
                        numRelated = 1
                        for ell in range(k+1,len(nonzeroIndices)):
                            indiv1,indiv2 = nonzeroIndices[k],nonzeroIndices[ell]
                            if relatedMatrix[indiv1][indiv2]:
                                srt2.append(ell)
                                numRelated += 1
                        # draw rectangle around group if bigger than one
                        if numRelated > 1:
                            xloc = currentPos*(1+padHorz) + offset[0] - padRelated
                            yloc = -(i+1)*(1+padVert) + offset[1] - padRelated
                            width = numRelated*(padHorz+1.) - padHorz + 2.*padRelated
                            height = 1 + 2.*padRelated
                            rect = pylab.Rectangle((xloc,yloc),                       \
                                                   width,height,fill=False,lw=relatedLinewidth,            \
                                                   ec=relatedLineColor)
                            #axes.add_artist(rect)
                            relatedRects.append(rect)
                    currentPos += numRelated
                # sort
                magsR = magsR[srt2]
                names = names[srt2]
            
            # draw the individual squares
            for j,mag in enumerate(magsR):
                xloc = j*(1+padHorz) + offset[0]
                yloc = -(i+1)*(1+padVert) + offset[1]
                if swapxy:
                    tmpYloc = copy.copy(yloc)
                    yloc = copy.copy(xloc)
                    xloc = -copy.copy(tmpYloc)
                rect = pylab.Rectangle((xloc,yloc),                         \
                                       1,1,color=cmap(int(mag)),ec=edgecolor,lw=linewidth)
                axes.add_artist(rect)
                if fontColor is 'auto':
                    fc = cmap((cmap.N/2 + int(mag))%cmap.N)
                elif fontColor is 'bw':
                    blackThresh = 1.2 #1.5
                    if sum(cmap(int(mag))[:-1]) > blackThresh: fc = 'black'
                    else: fc = 'white'
                else: fc = fontColor
                #if eraseNegatives: n = names[j][-2:]
                #else: n = names[j]
                pylab.annotate("$\mathrm{"+names[j]+"}$",                   \
                               (xloc+0.5,yloc+0.5),                                    \
                               color=fc,fontsize=fontsize,                             \
                               ha='center',va='center')
                xVals.append(xloc)
                yVals.append(yloc)
            if numBases is not None:
                #if scipy.iterable(rule[2]):
                #    raise Exception, "makeGroupsFigure: it appears the"     \
                #      +" given rules include no indices.  Cannot"           \
                #      +" associate numBases with given groups."
                pylab.annotate("$\mathrm{"+str(numBases[groupIndex])+"}$",  \
                               (-1*(1+padHorz)+offset[0]+0.5,yloc+0.5),color='black',  \
                               fontsize=fontsize,ha='center',va='center')
                xVals.append(-1*(1+padHorz)+offset[0])
            if includeTimesUsed:
                pylab.annotate(str(rule[1]),                                \
                               (-1*(1+padHorz)+offset[0]+0.5,yloc+0.5),color='black',  \
                               fontsize=fontsize,ha='center',va='center')
                xVals.append(-1*(1+padHorz)+offset[0])
            i += 1
            
            # draw the related rectangles last
            if relatedMatrix is not None:
                for rect in relatedRects: 
                    axes.add_artist(rect)
    
    pylab.axis('scaled')
    if not swapxy:
        pylab.axis(xmin=min(xVals)-padHorz-1,xmax=max(xVals)+1+padHorz+1,   \
                   ymin=min(yVals)-padVert-1,ymax=+1)
    else:
        pylab.axis(xmin=0,xmax=max(xVals)+padVert+1+1,                       \
                   ymin=min(yVals)-padHorz-1,ymax=max(yVals)+1+padHorz+1)
    pylab.xticks([])
    pylab.yticks([])
    pylab.show()
    
    return pylab.axis()


# 2.27.2015 stolen from SOC.py
# stolen from generativeModelFigures.py
def fightSizeDistributionPlot(samples,color='k',plotConfInt=True,
    plotErrorBars=False,log=False,alpha=0.4,
    makePlot=True,confIntP=0.95,removeZeros=False,removeOnes=False,
    multiple=1,verbose=True,maxSize=None,**kwargs):
    """
    multiple (1)            : Multiply probabilities by this 
                              factor.  Useful for plotting expected
                              number rather than probability.
    """
    
    #ell = len(samples[0])
    dist,confIntList = fightSizeDistribution(samples,               \
        confIntP=confIntP,removeZeros=removeZeros,removeOnes=removeOnes,
        maxSize=maxSize)
    ell = len(dist)
    
    dist,confIntList = multiple*dist, multiple*confIntList
    
    if makePlot:
        if plotConfInt:
            #for confInt in confIntList:
            #    if confInt[0] == 0.: confInt[0] = zeroEquiv
            #firstZero = pylab.find(dist==0)[2]
            firstZero = len(dist)
            pylab.fill_between(range(1,firstZero),                       \
                confIntList[:,0][1:firstZero],                           \
                confIntList[:,1][1:firstZero],color=color,alpha=alpha)
        if plotErrorBars:
            yerr = scipy.empty_like(confIntList.T)
            yerr[0] = dist - confIntList[:,1]
            yerr[1] = confIntList[:,0] - dist
            pylab.errorbar(range(ell),dist,yerr=yerr,color=color)
        
        pylab.plot(range(ell),dist,color=color,**kwargs)
        
        if log: plotFn = pylab.yscale('log')
    
    if verbose:
        print "sum(dist) =",sum(dist)
    return dist


# 3.18.2015 stolen from branchingProcess.py
# 3.11.2013 taken from AngularHessian.py
def sortedEig(matrix):
    """
    Note!  The sorted eigenvectors are numbered by their first index!
    (It's not eigvecSorted[:,0] but rather eigvecSorted[0] that gives the
    eigenvector with the smallest eigenvalue.)
    """
    eigvals,eigvecs = scipy.linalg.eig(matrix)
    sortedList = sorted( zip(eigvals,scipy.transpose(eigvecs)),                 \
                        key=lambda tup: tup[0] )
    sortedEigvals = [ sortedEigval for (sortedEigval,sortedEigvec) in sortedList]
    sortedEigvecs = [ sortedEigvec for (sortedEigval,sortedEigvec) in sortedList]
    return scipy.array(sortedEigvals), scipy.array(sortedEigvecs)

# 2.26.2016 taken from analyzeSparsenessProblem.py
# 4.19.2011
def RGBHdecimal2hex(RGBHdecimal):
    hexList = [ hex(int(256*x-1e-5))[2:] for x in RGBHdecimal ]
    hx = '#'
    for h in hexList:
        if len(h) == 1:
            hx = hx + '0' + h
        else:
            hx = hx + h
    return hx

# 2.26.2016 taken from analyzeSparsenessProblem.py
# 3.30.2011
def drawNetworkFromMatrix(mat,scoreListSize=None,scoreListColor=None,   \
    filename=None,nodeNames=None,cmap=pylab.cm.Greys,                   \
    nodeShape='ellipse',prog='dot',fontsize=14,edgeWidths=None,         \
    size=(),**kwargs):
    """
    prog:       'neato','fdp','dot'
    
    See also
    http://www.graphviz.org/doc/info/attrs.html
    """
    G = AGraph(maxiter=100000,epsilon=1e-7,**kwargs)
    num = len(mat)
    if scoreListSize is None:
        sizes = scipy.ones(num)/5.
    else:
        sizes = scipy.sqrt(scipy.array(scoreListSize))
    if scoreListColor is None:
        colors = scipy.ones(num)
    else:
        colors = scipy.array(scoreListColor)/max(scoreListColor)*cmap.N
    if nodeNames is None:
        nodeNames = scipy.repeat('',num)
    if edgeWidths is None:
        edgeWidths = scipy.ones_like(mat)
    for i in range(num):
      RGBHcolor = cmap(int(colors[i]))
      fillcolor=RGBHdecimal2hex(RGBHcolor)
      # change text color based on node color (from makeGroupsFigure)
      blackThresh = 1.2 #1.5
      if sum(RGBHcolor[:-1]) > blackThresh: fc = 'black'
      else: fc = 'white'
      G.add_node(i,width=sizes[i],height=sizes[i],label=nodeNames[i],   \
        fillcolor=fillcolor,style='filled',                             \
        shape=nodeShape,fontcolor=fc,fontsize=fontsize,margin=0.)
    for i in range(num):
      for j in range(i+1,num):
        if mat[i,j] > 0.:
          G.add_edge(i,j,len=mat[i,j],penwidth=edgeWidths[i,j])
    if filename is not None:
        G.draw(filename,prog=prog) # 'fdp' or 'neato'
    return G
