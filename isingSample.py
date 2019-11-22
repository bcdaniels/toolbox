# isingSample.py
#
# Bryan Daniels
# 1.24.2012
#

import scipy
import sys
if sys.version_info[0] < 3:
    import scipy.weave
else:
    print("toolbox.isingSample warning: weave is no longer supported")
import copy

dot,sum = scipy.dot,scipy.sum

# 11.8.2011
def metropolisSampleIsing(J,numSamples,T=1.,hext=0.,
                          seed=0,retall=False,nSkip=None,startConfig=None,
                          burnin=None,minSize=0):
    """
    Monte carlo Metropolis sampling on {0,1}^ell for Ising model.
    
    This is sped up in C compared to monteCarloSample.
    (Has the same functionality as inverseIsing.samplesFromJ,
    but over 200x faster...)
    
    Output should be filtered to remove effects of 'burn-in'
    and dependence.
    
    With retall=True, returns
    ( samples, energies, acceptance ratio )
    
    minSize (0)                 : Used for fight generation.  Samples are
                                  only recorded that have at least
                                  minSize 1s.  (5.1.2014 changed from
                                  removeZerosAndOnes, which corresponds
                                  to minSize = 2.)
    """
    
    if seed is not None:  scipy.random.seed(seed) 
    seed = int(seed)
    ell = len(J)
    T = float(T)
    
    #if removeZerosAndOnes: removeZerosAndOnes = 1
    #else: removeZerosAndOnes = 0
    
    JplusH = copy.copy(J)
    JplusH += scipy.diag(-hext*scipy.ones(ell))
    
    if nSkip is None: nSkip = ell*10
    if burnin is None: burnin = 10
    numSamples += burnin
    nSkip = int(nSkip)
    numSamples = int(numSamples)
    
    #acceptRand = scipy.random.random(nSkip*numSamples)
    #newStateRand = scipy.random.random(nSkip*numSamples)
    
    if startConfig is None:
        #currentState = scipy.zeros(ell,dtype=int)
        currentState = -1*scipy.ones(ell,dtype=int)
        while sum(currentState) < minSize:
            currentState = scipy.random.randint(0,2,ell)
    else:
        currentState = scipy.array(startConfig,dtype=int)
    if sum(currentState) < minSize:
        raise Exception("Cannot start with fewer than minSize participants.")
    newState = copy.copy(currentState)
    
    currentE = float( sum( dot(dot(currentState,JplusH),currentState) ) )
    
    n = int(numSamples)
    samplesList,EList = scipy.empty((n,ell),dtype=int),scipy.empty(n,dtype=float)
    movesAccepted = scipy.array([0])
    totalMoves = scipy.array([0])
    
    # 11.8.2011 use C code for Metropolis Ising sampling
    code = """
        int randomIndiv;
        double newE,deltaE;
        int indx = 0;
        double newStateRand,acceptRand; // 3.8.2013 moved RNG to C
        
        srand(seed);
        
        while(indx < n){
        
          for (int i=0; i<nSkip; i++){
                
                newStateRand = (double)rand() / (double)RAND_MAX;
                randomIndiv = floor( newStateRand*ell );
                newState = currentState;
                newState(randomIndiv) = (newState(randomIndiv) + 1)%2;
                
                newE = 0.;
                int fightSize = 0;
                for (int j=0; j<ell; j++){
                    for (int k=0; k<ell; k++){
                        newE += newState(j)*JplusH(j,k)*newState(k);
                    }
                    fightSize += newState(j);
                }
        
                if (fightSize < minSize){
                    // never accept
                    deltaE = 2.;
                    acceptRand = 2.;
                }
                else{
                    deltaE = newE - currentE;
                    acceptRand = (double)rand() / (double)RAND_MAX;
                }
                
                if ((deltaE < 0.) || (exp(-deltaE/T) > acceptRand)){
                    currentState = newState;
                    currentE = newE;
                    movesAccepted(0) += 1;
                }
                
                totalMoves(0) += 1;
        
          }
            
          //record state if valid
          //bool validState = true;
          //if (removeZerosAndOnes==1){
          //    int fightSize = 0;
          //    for (int m=0; m<ell; m++){
          //      fightSize += currentState(m);
          //    }
          //    if (fightSize < 2){
          //      validState = false;
          //    }
          //}
          //if (validState){
          
          //record state
          for (int m=0; m<ell; m++){
              samplesList(indx,m) = currentState(m);
          }
          EList(indx) = currentE;
          indx++;
          
          //}
        }
        """
    err = scipy.weave.inline(code,['ell',
        'T','currentState','newState','currentE','movesAccepted',
        'samplesList','EList','nSkip','n','JplusH',
        'minSize','seed','totalMoves'],                      
        type_converters = scipy.weave.converters.blitz)
    
    # 6.25.2014
    acceptanceRatio = float(movesAccepted[0])/totalMoves[0]
    if acceptanceRatio < 1./nSkip:
        print("metropolisSampleIsing WARNING: "\
              "acceptance ratio < 1/nSkip (a="+str(acceptanceRatio)+")")
    
    #samplesList,EList = scipy.array(samplesList),scipy.array(EList)
    if retall:
        #print "movesAccepted =",movesAccepted[0],",totalMoves =",totalMoves[0]
        return samplesList[burnin:],EList[burnin:],                     \
            float(movesAccepted[0])/totalMoves[0]
    return samplesList[burnin:]
