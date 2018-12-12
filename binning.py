# binning.py
#
# Bryan Daniels
# 12.12.2018
#
# Tools for binning, originally found in worms/explainableVariance.py
#

import scipy, pylab

def quantileBins(vals,nbins):
    """
    Produces edges for bins that include equal numbers of values 
    (or as close to equal as is possible).
    """
    quantiles = [ scipy.percentile(vals,i*100./nbins) for i in range(nbins+1) ]
    # want end values to include the min and max
    quantiles[0]  -= 1.
    quantiles[-1] += 1.
    return quantiles


def findBinIndices(inputvals,numbins):
    """
    inputvals has shape (# trials)
    
    returns binIndices, with length numbins, that lists the indices falling in each bin
    """

    inputBins = quantileBins(inputvals,numbins)

    binBools =   [ scipy.logical_and(scipy.less(inputBins[i],
                                                scipy.array(inputvals)),
                                     scipy.less_equal(scipy.array(inputvals),
                                                      inputBins[i+1])) \
                   for i in range(numbins) ]

    return [ pylab.find(b) for b in binBools ]
