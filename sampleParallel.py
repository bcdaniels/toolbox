# sampleParallel.py
#
# Bryan Daniels
# 1.23.2012
#
# Uses design pattern at 
# http://www.shocksolution.com/2010/04/managing-a-pool-of-mpi-processes-with-python-and-pypar/
# to run sampling in parallel.
#

#!/usr/bin/env python
from numpy import *
import pypar
import time

#from generateFightData import *
#from isingSample import *
from monteCarloSample import *
import sys



# Constants
MASTER_PROCESS = 0
WORK_TAG = 1
DIE_TAG = 2

MPI_myID = pypar.rank()
num_processors = pypar.size()

# read in arguments from command line file name
if len(sys.argv) < 2 or len(sys.argv) > 2:
    print "Usage: python sampleParallel.py paramsDictFile.data"
    exit()
paramsDictFile = sys.argv[1]
paramsDict = load(paramsDictFile)
numSamples = paramsDict.pop('numSamples')
outputFilename = paramsDict.pop('outputFilename')
seedStart = paramsDict.pop('seed')
num_work_processors = num_processors - 1 # can we make this num_processors?
numSamplesSingle = int(numSamples/num_work_processors)
numSamplesActual = numSamplesSingle*num_work_processors
# set up sampleFunction
modelType = paramsDict.pop('type')
if modelType == 'Ising':
    J = paramsDict.pop('J')
    sampleFunction = lambda seed:                                               \
        metropolisSampleIsing(J,numSamplesSingle,seed=seed,retall=True,**paramsDict)
    numIndivids = len(J)
elif modelType == 'Sparse':
    sparseBasis = paramsDict.pop('sparseBasis')
    sp = paramsDict.pop('sp')
    T = paramsDict.pop('T')
    sampleFunction = lambda seed: metropolisSampleSparse(                       \
        sparseBasis,sp,numSamplesSingle,T,seed=seed,retall=True,**paramsDict)
    numIndivids = sp.ell
else:
    raise Exception, "Unrecognized type: "+str(modelType)
# make empty arrays to hold results
allSamples = scipy.empty((numSamplesActual,numIndivids))
allEnergies = scipy.empty(numSamplesActual)
allAs = []

### Master Process ###
if MPI_myID == MASTER_PROCESS:
    from simplePickle import save

    num_processors = pypar.size()
    print "Master process found " + str(num_processors) + " worker processors."
    
    # Create a list of dummy arrays to pass to the worker processes
    #work_size = 10
    #work_array = range(0,work_size)
    #for i in range(len(work_array)):
    #    work_array[i] = arange(0.0, 10.0)
    
    # list of seeds to pass to the workers
    work_size = num_work_processors
    if seedStart is None:
        # 1.30.2012 in case you're trying to continue with a run,
        # which I don't think will work
        raise Exception, "seed = None is not accepted due to possible unwanted behavior."
    
    #work_array = range(seedStart,seedStart+num_work_processors)
    # 3.28.2015 changed max seed from 1e10 to 1e9 (now needs to be 32-bit integer?)
    scipy.random.seed(seedStart)
    work_array = scipy.random.random_integers(1e9,
        size=num_work_processors)
    
    # Dispatch jobs to worker processes
    work_index = 0
    num_completed = 0
    
    # Start all worker processes
    # BCD I messed with the following line...
    for i in range(1, num_work_processors+1):
        pypar.send(work_index, i, tag=WORK_TAG)
        pypar.send(work_array[work_index], i)
        print "Sent work index " + str(work_index) + " (" + str(work_array[work_index]) + ") to processor " + str(i)
        work_index += 1
    
    # Receive results from each worker, and send it new data
    # BCD We don't need this because the amount of work is equal to the number of processes.
    #for i in range(num_processors, work_size):
    #    results, status = pypar.receive(source=pypar.any_source, tag=pypar.any_tag, return_status=True)
    #    index = status.tag
    #    proc = status.source
    #    num_completed += 1
    #    
    #    # copy to output array
    #    begin = num_completed*numSamplesSingle
    #    end = (num_completed+1)*numSamplesSingle
    #    print(shape(samples))
    #    print(shape(allSamples[begin:end,:]))
    #    allSamples[begin:end,:] = samples
    #    
    #    #work_index += 1
    #    #pypar.send(work_index, proc, tag=WORK_TAG)
    #    #pypar.send(work_array[work_index], proc)
    #    #print "Sent work index " + str(work_index) + " to processor " + str(proc)
    
    # Get results from remaining worker processes
    while num_completed < work_size: #-1
        outputs, status = pypar.receive(source=pypar.any_source, tag=pypar.any_tag, return_status=True)
        num_completed += 1
        
        # copy to output array
        begin = (num_completed-1)*numSamplesSingle
        end = (num_completed)*numSamplesSingle
        samples,energies,a = outputs
        #print(shape(samples))
        #print(shape(allSamples[begin:end,:]))
        allSamples[begin:end,:] = samples
        allEnergies[begin:end] = energies
        allAs.append(a)
        
        
    
    # Shut down worker processes
    for proc in range(1, num_processors):
        print "Stopping worker process " + str(proc)
        pypar.send(-1, proc, tag=DIE_TAG)
        
    # Write data to file
    save((allSamples,allEnergies,scipy.mean(allAs)),outputFilename)

else:
    ### Worker Processes ###
    continue_working = True
    while continue_working:
        
        work_index, status =  pypar.receive(source=MASTER_PROCESS, tag=pypar.any_tag,
                                            return_status=True)
        
        if status.tag == DIE_TAG:
            continue_working = False
        else:
            work_array, status = pypar.receive(source=MASTER_PROCESS, tag=pypar.any_tag,
                                               return_status=True)
            work_index = status.tag
            
            # Code below simulates a task running
            #time.sleep(random.random_integers(low=0, high=5))
            #result_array = work_array.copy()
            
            workerSamples = sampleFunction(work_array)
            
            pypar.send(workerSamples, destination=MASTER_PROCESS, tag=work_index)
#### while
#### if worker

pypar.finalize()