# gifsicle.py
#
# Bryan Daniels
# 1.26.2016
#
# For constructing animated gifs using multiple pylab plots.
#

from subprocess import call
import os
import pylab

def gifsicleAnnotate(fileList,annotateList=None,filename='gifsicle_animation.gif',
    delay=25,openFile=True,convertOptions=[]):
    """
    Dependencies: gifsicle, imagemagick
    
    Converts image files in fileList to gif and
    assembles resulting images into a gif animation.
    
    annotateList (None)     : Optional text annotation added to each frame.
    delay (25)              : Delay between frames in hundredths
                              of a second.
    openFile (True)         : Use OSX's "open" command to show the resulting gif
    convertOptions ([])     : List of strings to pass to imageMagick's 'convert'.
                              (e.g. ["-resize","1000x1000","-density","1000"])
    """
    if annotateList is None:
        annotateList = [ None for file in fileList ]
    
    # () convert images to gifs
    gifFileList = []
    for file,annotation in zip(fileList,annotateList):
        
        # convert to gif using imagemagick's "convert"
        convertCall = ["convert"]
        convertCall += convertOptions
        if annotation is None:
            convertCall += [file,file+".gif"]
        else:
            # convert to gif and add annotation
            convertCall += [str(file),"label:'"+str(annotation)+"'",
                  "+swap","-gravity","Center","-append",str(file)+".gif"]
        call(convertCall)
        gifFileList.append(file+".gif")

    # () assemble gifs into animation using gifsicle
    cout = open(filename,'wb')
    call(["gifsicle","--disposal","background","--loop","-d",
         str(int(delay))]+gifFileList,stdout=cout)
    cout.close()
    print("gifsicleAnnotate: GIF animation written to "+filename)

    # () remove temporary gif files
    for file in gifFileList:
        os.remove(file)

    # () open the file externally
    if openFile:
        call(["open",filename])

def tempPrefix():
    """
    Create temporary folder for intermediate files.
    """
    tempFileDir = './.gifsiclePlot/'
    try: os.mkdir(tempFileDir)
    except OSError: pass
    return tempFileDir

# taken from neuralCorrelation.py
# 2.15.2013
# 8.28.2016 modified to use more general gifsicleAnnotate
def gifsiclePlot(plotFunc,argsList,filename='gifsicle_animation.gif',
    newFigures=True,delay=25,annotateList=None,figsize=None,**kwargs):
    """
    Dependencies: gifsicle, imagemagick
    (It seems that some backends for pylab will create gifs, in
    which case imagemagick wouldn't be needed, but some don't,
    so I convert from png instead using imagemagick.)
    
    Runs plotFunc for each set of args in argsList.
    Assembles resulting plots into a gif animation.
    
    newFigures (True)       : If True, call pylab.figure() before
                              each call of plotFunc (and 
                              pylab.close() after).
    delay (25)              : Delay between frames in hundredths
                              of a second.
    """
    if len(argsList) > 100000:
        # an arbitrary cutoff so that I know I can name temp files
        raise(Exception, "Too many args in argsList.")
    
    # () make plots and save gifs
    filenameList = []
    pid = os.getpid()
    tempFilePrefix = tempPrefix()+'gifsiclePlot_temp_'+str(pid)+'_'
    for i,args in enumerate(argsList):
        if newFigures: pylab.figure(figsize=figsize)
        plotFunc(args)
        
        numStr = '%(c)05d' % {'c':i}
        name = tempFilePrefix+numStr
        pylab.savefig(name+'.png')
        if newFigures: pylab.close()
        filenameList.append(name+".png")

    gifsicleAnnotate(filenameList,annotateList,filename=filename,delay=delay,**kwargs)

    # remove temporary files
    for name in filenameList:
        os.remove(name)
