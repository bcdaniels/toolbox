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

# taken from neuralCorrelation.py
# 2.15.2013
def gifsiclePlot(plotFunc,argsList,filename='gifsicle_animation.gif',   \
    newFigures=True,delay=25,openFile=True):
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
        raise Exception, "Too many args in argsList."
    
    # () make plots and save gifs
    pid = os.getpid()
    filenameList = []
    tempFileDir = './.gifsiclePlot/'
    try: os.mkdir(tempFileDir)
    except OSError: pass
    tempFilePrefix = 'gifsiclePlot_temp_'+str(pid)+'_'
    for i,args in enumerate(argsList):
        if newFigures: pylab.figure()
        plotFunc(args)
        
        numStr = '%(c)05d' % {'c':i}
        name = tempFileDir+tempFilePrefix+numStr
        pylab.savefig(name+'.png')
        if newFigures: pylab.close()
        # convert png to gif using imagemagick's "convert"
        call(["convert",name+".png",name+".gif"])
        os.remove(name+".png")
        filenameList.append(name+".gif")

    # () assemble gifs into animation using gifsicle
    cout = open(filename,'wb')
    call(["gifsicle","--loop","-d",str(int(delay))]+                \
         filenameList,stdout=cout)
    cout.close()
    print "gifsiclePlot: GIF animation written to "+filename

    # () remove temporary files
    for file in filenameList:
        os.remove(file)

    # () open the file externally
    if openFile:
        call(["open",filename])
