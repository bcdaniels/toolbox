# defaultFigure.py
#
# 4.5.2011
# Bryan Daniels
#
# branched from numNonzeroFigure.py
#
# 3.3.2016 taken from SparsenessCode and elsewhere
#

import pylab
import scipy
import copy
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker # to access colorbar ticks

def setDefaultParams(usetex=False):
    # 4.23.2012 for PNAS (sizeMultiple=2)
    params = {'axes.labelsize': 16,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 14,
        'text.usetex': usetex,
    }
    pylab.rcParams.update(params)

def makePretty(leg=None,ax=None,cbar=None,cbarNbins=6,frameLW=0.5):
    if ax is None: ax = pylab.gca()
    # set frame linewidth
    [i.set_linewidth(frameLW) for i in ax.spines.itervalues()]
    # set tick length
    ax.tick_params('both',width=frameLW,length=2)
    if leg is not None:
        # set legend frame linewidth
        leg.get_frame().set_linewidth(frameLW)
    if cbar is not None:
        # same for colorbar
        ax2 = cbar.ax
        tick_locator = ticker.MaxNLocator(nbins=cbarNbins)
        cbar.locator = tick_locator
        cbar.update_ticks()
        [i.set_linewidth(frameLW) for i in ax2.spines.itervalues()]
        cbar.outline.set_linewidth(frameLW)
        ax2.tick_params(which='both',width=frameLW,length=2)

def texize(string):
    string2 = '\mathrm{'+string.replace(' ',' \: ')+'}'
    # 6.27.2011 make nicer hyphens in place of '--'
    string2 = string2.replace('--','\\mbox{-}')
    oldString = ''
    newString = string2
    while newString != oldString:
        oldString = copy.copy(newString)
        newString = oldString.replace("$","}",1)
        newString = newString.replace("$","\mathrm{",1)
    return "$"+newString+"$"

def drawColorbar(cmap,vmin=0.,vmax=1.,                      \
    width=0.05,height=0.9,bottom=None,left=None):
    if bottom is None: bottom = (1.-height)/2.
    if left is None: left = (1.-width)/2.
    ax = pylab.axes([left,bottom,width,height])
    c = ColorbarBase(ax,cmap=cmap,norm=Normalize(vmin,vmax))
    pylab.show()

# taken from ellStar.py
def prettyErrorbar(ax,xList,yList,yerr,**kwargs):
    """
    See prettyConfInt for kwargs.
    """
    a = pylab.array
    x,y,err = a(xList),a(yList),a(yerr)
    prettyConfInt(ax,xList,yList,y-err,y+err,**kwargs)

def prettyConfInt(ax,xList,yList,yListLower,yListUpper,
                  color='blue',alpha=0.15,marker='',ls='-',lw=2,label=None):
    ax.plot(xList,yList,color=color,marker=marker,ls=ls,
            mec=color,label=label,lw=lw)
    ax.fill_between(xList,yListLower,yListUpper,facecolors=color,alpha=alpha)


# 10.18.2017 colormap scaling functions taken from plotMutualInfo.py

# 8.31.2012
# see http://stackoverflow.com/questions/6492514/non-linear-scaling-of-a-colormap-to-enhance-contrast
def scaledCmap(func,cmap):
    """
    func            : A function that maps (0,1) to (0,1).
                      If it doesn't map 0 to 0 and 1 to 1,
                      it will be shifted and linearly
                      scaled to do so.
    """
    cdict = cmap._segmentdata
    cdictNew = copy.copy(cdict)
    newName = cmap.name + '_scaled'
    for color in cdict.keys():
        newList = []
        for x,y0,y1 in cdict[color]:
            xnew = scaleTo01(func)(x)
            newList.append((xnew,y0,y1))
        cdictNew[color] = newList
    return LinearSegmentedColormap(newName,cdictNew)

# 8.31.2012
def contrastCmap(cmap,contrast=10.):
    scaleFunc = lambda x: scipy.arctanh((2.*x-1.)/(1.+1./contrast))
    return scaledCmap(scaleFunc,cmap)

# 8.31.2012
def scaleTo01(func):
    """
    Shifts and linearly scales func such that scaledFunc(0) = 0
    and scaledFunc(1) = 1.
    
    Used in scaledCmap.
    """
    f = func
    def gfunc(x):
        if x <= 0.: return 0.
        elif x >= 1.: return 1.
        else:
            return 0.5*( 1. - (f(0.)+f(1.)-2.*f(x))/(f(1.)-f(0.)) )
    return gfunc


