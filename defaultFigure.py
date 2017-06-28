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
import copy
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
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


