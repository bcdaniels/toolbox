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

def makePretty(leg=None,ax=None):
    if ax is None: ax = pylab.gca()
    # set frame linewidth
    [i.set_linewidth(0.5) for i in ax.spines.itervalues()]
    # set tick length
    ax.tick_params('both', length=2)
    if leg is not None:
        # set legend frame linewidth
        leg.get_frame().set_linewidth(0.5)

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