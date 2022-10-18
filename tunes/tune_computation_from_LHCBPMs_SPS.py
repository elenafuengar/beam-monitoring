# Computation of bunch-by-bunch tunes (in both vertical and horizontal plane) in the SPS, using position data acquired with the BPMs.

import datascout as ds
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.dates as mdates
import matplotlib
from datetime import datetime
import matplotlib.pyplot as plt
from pytz import timezone
from matplotlib import cm
from scipy.signal import butter,filtfilt
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pandas as pd
import helper_functions as hf
from scipy.fft import fft, fftfreq, fft2, fftshift
import matplotlib.gridspec as gridspec
from harpy.harmonic_analysis import HarmonicAnalysis
from harpy.harmonic_analysis import HarmonicPlotter
plt.rcParams.update({'font.size': 10})
import numpy.matlib as npm
import os
from collections import deque
from scipy.optimize import curve_fit
import scipy.signal
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import math


# In the CERN SPS there are only three BPMs able to acquire bunch-by-bunch positions. 
# Selection of the BPM:

bpmSelection=[]
#bpmSelection.append('SPS.BPMB.51303')
#bpmSelection.append('SPS.BPMB.51503')
bpmSelection.append('SPS.BPMB.51999')


class PlottingClassSPS():

    def __init__(self,figsize,pngfolder=None,*args,**kw):
        self.pngfolder = pngfolder
        self.figsize = figsize
        self.figname = self.__class__.__name__
        self._nrows = 1
        self._ncols = 1
        self._sharex = False
        self._sharey = False

    def createFigure(self):
        self.figure = plt.figure(self.figname,figsize=self.figsize)
        plt.show(block=False)

    def initializeFigure(self):
        self.createFigure()
        self.drawFigure()

    def clearFigure(self):
        self.figure.clear()

    def removeLines(self):
        axs = self._getAxes()
        for ax in axs:
            for line in ax.get_lines():
                line.remove()

    def clearAxes(self):
        axs = self._getAxes()
        [ax.cla() for ax in axs]

    def createSubplots(self,*args,**kwargs):
        nrows = self._nrows
        ncols = self._ncols
        num = self.figname
        sharex = self._sharex
        sharey = self._sharey
        f,axs = plt.subplots(nrows,ncols,num=num,
                    sharex=sharex,sharey=sharey,*args,**kwargs)
        self.axs = axs
        return f, axs

    def _getAxes(self):
        axs = self.figure.get_axes()
        axs = [axs] if not isinstance(axs, list) else axs
        return axs

    def setFigureSize(self):
        self.figure.set_size_inches(*self.figsize) 

    def drawFigure(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def generateTitleStr(self,data):
        ts_str = hf.getCycleStampLocalTz(data).strftime('(%d.%m.%Y - %H:%M:%S) ')
        user = hf.getSelector(data)
        return user+' '+ts_str

    def saveFigure(self,filename):
        pass


class LHCBPM_HARMONIC_ANALYSIS(PlottingClassSPS):

    def __init__(self, device='BPLOFSBA5/GetCapData', 
            bpmSelection='SPS.BPMB.51999', n_turns_to_analyse = {'horPosition': 16, 'verPosition': 16},  
            *args,**kw):
        # In the SPS usually 16 turns are used for the frequency computation with the FFT, but more turns can be used for other accelerators like the LHC.
        figsize = (15,10)
        super().__init__(figsize,*args,**kw)
        self.figname = f"{self.__class__.__name__}" 
        self._nrows = 2
        self._ncols = 2
        self._sharex = False
        self._sharey = False
        self.initializeFigure()
        self.device = device
        self.bpmSelection = bpmSelection
        self.n_turns_to_analyse = n_turns_to_analyse

    def plot(self, data):
        hnum=10 # Harmonic number: Number of modes used to perform the Harmonic Analysis with HarPy. If only the main frequency (tune) is wanted, just put hnum=1. 
        # hnum > 1 is useful to find lower amplitude peaks close to the main one. 
        caseUT='Plots/'
        cmap = matplotlib.cm.get_cmap('Spectral_r')

        if not plt.fignum_exists(self.figname):
            return
        self.clearFigure()
        plt.ion()
        figure, axs = self.createSubplots()
        axs_dict = {'horPositionraw': axs[0, 0], 'horPositionTunes': axs[0, 1], 'verPositionraw': axs[1, 0], 'verPositionTunes': axs[1, 1]}
        d = self.reshapeData(data[self.device]['value'])
        bpm = self.bpmSelection
        bpm_sel = d['bpmNames'].tolist().index(bpm)
        capScTime = data[self.device]['value']['capScTime']

        for par in ['horPosition', 'verPosition']:        
            arr = []    # Array of positions
            tunes = []  # Array of tunes
        
            msk = np.where(np.std(d[par][bpm_sel],axis=0))[0] # Mask to find the indexes of the bunches with signal.
            batch_upper_limit = np.where(np.diff(msk)>1)[0] + 1 
            batch_msk = np.split(msk, batch_upper_limit)
            batch_length = len(batch_msk[0])
            n_batches = len(batch_msk) # Number of batches 
        
            if par == 'horPosition':
                ini_turn = min(np.nonzero(d[par][bpm_sel][:,msk])[0]) + 14 # I find where the oscillations start, and additionally I can skip some more turns if the signal is not good enough at the beginning (in the H plane).
            else:
                ini_turn = min(np.nonzero(d[par][bpm_sel][:,msk])[0]) + 13 # I find where the oscillations start, and additionally I can skip some more turns if the signal is not good enough at the beginning (in the V plane).
          
            fin_turn = ini_turn + self.n_turns_to_analyse[par]
            tbt_data_all_raw = d[par][bpm_sel][ini_turn:fin_turn,msk] # Turn by turn data 
            tbt_data_all = tbt_data_all_raw - (tbt_data_all_raw).mean(axis=0) # Turn by turn data centered at zero (this correction is needed to compute the frequency/tune afterwards).
            number_of_bunches = tbt_data_all.shape[1]
            
            label = par.replace('horPosition','Horizontal').replace('verPosition','Vertical') # Label for the plots' axes
            
            ax_raw = axs_dict[f'{par}raw']
            ax_raw.text(0.5,0.9,bpm,ha='center',va='bottom',transform=ax_raw.transAxes, color='black')            
            ax_raw.set_xlabel('Turn',fontsize=14)
            ax_raw.set_ylabel(label + ' Position (ALL batches) [mm]',fontsize=14)

            ax_res = axs_dict[f'{par}Tunes']
            ax_res.set_ylabel(label + ' fractional tune',fontsize=18)
            ax_res.set_xlabel('Bunch number',fontsize=18)


            
            # Computation of the tune
            for b in range(number_of_bunches): 
                arr.append(tbt_data_all[:,b])
                ha = HarmonicAnalysis(arr[-1][:]) 
                tunes.append(ha.laskar_method(num_harmonics=hnum)[0][0]) # Tunes of each bunch. 
                      
            tunes = np.abs(np.array(tunes)-np.round(tunes)) # I only save the decimal part of the tune.
            median_tunes = np.zeros(n_batches) # Array where the median tune of each batch will be saved.
            text = 'Median tunes = '
            for i in range(n_batches):
                median_tunes[i] = np.median(tunes[i*batch_length:(i+1)*batch_length])
                text += '%.3f, '%median_tunes[i]
            arr = np.array(arr) 

            ax_raw.plot(arr.T) 
            plt.ticklabel_format(axis='both', style='sci', scilimits=(-4,4))
            plt.tight_layout()

            # To compute the frequency of the peaks with lower amplitude than the main frequency 
            arr = []
            tunesx = []
            ampx= []

            for b in range(number_of_bunches): 
                arr.append(tbt_data_all[:,b])
                ha = HarmonicAnalysis(arr[-1][:])
                tunesx.append(ha.laskar_method(num_harmonics=hnum)[0])
                ampx.append(ha.laskar_method(num_harmonics=hnum)[1])

            font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 14}
        
            # To show the real bunch number in the plot
            nbOfCapBunches = data[self.device]['value']['nbOfCapBunches']
            nbOfCapTurns = data[self.device]['value']['nbOfCapTurns']
            hor_id = data[self.device]['value']['horBunchId'][bpm_sel]
            bunch_ids = hor_id.reshape(nbOfCapTurns,nbOfCapBunches)
            bunch_number = bunch_ids[0][msk]

            x=np.transpose(np.matlib.repmat(bunch_number,hnum,1))
            yx=[]

            for pp in range(number_of_bunches):
                for ii in range(hnum):
                    yx0=np.abs(tunesx[pp][ii]-round(tunesx[pp][ii]))
                    yx.append(yx0)

            yx=(np.reshape(yx,[number_of_bunches,hnum]))
            zx=20*np.log10(np.abs((ampx))/np.max(np.abs((ampx))))

            i_sorted = np.argsort(zx.flatten())[::1] 
            x_sorted = x.flatten()[i_sorted]
            y_sorted = yx.flatten()[i_sorted]
            z_sorted = zx.flatten()[i_sorted]

            # Add colorbar        
            divider = make_axes_locatable(ax_res)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im=ax_res.scatter(x_sorted,y_sorted,c=z_sorted, cmap='gray_r', vmin=-18, vmax=0)
            ax_res.plot(bunch_number, tunes,'r.',label='Computed tune')
            cbar = plt.colorbar(im, cax=cax)    
            cbar.ax.set_ylabel('Relative FFT Amp. [dB]', fontsize=18)
            if ax_res == axs_dict['horPositionTunes']:
                ax_res.set_ylim(0.1,0.18)
            else:
                ax_res.set_ylim(0.05,0.3)   

            ax_res.text(0.94,0.9, text ,ha='right',va='top',transform=ax_res.transAxes, color='tab:blue', weight='bold', bbox = dict(facecolor = 'white', alpha = 0.6))

            ax_res.tick_params(labelsize=17)
            cbar.ax.tick_params(labelsize=16)

        title = self.generateTitleStr(data[self.device])
        self.figure.suptitle(title, fontsize=16)
        plt.tight_layout()
        #plt.savefig(caseUT+title +'.png')
        self.drawFigure()


    # Function needed to reshape the data in the .paquet files since the information of the bunches position is stored in a 1-D array
    def reshapeData(self,data):
        nbOfCapBunches = data['nbOfCapBunches']
        nbOfCapTurns = data['nbOfCapTurns']
        for par in ['horPosition','verPosition']:
            data[par]=[pos.reshape(nbOfCapTurns,nbOfCapBunches) for pos in data[par]]
        return data


# Uncomment the following part in order to do the analysis in real time in the CERN CCC.
"""
from pyjapcscout import PyJapcScout
from helper_functions import callback_core
import PlottingClassesSPS as pc


def outer_callback(plot_func_list):
    def callback(data, h):
        callback_core(data, h, plot_func_list)
    return callback


# Set up monitor
selector = 'SPS.USER.HIRADMT2' #'SPS.USER.MD5'
device = 'BPLOFSBA5/GetCapData'
bpmSelection = 'SPS.BPMB.51999'
plot_func_list = [LHCBPM_HARMONIC_ANALYSIS(bpmSelection=bpmSelection)]


# start PyJapcScout and so incaify Python instance
myPyJapc = PyJapcScout(incaAcceleratorName='SPS')

myMonitor = myPyJapc.PyJapcScoutMonitor(selector, device,
                   onValueReceived=outer_callback(plot_func_list))#,
                   #groupStrategy='extended',strategyTimeout=10000)

# Saving data configuration
myMonitor.saveDataPath = f'./data/lhcbpm/{selector}'
myMonitor.saveData = True
myMonitor.saveDataFormat = 'parquet'

# Start acquisition
myMonitor.startMonitor()
"""

# Using files already existing and saved.

lchbpm = LHCBPM_HARMONIC_ANALYSIS(bpmSelection='SPS.BPMB.51999', n_turns_to_analyse = {'horPosition': 32, 'verPosition': 16})
#data = ds.parquet_to_dict('/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/lhcbpm/SPS.USER.LHC1/2022.06.03.17.30.53.562717.parquet')
#data = ds.parquet_to_dict('/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/lhcbpm/SPS.USER.LHC1/2022.06.03.16.30.51.232588.parquet')
#data = ds.parquet_to_dict('/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/lhcbpm/SPS.USER.LHC1/2022.06.01.12.55.32.114758.parquet')
data = ds.parquet_to_dict('/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/lhcbpm/SPS.USER.LHC25NS/2022.04.20.20.34.34.275787.parquet')

lchbpm.plot(data)



