'''
Growth rate analysis
----

Script for growth rate analysis of the vertical plane OFFLINE

* Click on the instability start when the first plot pops
* Run using ipython>run analysis_growthrate.py

*Oct 22*
'''

import os
import glob
import time
import datascout as ds
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import csv

# LSA function settings
cycle_offset = 4015 #default 1015
cycle_start = 200
cycle_end = 3000


# helper functions
def get_QP_knob():
    global parquet

    data = ds.parquet_to_dict(parquet)

    QPV = data['SPSBEAM/QPV']
    QPH = data['SPSBEAM/QPH']
    QPHval = get_LSA_time_value(QPH)
    QPVval = get_LSA_time_value(QPV)
    return QPHval, QPVval

def get_LSA_time_value(dict, cycle_offset=cycle_offset, cycle_start=cycle_start, cycle_end=cycle_end):
    X = dict['value']['JAPC_FUNCTION']['X'] - cycle_offset
    Y = dict['value']['JAPC_FUNCTION']['Y']
    vals = Y[X > cycle_start]
    return float(vals[0])

def onclick(event):
    global grstart
    global flag_unstable_beam

    flag_unstable_beam = True
    grstart = event.xdata
    plt.close()

def get_intensity(parquet):
    global flag_intensity
    global xmax
    global grstart
    global xstart
    global xend
    global xinty
    global yinty
    global exp
    global grinty

    global md
    i_path = f'/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/bct/SPS.USER.{md}/'
    
    #gr parquet
    gr = ds.parquet_to_dict(parquet)
    vpos=gr['SPS.BQ.KICKED/ContinuousAcquisition']['value']['rawDataV']
    revf=gr['SPS.BQ.KICKED/ContinuousAcquisition']['value']['frevFreq']
    turns=range(len(vpos))
    ms=turns/np.mean(revf)

    filename = parquet.split('/')[-1]
    time_stamp = filename.split('.parquet')[0]
    time = time_stamp.split('.')
    filter = time[0]+'.'+time[1]+'.'+time[2]+'.'+time[3]+'.'+time[4]+'.'+time[5][0]+'*'

    #plot
    global xmax
    global xmin
    global QPH_knob
    global QPV_knob
    fig, ax = plt.subplots(1,1, num=1)
    fig.tight_layout()
    fig.set_size_inches(6,6)
    ax.plot(ms, vpos, color='b')
    ax.set_xlim(xmin, xmax)
    #ax.set_ylim(-1.0e8, 1.0e8)
    ax.set_ylabel('Vertical position', color = 'b')
    ax.set_xlabel('Time [ms]')
    ax.tick_params(axis='y', colors='b')
    ax.set_title('QPH'+str(QPH_knob)+'_QPV'+str(QPV_knob)+'N3e10', fontweight='bold')

    #get instability start 
    grstart = 0 
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    xstart = np.where(ms >= grstart)[0][0]
    xpos = np.where(ms >= grstart)[0]
    xend = np.argmax(vpos[xpos])+xstart+1

    try:
        #intensity 
        inty_parquet = sorted(glob.glob(i_path+filter))[0]
        inty = ds.parquet_to_dict(inty_parquet)

        yinty=inty['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity']
        xinty=inty['SPS.BCTDC.51456/Acquisition']['value']['measStamp']
        exp=inty['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity_unitExponent']
        grinty = yinty[np.where(xinty >= grstart)[0][0]]

        flag_intensity = True
        
    except:
        print(f'intentsity file for time: {filter} not found')
        flag_intensity = False


def run_STFT_analysis(parquet):

    global path
    global folder
    global flag_unstable_beam

    cwd = path
    flag_unstable_beam = False

    # Read file
    print('    - analysing ' + parquet.split('/')[-1] + ' file...')
    data = ds.parquet_to_dict(parquet)
    filename = parquet.split('/')[-1]
    time_stamp = filename.split('.parquet')[0]

    #check empty parquet
    try:
        vpos=data['SPS.BQ.KICKED/ContinuousAcquisition']['value']['rawDataV']
    except:
        return

    # Retrieve data
    acqPeriod = data['SPS.BQ.KICKED/ContinuousAcquisition']['header']['acqStamp']
    vpos=data['SPS.BQ.KICKED/ContinuousAcquisition']['value']['rawDataV']
    revf=data['SPS.BQ.KICKED/ContinuousAcquisition']['value']['frevFreq']
    turns=range(len(vpos))
    ms=turns/np.mean(revf)

    # Start analysis
    get_intensity(parquet)

    if flag_unstable_beam:
        
        #plot the raw position
        global xmax
        global xmin
        global grstart
        global xstart
        global xend

        fig, (ax1, ax3) = plt.subplots(2, 1, num=1)
        ax1.axhline(grstart,c='r')
        ax1.plot(ms, vpos, color='b', alpha = 0.5)
        ax1.plot(ms[xstart:xend], vpos[xstart:xend], color='b') 
        ax1.axvline(grstart,c='r')
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(-np.max(vpos[xstart:xend]), np.max(vpos[xstart:xend])*1.2)
        ax1.set_ylabel('Vertical position', color = 'b')
        ax1.set_xlabel('Time [ms]')
        ax1.tick_params(axis='y', colors='b')
        
        #plot intensity
        global flag_intensity
        if flag_intensity:

            global xinty
            global yinty
            global exp
            global grinty

            ax2=ax1.twinx()
            ax2.plot(xinty, yinty*10**exp, color='orange', alpha = 0.5)
            ax2.set_ylabel('Intensity [p/b]', color = 'orange')
            ax2.text(0.1, 0.1, 'Intensity = %.2E' %(grinty*10**exp),transform=ax2.transAxes,va='top', color='orange', bbox=dict(facecolor = 'white', alpha = 0.9, edgecolor='orange'), fontsize=11)
            ax2.tick_params(axis='y', colors='orange')

        # Perform STFT
        global nperseg
        global noverlap
        
        pad = 5
        vposgr = vpos[xstart-pad:xend+pad]
        MWFFT = scipy.signal.stft(vpos, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
        MWFFTgr = scipy.signal.stft(vposgr, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
        conv = int(nperseg - noverlap)
        points = MWFFT[1]

        # Obtain the log of the STFT amplitude
        log_peak_amplitude = []
        log_peak_amplitude_gr = []
        for i in range(len(MWFFT[1])):
            log_peak_amplitude.append(np.log(max(np.abs(MWFFT[2][:,i]))))
        for i in range(len(MWFFTgr[1])):
            log_peak_amplitude_gr.append(np.log(max(np.abs(MWFFTgr[2][:,i]))))

        log_peak_amplitude = np.array(log_peak_amplitude)
        gry = np.array(log_peak_amplitude_gr)[pad:-pad]
        grx = MWFFTgr[1][pad:-pad]+xstart

        # linear fit
        coef, cov = np.polyfit(grx/np.mean(revf) , gry, 1, cov=True)
        poly1d_fn = np.poly1d(coef)
        error= np.sqrt(np.diag(cov)[0])

        ax3.plot(points/np.mean(revf),log_peak_amplitude,'.-b', label='MWFFT')
        ax3.plot(grx/np.mean(revf) , gry, '.-r', label='points for fit')
        ax3.plot(grx/np.mean(revf) ,poly1d_fn(grx/np.mean(revf)), '--k', lw=2.0, label='fit masked data')

        ax3.text(0.1,0.1," $\u03C4^{-1}$ = Slope = %.2E +/- %.E" %(coef[0]/np.mean(revf), error),va='top',transform=ax3.transAxes, color='black',\
        bbox=dict(boxstyle='round',facecolor='white',alpha=0.9), fontsize=11)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('log (Amplitude of Peak in FFT)', color = 'red')
        ax3.tick_params(axis='y', colors='red')
        ax3.set_xlim(xmin, xmax)
        ax3.legend()

        global QPH_knob
        global QPV_knob

        slope = float(coef[0])/np.mean(revf)
        uncertainty = float(error)
        nturns = len(vpos)

        fig.suptitle('QPH'+str(QPH_knob)+'_QPV'+str(QPV_knob)+'N3e10', fontweight='bold')
        fig.set_size_inches(6,10)
        fig.tight_layout()

        '''
        print(f'    QPH = {QPH_knob} , QPV = {QPV_knob}')
        print('    GR = %.2E' %slope)
        print('    Intensity = %.2E' %(grinty*10**exp))
        '''
        plt.show()

        # Save data in csv
        global fname
        header=['time', 'QPH', 'QPV', 'grstart', 'grend', 'slope', 'intensity', 'error', 'nturns', 'acqPeriod']
        if not os.path.exists(fname):
            with open(fname, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                f.close()

        if flag_intensity:
            with open(fname, 'a') as f:  
                writer=csv.writer(f, delimiter=',')
                writer.writerow([time_stamp, QPH_knob, QPV_knob, grstart, ms[xend], slope, grinty, uncertainty, nturns, acqPeriod])
        else: 
            with open(fname, 'a') as f:  
                writer=csv.writer(f, delimiter=',')
                writer.writerow([time_stamp, QPH_knob, QPV_knob, grstart, ms[xend], slope, np.nan,  uncertainty, nturns, acqPeriod])

        # Save fig
        if not os.path.exists(path+folder+'plots/'):
            os.mkdir(path+folder+'plots/')

        plt.savefig(path+folder+'plots/'+time_stamp +'.png')

    else: 
        print('     [WARNING] File '+ time_stamp + ' has no unstable beam ')

#------------------------------------------------------------------------
# Set up

#----- User variables -----#

#path parameters
md = 'MD3'
date = '2022-08-30' 
folder = 'intensity/'                       #folder to save the plots and the csv
fname = 'growthrate_220830.csv'   # csv title to save the values

#FFT parameters
nperseg = 65    #nturns per window in the MWFFT
noverlap = 60   #overlap property in the MWFFT

# visualization
xmin = 5     #default is 5 ms
xmax = 100     #ms to plot of the raw data

#--------------------------#

print('Start analysis:')

path ='/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/growthrate/SPS.USER.'+md+'/'+date+'/'


# to analyze only one file at a time
parquet=input('input parquet name: ')
parquet_list=[path+parquet] 


# to analyze all parquets of the day
#parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime) 
#parquet_list=sorted(glob.glob(path+'QPH0.05_QPV-1.7N3e10/*'), key=os.path.getmtime) #TODO


for parquet in parquet_list:
    QPH_knob, QPV_knob = get_QP_knob()
    run_STFT_analysis(parquet)
    #plt.close()

print('Analysis finished')
#-----------------------------------------------------------------------

