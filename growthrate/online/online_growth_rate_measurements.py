'''
Script for online growth rate aalysis of the vertical plane

* First run the acquisition script ipython>run record_GR_{MD}.py
* Check the path
* Set the variable max_count for the number of consecutive cycles to analyze
* Run using ipython>run online_growth_rate_measurements.py

Aug 22
'''

import os
import glob
import time
import datascout as ds
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import csv
import math
from scipy.interpolate import interp1d
from datetime import date
import csv

# LSA function settings
cycle_offset = 2015 #default 1015
cycle_start = 200
cycle_end = 3000


# helper functions
def get_QP_knob():
    global path
    parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime)
    parquet = parquet_list[-1]

    data = ds.parquet_to_dict(parquet)

    QPV = data['SPSBEAM/QPV']
    QPH = data['SPSBEAM/QPH']
    QPHval = get_LSA_time_value(QPH)
    QPVval = get_LSA_time_value(QPV)
    return QPHval, QPVval


def get_LSA_time_value(dict, cycle_offset=2000, cycle_start=cycle_start, cycle_end=cycle_end):
    X = dict['value']['JAPC_FUNCTION']['X'] - cycle_offset
    Y = dict['value']['JAPC_FUNCTION']['Y']
    vals = Y[X > cycle_start]
    return float(vals[0])

def run_STFT_analysis(parquet):

    global path
    global folder
    global skip_turns

    cwd = path
    ni = skip_turns #mask injection noise

    # Read file
    print('    - analysing ' + parquet.split('/')[-1] + ' file...')
    data = ds.parquet_to_dict(parquet)
    filename = parquet.split('/')[-1]
    time_stamp = filename.split('.parquet')[0]


    # Retrieve data
    vertical_position_all=data['SPS.BQ.KICKED/ContinuousAcquisition']['value']['rawDataV']
    nf = len(vertical_position_all)
    if nf > 2**17: #for duplicated signals
        nf = nf//2
    vertical_position = vertical_position_all[ni:nf]
    acqPeriod = data['SPS.BQ.KICKED/ContinuousAcquisition']['header']['acqStamp']
    turns =np.arange(ni, nf, 1)

    # Start analysis
    if max(vertical_position) > 1e5:

        # mask noise before instability by 2%
        mask_noise = abs(vertical_position) > max(vertical_position)*0.02
        igr = np.where(mask_noise)[0]

        #plot the raw position
        global xlim
        global ylim
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8,11), num=time_stamp)
        ax1.set_xlabel('Number of turns')
        ax1.set_ylabel('Vertical position')
        ax1.plot(vertical_position_all, 'k')
        ax1.plot(turns[mask_noise], vertical_position[mask_noise], 'b') #masked noise
        ax1.set_xlim(1000,xlim)
        ax1.set_ylim(-ylim,ylim)

        # Perform STFT
        global nperseg
        global noverlap

        MWFFT = scipy.signal.stft(vertical_position[mask_noise], fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)

        # Obtain the log of the STFT amplitude
        log_peak_amplitude = []
        for i in range(len(MWFFT[1])):
            log_peak_amplitude.append(np.log(max(np.abs(MWFFT[2][:,i]))))

        log_peak_amplitude = np.array(log_peak_amplitude)
        ax3.set_xlabel('Number of STFT point')
        ax3.set_ylabel('log (Amplitude of Peak in FFT)')
        ax3.plot(MWFFT[1],log_peak_amplitude,'.-b', label='measurements')

        global flag_manual
        if flag_manual:
            global nstart
            global nend
            gr = log_peak_amplitude[nstart:nend]
            points = MWFFT[1][nstart:nend]

            coef, cov = np.polyfit(points, gr,1, cov=True)
            poly1d_fn = np.poly1d(coef)
            error= np.sqrt(np.diag(cov)[0])

            ax3.plot(points , gr, '.-r', label='mask')
            ax3.plot(points ,poly1d_fn(points), '--k', lw=2.0, label='fit masked data')
            ax3.text(0.82,0.4," $\u03C4^{-1}$ = Slope = %.2E +/- %.E" %(coef[0], error),ha='right',va='top',transform=ax3.transAxes, color='black',\
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.9), fontsize=12)

        else:
            # Mask log amplitude after maximum
            imax = np.argmax(log_peak_amplitude)
            mask_amp = log_peak_amplitude > 0
            mask_amp[imax+1:]=False

            # Mask log_pak_amplitude to get only values with positive slope
            df = MWFFT[1][2]-MWFFT[1][1]
            ddf = np.zeros_like(log_peak_amplitude)
            for i in range(2, len(MWFFT[1])-2):
                #fourth order scheme
                ddf[i+1] = (-log_peak_amplitude[i+2]+8*log_peak_amplitude[i+1]-8*log_peak_amplitude[i-1]+log_peak_amplitude[i-2])/(12*df)

            mask_pos=ddf > 0

            # Obtain final mask and plot
            mask = np.logical_and(mask_pos, mask_amp)
            gr = log_peak_amplitude[mask]
            points = MWFFT[1][mask]

            if np.count_nonzero(mask) > 2.0:

                # Perform the first linear fit 
                global ifit
                coef, cov = np.polyfit(points[ifit:], gr[ifit:],1, cov=True)
                poly1d_fn = np.poly1d(coef)
                error= np.sqrt(np.diag(cov)[0])

                if abs(float(error)/float(coef[0])) > 0.01: #for bad signals, take only the last ifit points
                    print('Fit with large error, taking only the last 10 points')
                    global efit
                    ifit = len(gr)-efit
                    coef = np.polyfit(points[ifit:], gr[ifit:],1, cov=False)
                    poly1d_fn = np.poly1d(coef)
                    ax3.plot(points[ifit:] , gr[ifit:], '.-r', label='mask')
                    ax3.plot(points[ifit:] ,poly1d_fn(points[ifit:]), '--k', label='fit masked data')

                else:
                    ax3.plot(points[nfit:] , gr[nfit:], '.-r', label='mask')
                    ax3.plot(points[nfit:] ,poly1d_fn(points)[nfit:], '--k', lw=2.0, label='fit masked data')

                ax3.text(0.82,0.4," $\u03C4^{-1}$ = Slope = %.2E +/- %.E" %(coef[0], error),ha='right',va='top',transform=ax3.transAxes, color='black',\
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.9), fontsize=12)

                global QPH_knob
                global QPV_knob

                slope = float(coef[0])
                uncertainty = float(error)
                nturns = len(vertical_position)
                plt.title('QPH'+str(QPH_knob)+'_QPV'+str(QPV_knob)+'N3e10')
                plt.legend()
                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                print(f' GR = {slope}')

                # Save data in csv
                global fname
                header=['time', 'QPH', 'QPV', 'slope','error', 'nturns', 'acqPeriod']
                if not os.path.exists(path+folder+fname):
                    f=open(path+folder+fname, 'w')
                    writer = csv.writer(f)
                    writer.writerow(header)
                    f.close()

                with open(path+folder+fname, 'a') as f:
                    writer=csv.writer(f, delimiter=',')
                    writer.writerow([time_stamp, QPH_knob, QPV_knob, slope, uncertainty, nturns, acqPeriod])

                # Save fig
                if not os.path.exists(path+folder+'plots/'):
                    os.mkdir(path+folder+'plots/')
                plt.savefig(path+folder+'plots/'+time_stamp +'.png')

            else:
                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()

                print('[WARNING] File '+ time_stamp + ' contains no valid data for fit')
    else:
        print('[WARNING] File '+ time_stamp + ' has no beam ')



def move_files_to_folder(files, folder):
    global path
    if not os.path.exists(path+folder):
        os.mkdir(path+folder)
    for file in files:
        os.replace(file, path+folder+'/'+file.split('/')[-1])

#------------------------------------------------------------------------

#----- User variables -----#

#number of files to analyze
count_max = 1

#FFT parameters
nperseg = 100             #nturns per window in the MWFFT
noverlap = 60            #overlap property in the MWFFT
skip_turns = 0       #nturns to skip for the MWFFT

# automatic fit
ifit = 0              #points to skip before the fit
efit = 100             #points to use before the maximum

# manual fit
flag_manual = False    #activate the manual fit
nstart = 57            #where to start the fit
nend = 67            #where to end the fit

# visualization
xlim = 120000             #number of turns to plot of the raw data
ylim = 1e6

#csv
fname = 'growth_rate_values.csv'

#--------------------------#

print('Start analysis:')

# Set up path
date = str(date.today())
path ='/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/growthrate/SPS.USER.MD3/'+date+'/'

# set folder name
QPH_knob, QPV_knob = get_QP_knob()
folder = 'QPH'+str(QPH_knob)+'_QPV'+str(QPV_knob)+'N7e9'

parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime)
parquet = parquet_list[-1]
parquets_analyzed = []
count = 0
plt.ion()

while count < count_max:
    count+=1
    run_STFT_analysis(parquet)
    
    if count_max > 1:
        while True:
            parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime)
            if parquet_list[-1] != parquet:
                parquets_analyzed.append(parquet)
                parquet = parquet_list[-1]
                time.sleep(2)
                break
    
    time.sleep(2)
    plt.close()
    
plt.close()
move_files_to_folder(parquets_analyzed, folder)

print('Analysis finished')
#-----------------------------------------------------------------------

